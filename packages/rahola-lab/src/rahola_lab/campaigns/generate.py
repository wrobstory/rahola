"""Chunked deterministic reference-campaign generation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from rahola.simulate import simulate_batch
from rahola.storage import write_dataset
from rahola_lab.campaigns.definition import CampaignDefinition
from rahola_lab.evaluation.splits import seeds_for


@dataclass(frozen=True)
class GenerationResult:
    manifest_path: Path
    elapsed_s: float
    bytes_written: int
    capsize_fractions: dict[str, float]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_campaign(
    definition: CampaignDefinition,
    output_root: str | Path,
    *,
    chunk_size: int = 256,
) -> GenerationResult:
    """Generate all declared non-reserve splits without holding a campaign in memory."""
    started = time.perf_counter()
    root = Path(output_root) / definition.name
    root.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, object] = {}
    fractions: dict[str, float] = {}
    for split in definition.splits:
        seeds = seeds_for(split.block, split.count, offset=split.offset)
        chunks: list[dict[str, object]] = []
        capsized = 0
        for chunk_number, start in enumerate(range(0, len(seeds), chunk_size)):
            chunk_seeds = seeds[start : start + chunk_size]
            dataset = simulate_batch(definition.simulation, chunk_seeds)
            capsized += int(dataset.capsized.sum())
            chunk_dir = root / str(split.block) / f"chunk-{chunk_number:05d}"
            manifest = write_dataset(dataset, chunk_dir, shard_size=chunk_size)
            chunks.append(
                {
                    "path": str(manifest.relative_to(root)),
                    "sha256": _sha256(manifest),
                    "rows": len(chunk_seeds),
                }
            )
        fractions[str(split.block)] = capsized / split.count
        split_records[str(split.block)] = {
            "count": split.count,
            "offset": split.offset,
            "capsized": capsized,
            "capsize_fraction": fractions[str(split.block)],
            "chunks": chunks,
        }
    top_manifest = {
        "schema_version": 1,
        "name": definition.name,
        "role": definition.role,
        "rationale": definition.rationale,
        "simulation": definition.simulation.to_dict(),
        "splits": split_records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(top_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    bytes_written = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return GenerationResult(
        manifest_path=manifest_path,
        elapsed_s=time.perf_counter() - started,
        bytes_written=bytes_written,
        capsize_fractions=fractions,
    )
