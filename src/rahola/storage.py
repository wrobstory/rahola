"""Deterministic sharded Parquet storage with checksummed JSON manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rahola.dataset import SimulationDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(
    dataset: SimulationDataset, output_dir: str | Path, *, shard_size: int = 256
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, str | int]] = []
    time_values = dataset.time_s.tolist()
    for shard_number, start in enumerate(range(0, dataset.batch_size, shard_size)):
        end = min(start + shard_size, dataset.batch_size)
        path = output / f"part-{shard_number:05d}.parquet"
        table = pa.table(
            {
                "seed": pa.array(dataset.seeds[start:end], type=pa.uint64()),
                "capsized": pa.array(dataset.capsized[start:end], type=pa.bool_()),
                "t_capsize_s": pa.array(dataset.t_capsize_s[start:end], type=pa.float64()),
                "time_s": pa.array([time_values] * (end - start), type=pa.list_(pa.float64())),
                "angle_rad": pa.array(
                    dataset.angle_rad[start:end].tolist(), type=pa.list_(pa.float64())
                ),
                "rate_rad_s": pa.array(
                    dataset.rate_rad_s[start:end].tolist(), type=pa.list_(pa.float64())
                ),
                "metadata_json": pa.array(
                    [
                        json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        for item in dataset.metadata[start:end]
                    ],
                    type=pa.string(),
                ),
            }
        )
        pq.write_table(
            table,
            path,
            compression="NONE",
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
        )
        shards.append({"file": path.name, "rows": end - start, "sha256": _sha256(path)})
    counts = Counter(
        f"{item['family']}:{item['protocol']}:{'capsized' if item['capsized'] else 'safe'}"
        for item in dataset.metadata
    )
    manifest = {
        "schema_version": 1,
        "config": dataset.config,
        "seeds": [int(seed) for seed in dataset.seeds],
        "git_commit": dataset.metadata[0]["git_commit"],
        "package_version": dataset.metadata[0]["package_version"],
        "shards": shards,
        "summary": {
            "trajectories": dataset.batch_size,
            "capsized": int(dataset.capsized.sum()),
            "counts": dict(sorted(counts.items())),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path
