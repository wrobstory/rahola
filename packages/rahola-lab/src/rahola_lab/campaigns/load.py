"""Load a generated campaign split into the core dense dataset type."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rahola.config import SimulationConfig
from rahola.dataset import SimulationDataset
from rahola_lab.constants import SEED_BLOCK_START, SeedBlock
from rahola_lab.evaluation.splits import ReserveBlockError, assert_seed_membership

_REFERENCE_CHECKSUMS = json.loads(
    Path(__file__).with_name("reference_checksums.json").read_text(encoding="utf-8")
)


def _read_verified_bytes(path: Path, expected: str, *, kind: str) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        blocks = []
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            blocks.append(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"{kind} hash mismatch for {path}: expected {expected}, got {actual}")
    return b"".join(blocks)


def _contained_path(root: Path, base: Path, relative: object, *, kind: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{kind} path must be a nonempty relative string")
    candidate = base / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{kind} path escapes campaign root: {relative!r}") from error
    return candidate


def load_campaign_split(
    campaign_dir: str | Path,
    block: SeedBlock | str,
    *,
    limit: int | None = None,
    allow_unanchored: bool = False,
) -> SimulationDataset:
    """Load an anchored reference split and verify every stored seed belongs to it."""
    selected = SeedBlock(block)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if selected in {SeedBlock.RESERVE, SeedBlock.RESERVE2}:
        raise ReserveBlockError(f"{selected} data may not be inspected by development paths")
    root = Path(campaign_dir)
    manifest_path = root / "manifest.json"
    if not allow_unanchored and root.name not in _REFERENCE_CHECKSUMS:
        raise ValueError(f"campaign {root.name!r} has no tracked reference anchor")
    if not allow_unanchored:
        manifest_bytes = _read_verified_bytes(
            manifest_path, _REFERENCE_CHECKSUMS[root.name], kind="anchored reference manifest"
        )
    else:
        manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = manifest["splits"][str(selected)]
    expected_total = int(records["count"])
    declared_total = sum(int(chunk["rows"]) for chunk in records["chunks"])
    if declared_total != expected_total:
        raise ValueError(
            f"split row-count mismatch: expected {expected_total}, chunks declare {declared_total}"
        )
    expected_loaded = expected_total if limit is None else min(limit, expected_total)
    expected_config_hash = SimulationConfig.from_dict(manifest["simulation"]).config_hash
    seeds: list[int] = []
    capsized: list[bool] = []
    cap_times: list[float] = []
    angle: list[list[float]] = []
    rate: list[list[float]] = []
    metadata: list[dict[str, object]] = []
    time_s: np.ndarray | None = None
    for chunk in records["chunks"]:
        chunk_manifest_path = _contained_path(
            root, root, chunk["path"], kind="chunk manifest"
        )
        chunk_manifest = json.loads(
            _read_verified_bytes(
                chunk_manifest_path, chunk["sha256"], kind="chunk manifest"
            )
        )
        for shard in chunk_manifest["shards"]:
            shard_path = _contained_path(
                root, chunk_manifest_path.parent, shard["file"], kind="Parquet shard"
            )
            shard_bytes = _read_verified_bytes(
                shard_path, shard["sha256"], kind="Parquet shard"
            )
            table = pq.read_table(pa.BufferReader(shard_bytes))
            if table.num_rows != int(shard["rows"]):
                raise ValueError(f"row-count mismatch for {shard_path}")
            payload = table.to_pydict()
            for row in range(table.num_rows):
                if limit is not None and len(seeds) >= limit:
                    break
                row_time = np.asarray(payload["time_s"][row], dtype=np.float64)
                if time_s is None:
                    time_s = row_time
                elif not np.array_equal(row_time, time_s):
                    raise ValueError(f"time-vector mismatch for row {row} in {shard_path}")
                row_metadata = json.loads(payload["metadata_json"][row])
                if row_metadata.get("config_hash") != expected_config_hash:
                    raise ValueError(f"metadata config hash mismatch for row {row} in {shard_path}")
                seeds.append(payload["seed"][row])
                capsized.append(payload["capsized"][row])
                cap_times.append(payload["t_capsize_s"][row])
                angle.append(payload["angle_rad"][row])
                rate.append(payload["rate_rad_s"][row])
                metadata.append(row_metadata)
            if limit is not None and len(seeds) >= limit:
                break
        if limit is not None and len(seeds) >= limit:
            break
    if time_s is None:
        raise ValueError("campaign split contains no trajectories")
    if len(seeds) != expected_loaded:
        raise ValueError(f"loaded {len(seeds)} rows; expected {expected_loaded}")
    seed_array = np.asarray(seeds, dtype=np.uint64)
    assert_seed_membership(seed_array, selected)
    split_start = SEED_BLOCK_START[selected] + int(records["offset"])
    split_stop = split_start + expected_total
    if np.any(seed_array < split_start) or np.any(seed_array >= split_stop):
        raise ValueError("dataset contains seeds outside the declared split range")
    return SimulationDataset(
        time_s=time_s,
        angle_rad=np.asarray(angle, dtype=np.float64),
        rate_rad_s=np.asarray(rate, dtype=np.float64),
        seeds=seed_array,
        capsized=np.asarray(capsized, dtype=np.bool_),
        t_capsize_s=np.asarray(cap_times, dtype=np.float64),
        metadata=tuple(metadata),
        config=manifest["simulation"],
    )
