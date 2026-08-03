"""Load a generated campaign split into the core dense dataset type."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rahola.dataset import SimulationDataset
from rahola_lab.constants import SeedBlock
from rahola_lab.evaluation.splits import assert_seed_membership


def load_campaign_split(
    campaign_dir: str | Path,
    block: SeedBlock | str,
    *,
    limit: int | None = None,
) -> SimulationDataset:
    """Load a named split and verify every stored seed belongs to it."""
    selected = SeedBlock(block)
    root = Path(campaign_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["splits"][str(selected)]
    seeds: list[int] = []
    capsized: list[bool] = []
    cap_times: list[float] = []
    angle: list[list[float]] = []
    rate: list[list[float]] = []
    metadata: list[dict[str, object]] = []
    time_s: np.ndarray | None = None
    for chunk in records["chunks"]:
        chunk_manifest_path = root / chunk["path"]
        chunk_manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
        for shard in chunk_manifest["shards"]:
            table = pq.read_table(chunk_manifest_path.parent / shard["file"])
            payload = table.to_pydict()
            for row in range(table.num_rows):
                if limit is not None and len(seeds) >= limit:
                    break
                seeds.append(payload["seed"][row])
                capsized.append(payload["capsized"][row])
                cap_times.append(payload["t_capsize_s"][row])
                angle.append(payload["angle_rad"][row])
                rate.append(payload["rate_rad_s"][row])
                metadata.append(json.loads(payload["metadata_json"][row]))
                if time_s is None:
                    time_s = np.asarray(payload["time_s"][row], dtype=np.float64)
            if limit is not None and len(seeds) >= limit:
                break
        if limit is not None and len(seeds) >= limit:
            break
    if time_s is None:
        raise ValueError("campaign split contains no trajectories")
    seed_array = np.asarray(seeds, dtype=np.uint64)
    assert_seed_membership(seed_array, selected)
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
