from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from rahola_lab.campaigns import load_campaign_definition, load_campaign_split
from rahola_lab.campaigns.load import _contained_path
from rahola_lab.constants import EWS_HORIZON_PERIODS, EWS_WINDOW_PERIODS, SeedBlock
from rahola_lab.evaluation import ReserveBlockError

from rahola.dataset import SimulationDataset
from rahola.storage import write_dataset

CONFIG_DIR = Path(__file__).parents[1] / "src" / "rahola_lab" / "campaigns" / "configs"


def test_frozen_campaign_grid_and_duration_budget() -> None:
    definitions = [load_campaign_definition(path) for path in sorted(CONFIG_DIR.glob("*.yaml"))]
    assert len(definitions) == 15
    roles = [definition.role for definition in definitions]
    assert roles.count("stationary_training") == 3
    assert roles.count("prototype2_ramp") == 3
    assert roles.count("prototype2_bandwidth") == 5
    assert roles.count("sea_state_transition") == 1
    assert roles.count("rare_event_evaluation") == 3
    for definition in definitions:
        required = (
            EWS_WINDOW_PERIODS + EWS_HORIZON_PERIODS
        ) * definition.simulation.natural_period_s
        assert definition.simulation.duration_s > required
        assert all(
            split.block not in {SeedBlock.RESERVE, SeedBlock.RESERVE2}
            for split in definition.splits
        )


def test_campaign_counts_match_frozen_size_ranges() -> None:
    definitions = [load_campaign_definition(path) for path in sorted(CONFIG_DIR.glob("*.yaml"))]
    for definition in definitions:
        count = sum(split.count for split in definition.splits)
        if definition.role == "stationary_training":
            assert 2_000 <= count <= 5_000
        elif definition.role == "rare_event_evaluation":
            assert 5_000 <= count <= 10_000
        elif definition.role == "prototype2_bandwidth":
            assert count == 2_400


def test_campaign_loader_rejects_mutated_shard(tmp_path: Path) -> None:
    chunk_root = tmp_path / "train" / "chunk-00000"
    dataset = SimulationDataset(
        time_s=np.array([0.0, 1.0]),
        angle_rad=np.zeros((1, 2)),
        rate_rad_s=np.zeros((1, 2)),
        seeds=np.array([0], dtype=np.uint64),
        capsized=np.array([False]),
        t_capsize_s=np.array([np.nan]),
        metadata=({
            "family": "softening",
            "protocol": "stationary",
            "capsized": False,
            "git_commit": "test",
            "package_version": "test",
        },),
        config={"natural_period_s": 1.0, "family": "softening"},
    )
    chunk_manifest = write_dataset(dataset, chunk_root)
    chunk_sha = hashlib.sha256(chunk_manifest.read_bytes()).hexdigest()
    outer = {
        "simulation": dataset.config,
        "splits": {
            "train": {
                "chunks": [
                    {
                        "path": "train/chunk-00000/manifest.json",
                        "rows": 1,
                        "sha256": chunk_sha,
                    }
                ]
            }
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(outer, allow_nan=False), encoding="utf-8"
    )
    shard = chunk_root / "part-00000.parquet"
    shard.write_bytes(shard.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="Parquet shard hash mismatch"):
        load_campaign_split(tmp_path, SeedBlock.TRAIN, allow_unanchored=True)


def test_reference_manifest_requires_tracked_anchor(tmp_path: Path) -> None:
    root = tmp_path / "softening_stationary"
    root.mkdir()
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="anchored reference manifest hash mismatch"):
        load_campaign_split(root, SeedBlock.TRAIN)


def test_unknown_campaign_fails_closed_without_anchor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no tracked reference anchor"):
        load_campaign_split(tmp_path / "renamed-reference", SeedBlock.TRAIN)


@pytest.mark.parametrize("block", [SeedBlock.RESERVE, SeedBlock.RESERVE2])
def test_campaign_loader_refuses_reserve_before_reading_files(
    tmp_path: Path, block: SeedBlock
) -> None:
    with pytest.raises(ReserveBlockError, match="may not be inspected"):
        load_campaign_split(tmp_path / "missing", block, allow_unanchored=True)


def test_campaign_paths_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes campaign root"):
        _contained_path(root, root, "../outside/manifest.json", kind="chunk manifest")


@pytest.mark.parametrize("field,value", [("count", 1.5), ("offset", True)])
def test_campaign_split_controls_require_integers(
    tmp_path: Path, field: str, value: object
) -> None:
    raw = yaml.safe_load(next(CONFIG_DIR.glob("*.yaml")).read_text(encoding="utf-8"))
    split = next(iter(raw["splits"].values()))
    split[field] = value
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="non-boolean integer"):
        load_campaign_definition(path)


def test_campaign_split_must_fit_seed_block(tmp_path: Path) -> None:
    raw = yaml.safe_load(next(CONFIG_DIR.glob("*.yaml")).read_text(encoding="utf-8"))
    split = next(iter(raw["splits"].values()))
    split["count"] = 1
    split["offset"] = 100_000
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fit inside"):
        load_campaign_definition(path)
