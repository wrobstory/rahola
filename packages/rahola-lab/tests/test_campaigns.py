from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from rahola_lab.campaigns import (
    CampaignDefinition,
    generate_campaign,
    load_campaign_definition,
    load_campaign_split,
)
from rahola_lab.campaigns.definition import SplitDefinition
from rahola_lab.campaigns.load import _contained_path
from rahola_lab.constants import EWS_HORIZON_PERIODS, EWS_WINDOW_PERIODS, SeedBlock
from rahola_lab.evaluation import ReserveBlockError

from rahola.config import ForcingConfig, SimulationConfig
from rahola.dataset import SimulationDataset
from rahola.storage import write_dataset

CONFIG_DIR = Path(__file__).parents[1] / "src" / "rahola_lab" / "campaigns" / "configs"


def _write_loader_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    config = SimulationConfig(duration_s=2.0, natural_period_s=2.0, output_rate_hz=1.0)
    metadata = tuple(
        {
            "seed": seed,
            "family": str(config.family),
            "protocol": str(config.protocol.kind),
            "capsized": False,
            "git_commit": "test",
            "package_version": "test",
            "config_hash": config.config_hash,
        }
        for seed in (0, 1)
    )
    dataset = SimulationDataset(
        time_s=np.array([0.0, 1.0, 2.0]),
        angle_rad=np.zeros((2, 3)),
        rate_rad_s=np.zeros((2, 3)),
        seeds=np.array([0, 1], dtype=np.uint64),
        capsized=np.array([False, False]),
        t_capsize_s=np.array([np.nan, np.nan]),
        metadata=metadata,
        config=config.to_dict(),
    )
    chunk_root = tmp_path / "train" / "chunk-00000"
    chunk_manifest_path = write_dataset(dataset, chunk_root)
    outer: dict[str, object] = {
        "simulation": dataset.config,
        "splits": {
            "train": {
                "count": 2,
                "offset": 0,
                "chunks": [
                    {
                        "path": "train/chunk-00000/manifest.json",
                        "rows": 2,
                        "sha256": hashlib.sha256(chunk_manifest_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(outer, allow_nan=False), encoding="utf-8"
    )
    return chunk_root / "part-00000.parquet", chunk_manifest_path, outer


def _rewrite_fixture_shard(
    tmp_path: Path,
    shard_path: Path,
    chunk_manifest_path: Path,
    outer: dict[str, object],
    payload: dict[str, list[object]],
) -> None:
    table = pq.read_table(shard_path)
    pq.write_table(pa.Table.from_pydict(payload, schema=table.schema), shard_path)
    chunk_manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    chunk_manifest["shards"][0]["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    chunk_manifest_path.write_text(
        json.dumps(chunk_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outer["splits"]["train"]["chunks"][0]["sha256"] = hashlib.sha256(
        chunk_manifest_path.read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(outer, allow_nan=False), encoding="utf-8"
    )


def test_frozen_campaign_grid_and_duration_budget() -> None:
    definitions = [load_campaign_definition(path) for path in sorted(CONFIG_DIR.glob("*.yaml"))]
    assert len(definitions) == 16
    roles = [definition.role for definition in definitions]
    assert roles.count("stationary_training") == 3
    assert roles.count("prototype2_ramp") == 3
    assert roles.count("prototype2_bandwidth") == 5
    assert roles.count("sea_state_transition") == 1
    assert roles.count("sea_state_transition_v02") == 1
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


def test_tiny_campaign_regeneration_is_byte_deterministic(tmp_path: Path) -> None:
    definition = CampaignDefinition(
        name="tiny-regeneration",
        role="test_fixture",
        rationale="small deterministic regeneration check",
        simulation=SimulationConfig(
            duration_s=1.0,
            natural_period_s=1.0,
            output_rate_hz=1.0,
            forcing=ForcingConfig(effective_wave_slope=0.0),
        ),
        splits=(SplitDefinition(block=SeedBlock.TRAIN, count=2, offset=0),),
    )
    first = generate_campaign(definition, tmp_path / "first", chunk_size=2)
    second = generate_campaign(definition, tmp_path / "second", chunk_size=2)
    first_files = sorted(
        path.relative_to(first.manifest_path.parent)
        for path in first.manifest_path.parent.rglob("*")
        if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second.manifest_path.parent)
        for path in second.manifest_path.parent.rglob("*")
        if path.is_file()
    )
    assert first_files == second_files
    for relative in first_files:
        assert (first.manifest_path.parent / relative).read_bytes() == (
            second.manifest_path.parent / relative
        ).read_bytes()


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
            "seed": 0,
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
                "count": 1,
                "offset": 0,
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


def test_campaign_loader_rejects_row_time_mismatch(tmp_path: Path) -> None:
    shard, chunk_manifest, outer = _write_loader_fixture(tmp_path)
    payload = pq.read_table(shard).to_pydict()
    payload["time_s"][1] = [0.0, 1.0, 3.0]
    _rewrite_fixture_shard(tmp_path, shard, chunk_manifest, outer, payload)
    with pytest.raises(ValueError, match="time-vector mismatch"):
        load_campaign_split(tmp_path, SeedBlock.TRAIN, allow_unanchored=True)


def test_campaign_loader_rejects_metadata_config_hash_mismatch(tmp_path: Path) -> None:
    shard, chunk_manifest, outer = _write_loader_fixture(tmp_path)
    payload = pq.read_table(shard).to_pydict()
    metadata = json.loads(payload["metadata_json"][1])
    metadata["config_hash"] = "corrupted"
    payload["metadata_json"][1] = json.dumps(metadata)
    _rewrite_fixture_shard(tmp_path, shard, chunk_manifest, outer, payload)
    with pytest.raises(ValueError, match="metadata config hash mismatch"):
        load_campaign_split(tmp_path, SeedBlock.TRAIN, allow_unanchored=True)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [([0, 0], "unique"), ([0, 3], "declared split range")],
)
def test_campaign_loader_rejects_invalid_seed_sets(
    tmp_path: Path, replacement: list[int], message: str
) -> None:
    shard, chunk_manifest, outer = _write_loader_fixture(tmp_path)
    payload = pq.read_table(shard).to_pydict()
    payload["seed"] = replacement
    for row, seed in enumerate(replacement):
        metadata = json.loads(payload["metadata_json"][row])
        metadata["seed"] = seed
        payload["metadata_json"][row] = json.dumps(metadata)
    _rewrite_fixture_shard(tmp_path, shard, chunk_manifest, outer, payload)
    with pytest.raises(ValueError, match=message):
        load_campaign_split(tmp_path, SeedBlock.TRAIN, allow_unanchored=True)


def test_campaign_loader_rejects_declared_total_mismatch(tmp_path: Path) -> None:
    _, _, outer = _write_loader_fixture(tmp_path)
    outer["splits"]["train"]["count"] = 3
    (tmp_path / "manifest.json").write_text(
        json.dumps(outer, allow_nan=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="split row-count mismatch"):
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
