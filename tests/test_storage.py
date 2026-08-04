from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rahola.config import SimulationConfig
from rahola.dataset import SimulationDataset
from rahola.simulate import simulate_batch
from rahola.storage import write_dataset


def test_same_inputs_produce_byte_identical_dataset(tmp_path: Path) -> None:
    config = SimulationConfig(duration_s=4.0, natural_period_s=2.0, output_rate_hz=2.0)
    dataset = simulate_batch(config, [4, 5])
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_dataset(dataset, first, shard_size=1)
    write_dataset(dataset, second, shard_size=1)
    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def _valid_dense_dataset(**updates: object) -> SimulationDataset:
    values = {
        "time_s": np.array([0.0, 1.0, 2.0]),
        "angle_rad": np.array([[0.0, 0.1, np.nan], [0.0, 0.1, 0.2]]),
        "rate_rad_s": np.array([[0.0, 0.1, np.nan], [0.0, 0.1, 0.2]]),
        "seeds": np.array([4, 5], dtype=np.uint64),
        "capsized": np.array([True, False]),
        "t_capsize_s": np.array([1.0, np.nan]),
        "metadata": ({"seed": 4}, {"seed": 5}),
        "config": {"natural_period_s": 2.0},
    }
    values.update(updates)
    return SimulationDataset(**values)


def test_dense_dataset_rejects_corrupted_per_trajectory_fields() -> None:
    with pytest.raises(ValueError, match="per-trajectory fields"):
        _valid_dense_dataset(t_capsize_s=np.array([1.0]))
    with pytest.raises(ValueError, match="consistent"):
        _valid_dense_dataset(capsized=np.array([False, False]))


def test_dense_dataset_rejects_time_seed_and_metadata_corruption() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _valid_dense_dataset(time_s=np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="unique"):
        _valid_dense_dataset(seeds=np.array([4, 4], dtype=np.uint64))
    with pytest.raises(ValueError, match="metadata seed mismatch"):
        _valid_dense_dataset(metadata=({"seed": 9}, {"seed": 5}))


def test_dense_dataset_rejects_invalid_nan_onset() -> None:
    with pytest.raises(ValueError, match="post-capsize samples"):
        _valid_dense_dataset(angle_rad=np.array([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]))
    with pytest.raises(ValueError, match="must be finite"):
        _valid_dense_dataset(
            angle_rad=np.array([[0.0, 0.1, np.nan], [0.0, np.nan, np.nan]])
        )
