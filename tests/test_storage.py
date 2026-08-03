from __future__ import annotations

from pathlib import Path

from rahola.config import SimulationConfig
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
