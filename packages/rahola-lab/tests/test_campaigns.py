from __future__ import annotations

from pathlib import Path

from rahola_lab.campaigns import load_campaign_definition
from rahola_lab.constants import EWS_HORIZON_PERIODS, EWS_WINDOW_PERIODS, SeedBlock

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
        assert all(split.block != SeedBlock.RESERVE for split in definition.splits)


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
