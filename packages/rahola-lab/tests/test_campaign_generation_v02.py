from __future__ import annotations

from pathlib import Path

from rahola_lab.campaigns import versioned_definitions

CONFIG_ROOT = Path(__file__).parents[1] / "src" / "rahola_lab" / "campaigns" / "configs"


def test_versioned_definitions_preserve_v01_configs_and_replace_step_geometry() -> None:
    definitions = versioned_definitions(
        CONFIG_ROOT, ["softening_ramp", "softening_step"]
    )
    assert [definition.name for definition in definitions] == [
        "softening_ramp_v02",
        "softening_step_v02",
    ]
    assert definitions[0].simulation.duration_s == 600.0
    assert definitions[1].simulation.duration_s == 900.0
    assert definitions[1].simulation.protocol.steps[0].time_s == 300.0
