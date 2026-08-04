from __future__ import annotations

from rahola_lab.experiments.forcing_invariance_v02 import affected_campaigns


def test_forcing_regeneration_rule_is_strictly_above_one_point() -> None:
    rows = [
        {"campaign": "at_limit", "prevalence_shift": 0.01},
        {"campaign": "affected", "prevalence_shift": -0.010_001},
        {"campaign": "stable", "prevalence_shift": 0.002},
    ]
    assert affected_campaigns(rows) == ["affected"]
