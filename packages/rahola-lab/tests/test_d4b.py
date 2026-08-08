from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.experiments.d4b import cluster_groups, detect_groups


def test_hand_placed_wave_groups_recover_count_and_parameters() -> None:
    dt_s = 0.05
    time_s = np.arange(0.0, 140.0 + dt_s, dt_s)
    carrier_period_s = 4.0
    envelope = (
        0.01
        + 2.0 * np.exp(-0.5 * np.square((time_s - 30.0) / 5.0))
        + 2.5 * np.exp(-0.5 * np.square((time_s - 100.0) / 5.0))
    )
    elevation = envelope * np.cos(2.0 * np.pi * time_s / carrier_period_s)

    groups = detect_groups(
        time_s,
        elevation,
        source_seed=7,
        significant_height_m=4.0,
        peak_period_s=carrier_period_s,
    )

    assert len(groups) == 2
    assert [group.carrier_period_s for group in groups] == pytest.approx(
        [carrier_period_s, carrier_period_s], rel=1e-5
    )
    assert [group.central_height_m for group in groups] == pytest.approx([4.02, 5.02], rel=1e-5)
    assignments, medoids, _, _ = cluster_groups(list(groups), 2)
    assert sorted(assignments.tolist()) == [0, 1]
    assert sorted(medoids.tolist()) == [0, 1]
