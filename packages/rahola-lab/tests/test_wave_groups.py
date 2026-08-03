from __future__ import annotations

import numpy as np
from rahola_lab.evaluation import identify_wave_groups, intervals_overlap


def test_wave_group_run_length_and_height_are_hand_checkable() -> None:
    times = np.arange(0.0, 40.0, 0.1)
    carrier = np.cos(2.0 * np.pi * times / 2.0)
    amplitude = np.where((times >= 8.0) & (times < 16.0), 2.0, 0.2)
    groups = identify_wave_groups(
        times,
        amplitude * carrier,
        significant_height_m=4.0,
        peak_period_s=2.0,
        height_fraction=0.75,
        minimum_periods=2.0,
    )

    assert len(groups) == 1
    assert 7.5 <= groups[0].start_s <= 8.5
    assert 15.5 <= groups[0].end_s <= 16.5
    assert groups[0].maximum_height_m > 3.5


def test_interval_overlap_includes_touching_boundaries() -> None:
    assert intervals_overlap(1.0, 2.0, 2.0, 3.0)
    assert not intervals_overlap(1.0, 1.9, 2.0, 3.0)
