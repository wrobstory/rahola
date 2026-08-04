from __future__ import annotations

import numpy as np
from rahola_lab.evaluation import (
    TrajectoryScores,
    WaveGroup,
    identify_wave_groups,
    intervals_overlap,
)
from rahola_lab.experiments.d4 import _false_group_fraction


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


def test_d4_excludes_false_episodes_after_evaluable_record_end() -> None:
    trajectory = TrajectoryScores(
        times_s=np.array([10.0, 20.0, 30.0, 110.0, 120.0, 130.0]),
        scores=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        record_start_s=10.0,
        record_end_s=30.0,
    )
    result = _false_group_fraction(
        [trajectory],
        [(WaveGroup(start_s=100.0, end_s=140.0, maximum_height_m=5.0),)],
        threshold=0.5,
        decorrelation_s=0.0,
        horizon_s=20.0,
    )
    assert result["false_episode_count"] == 0
    assert result["coincident_false_episodes"] == 0


def test_d4_does_not_fill_quiet_gap_when_testing_group_coincidence() -> None:
    trajectory = TrajectoryScores(
        times_s=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 110.0, 120.0, 130.0]),
        scores=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        record_start_s=10.0,
        record_end_s=130.0,
    )
    result = _false_group_fraction(
        [trajectory],
        [(WaveGroup(start_s=70.0, end_s=90.0, maximum_height_m=5.0),)],
        threshold=0.5,
        decorrelation_s=100.0,
        horizon_s=20.0,
    )
    assert result["false_episode_count"] == 1
    assert result["coincident_false_episodes"] == 0
