from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.evaluation import (
    EpisodeConfig,
    TrajectoryScores,
    alarm_episodes,
    clopper_pearson_interval,
    evaluate_alarms,
    operating_curve,
)


def test_episode_debounce_and_refractory_by_hand() -> None:
    times = np.arange(13, dtype=np.float64)
    scores = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=np.float64)
    episodes = alarm_episodes(
        times,
        scores,
        EpisodeConfig(threshold=0.5, debounce_windows=2, refractory_windows=2),
    )
    assert [(item.start_index, item.end_index) for item in episodes] == [(3, 7), (10, 12)]


def test_episode_metrics_known_answer() -> None:
    config = EpisodeConfig(threshold=0.5, debounce_windows=2, refractory_windows=2)
    trajectories = [
        TrajectoryScores(
            times_s=np.arange(10, dtype=np.float64) * 10,
            scores=np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            record_end_s=100.0,
            t_capsize_s=60.0,
        ),
        TrajectoryScores(
            times_s=np.arange(10, dtype=np.float64) * 10,
            scores=np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
            record_end_s=100.0,
        ),
    ]
    metrics = evaluate_alarms(trajectories, config, horizon_s=40.0)
    assert metrics.sensitivity == 1.0
    assert metrics.lead_times_s.tolist() == [40.0]
    assert metrics.false_episode_count == 1
    assert metrics.exposure_hours == pytest.approx(160.0 / 3600.0)
    assert metrics.false_positives_per_hour == pytest.approx(22.5)


def test_sustained_episode_overlapping_horizon_is_a_detection() -> None:
    trajectory = TrajectoryScores(
        times_s=np.arange(10, dtype=np.float64) * 10,
        scores=np.ones(10, dtype=np.float64),
        record_end_s=100.0,
        t_capsize_s=90.0,
    )
    metrics = evaluate_alarms(
        [trajectory],
        EpisodeConfig(threshold=0.5, debounce_windows=2, refractory_windows=2),
        horizon_s=30.0,
    )
    assert metrics.sensitivity == 1.0
    assert metrics.false_episode_count == 0
    assert metrics.lead_times_s.tolist() == [90.0]


def test_repeated_episodes_inside_event_horizon_are_not_false() -> None:
    trajectory = TrajectoryScores(
        times_s=np.arange(9, dtype=np.float64) * 10,
        scores=np.array([0, 0, 0, 1, 1, 0, 1, 1, 0], dtype=np.float64),
        record_end_s=90.0,
        t_capsize_s=85.0,
    )
    metrics = evaluate_alarms(
        [trajectory],
        EpisodeConfig(threshold=0.5, debounce_windows=2, refractory_windows=1),
        horizon_s=60.0,
    )
    assert metrics.sensitivity == 1.0
    assert metrics.false_episode_count == 0
    assert metrics.lead_times_s.tolist() == [55.0]


def test_exposure_and_events_begin_at_first_scorable_time() -> None:
    early = TrajectoryScores(
        times_s=np.array([120.0]),
        scores=np.array([0.0]),
        record_start_s=120.0,
        record_end_s=600.0,
        t_capsize_s=50.0,
    )
    at_risk = TrajectoryScores(
        times_s=np.array([120.0]),
        scores=np.array([0.0]),
        record_start_s=120.0,
        record_end_s=600.0,
        t_capsize_s=300.0,
    )
    metrics = evaluate_alarms([early, at_risk], EpisodeConfig(threshold=0.5), horizon_s=60.0)
    assert metrics.capsize_count == 1
    assert metrics.exposure_hours == pytest.approx(180.0 / 3600.0)


def test_operating_curve_sweeps_thresholds() -> None:
    trajectory = TrajectoryScores(
        times_s=np.arange(5, dtype=np.float64),
        scores=np.array([0.0, 0.4, 0.6, 0.8, 0.0]),
        record_end_s=5.0,
    )
    points = operating_curve(
        [trajectory],
        EpisodeConfig(threshold=0.0, debounce_windows=1, refractory_windows=1),
        np.array([0.5, 0.9]),
        horizon_s=2.0,
    )
    assert points[0].metrics.false_episode_count == 1
    assert points[1].metrics.false_episode_count == 0


def test_clopper_pearson_interval_handles_edge_counts() -> None:
    none = clopper_pearson_interval(0, 10)
    all_success = clopper_pearson_interval(10, 10)
    assert none.lower == 0.0
    assert none.upper == pytest.approx(1.0 - 0.025 ** (1.0 / 10.0))
    assert all_success.lower == pytest.approx(0.025 ** (1.0 / 10.0))
    assert all_success.upper == 1.0
