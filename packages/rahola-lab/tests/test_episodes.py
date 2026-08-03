from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.evaluation import (
    EpisodeConfig,
    TrajectoryScores,
    alarm_episodes,
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
