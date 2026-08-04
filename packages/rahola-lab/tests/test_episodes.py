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
    assert [(item.start_index, item.end_index) for item in episodes] == [(4, 7), (11, 12)]


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
    assert metrics.lead_times_s.tolist() == [30.0]
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
    assert metrics.lead_times_s.tolist() == [80.0]


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
    assert metrics.lead_times_s.tolist() == [45.0]


def test_exposure_and_events_begin_at_first_scorable_time() -> None:
    early = TrajectoryScores(
        times_s=np.array([120.0, 130.0, 140.0]),
        scores=np.zeros(3),
        record_start_s=120.0,
        record_end_s=600.0,
        t_capsize_s=50.0,
    )
    at_risk = TrajectoryScores(
        times_s=np.array([120.0, 130.0, 140.0, 250.0]),
        scores=np.zeros(4),
        record_start_s=120.0,
        record_end_s=600.0,
        t_capsize_s=300.0,
    )
    metrics = evaluate_alarms([early, at_risk], EpisodeConfig(threshold=0.5), horizon_s=60.0)
    assert metrics.capsize_count == 1
    assert metrics.exposure_hours == pytest.approx(180.0 / 3600.0)


def test_event_before_debounce_can_open_is_outside_sensitivity_risk_set() -> None:
    trajectory = TrajectoryScores(
        times_s=np.array([120.0, 130.0]),
        scores=np.ones(2),
        record_start_s=120.0,
        record_end_s=400.0,
        t_capsize_s=135.0,
    )
    metrics = evaluate_alarms(
        [trajectory], EpisodeConfig(threshold=0.5, debounce_windows=3), horizon_s=200.0
    )
    assert metrics.capsize_count == 0
    assert metrics.exposure_hours == 0.0


def test_event_without_scored_endpoint_in_horizon_is_right_censored() -> None:
    trajectory = TrajectoryScores(
        times_s=np.array([120.0, 130.0, 140.0]),
        scores=np.ones(3),
        record_start_s=120.0,
        record_end_s=140.0,
        t_capsize_s=400.0,
    )
    metrics = evaluate_alarms(
        [trajectory], EpisodeConfig(threshold=0.5, debounce_windows=3), horizon_s=200.0
    )
    assert metrics.capsize_count == 0
    assert metrics.exposure_hours == pytest.approx(20.0 / 3600.0)


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


def test_episode_config_rejects_non_finite_threshold() -> None:
    with pytest.raises(ValueError, match="finite"):
        EpisodeConfig(threshold=float("nan"))


@pytest.mark.parametrize("value", [True, 1.5])
def test_episode_config_requires_integer_window_controls(value: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        EpisodeConfig(threshold=0.5, debounce_windows=value)  # type: ignore[arg-type]


def test_alarm_episodes_reject_non_finite_scores_and_accept_empty_vectors() -> None:
    config = EpisodeConfig(threshold=0.5)
    with pytest.raises(ValueError, match="finite"):
        alarm_episodes(np.array([0.0]), np.array([np.nan]), config)
    assert alarm_episodes(np.empty(0), np.empty(0), config) == ()


def test_alarms_after_the_risk_interval_are_ignored() -> None:
    trajectory = TrajectoryScores(
        times_s=np.array([10.0, 20.0, 30.0, 110.0, 120.0]),
        scores=np.array([0.0, 0.0, 0.0, 1.0, 1.0]),
        record_start_s=10.0,
        record_end_s=30.0,
    )
    metrics = evaluate_alarms(
        [trajectory],
        EpisodeConfig(threshold=0.5, debounce_windows=2, refractory_windows=1),
        horizon_s=20.0,
    )
    assert metrics.false_episode_count == 0
    assert metrics.alarm_opportunity_count == 3


def test_clock_only_tail_signal_cannot_exploit_outcome_followup() -> None:
    times = np.arange(240.0, 541.0, 10.0)
    scores = (times >= 410.0).astype(np.float64)
    trajectories = [
        TrajectoryScores(
            times_s=times,
            scores=scores,
            record_start_s=240.0,
            record_end_s=400.0,
            t_capsize_s=550.0,
        ),
        TrajectoryScores(
            times_s=times,
            scores=scores,
            record_start_s=240.0,
            record_end_s=400.0,
        ),
    ]
    metrics = evaluate_alarms(
        trajectories,
        EpisodeConfig(threshold=0.5, debounce_windows=3, refractory_windows=3),
        horizon_s=200.0,
    )
    assert metrics.capsize_count == 1
    assert metrics.sensitivity == 0.0
    assert metrics.false_episode_count == 0
