from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.evaluation import (
    EpisodeConfig,
    TrajectoryScores,
    bootstrap_alarm_metrics,
    clopper_pearson_interval,
    trajectory_block_bootstrap,
)
from rahola_lab.experiments.detector_common import bootstrap_window_auc


def test_trajectory_bootstrap_captures_cluster_dependence_that_binomial_misses() -> None:
    trajectories = [np.ones(50) for _ in range(14)] + [np.zeros(50) for _ in range(6)]

    def pooled_window_mean(sample: list[np.ndarray]) -> float:
        return float(np.mean(np.concatenate(sample)))

    interval = trajectory_block_bootstrap(
        trajectories, pooled_window_mean, replicates=1_000, seed=91
    )
    naive = clopper_pearson_interval(14 * 50, 20 * 50)

    assert naive.lower > 0.5
    assert interval.lower[0] <= 0.5 <= interval.upper[0]
    assert interval.upper[0] - interval.lower[0] > naive.upper - naive.lower


def test_stratified_bootstrap_holds_campaign_weights_fixed() -> None:
    items = [("a", index) for index in range(3)] + [("b", index) for index in range(7)]

    def campaign_a_fraction(sample: list[tuple[str, int]]) -> float:
        return sum(campaign == "a" for campaign, _ in sample) / len(sample)

    result = trajectory_block_bootstrap(
        items,
        campaign_a_fraction,
        strata=[campaign for campaign, _ in items],
        replicates=1_000,
        seed=17,
    )
    np.testing.assert_allclose(result.estimate, [0.3])
    np.testing.assert_allclose(result.lower, [0.3])
    np.testing.assert_allclose(result.upper, [0.3])


def test_bootstrap_rejects_too_few_replicates() -> None:
    with pytest.raises(ValueError, match="at least 1,000"):
        trajectory_block_bootstrap([1, 2], lambda values: float(np.mean(values)), replicates=999)


def test_alarm_bootstrap_recomputes_episode_metrics_by_trajectory() -> None:
    trajectories = [
        TrajectoryScores(
            times_s=np.array([0.0, 10.0, 20.0, 30.0]),
            scores=np.array([0.0, 1.0, 1.0, 0.0]),
            record_end_s=30.0,
            t_capsize_s=30.0 if index < 10 else None,
        )
        for index in range(20)
    ]
    intervals = bootstrap_alarm_metrics(
        trajectories,
        EpisodeConfig(threshold=0.5, debounce_windows=2, refractory_windows=1),
        horizon_s=20.0,
        replicates=1_000,
        seed=3,
    )
    assert intervals.valid_replicates == (1_000, 1_000)
    assert intervals.sensitivity.lower == pytest.approx(1.0)
    assert intervals.sensitivity.upper == pytest.approx(1.0)


def test_window_auc_bootstrap_keeps_each_trajectory_windows_together() -> None:
    trajectories = [
        TrajectoryScores(
            times_s=np.array([10.0, 20.0]),
            scores=np.array([float(index < 10), float(index < 10)]),
            record_end_s=20.0,
            t_capsize_s=30.0 if index < 10 else None,
        )
        for index in range(20)
    ]
    payload = bootstrap_window_auc(trajectories, horizon_s=20.0, buffer_s=0.0)
    assert payload["auc"] == pytest.approx(1.0)
    assert payload["auc_trajectory_bootstrap_interval"] == pytest.approx([1.0, 1.0])
