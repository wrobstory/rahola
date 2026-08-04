from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.detectors import (
    JaxTemporalCNN,
    classical_ews_scores,
    extract_detector_windows,
    galeazzi_roll_power_glrt,
    neighbor_count_scores,
)
from rahola_lab.experiments.d3 import _all_motion_skill_collapses

from rahola.dataset import SimulationDataset
from rahola.windowing import binary_auc


def test_neighbor_score_matches_hand_count_and_thesis_direction() -> None:
    features = np.zeros((2, 8, 2))
    features[0, :5, 0] = [0.0, 0.1, 0.2, 2.0, 3.0]
    features[0, -1, 0] = 0.15
    features[1, :5, 0] = 4.0
    features[1, -1, 0] = 0.0
    scores = neighbor_count_scores(features, radius=0.11, samples_per_period=3)
    assert np.array_equal(scores, [-2.0, 0.0])
    assert scores[1] > scores[0]


def test_classical_variance_trend_detects_increasing_scale() -> None:
    rng = np.random.default_rng(17)
    stationary = rng.normal(size=120)
    increasing = rng.normal(size=120) * np.linspace(0.2, 3.0, 120)
    features = np.zeros((2, 120, 2))
    features[:, :, 0] = [stationary, increasing]
    scores = classical_ews_scores(features, statistic="variance", subwindow_fraction=0.25)
    assert scores[1] > 0.5
    assert scores[1] > scores[0]


def test_roll_power_glrt_increases_for_late_resonant_power() -> None:
    samples_per_period = 8
    length = 60 * samples_per_period
    time = np.arange(length)
    base = np.sin(2.0 * np.pi * time / samples_per_period)
    shifted = base.copy()
    shifted[-4 * samples_per_period :] *= 5.0
    features = np.zeros((2, length, 2))
    features[:, :, 0] = [base, shifted]
    scores = galeazzi_roll_power_glrt(features, samples_per_period=samples_per_period)
    assert scores[1] > scores[0]


def test_roll_power_glrt_ignores_late_scale_decrease() -> None:
    samples_per_period = 8
    length = 60 * samples_per_period
    time = np.arange(length)
    decreasing = np.sin(2.0 * np.pi * time / samples_per_period)
    decreasing[-4 * samples_per_period :] *= 0.1
    features = np.zeros((1, length, 2))
    features[0, :, 0] = decreasing
    score = galeazzi_roll_power_glrt(features, samples_per_period=samples_per_period)
    np.testing.assert_allclose(score, [0.0], atol=1e-12)


def test_cnn_stays_under_parameter_cap_and_learns_simple_signal() -> None:
    rng = np.random.default_rng(22)
    features = rng.normal(size=(64, 80, 2)).astype(np.float32)
    labels = np.repeat([0, 1], 32).astype(np.int8)
    features[labels == 1, :, 0] += 1.0
    model = JaxTemporalCNN(channels=(6, 10), kernel_size=5, epochs=3, batch_size=32)
    model.fit(features, labels)
    assert model.parameter_count() < 100_000
    assert binary_auc(labels, model.predict_scores(features)) > 0.75


def test_detector_feature_pipeline_blocks_future_only_signal() -> None:
    rng = np.random.default_rng(91)
    rows, samples = 256, 900
    labels = np.repeat([0, 1], rows // 2)
    angle = rng.normal(size=(rows, samples))
    angle[:, 480:] += np.where(labels[:, None] == 1, 20.0, -20.0)
    rate = rng.normal(size=(rows, samples))
    cap_times = np.where(labels == 1, 300.0, np.nan)
    dataset = SimulationDataset(
        time_s=np.arange(samples, dtype=np.float64) * 0.5,
        angle_rad=angle,
        rate_rad_s=rate,
        seeds=np.arange(rows, dtype=np.uint64),
        capsized=np.isfinite(cap_times),
        t_capsize_s=cap_times,
        metadata=tuple({"row": row} for row in range(rows)),
        config={"natural_period_s": 4.0, "family": "softening"},
    )
    windows = extract_detector_windows(dataset, stride_s=500.0, max_windows_per_trajectory=1)
    causal_scores = windows.features[:, :, 0].mean(axis=1)
    assert binary_auc(windows.labels, causal_scores) == pytest.approx(0.5, abs=0.08)

    full_mean = angle.mean(axis=1, keepdims=True)
    full_std = angle.std(axis=1, keepdims=True)
    leaky_scores = ((angle - full_mean) / full_std)[:, :480].mean(axis=1)
    assert min(binary_auc(labels, leaky_scores), 1.0 - binary_auc(labels, leaky_scores)) < 0.05


def test_detector_windows_drop_censored_negatives_but_allow_inference_features() -> None:
    samples = 121
    angle = np.sin(np.arange(samples, dtype=np.float64))[None, :]
    dataset = SimulationDataset(
        time_s=np.arange(samples, dtype=np.float64),
        angle_rad=angle,
        rate_rad_s=np.zeros_like(angle),
        seeds=np.array([1], dtype=np.uint64),
        capsized=np.array([False]),
        t_capsize_s=np.array([np.nan]),
        metadata=({},),
        config={"natural_period_s": 1.0, "family": "softening"},
    )
    supervised = extract_detector_windows(dataset, stride_s=1.0)
    inference = extract_detector_windows(
        dataset, stride_s=1.0, allow_censored_for_inference=True
    )
    assert np.max(supervised.end_times_s) == 70.0
    late = inference.end_times_s > 70.0
    assert np.any(late)
    assert np.all(inference.labels == -1)

    capsizing = SimulationDataset(
        time_s=dataset.time_s,
        angle_rad=dataset.angle_rad,
        rate_rad_s=dataset.rate_rad_s,
        seeds=dataset.seeds,
        capsized=np.array([True]),
        t_capsize_s=np.array([100.0]),
        metadata=dataset.metadata,
        config=dataset.config,
    )
    capsizing_windows = extract_detector_windows(capsizing, stride_s=1.0)
    assert np.max(capsizing_windows.end_times_s) == 70.0
    assert np.any(capsizing_windows.labels == 1)

    capsizing_inference = extract_detector_windows(
        capsizing, stride_s=1.0, allow_censored_for_inference=True
    )
    np.testing.assert_array_equal(
        capsizing_inference.end_times_s,
        inference.end_times_s[inference.end_times_s < 100.0],
    )


def test_d3_collapse_requires_every_motion_detector_to_reach_b1_floor() -> None:
    methods = {
        "classical_ews": {"auc": 0.50},
        "cnn": {"auc": 0.51},
        "cnn_cross_gamma": {"auc": 0.51},
        "galeazzi_glrt": {"auc": 0.90},
    }
    assert not _all_motion_skill_collapses(methods)
    methods["galeazzi_glrt"]["auc"] = 0.52
    assert _all_motion_skill_collapses(methods)
