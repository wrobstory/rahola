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
