from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.detectors import DetectorWindowDataset
from rahola_lab.experiments.ceiling import (
    _calibration,
    _clock_quartile_scores,
    _sample_stratified,
    _stratum_counts,
)


def _windows() -> DetectorWindowDataset:
    rows = 24
    return DetectorWindowDataset(
        features=np.zeros((rows, 4, 2), dtype=np.float32),
        labels=np.repeat(np.array([0, 1], dtype=np.int8), rows // 2),
        family_labels=np.zeros(rows, dtype=np.int8),
        trajectory_indices=np.arange(rows, dtype=np.int64),
        end_times_s=np.tile(np.array([10.0, 30.0, 55.0, 80.0]), 6),
        raw_angle_rad=np.zeros(rows),
        raw_rate_rad_s=np.zeros(rows),
    )


def test_stratified_sample_weights_reconstruct_each_population_stratum() -> None:
    windows = _windows()
    sampled, weights = _sample_stratified(windows, 100.0, 16, seed=9)
    full_groups = 4 * windows.labels + np.minimum(
        (4.0 * windows.end_times_s / 100.0).astype(int), 3
    )
    sampled_groups = 4 * sampled.labels + np.minimum(
        (4.0 * sampled.end_times_s / 100.0).astype(int), 3
    )
    for group in np.unique(full_groups):
        assert np.sum(weights[sampled_groups == group]) == pytest.approx(
            np.sum(full_groups == group)
        )


def test_stratum_counts_use_label_and_absolute_time_quartile() -> None:
    counts = _stratum_counts(_windows(), 100.0)
    assert counts == {
        "label=0,time_quartile=0": 3,
        "label=0,time_quartile=1": 3,
        "label=0,time_quartile=2": 3,
        "label=0,time_quartile=3": 3,
        "label=1,time_quartile=0": 3,
        "label=1,time_quartile=1": 3,
        "label=1,time_quartile=2": 3,
        "label=1,time_quartile=3": 3,
    }


def test_clock_baseline_uses_the_declared_protocol_time_quartiles() -> None:
    assert _clock_quartile_scores(_windows(), 100.0).tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
    ] * 6


def test_weighted_calibration_matches_hand_calculation() -> None:
    labels = np.array([0, 0, 1], dtype=np.int8)
    scores = np.array([0.1, 0.4, 0.8])
    weights = np.array([10.0, 1.0, 1.0])
    result = _calibration(labels, scores, weights)
    expected = np.average((scores - labels) ** 2, weights=weights)
    assert result["weighted_brier"] == pytest.approx(expected)
    assert result["weighted_ece_10_bin"] == pytest.approx(
        np.average(np.abs(scores - labels), weights=weights)
    )


def test_stratified_sample_requires_one_draw_per_nonempty_stratum() -> None:
    with pytest.raises(ValueError, match="nonempty stratum"):
        _sample_stratified(_windows(), 100.0, 1, seed=9)
