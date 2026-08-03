from __future__ import annotations

import numpy as np
import pytest

from rahola.dataset import SimulationDataset
from rahola.windowing import CausalTransformer, WindowConfig, binary_auc, make_windows


def _dataset(angle: np.ndarray, cap_times: np.ndarray) -> SimulationDataset:
    rows, samples = angle.shape
    return SimulationDataset(
        time_s=np.arange(samples, dtype=np.float64),
        angle_rad=angle,
        rate_rad_s=np.zeros_like(angle),
        seeds=np.arange(rows, dtype=np.uint64),
        capsized=np.isfinite(cap_times),
        t_capsize_s=cap_times,
        metadata=tuple({"row": row} for row in range(rows)),
        config={"natural_period_s": 2.0},
    )


def test_horizon_buffer_and_post_capsize_rules() -> None:
    angle = np.tile(np.sin(np.arange(20)), (2, 1))
    dataset = _dataset(angle, np.array([15.0, np.nan]))
    windows = make_windows(
        dataset,
        WindowConfig(
            length_periods=2,
            horizon_periods=2,
            exclusion_buffer_periods=1,
        ),
    )
    first = windows.trajectory_indices == 0
    assert np.all(windows.end_times_s[first] < 15.0)
    assert np.all(windows.labels[windows.trajectory_indices == 1] == 0)
    assert not np.any((windows.end_times_s[first] >= 9) & (windows.end_times_s[first] < 11))


def test_future_only_leakage_probe_has_teeth() -> None:
    rng = np.random.default_rng(91)
    labels = np.repeat([0, 1], 128)
    prefix = rng.normal(size=(256, 40))
    future = np.where(labels[:, None] == 1, 20.0, -20.0) * np.ones((256, 20))
    series = np.concatenate((prefix, future), axis=1)

    causal_scores = np.array(
        [CausalTransformer(detrend=False).transform(row)[:40].mean() for row in series]
    )
    full_mean = series.mean(axis=1, keepdims=True)
    full_std = series.std(axis=1, keepdims=True)
    leaky_scores = ((series - full_mean) / full_std)[:, :40].mean(axis=1)
    causal_auc = binary_auc(labels, causal_scores)
    leaky_auc = binary_auc(labels, leaky_scores)
    assert causal_auc == pytest.approx(0.5, abs=0.08)
    assert min(leaky_auc, 1.0 - leaky_auc) < 0.05
