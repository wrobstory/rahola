from __future__ import annotations

import numpy as np
import pytest

from rahola.dataset import SimulationDataset
from rahola.windowing import CausalTransformer, WindowConfig, binary_auc, make_windows


def _dataset(angle: np.ndarray, cap_times: np.ndarray) -> SimulationDataset:
    rows, samples = angle.shape
    angle = angle.copy()
    rate = np.zeros_like(angle)
    time_s = np.arange(samples, dtype=np.float64)
    for row, capsize_time in enumerate(cap_times):
        if np.isfinite(capsize_time):
            angle[row, time_s > capsize_time] = np.nan
            rate[row, time_s > capsize_time] = np.nan
    return SimulationDataset(
        time_s=time_s,
        angle_rad=angle,
        rate_rad_s=rate,
        seeds=np.arange(rows, dtype=np.uint64),
        capsized=np.isfinite(cap_times),
        t_capsize_s=cap_times,
        metadata=tuple({"row": row, "seed": row} for row in range(rows)),
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
    assert np.max(windows.end_times_s[first]) <= 15.0
    assert np.all(windows.labels[windows.trajectory_indices == 1] == 0)
    assert not np.any((windows.end_times_s[first] >= 9) & (windows.end_times_s[first] < 11))
    safe = windows.trajectory_indices == 1
    assert np.max(windows.end_times_s[safe]) == 15.0


def test_capsize_and_non_event_windows_share_horizon_complete_support() -> None:
    samples = 121
    angle = np.sin(np.arange(samples, dtype=np.float64))[None, :]
    dataset = _dataset(angle, np.array([100.0]))
    windows = make_windows(
        dataset,
        WindowConfig(
            length_periods=30.0,
            horizon_periods=25.0,
            exclusion_buffer_periods=1.0,
        ),
    )
    assert np.max(windows.end_times_s) == 70.0
    assert np.any(windows.labels == 1)


def test_auc_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        binary_auc(np.array([0, 1]), np.array([0.0, np.nan]))


@pytest.mark.parametrize("stride", [float("nan"), 1.5])
def test_window_stride_must_be_an_integer(stride: float) -> None:
    with pytest.raises(ValueError, match="integer"):
        WindowConfig(length_periods=2.0, horizon_periods=2.0, stride_samples=stride)


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


def test_vectorized_causal_transformer_is_bitwise_equivalent_to_reference() -> None:
    def reference(values: np.ndarray, *, detrend: bool, epsilon: float = 1e-12) -> np.ndarray:
        result = np.zeros_like(values)
        sum_y = sum_y2 = sum_t = sum_t2 = sum_ty = 0.0
        for index, value in enumerate(values):
            count = index
            if not np.isfinite(value):
                result[index:] = np.nan
                break
            if count < 2:
                residual, scale = 0.0, 1.0
            else:
                mean = sum_y / count
                if detrend:
                    denominator = count * sum_t2 - sum_t**2
                    slope = (count * sum_ty - sum_t * sum_y) / max(denominator, epsilon)
                    intercept = (sum_y - slope * sum_t) / count
                    prediction = intercept + slope * index
                else:
                    prediction = mean
                variance = max((sum_y2 - count * mean**2) / (count - 1), 0.0)
                scale = max(np.sqrt(variance), epsilon)
                residual = value - prediction
            result[index] = residual / scale
            sum_y += value
            sum_y2 += value * value
            sum_t += index
            sum_t2 += index * index
            sum_ty += index * value
        return result

    rng = np.random.default_rng(812)
    corpus = [
        np.zeros(32),
        np.arange(32, dtype=np.float64),
        rng.normal(size=257),
        np.concatenate((rng.normal(size=81), [np.nan], np.full(12, np.nan))),
    ]
    for detrend in (False, True):
        for values in corpus:
            expected = reference(values, detrend=detrend)
            actual = CausalTransformer(detrend=detrend).transform(values)
            assert np.array_equal(actual, expected, equal_nan=True)
