"""Structurally causal transforms and horizon-aware window labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rahola.dataset import SimulationDataset, WindowDataset

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WindowConfig:
    length_periods: float
    horizon_periods: float
    exclusion_buffer_periods: float = 1.0
    stride_samples: int = 1
    detrend: bool = True

    def __post_init__(self) -> None:
        if self.length_periods <= 0 or self.horizon_periods <= 0:
            raise ValueError("window length and horizon must be positive")
        if self.exclusion_buffer_periods < 0 or self.stride_samples < 1:
            raise ValueError("buffer must be nonnegative and stride at least one")


class CausalTransformer:
    """Transform each sample using only state fitted strictly before that sample."""

    def __init__(self, *, detrend: bool = True, epsilon: float = 1e-12) -> None:
        self.detrend = detrend
        self.epsilon = epsilon

    def transform(self, values: FloatArray) -> FloatArray:
        source = np.asarray(values, dtype=np.float64)
        if source.ndim != 1:
            raise ValueError("CausalTransformer accepts one trajectory at a time")
        result = np.zeros_like(source)
        sum_y = 0.0
        sum_y2 = 0.0
        sum_t = 0.0
        sum_t2 = 0.0
        sum_ty = 0.0
        for index, value in enumerate(source):
            count = index
            if not np.isfinite(value):
                result[index:] = np.nan
                break
            if count < 2:
                residual = 0.0
                scale = 1.0
            else:
                mean = sum_y / count
                if self.detrend:
                    denominator = count * sum_t2 - sum_t**2
                    slope = (count * sum_ty - sum_t * sum_y) / max(denominator, self.epsilon)
                    intercept = (sum_y - slope * sum_t) / count
                    prediction = intercept + slope * index
                else:
                    prediction = mean
                variance = max((sum_y2 - count * mean**2) / (count - 1), 0.0)
                scale = max(np.sqrt(variance), self.epsilon)
                residual = value - prediction
            result[index] = residual / scale
            sum_y += value
            sum_y2 += value * value
            sum_t += index
            sum_t2 += index * index
            sum_ty += index * value
        return result


def make_windows(dataset: SimulationDataset, config: WindowConfig) -> WindowDataset:
    natural_period_s = float(dataset.config["natural_period_s"])
    sample_dt = float(np.median(np.diff(dataset.time_s)))
    length = max(1, round(config.length_periods * natural_period_s / sample_dt))
    horizon_s = config.horizon_periods * natural_period_s
    buffer_s = config.exclusion_buffer_periods * natural_period_s
    transformer = CausalTransformer(detrend=config.detrend)
    values: list[np.ndarray] = []
    labels: list[int] = []
    trajectory_indices: list[int] = []
    end_times: list[float] = []
    for trajectory_index in range(dataset.batch_size):
        transformed = transformer.transform(dataset.angle_rad[trajectory_index])
        cap_time = dataset.t_capsize_s[trajectory_index]
        for end_index in range(length - 1, len(dataset.time_s), config.stride_samples):
            end_time = dataset.time_s[end_index]
            if np.isfinite(cap_time) and end_time >= cap_time:
                break
            window = transformed[end_index - length + 1 : end_index + 1]
            if not np.all(np.isfinite(window)):
                break
            if not np.isfinite(cap_time):
                label = 0
            else:
                lead = cap_time - end_time
                if 0 < lead <= horizon_s:
                    label = 1
                elif lead <= horizon_s + buffer_s:
                    continue
                else:
                    label = 0
            values.append(window)
            labels.append(label)
            trajectory_indices.append(trajectory_index)
            end_times.append(end_time)
    shape = (0, length) if not values else None
    value_array = np.empty(shape, dtype=np.float64) if shape else np.stack(values)
    return WindowDataset(
        values=value_array,
        labels=np.asarray(labels, dtype=np.int8),
        trajectory_indices=np.asarray(trajectory_indices, dtype=np.int64),
        end_times_s=np.asarray(end_times, dtype=np.float64),
    )


def binary_auc(labels: NDArray[np.integer], scores: FloatArray) -> float:
    """Dependency-free Mann-Whitney AUC, with average ranks for ties."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    n_pos, n_neg = int(np.sum(positives)), int(np.sum(~positives))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((np.sum(ranks[positives]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
