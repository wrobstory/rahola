"""Structurally causal transforms and horizon-aware window labels."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

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
        if isinstance(self.stride_samples, bool) or not isinstance(
            self.stride_samples, Integral
        ):
            raise ValueError("stride_samples must be an integer")
        numeric = (self.length_periods, self.horizon_periods, self.exclusion_buffer_periods)
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("window values must be finite")
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
        result = np.full_like(source, np.nan)
        finite = np.flatnonzero(~np.isfinite(source))
        stop = int(finite[0]) if len(finite) else len(source)
        if stop == 0:
            return result

        current = source[:stop]
        index = np.arange(stop, dtype=np.float64)
        count = index

        def prior_sum(terms: FloatArray) -> FloatArray:
            return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(terms[:-1])))

        offset = current[0]
        centered = current - offset
        sum_centered = prior_sum(centered)
        sum_centered2 = prior_sum(centered * centered)
        sum_index_centered = prior_sum(index * centered)
        safe_count = np.maximum(count, 1.0)
        centered_mean = sum_centered / safe_count
        mean = offset + centered_mean
        if self.detrend:
            time_mean = (count - 1.0) / 2.0
            denominator = count * (count**2 - 1.0) / 12.0
            covariance = sum_index_centered - time_mean * sum_centered
            slope = covariance / np.maximum(denominator, self.epsilon)
            prediction = offset + centered_mean + slope * (index - time_mean)
        else:
            prediction = mean
        variance = np.maximum(
            (sum_centered2 - sum_centered * centered_mean)
            / np.maximum(count - 1.0, 1.0),
            0.0,
        )
        scale = np.maximum(np.sqrt(variance), self.epsilon)
        transformed = (current - prediction) / scale
        transformed[count < 2] = 0.0
        result[:stop] = transformed
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
            if end_time + horizon_s > dataset.time_s[-1]:
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
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("labels and scores must be non-empty matching vectors")
    if not np.all(np.isin(labels, (0, 1))) or not np.all(np.isfinite(scores)):
        raise ValueError("AUC requires binary labels and finite scores")
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
