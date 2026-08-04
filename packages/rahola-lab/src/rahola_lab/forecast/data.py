"""Causal history/maximum-future-roll target extraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rahola.dataset import SimulationDataset


@dataclass(frozen=True)
class ForecastDataset:
    histories: NDArray[np.float64]
    targets_rad: NDArray[np.float64]
    seeds: NDArray[np.uint64]
    history_end_s: NDArray[np.float64]
    trajectory_indices: NDArray[np.int64]
    horizons_s: tuple[float, ...]
    escape_angle_rad: float
    sample_dt_s: float


def absolute_roll_escape_angle(config: Mapping[str, object]) -> float:
    """Return the conservative escape magnitude used by scalar |roll| targets.

    Asymmetric families use the tighter side for both target extraction and
    alarm normalization. A future signed-target design may instead select the
    side from excursion direction, but scalar targets must not mix conventions.
    """
    positive_escape = float(config["escape_angle_rad"])
    configured_negative = config.get("negative_escape_angle_rad")
    negative_escape = (
        positive_escape if configured_negative is None else float(configured_negative)
    )
    return min(positive_escape, negative_escape)


def extract_forecast_dataset(
    dataset: SimulationDataset,
    *,
    history_s: float,
    horizons_s: tuple[float, ...],
    stride_s: float = 10.0,
    max_samples_per_trajectory: int | None = None,
    first_history_end_s: float | None = None,
) -> ForecastDataset:
    """Extract histories and future maximum-|roll| targets without future leakage.

    Candidate histories must be fully finite and horizons truncated by the
    nominal record end are dropped. If capsize occurs inside a horizon, its
    target is raised to at least the relevant escape angle even when absorbing
    termination leaves no sample exactly at the event time. For asymmetric
    biased-family runs, the scalar absolute-roll target conservatively uses the
    smaller of the positive and negative escape magnitudes for both directions.
    A positive excursion is therefore judged against the tighter negative-side
    margin; a signed target is deferred rather than implying side correctness.
    """
    if history_s <= 0 or not horizons_s or min(horizons_s) <= 0 or stride_s <= 0:
        raise ValueError("history, horizons, and stride must be positive")
    sample_dt = float(np.median(np.diff(dataset.time_s)))
    history_samples = round(history_s / sample_dt)
    stride_samples = max(1, round(stride_s / sample_dt))
    horizon_samples = tuple(round(value / sample_dt) for value in horizons_s)
    relevant_escape = absolute_roll_escape_angle(dataset.config)
    histories: list[np.ndarray] = []
    targets: list[list[float]] = []
    seeds: list[int] = []
    end_times: list[float] = []
    trajectory_indices: list[int] = []
    last_horizon = max(horizon_samples)
    first_end_index = history_samples - 1
    if first_history_end_s is not None:
        if first_history_end_s < history_s - sample_dt:
            raise ValueError("first history end cannot precede a complete history")
        first_end_index = max(first_end_index, round(first_history_end_s / sample_dt))
    for trajectory_index in range(dataset.batch_size):
        accepted = 0
        cap_time = dataset.t_capsize_s[trajectory_index]
        for end_index in range(first_end_index, len(dataset.time_s), stride_samples):
            end_time = float(dataset.time_s[end_index])
            if np.isfinite(cap_time) and end_time >= cap_time:
                break
            if end_index + last_horizon >= len(dataset.time_s):
                break
            start_index = end_index - history_samples + 1
            history = np.stack(
                (
                    dataset.angle_rad[trajectory_index, start_index : end_index + 1],
                    dataset.rate_rad_s[trajectory_index, start_index : end_index + 1],
                ),
                axis=-1,
            )
            if not np.all(np.isfinite(history)):
                continue
            row_targets: list[float] = []
            for horizon_s, horizon_count in zip(horizons_s, horizon_samples, strict=True):
                future = dataset.angle_rad[
                    trajectory_index, end_index + 1 : end_index + horizon_count + 1
                ]
                finite = np.abs(future[np.isfinite(future)])
                target = float(np.max(finite)) if len(finite) else 0.0
                if np.isfinite(cap_time) and 0.0 < cap_time - end_time <= horizon_s:
                    target = max(target, relevant_escape)
                row_targets.append(target)
            histories.append(history)
            targets.append(row_targets)
            seeds.append(int(dataset.seeds[trajectory_index]))
            end_times.append(end_time)
            trajectory_indices.append(trajectory_index)
            accepted += 1
            if max_samples_per_trajectory is not None and accepted >= max_samples_per_trajectory:
                break
    history_shape = (0, history_samples, 2)
    return ForecastDataset(
        histories=np.stack(histories) if histories else np.empty(history_shape, dtype=np.float64),
        targets_rad=np.asarray(targets, dtype=np.float64).reshape(-1, len(horizons_s)),
        seeds=np.asarray(seeds, dtype=np.uint64),
        history_end_s=np.asarray(end_times, dtype=np.float64),
        trajectory_indices=np.asarray(trajectory_indices, dtype=np.int64),
        horizons_s=horizons_s,
        escape_angle_rad=relevant_escape,
        sample_dt_s=sample_dt,
    )
