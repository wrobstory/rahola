"""Causally normalized roll/rate windows shared by every learned detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rahola.dataset import SimulationDataset
from rahola.windowing import CausalTransformer
from rahola_lab.constants import (
    EWS_HORIZON_PERIODS,
    EWS_WINDOW_PERIODS,
    EXCLUSION_BUFFER_PERIODS,
)


@dataclass(frozen=True)
class DetectorWindowDataset:
    features: NDArray[np.float32]
    labels: NDArray[np.int8]
    family_labels: NDArray[np.int8]
    trajectory_indices: NDArray[np.int64]
    end_times_s: NDArray[np.float64]
    raw_angle_rad: NDArray[np.float64]
    raw_rate_rad_s: NDArray[np.float64]


_FAMILY_LABEL = {"softening": 0, "parametric": 1, "biased": 2}


def extract_detector_windows(
    dataset: SimulationDataset,
    *,
    stride_s: float = 10.0,
    max_windows_per_trajectory: int | None = None,
) -> DetectorWindowDataset:
    """Extract frozen 60-period histories and 50-period horizon labels.

    Each channel is standardized and detrended sample-by-sample using statistics
    fitted strictly before that sample. No per-window or per-trajectory future
    statistic enters the feature tensor.
    """
    time = dataset.time_s
    dt = float(np.median(np.diff(time)))
    period = float(dataset.config["natural_period_s"])
    length = round(EWS_WINDOW_PERIODS * period / dt)
    stride = max(1, round(stride_s / dt))
    horizon_s = EWS_HORIZON_PERIODS * period
    buffer_s = EXCLUSION_BUFFER_PERIODS * period
    transformer = CausalTransformer(detrend=True)
    features: list[np.ndarray] = []
    labels: list[int] = []
    families: list[int] = []
    trajectories: list[int] = []
    end_times: list[float] = []
    raw_angle: list[float] = []
    raw_rate: list[float] = []
    family = _FAMILY_LABEL[str(dataset.config["family"])]
    for trajectory in range(dataset.batch_size):
        angle = dataset.angle_rad[trajectory]
        rate = dataset.rate_rad_s[trajectory]
        normalized = np.column_stack((transformer.transform(angle), transformer.transform(rate)))
        cap_time = float(dataset.t_capsize_s[trajectory])
        candidates: list[tuple[int, int]] = []
        for end in range(length - 1, len(time), stride):
            end_time = float(time[end])
            if np.isfinite(cap_time) and end_time >= cap_time:
                break
            window = normalized[end - length + 1 : end + 1]
            if not np.all(np.isfinite(window)):
                break
            if not np.isfinite(cap_time):
                label = 0
            else:
                lead = cap_time - end_time
                if 0.0 < lead <= horizon_s:
                    label = 1
                elif lead <= horizon_s + buffer_s:
                    continue
                else:
                    label = 0
            candidates.append((end, label))
        if max_windows_per_trajectory is not None and len(candidates) > max_windows_per_trajectory:
            chosen = np.linspace(0, len(candidates) - 1, max_windows_per_trajectory, dtype=np.int64)
            candidates = [candidates[index] for index in chosen]
        for end, label in candidates:
            features.append(normalized[end - length + 1 : end + 1].astype(np.float32))
            labels.append(label)
            families.append(family)
            trajectories.append(trajectory)
            end_times.append(float(time[end]))
            raw_angle.append(float(angle[end]))
            raw_rate.append(float(rate[end]))
    empty = (0, length, 2)
    return DetectorWindowDataset(
        features=np.stack(features) if features else np.empty(empty, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        family_labels=np.asarray(families, dtype=np.int8),
        trajectory_indices=np.asarray(trajectories, dtype=np.int64),
        end_times_s=np.asarray(end_times, dtype=np.float64),
        raw_angle_rad=np.asarray(raw_angle, dtype=np.float64),
        raw_rate_rad_s=np.asarray(raw_rate, dtype=np.float64),
    )
