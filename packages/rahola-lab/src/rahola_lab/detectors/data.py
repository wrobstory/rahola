"""Selectable past-only preprocessing for detector roll/rate windows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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


class NormalizationMode(StrEnum):
    """Frozen v0.2 detector preprocessing modes."""

    PHYSICAL = "physical"
    FIXED_WINDOW_CAUSAL = "fixed_window_causal"
    CUMULATIVE_ONLINE = "cumulative_online"


def _fixed_window_causal(values: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Fit one linear trend and residual scale to a scored past-only window."""
    source = np.asarray(values, dtype=np.float64)
    index = np.arange(len(source), dtype=np.float64)
    centered_index = index - np.mean(index)
    centered_source = source - np.mean(source)
    denominator = float(centered_index @ centered_index)
    slope = float(centered_index @ centered_source) / max(denominator, epsilon)
    residual = centered_source - slope * centered_index
    scale = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    return residual / max(scale, epsilon)


def _window_features(
    angle: np.ndarray,
    rate: np.ndarray,
    *,
    mode: NormalizationMode,
    escape_angle_rad: float,
    omega_n_rad_s: float,
) -> np.ndarray:
    if mode == NormalizationMode.PHYSICAL:
        return np.column_stack(
            (
                angle / escape_angle_rad,
                rate / (omega_n_rad_s * escape_angle_rad),
            )
        )
    if mode == NormalizationMode.FIXED_WINDOW_CAUSAL:
        return np.column_stack(
            (_fixed_window_causal(angle), _fixed_window_causal(rate))
        )
    raise ValueError("cumulative-online windows must be prepared from the full prior record")


def extract_detector_windows(
    dataset: SimulationDataset,
    *,
    stride_s: float = 10.0,
    max_windows_per_trajectory: int | None = None,
    allow_censored_for_inference: bool = False,
    normalization_mode: NormalizationMode | str = NormalizationMode.CUMULATIVE_ONLINE,
) -> DetectorWindowDataset:
    """Extract frozen 60-period histories and 50-period horizon labels.

    All modes exclude motion after the scoring endpoint. Physical mode applies
    the configured nondimensionalization. Fixed-window mode fits one trend and
    scale to the complete scored history and applies them uniformly. Cumulative
    mode retains the historical sample-by-sample prior-only transformer.
    """
    mode = NormalizationMode(normalization_mode)
    time = dataset.time_s
    dt = float(np.median(np.diff(time)))
    period = float(dataset.config["natural_period_s"])
    length = round(EWS_WINDOW_PERIODS * period / dt)
    stride = max(1, round(stride_s / dt))
    horizon_s = EWS_HORIZON_PERIODS * period
    buffer_s = EXCLUSION_BUFFER_PERIODS * period
    transformer = CausalTransformer(detrend=True)
    escape_angle = float(dataset.config["escape_angle_rad"])
    omega_n = 2.0 * np.pi / period
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
        cumulative = None
        if mode == NormalizationMode.CUMULATIVE_ONLINE:
            cumulative = np.column_stack(
                (transformer.transform(angle), transformer.transform(rate))
            )
        cap_time = float(dataset.t_capsize_s[trajectory])
        candidates: list[tuple[int, int]] = []
        for end in range(length - 1, len(time), stride):
            end_time = float(time[end])
            if np.isfinite(cap_time) and end_time >= cap_time:
                break
            window_slice = slice(end - length + 1, end + 1)
            raw_angle_window = angle[window_slice]
            raw_rate_window = rate[window_slice]
            if not np.all(np.isfinite(raw_angle_window)) or not np.all(
                np.isfinite(raw_rate_window)
            ):
                break
            if allow_censored_for_inference:
                label = -1
                candidates.append((end, label))
                continue
            if end_time + horizon_s > float(time[-1]):
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
            window_slice = slice(end - length + 1, end + 1)
            window = (
                cumulative[window_slice]
                if cumulative is not None
                else _window_features(
                    angle[window_slice],
                    rate[window_slice],
                    mode=mode,
                    escape_angle_rad=escape_angle,
                    omega_n_rad_s=omega_n,
                )
            )
            if not np.all(np.isfinite(window)):
                raise ValueError("normalization produced non-finite detector features")
            features.append(window.astype(np.float32))
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


def acausal_whole_record_features(
    dataset: SimulationDataset, windows: DetectorWindowDataset
) -> NDArray[np.float32]:
    """Rebuild selected windows using deliberately acausal record normalization.

    Each trajectory's mean and scale use every finite sample in that record,
    including samples after the scored window. This helper exists only for the
    Prototype #3 diagnostic appendix and must never feed an operational result.
    """
    length = windows.features.shape[1]
    output = np.empty_like(windows.features)
    for trajectory in np.unique(windows.trajectory_indices):
        selected = np.flatnonzero(windows.trajectory_indices == trajectory)
        channels = np.column_stack(
            (dataset.angle_rad[trajectory], dataset.rate_rad_s[trajectory])
        ).astype(np.float64)
        normalized = np.empty_like(channels)
        for channel in range(2):
            finite = np.isfinite(channels[:, channel])
            mean = float(np.mean(channels[finite, channel]))
            scale = float(np.std(channels[finite, channel], ddof=1))
            normalized[:, channel] = (channels[:, channel] - mean) / max(scale, 1e-12)
        for row in selected:
            end = int(np.searchsorted(dataset.time_s, windows.end_times_s[row]))
            output[row] = normalized[end - length + 1 : end + 1]
    return output
