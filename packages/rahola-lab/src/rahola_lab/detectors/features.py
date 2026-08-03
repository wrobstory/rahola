"""Engineered motion features for the fixed Prototype #3 XGBoost ablation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

from rahola.config import SimulationConfig
from rahola_lab.detectors.data import DetectorWindowDataset
from rahola_lab.detectors.ews import classical_ews_scores
from rahola_lab.detectors.glrt import galeazzi_roll_power_glrt
from rahola_lab.detectors.neighbor import neighbor_count_scores
from rahola_lab.forecast import fit_piecewise_linear_restoring

ENGINEERED_FEATURE_NAMES = (
    "roll_variance",
    "roll_ac1",
    "variance_kendall_tau",
    "ac1_kendall_tau",
    "envelope_mean",
    "envelope_std",
    "envelope_max",
    "envelope_trend",
    "estimated_period_s",
    "danger_margin",
    "neighbor_count_score",
    "glrt_statistic",
    "absolute_roll",
    "absolute_roll_rate",
)


def engineered_features(
    windows: DetectorWindowDataset,
    config: SimulationConfig,
    *,
    neighbor_radius: float,
) -> NDArray[np.float64]:
    values = np.asarray(windows.features, dtype=np.float64)
    roll = values[:, :, 0]
    centered_left = roll[:, :-1] - np.mean(roll[:, :-1], axis=1, keepdims=True)
    centered_right = roll[:, 1:] - np.mean(roll[:, 1:], axis=1, keepdims=True)
    denominator = np.sqrt(np.mean(centered_left**2, axis=1) * np.mean(centered_right**2, axis=1))
    ac1 = np.divide(
        np.mean(centered_left * centered_right, axis=1),
        denominator,
        out=np.zeros(len(roll)),
        where=denominator > 1e-12,
    )
    envelope = np.abs(hilbert(roll, axis=1))
    index = np.arange(roll.shape[1], dtype=np.float64)
    index -= np.mean(index)
    envelope_trend = envelope @ index / np.sum(index**2)
    dt_s = 1.0 / config.output_rate_hz
    spectrum = np.abs(np.fft.rfft(roll - np.mean(roll, axis=1, keepdims=True), axis=1))
    frequencies = np.fft.rfftfreq(roll.shape[1], dt_s)
    spectrum[:, 0] = 0.0
    peak_frequency = frequencies[np.argmax(spectrum, axis=1)]
    estimated_period = np.divide(
        1.0,
        peak_frequency,
        out=np.full(len(roll), config.natural_period_s),
        where=peak_frequency > 0.0,
    )
    samples_per_period = round(config.natural_period_s / dt_s)
    danger = fit_piecewise_linear_restoring(config.to_dict()).danger_score(
        windows.raw_angle_rad, windows.raw_rate_rad_s
    )
    columns = (
        np.var(roll, axis=1, ddof=1),
        ac1,
        classical_ews_scores(values, statistic="variance", subwindow_fraction=0.35),
        classical_ews_scores(values, statistic="ac1", subwindow_fraction=0.35),
        np.mean(envelope, axis=1),
        np.std(envelope, axis=1, ddof=1),
        np.max(envelope, axis=1),
        envelope_trend,
        estimated_period,
        danger,
        neighbor_count_scores(
            values, radius=neighbor_radius, samples_per_period=samples_per_period
        ),
        galeazzi_roll_power_glrt(values, samples_per_period=samples_per_period),
        np.abs(windows.raw_angle_rad),
        np.abs(windows.raw_rate_rad_s),
    )
    return np.column_stack(columns)
