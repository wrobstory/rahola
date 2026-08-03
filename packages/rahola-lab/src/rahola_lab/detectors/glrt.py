"""Roll-only signal-power adaptation of Galeazzi's double-Weibull GLRT."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def galeazzi_roll_power_glrt(
    features: NDArray[np.floating],
    *,
    samples_per_period: int,
    shape: float = 0.55,
    band_fraction: float = 0.20,
) -> NDArray[np.float64]:
    """Score a scale increase in roll motion near its natural frequency.

    Galeazzi et al.'s original W2-GLRT monitors ``d=roll^2*pitch``. Rahola
    has no pitch channel and prohibits wave inputs, so this detector applies
    their equal-shape scale-change statistic (Eqs. 37 and 42) to band-limited
    roll. The detection segment remains the published four roll periods.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("features must have shape (windows, time, 2)")
    if samples_per_period < 2 or shape <= 0.0 or not 0.0 < band_fraction < 1.0:
        raise ValueError("invalid period, shape, or band")
    length = values.shape[1]
    detection_length = 4 * samples_per_period
    if detection_length >= length:
        raise ValueError("history must exceed the four-period detection segment")
    frequency = np.fft.rfftfreq(length, d=1.0)
    center = 1.0 / samples_per_period
    mask = (frequency >= center * (1.0 - band_fraction)) & (
        frequency <= center * (1.0 + band_fraction)
    )
    spectrum = np.fft.rfft(values[:, :, 0], axis=1)
    filtered = np.fft.irfft(spectrum * mask[None, :], n=length, axis=1)
    reference = filtered[:, :-detection_length]
    detection = filtered[:, -detection_length:]
    epsilon = 1e-12
    scale0 = np.maximum(np.mean(np.abs(reference) ** shape, axis=1), epsilon) ** (1.0 / shape)
    scale1 = np.maximum(np.mean(np.abs(detection) ** shape, axis=1), epsilon) ** (1.0 / shape)
    count = detection.shape[1]
    ratio = np.maximum(scale1 / scale0, epsilon)
    return count * (-shape * np.log(ratio) + ratio**shape - 1.0)
