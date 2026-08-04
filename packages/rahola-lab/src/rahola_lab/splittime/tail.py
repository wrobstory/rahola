"""Physics-informed exponential tail with Gamma-rate shrinkage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GammaRatePrior:
    """Gamma prior in shape-rate form for the exponential rate ``theta``."""

    shape: float
    rate: float
    threshold_w: float | None = None
    exceedance_probability: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.shape) or not np.isfinite(self.rate):
            raise ValueError("Gamma prior parameters must be finite")
        if self.shape <= 0.0 or self.rate <= 0.0:
            raise ValueError("Gamma prior parameters must be positive")
        if self.threshold_w is not None and not 0.0 <= self.threshold_w < 1.0:
            raise ValueError("fixed tail threshold must lie in [0, 1)")
        if self.exceedance_probability is not None and not (
            0.0 < self.exceedance_probability <= 1.0
        ):
            raise ValueError("prior exceedance probability must lie in (0, 1]")
        if (self.threshold_w is None) != (self.exceedance_probability is None):
            raise ValueError("fixed threshold and exceedance probability must be supplied together")

    @property
    def mean_rate(self) -> float:
        return self.shape / self.rate

    @classmethod
    def from_mean(
        cls,
        mean_rate: float,
        strength: float,
        *,
        threshold_w: float | None = None,
        exceedance_probability: float | None = None,
    ) -> GammaRatePrior:
        if not np.isfinite(mean_rate) or mean_rate <= 0.0:
            raise ValueError("prior mean rate must be finite and positive")
        return cls(
            shape=float(strength),
            rate=float(strength) / mean_rate,
            threshold_w=threshold_w,
            exceedance_probability=exceedance_probability,
        )


@dataclass(frozen=True)
class ExponentialTailEstimate:
    threshold_w: float
    crossing_count: int
    exceedance_count: int
    exceedance_sum: float
    posterior_shape: float
    posterior_rate: float
    predictive_exceedance: float
    critical_probability: float
    threshold_clipped: bool

    @property
    def posterior_mean_rate(self) -> float:
        return self.posterior_shape / self.posterior_rate


def _threshold(values: NDArray[np.float64], quantile: float) -> tuple[float, bool]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("tail quantile must lie in (0, 1)")
    threshold = float(np.quantile(values, quantile))
    limit = float(np.nextafter(1.0, -np.inf))
    clipped = threshold >= 1.0
    return (limit if clipped else threshold), clipped


def exponential_rate_mle(severities_u: NDArray[np.floating], *, quantile: float) -> float:
    """Fit the exponential rate over pooled calibration exceedances."""
    values = np.asarray(severities_u, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("severities must be a nonempty finite vector")
    threshold, _ = _threshold(values, quantile)
    exceedances = values[values > threshold] - threshold
    total = float(exceedances.sum())
    if len(exceedances) == 0 or total <= 0.0:
        raise ValueError("calibration data contain no positive tail exceedances")
    return len(exceedances) / total


def estimate_exponential_tail(
    severities_u: NDArray[np.floating],
    *,
    quantile: float,
    prior: GammaRatePrior,
    critical_level: float = 1.0,
) -> ExponentialTailEstimate:
    """Return the conjugate posterior-predictive critical probability."""
    values = np.asarray(severities_u, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("severities must be a finite vector")
    if not len(values) and prior.threshold_w is None:
        raise ValueError("empty severities require a pooled fixed-threshold prior")
    if critical_level != 1.0:
        raise ValueError("U1 fixes normalized critical severity at one")
    if prior.threshold_w is None:
        threshold, clipped = _threshold(values, quantile)
    else:
        threshold = prior.threshold_w
        clipped = False
    exceedances = values[values > threshold] - threshold
    exceedance_sum = float(exceedances.sum())
    posterior_shape = prior.shape + len(exceedances)
    posterior_rate = prior.rate + exceedance_sum
    distance = critical_level - threshold
    predictive = (posterior_rate / (posterior_rate + distance)) ** posterior_shape
    exceedance_probability = (
        len(exceedances) / len(values)
        if prior.exceedance_probability is None
        else prior.exceedance_probability
    )
    critical_probability = exceedance_probability * predictive
    return ExponentialTailEstimate(
        threshold_w=threshold,
        crossing_count=len(values),
        exceedance_count=len(exceedances),
        exceedance_sum=exceedance_sum,
        posterior_shape=posterior_shape,
        posterior_rate=posterior_rate,
        predictive_exceedance=predictive,
        critical_probability=critical_probability,
        threshold_clipped=clipped,
    )
