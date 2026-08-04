"""Causal assembly of the online split-time capsize-rate estimate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rahola_lab.constants import (
    U1_DECORRELATION_SIGNIFICANCE,
    U1_EMISSION_CADENCE_S,
    U1_INTERVAL_CADENCE_S,
    U1_MIN_EXCEEDANCES,
    U1_PARAMETRIC_BOOTSTRAP_DRAWS,
    U1_PARAMETRIC_BOOTSTRAP_SEED,
)
from rahola_lab.forecast import DangerMarginFit
from rahola_lab.splittime.crossings import (
    Crossing,
    decluster_crossings,
    detect_crossings,
    roll_decorrelation_time,
)
from rahola_lab.splittime.tail import (
    ExponentialTailEstimate,
    GammaRatePrior,
    estimate_exponential_tail,
)


@dataclass(frozen=True)
class SplitTimeConfig:
    tail_quantile: float
    trailing_window_s: float | None
    emission_cadence_s: float = U1_EMISSION_CADENCE_S
    interval_cadence_s: float = U1_INTERVAL_CADENCE_S
    bootstrap_draws: int = U1_PARAMETRIC_BOOTSTRAP_DRAWS
    bootstrap_seed: int = U1_PARAMETRIC_BOOTSTRAP_SEED
    minimum_exceedances: int = U1_MIN_EXCEEDANCES
    decorrelation_significance: float = U1_DECORRELATION_SIGNIFICANCE

    def __post_init__(self) -> None:
        if not 0.0 < self.tail_quantile < 1.0:
            raise ValueError("tail quantile must lie in (0, 1)")
        if self.trailing_window_s is not None and self.trailing_window_s <= 0.0:
            raise ValueError("trailing window must be positive")
        if self.emission_cadence_s <= 0.0 or self.interval_cadence_s <= 0.0:
            raise ValueError("emission and interval cadences must be positive")
        if self.bootstrap_draws < 500:
            raise ValueError("parametric bootstrap requires at least 500 draws")
        if self.minimum_exceedances < 1:
            raise ValueError("minimum exceedances must be positive")


@dataclass(frozen=True)
class RateEmission:
    time_s: float
    rate_per_hour: float
    interval_lower_per_hour: float
    interval_upper_per_hour: float
    exposure_s: float
    decorrelation_time_s: float
    retained_crossings: int
    exceedances: int
    positive_crossings: int
    negative_crossings: int
    tail_threshold_w: float
    critical_probability: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class RateTrajectory:
    emissions: tuple[RateEmission, ...]
    integrated_count: float
    integrated_count_draws: NDArray[np.float64]
    all_crossings: tuple[Crossing, ...]

    @property
    def integrated_interval(self) -> tuple[float, float]:
        if not len(self.integrated_count_draws):
            return (0.0, 0.0)
        return tuple(
            float(value) for value in np.quantile(self.integrated_count_draws, [0.025, 0.975])
        )


def _finite_stop(angle: NDArray[np.float64], rate: NDArray[np.float64]) -> int:
    finite = np.isfinite(angle) & np.isfinite(rate)
    invalid = np.flatnonzero(~finite)
    stop = int(invalid[0]) if len(invalid) else len(angle)
    if np.any(finite[stop:]):
        raise ValueError("non-finite motion samples must end the stream")
    return stop


def _tail_for_crossings(
    crossings: tuple[Crossing, ...], config: SplitTimeConfig, prior: GammaRatePrior
) -> ExponentialTailEstimate | None:
    if not crossings:
        return None
    values = np.asarray([event.severity_u for event in crossings], dtype=np.float64)
    estimate = estimate_exponential_tail(values, quantile=config.tail_quantile, prior=prior)
    return estimate if estimate.exceedance_count >= config.minimum_exceedances else None


def _bootstrap_rate(
    tail: ExponentialTailEstimate,
    exposure_s: float,
    *,
    draws: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    count_star = rng.poisson(tail.crossing_count, size=draws)
    exceedance_fraction = tail.exceedance_count / tail.crossing_count
    exceedance_star = rng.binomial(count_star, exceedance_fraction)
    theta_star = rng.gamma(
        shape=tail.posterior_shape,
        scale=1.0 / tail.posterior_rate,
        size=draws,
    )
    conditional = np.exp(-theta_star * (1.0 - tail.threshold_w))
    return exceedance_star / exposure_s * conditional * 3_600.0


def estimate_rate_trajectory(
    time_s: NDArray[np.floating],
    angle_rad: NDArray[np.floating],
    rate_rad_s: NDArray[np.floating],
    fit: DangerMarginFit,
    *,
    prior: GammaRatePrior,
    config: SplitTimeConfig,
) -> RateTrajectory:
    """Emit a causal rate path through the finite portion of one trajectory."""
    time = np.asarray(time_s, dtype=np.float64)
    angle = np.asarray(angle_rad, dtype=np.float64)
    rate = np.asarray(rate_rad_s, dtype=np.float64)
    if time.ndim != 1 or angle.shape != time.shape or rate.shape != time.shape:
        raise ValueError("time, angle, and rate must be matching vectors")
    if len(time) < 3 or not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be a finite, strictly increasing vector")
    stop = _finite_stop(angle, rate)
    if stop < 3:
        return RateTrajectory((), 0.0, np.zeros(config.bootstrap_draws), ())
    finite_end_s = float(time[stop - 1])
    sample_interval_s = float(np.median(np.diff(time)))
    all_crossings = detect_crossings(time, angle, rate, fit)
    emission_times = np.arange(
        float(time[0]) + config.emission_cadence_s,
        finite_end_s + 0.5 * config.emission_cadence_s,
        config.emission_cadence_s,
    )
    rng = np.random.default_rng(config.bootstrap_seed)
    emissions: list[RateEmission] = []
    emission_draws: list[NDArray[np.float64]] = []
    last_interval_time = -np.inf
    current_draws: NDArray[np.float64] | None = None

    for emission_time in emission_times:
        end = int(np.searchsorted(time[:stop], emission_time, side="right"))
        if end < 3:
            continue
        configured_start_s = (
            float(time[0])
            if config.trailing_window_s is None
            else max(float(time[0]), emission_time - config.trailing_window_s)
        )

        def retained_from(
            start_s: float,
            *,
            current_end: int = end,
            current_emission_time: float = float(emission_time),
        ) -> tuple[tuple[Crossing, ...], float]:
            start = int(np.searchsorted(time[:current_end], start_s, side="left"))
            history = angle[start:current_end]
            decorrelation = roll_decorrelation_time(
                history,
                sample_interval_s,
                significance_level=config.decorrelation_significance,
            )
            candidates = tuple(
                event
                for event in all_crossings
                if start_s <= event.time_s <= current_emission_time
            )
            return decluster_crossings(candidates, decorrelation), decorrelation

        retained, decorrelation_time = retained_from(configured_start_s)
        tail = _tail_for_crossings(retained, config, prior)
        flags: list[str] = []
        start_s = configured_start_s
        if tail is None and configured_start_s > float(time[0]):
            start_s = float(time[0])
            retained, decorrelation_time = retained_from(start_s)
            tail = _tail_for_crossings(retained, config, prior)
            flags.append("full_history_tail_fallback")
        if tail is None:
            continue
        if tail.threshold_clipped:
            flags.append("tail_threshold_clipped")
        exposure_s = emission_time - start_s
        if exposure_s <= 0.0:
            continue
        rate_per_hour = (
            tail.crossing_count / exposure_s * tail.critical_probability * 3_600.0
        )
        if (
            current_draws is None
            or emission_time - last_interval_time >= config.interval_cadence_s - 1e-9
        ):
            current_draws = _bootstrap_rate(
                tail,
                exposure_s,
                draws=config.bootstrap_draws,
                rng=rng,
            )
            last_interval_time = emission_time
        lower, upper = np.quantile(current_draws, [0.025, 0.975])
        positive = sum(event.side == 1 for event in retained)
        negative = len(retained) - positive
        emissions.append(
            RateEmission(
                time_s=float(emission_time),
                rate_per_hour=float(rate_per_hour),
                interval_lower_per_hour=float(lower),
                interval_upper_per_hour=float(upper),
                exposure_s=float(exposure_s),
                decorrelation_time_s=float(decorrelation_time),
                retained_crossings=tail.crossing_count,
                exceedances=tail.exceedance_count,
                positive_crossings=positive,
                negative_crossings=negative,
                tail_threshold_w=tail.threshold_w,
                critical_probability=tail.critical_probability,
                flags=tuple(flags),
            )
        )
        emission_draws.append(current_draws.copy())

    integrated = 0.0
    integrated_draws = np.zeros(config.bootstrap_draws, dtype=np.float64)
    for index, emission in enumerate(emissions):
        next_time = emissions[index + 1].time_s if index + 1 < len(emissions) else finite_end_s
        duration_s = max(0.0, next_time - emission.time_s)
        integrated += emission.rate_per_hour * duration_s / 3_600.0
        integrated_draws += emission_draws[index] * duration_s / 3_600.0
    return RateTrajectory(
        emissions=tuple(emissions),
        integrated_count=float(integrated),
        integrated_count_draws=integrated_draws,
        all_crossings=all_crossings,
    )
