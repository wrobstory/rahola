"""Episode-level alarm metrics shared by Rahola prototypes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta

from rahola_lab.evaluation.episodes import AlarmEpisode, EpisodeConfig, alarm_episodes


@dataclass(frozen=True)
class TrajectoryScores:
    times_s: NDArray[np.float64]
    scores: NDArray[np.float64]
    record_end_s: float
    t_capsize_s: float | None = None
    record_start_s: float = 0.0


@dataclass(frozen=True)
class RateInterval:
    lower: float
    upper: float
    confidence_level: float = 0.95


@dataclass(frozen=True)
class AlarmMetrics:
    false_positives_per_hour: float
    false_positives_per_hour_interval: RateInterval
    sensitivity: float
    sensitivity_interval: RateInterval
    lead_times_s: NDArray[np.float64]
    false_episode_count: int
    capsize_count: int
    detected_capsize_count: int
    exposure_hours: float
    alarm_opportunity_count: int


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    metrics: AlarmMetrics
    lead_time_quantiles_s: tuple[float, float, float]


def clopper_pearson_interval(
    successes: int, trials: int, *, confidence_level: float = 0.95
) -> RateInterval:
    """Exact equal-tailed interval for a binomial success probability."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes must lie between zero and trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")
    if trials == 0:
        return RateInterval(float("nan"), float("nan"), confidence_level)
    tail = (1.0 - confidence_level) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - tail, successes + 1, trials - successes))
    )
    return RateInterval(lower, upper, confidence_level)


def _associated_episodes(
    episodes: tuple[AlarmEpisode, ...], t_capsize_s: float, horizon_s: float
) -> tuple[AlarmEpisode, ...]:
    horizon_start = t_capsize_s - horizon_s
    return tuple(
        episode
        for episode in episodes
        if episode.start_s < t_capsize_s and episode.end_s >= horizon_start
    )


def evaluate_alarms(
    trajectories: list[TrajectoryScores],
    episode_config: EpisodeConfig,
    *,
    horizon_s: float,
) -> AlarmMetrics:
    """Compute episode metrics using pre-event exposure.

    Exposure is the sum of each trajectory's scorable non-capsized duration:
    ``min(record_end_s, t_capsize_s) - record_start_s`` for capsizes after the
    observation start and ``record_end_s - record_start_s`` otherwise. Events
    before observation starts are outside the risk set. An episode is false
    unless it overlaps the open pre-capsize horizon
    ``[t_capsize-horizon_s, t_capsize)``; *all* overlapping episodes are event
    associated, so repeated warnings inside that horizon are not false alarms.
    Sensitivity is the fraction of capsize events with at least one associated
    episode. Lead time uses the earliest associated episode and may exceed the
    nominal horizon.

    Reported 95% intervals use Clopper--Pearson binomial quantiles. Sensitivity
    trials are observable capsize events. For false episodes, each scorable
    window is an alarm-opening opportunity and the probability interval is
    rescaled by observed opportunities per exposure hour. Debounce/refractory
    decluster alarms, but residual serial dependence means this interval is a
    binomial-opportunity convention, not a proof of independent trials; full
    decorrelation-time declustering is deferred to Prototype #2.
    """
    if horizon_s <= 0 or not trajectories:
        raise ValueError("a positive horizon and at least one trajectory are required")
    exposure_s = 0.0
    false_episodes = 0
    capsize_count = 0
    detected = 0
    alarm_opportunities = 0
    lead_times: list[float] = []
    for trajectory in trajectories:
        capsize = trajectory.t_capsize_s
        observable_capsize = (
            capsize if capsize is not None and capsize > trajectory.record_start_s else None
        )
        if capsize is not None and capsize <= trajectory.record_start_s:
            event_end = trajectory.record_start_s
        else:
            event_end = min(trajectory.record_end_s, observable_capsize or trajectory.record_end_s)
        exposure_s += max(0.0, event_end - trajectory.record_start_s)
        alarm_opportunities += int(
            np.sum(
                (trajectory.times_s >= trajectory.record_start_s)
                & (trajectory.times_s <= event_end)
            )
        )
        episodes = alarm_episodes(trajectory.times_s, trajectory.scores, episode_config)
        associated: tuple[AlarmEpisode, ...] = ()
        if observable_capsize is not None:
            capsize_count += 1
            associated = _associated_episodes(episodes, observable_capsize, horizon_s)
            if associated:
                detected += 1
                first = min(associated, key=lambda episode: episode.start_s)
                lead_times.append(observable_capsize - first.start_s)
        false_episodes += sum(episode not in associated for episode in episodes)
    exposure_hours = exposure_s / 3600.0
    sensitivity_interval = clopper_pearson_interval(detected, capsize_count)
    false_probability_interval = clopper_pearson_interval(false_episodes, alarm_opportunities)
    opportunities_per_hour = alarm_opportunities / exposure_hours if exposure_hours else 0.0
    return AlarmMetrics(
        false_positives_per_hour=false_episodes / exposure_hours if exposure_hours else 0.0,
        false_positives_per_hour_interval=RateInterval(
            false_probability_interval.lower * opportunities_per_hour,
            false_probability_interval.upper * opportunities_per_hour,
        ),
        sensitivity=detected / capsize_count if capsize_count else 0.0,
        sensitivity_interval=sensitivity_interval,
        lead_times_s=np.asarray(lead_times, dtype=np.float64),
        false_episode_count=false_episodes,
        capsize_count=capsize_count,
        detected_capsize_count=detected,
        exposure_hours=exposure_hours,
        alarm_opportunity_count=alarm_opportunities,
    )


def operating_curve(
    trajectories: list[TrajectoryScores],
    episode_config: EpisodeConfig,
    thresholds: NDArray[np.floating],
    *,
    horizon_s: float,
) -> tuple[OperatingPoint, ...]:
    """Sweep score thresholds and retain sensitivity, FPR/h, and lead quantiles."""
    points: list[OperatingPoint] = []
    for threshold in np.asarray(thresholds, dtype=np.float64):
        metrics = evaluate_alarms(
            trajectories,
            replace(episode_config, threshold=float(threshold)),
            horizon_s=horizon_s,
        )
        quantiles = (
            tuple(float(value) for value in np.quantile(metrics.lead_times_s, [0.1, 0.5, 0.9]))
            if len(metrics.lead_times_s)
            else (float("nan"),) * 3
        )
        points.append(OperatingPoint(float(threshold), metrics, quantiles))
    return tuple(points)
