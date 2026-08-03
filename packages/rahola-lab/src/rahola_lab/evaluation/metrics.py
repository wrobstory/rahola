"""Episode-level alarm metrics shared by Rahola prototypes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from rahola_lab.evaluation.episodes import AlarmEpisode, EpisodeConfig, alarm_episodes


@dataclass(frozen=True)
class TrajectoryScores:
    times_s: NDArray[np.float64]
    scores: NDArray[np.float64]
    record_end_s: float
    t_capsize_s: float | None = None
    record_start_s: float = 0.0


@dataclass(frozen=True)
class AlarmMetrics:
    false_positives_per_hour: float
    sensitivity: float
    lead_times_s: NDArray[np.float64]
    false_episode_count: int
    capsize_count: int
    detected_capsize_count: int
    exposure_hours: float


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    metrics: AlarmMetrics
    lead_time_quantiles_s: tuple[float, float, float]


def _associated_episode(
    episodes: tuple[AlarmEpisode, ...], t_capsize_s: float, horizon_s: float
) -> AlarmEpisode | None:
    horizon_start = t_capsize_s - horizon_s
    eligible = [
        episode
        for episode in episodes
        if episode.start_s < t_capsize_s and episode.end_s >= horizon_start
    ]
    return min(eligible, key=lambda episode: episode.start_s) if eligible else None


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
    unless it overlaps the open pre-capsize
    horizon ``[t_capsize-horizon_s, t_capsize)``; a sustained alarm does not
    become false merely because it opened earlier. Sensitivity is the fraction
    of capsize events with such an episode. Lead time uses its first sustained
    alarm time and may therefore exceed the nominal horizon.
    """
    if horizon_s <= 0 or not trajectories:
        raise ValueError("a positive horizon and at least one trajectory are required")
    exposure_s = 0.0
    false_episodes = 0
    capsize_count = 0
    detected = 0
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
        episodes = alarm_episodes(trajectory.times_s, trajectory.scores, episode_config)
        associated: AlarmEpisode | None = None
        if observable_capsize is not None:
            capsize_count += 1
            associated = _associated_episode(episodes, observable_capsize, horizon_s)
            if associated is not None:
                detected += 1
                lead_times.append(observable_capsize - associated.start_s)
        false_episodes += sum(episode is not associated for episode in episodes)
    exposure_hours = exposure_s / 3600.0
    return AlarmMetrics(
        false_positives_per_hour=false_episodes / exposure_hours if exposure_hours else 0.0,
        sensitivity=detected / capsize_count if capsize_count else 0.0,
        lead_times_s=np.asarray(lead_times, dtype=np.float64),
        false_episode_count=false_episodes,
        capsize_count=capsize_count,
        detected_capsize_count=detected,
        exposure_hours=exposure_hours,
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
