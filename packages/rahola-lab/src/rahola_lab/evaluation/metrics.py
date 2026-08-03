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
    eligible = [episode for episode in episodes if 0.0 < t_capsize_s - episode.start_s <= horizon_s]
    return min(eligible, key=lambda episode: episode.start_s) if eligible else None


def evaluate_alarms(
    trajectories: list[TrajectoryScores],
    episode_config: EpisodeConfig,
    *,
    horizon_s: float,
) -> AlarmMetrics:
    """Compute episode metrics using pre-event exposure.

    Exposure is the sum of each trajectory's observable non-capsized duration:
    ``min(record_end_s, t_capsize_s)`` for capsized runs and ``record_end_s``
    otherwise. An episode is false unless its start precedes a capsize by no more
    than ``horizon_s``. Sensitivity is the fraction of capsize events with such
    an episode. Lead time uses the first sustained eligible alarm.
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
        exposure_s += (
            min(trajectory.record_end_s, capsize)
            if capsize is not None
            else trajectory.record_end_s
        )
        episodes = alarm_episodes(trajectory.times_s, trajectory.scores, episode_config)
        associated: AlarmEpisode | None = None
        if capsize is not None:
            capsize_count += 1
            associated = _associated_episode(episodes, capsize, horizon_s)
            if associated is not None:
                detected += 1
                lead_times.append(capsize - associated.start_s)
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
