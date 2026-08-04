"""Convert raw window scores into debounced alarm episodes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EpisodeConfig:
    threshold: float
    debounce_windows: int = 3
    refractory_windows: int = 3

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        for name, value in (
            ("debounce_windows", self.debounce_windows),
            ("refractory_windows", self.refractory_windows),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.debounce_windows < 1 or self.refractory_windows < 1:
            raise ValueError("debounce and refractory windows must be positive")


@dataclass(frozen=True)
class AlarmEpisode:
    start_index: int
    end_index: int
    start_s: float
    end_s: float
    maximum_score: float


def alarm_episodes(
    times_s: NDArray[np.floating],
    scores: NDArray[np.floating],
    config: EpisodeConfig,
) -> tuple[AlarmEpisode, ...]:
    """Apply threshold, debounce, and refractory logic.

    An episode starts at the window that confirms ``debounce_windows``
    consecutive flags, the first time the alarm can be issued. It remains open until
    ``refractory_windows`` consecutive unflagged windows confirm closure; its
    recorded end is the last flagged window, not the confirmation delay.
    """
    times = np.asarray(times_s, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("times and scores must be matching vectors")
    if not len(times):
        return ()
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("times and scores must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("window times must increase strictly")
    flagged = values >= config.threshold
    episodes: list[AlarmEpisode] = []
    opening_run = 0
    opening_start = -1
    active_start = -1
    last_flag = -1
    unflagged_run = 0
    maximum = -np.inf
    for index, is_flagged in enumerate(flagged):
        if active_start < 0:
            if is_flagged:
                if opening_run == 0:
                    opening_start = index
                opening_run += 1
                if opening_run >= config.debounce_windows:
                    active_start = index
                    last_flag = index
                    maximum = float(np.max(values[opening_start : index + 1]))
            else:
                opening_run = 0
                opening_start = -1
            continue
        if is_flagged:
            last_flag = index
            unflagged_run = 0
            maximum = max(maximum, float(values[index]))
        else:
            unflagged_run += 1
            if unflagged_run >= config.refractory_windows:
                episodes.append(
                    AlarmEpisode(
                        start_index=active_start,
                        end_index=last_flag,
                        start_s=float(times[active_start]),
                        end_s=float(times[last_flag]),
                        maximum_score=maximum,
                    )
                )
                active_start = -1
                last_flag = -1
                unflagged_run = 0
                opening_run = 0
                opening_start = -1
                maximum = -np.inf
    if active_start >= 0:
        episodes.append(
            AlarmEpisode(
                start_index=active_start,
                end_index=last_flag,
                start_s=float(times[active_start]),
                end_s=float(times[last_flag]),
                maximum_score=maximum,
            )
        )
    return tuple(episodes)
