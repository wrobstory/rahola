"""In-memory simulation and window datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulationDataset:
    """Dense batch output; samples after capsize are represented by NaN."""

    time_s: NDArray[np.float64]
    angle_rad: NDArray[np.float64]
    rate_rad_s: NDArray[np.float64]
    seeds: NDArray[np.uint64]
    capsized: NDArray[np.bool_]
    t_capsize_s: NDArray[np.float64]
    metadata: tuple[dict[str, Any], ...]
    config: dict[str, Any]

    def __post_init__(self) -> None:
        if self.angle_rad.ndim != 2:
            raise ValueError("angle_rad must be a two-dimensional dense array")
        batch, samples = self.angle_rad.shape
        if self.rate_rad_s.shape != (batch, samples) or self.time_s.shape != (samples,):
            raise ValueError("inconsistent dense trajectory shapes")
        if (
            self.seeds.shape != (batch,)
            or self.capsized.shape != (batch,)
            or self.t_capsize_s.shape != (batch,)
            or len(self.metadata) != batch
        ):
            raise ValueError("per-trajectory fields do not match batch size")
        if not np.all(np.isfinite(self.time_s)) or not np.all(np.diff(self.time_s) > 0.0):
            raise ValueError("time_s must be finite and strictly increasing")
        if len(np.unique(self.seeds)) != batch:
            raise ValueError("trajectory seeds must be unique")
        if not np.array_equal(self.capsized, np.isfinite(self.t_capsize_s)):
            raise ValueError("capsized and t_capsize_s must be consistent")

        for row, (seed, did_capsize, capsize_time, metadata) in enumerate(
            zip(
                self.seeds,
                self.capsized,
                self.t_capsize_s,
                self.metadata,
                strict=True,
            )
        ):
            if int(metadata.get("seed", -1)) != int(seed):
                raise ValueError(f"metadata seed mismatch in trajectory {row}")
            angle = self.angle_rad[row]
            rate = self.rate_rad_s[row]
            if did_capsize:
                if not self.time_s[0] <= capsize_time <= self.time_s[-1]:
                    raise ValueError(f"capsize time outside record in trajectory {row}")
                after_capsize = self.time_s > capsize_time
                if not np.all(np.isnan(angle[after_capsize])) or not np.all(
                    np.isnan(rate[after_capsize])
                ):
                    raise ValueError(f"post-capsize samples must be NaN in trajectory {row}")
                if not np.all(np.isfinite(angle[~after_capsize])) or not np.all(
                    np.isfinite(rate[~after_capsize])
                ):
                    raise ValueError(f"pre-capsize samples must be finite in trajectory {row}")
            elif not np.all(np.isfinite(angle)) or not np.all(np.isfinite(rate)):
                raise ValueError(f"non-capsizing trajectory {row} must be finite")

    @property
    def batch_size(self) -> int:
        return self.angle_rad.shape[0]


@dataclass(frozen=True)
class TangentRollout:
    """A base trajectory plus nondimensional local state-transition matrices."""

    dataset: SimulationDataset
    transition_matrices: NDArray[np.float64]
    effective_stiffness: NDArray[np.float64]

    def __post_init__(self) -> None:
        expected = (self.dataset.batch_size, len(self.dataset.time_s) - 1, 2, 2)
        if self.transition_matrices.shape != expected:
            raise ValueError(f"transition_matrices must have shape {expected}")
        if not np.all(np.isfinite(self.transition_matrices)):
            raise ValueError("transition_matrices must be finite")
        stiffness_shape = (self.dataset.batch_size, len(self.dataset.time_s))
        if self.effective_stiffness.shape != stiffness_shape:
            raise ValueError(f"effective_stiffness must have shape {stiffness_shape}")
        if not np.all(np.isfinite(self.effective_stiffness)):
            raise ValueError("effective_stiffness must be finite")


@dataclass(frozen=True)
class WindowDataset:
    values: NDArray[np.float64]
    labels: NDArray[np.int8]
    trajectory_indices: NDArray[np.int64]
    end_times_s: NDArray[np.float64]
