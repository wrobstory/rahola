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
        batch, samples = self.angle_rad.shape
        if self.rate_rad_s.shape != (batch, samples) or self.time_s.shape != (samples,):
            raise ValueError("inconsistent dense trajectory shapes")
        if len(self.seeds) != batch or len(self.metadata) != batch:
            raise ValueError("per-trajectory fields do not match batch size")

    @property
    def batch_size(self) -> int:
        return self.angle_rad.shape[0]


@dataclass(frozen=True)
class WindowDataset:
    values: NDArray[np.float64]
    labels: NDArray[np.int8]
    trajectory_indices: NDArray[np.int64]
    end_times_s: NDArray[np.float64]
