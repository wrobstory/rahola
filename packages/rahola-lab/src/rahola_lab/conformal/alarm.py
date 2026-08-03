"""Bridge conformal roll bounds to the shared episode detector."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rahola_lab.constants import ALARM_THRESHOLD_ESCAPE_FRACTION


def normalized_alarm_scores(
    upper_bounds_rad: NDArray[np.floating], escape_angle_rad: float
) -> NDArray[np.float64]:
    """Return bound/threshold scores; the frozen alarm threshold is exactly 1."""
    if escape_angle_rad <= 0:
        raise ValueError("escape angle must be positive")
    bounds = np.asarray(upper_bounds_rad, dtype=np.float64)
    return bounds / (ALARM_THRESHOLD_ESCAPE_FRACTION * escape_angle_rad)
