"""Bridge conformal roll bounds to the shared episode detector."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rahola_lab.constants import ALARM_THRESHOLD_ESCAPE_FRACTION


def normalized_alarm_scores(
    upper_bounds_rad: NDArray[np.floating], escape_angle_rad: float
) -> NDArray[np.float64]:
    """Return bound/threshold scores; the frozen alarm threshold is exactly 1."""
    if not np.isfinite(escape_angle_rad) or escape_angle_rad <= 0:
        raise ValueError("escape angle must be finite and positive")
    bounds = np.asarray(upper_bounds_rad, dtype=np.float64)
    if np.any(np.isnan(bounds)):
        raise ValueError("alarm bounds must not contain NaN")
    scores = bounds / (ALARM_THRESHOLD_ESCAPE_FRACTION * escape_angle_rad)
    # Exterior conformal levels have defined operational semantics: +inf is
    # always-alarm and -inf is never-alarm. Keep the episode API finite-valued.
    return np.nan_to_num(
        scores,
        posinf=np.finfo(np.float64).max,
        neginf=-np.finfo(np.float64).max,
    )
