"""Split conformal correction of an upper conditional quantile.

Implements the upper-tail specialization in Theorem 2, equation (16), of
Romano, Patterson & Candès (2019), *Conformalized Quantile Regression*:
https://arxiv.org/abs/1905.03222. The inflated order statistic is the standard
finite-sample conformal rank from Vovk, Gammerman & Shafer (2005).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def conformal_quantile(scores: FloatArray, alpha: float) -> float:
    """Return rank ceil((n+1)(1-alpha)); infinities handle ACI's exterior levels."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("conformity scores must be a non-empty finite vector")
    if alpha <= 0.0:
        return float("inf")
    if alpha >= 1.0:
        return float("-inf")
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    if rank > len(values):
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])


@dataclass(frozen=True)
class SplitCQRUpper:
    """One-sided CQR calibration for a future maximum-roll upper bound."""

    scores: FloatArray

    @classmethod
    def calibrate(
        cls, calibration_targets: FloatArray, predicted_upper: FloatArray
    ) -> SplitCQRUpper:
        targets = np.asarray(calibration_targets, dtype=np.float64)
        prediction = np.asarray(predicted_upper, dtype=np.float64)
        if targets.shape != prediction.shape or targets.ndim != 1:
            raise ValueError("calibration targets and predictions must be matching vectors")
        return cls(scores=targets - prediction)

    def correction(self, alpha: float) -> float:
        return conformal_quantile(self.scores, alpha)

    def upper_bound(self, predicted_upper: FloatArray, alpha: float) -> FloatArray:
        return np.asarray(predicted_upper, dtype=np.float64) + self.correction(alpha)
