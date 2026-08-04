"""Piecewise-linear split-time danger margin from measured roll state.

The critical-rate formula is equation (13), with the optional particular-
solution correction from equation (15), of Belenky et al. (2024), Ocean
Engineering 292, 116452: https://doi.org/10.1016/j.oceaneng.2023.116452.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RestoringSideFit:
    direction: int
    threshold_angle_rad: float
    vanishing_angle_rad: float
    threshold_distance_rad: float
    vanishing_distance_rad: float
    repeller_slope: float
    growth_rate_s: float

    def critical_rate_at_threshold(
        self,
        particular_angle_rad: float = 0.0,
        particular_rate_rad_s: float = 0.0,
    ) -> float:
        """Eq. 13, or Eq. 15 when a forced particular solution is supplied."""
        return (
            self.growth_rate_s
            * (
                self.vanishing_distance_rad
                - self.threshold_distance_rad
                + self.direction * particular_angle_rad
            )
            + self.direction * particular_rate_rad_s
        )


@dataclass(frozen=True)
class DangerMarginFit:
    equilibrium_angle_rad: float
    central_omega_rad_s: float
    damping_coefficient_s: float
    positive: RestoringSideFit
    negative: RestoringSideFit

    def safety_margin(
        self, angle_rad: NDArray[np.floating], rate_rad_s: NDArray[np.floating]
    ) -> FloatArray:
        """Minimum margin to either fitted escape-side separatrix.

        The positive-side margin is ``vcrit,+ (phi) - phi_dot`` and the
        negative-side margin is ``vcrit,- (phi) + phi_dot``. Taking their
        minimum avoids choosing an escape side from angle alone: a sufficiently
        large rate toward either boundary must reduce the reported margin.
        """
        angle = np.asarray(angle_rad, dtype=np.float64)
        rate = np.asarray(rate_rad_s, dtype=np.float64)
        if angle.shape != rate.shape:
            raise ValueError("angle and rate arrays must match")
        displacement = angle - self.equilibrium_angle_rad
        positive_critical_rate = self.positive.growth_rate_s * np.maximum(
            self.positive.vanishing_distance_rad - displacement, 0.0
        )
        negative_critical_rate = self.negative.growth_rate_s * np.maximum(
            self.negative.vanishing_distance_rad + displacement, 0.0
        )
        positive_margin = positive_critical_rate - rate
        negative_margin = negative_critical_rate + rate
        return np.minimum(positive_margin, negative_margin)

    def danger_score(
        self, angle_rad: NDArray[np.floating], rate_rad_s: NDArray[np.floating]
    ) -> FloatArray:
        """Negative safety margin, so larger scores mean greater danger."""
        return -self.safety_margin(angle_rad, rate_rad_s)


def _real_roots(coefficients: list[float]) -> list[float]:
    first = next((index for index, value in enumerate(coefficients) if value != 0.0), None)
    if first is None:
        return []
    return sorted(
        float(root.real) for root in np.roots(coefficients[first:]) if abs(root.imag) < 1e-9
    )


def fit_piecewise_linear_restoring(config: dict[str, Any]) -> DangerMarginFit:
    """Match equilibrium slope, first restoring peak, and each escape angle.

    The biased family is translated to its stable static equilibrium and fitted
    separately on each side. Configured asymmetric escape angles are treated as
    the operational vanishing angles, even when an absorbing boundary truncates
    the underlying biased polynomial before its mathematical zero.
    """
    escape = float(config["escape_angle_rad"])
    negative_escape = float(config.get("negative_escape_angle_rad") or escape)
    bias = float(config.get("bias_moment", 0.0))
    quintic = float(config.get("quintic_coefficient", 0.0))
    restoring_roots = _real_roots([quintic, 0.0, -1.0, 0.0, 1.0, -bias])
    stable_roots = [
        root
        for root in restoring_roots
        if 1.0 - 3.0 * root**2 + 5.0 * quintic * root**4 > 0.0
        and -negative_escape / escape < root < 1.0
    ]
    if not stable_roots:
        raise ValueError("restoring curve has no stable equilibrium inside escape angles")
    initial_x = float(config.get("initial_angle_rad", 0.0)) / escape
    equilibrium_x = min(stable_roots, key=lambda root: abs(root - initial_x))
    central_slope = 1.0 - 3.0 * equilibrium_x**2 + 5.0 * quintic * equilibrium_x**4
    base_omega = 2.0 * math.pi / float(config["natural_period_s"])
    central_omega = base_omega * math.sqrt(central_slope)
    damping = float(config["damping_ratio"]) * base_omega
    derivative_roots = _real_roots([5.0 * quintic, 0.0, -3.0, 0.0, 1.0])
    equilibrium_angle = equilibrium_x * escape

    def fit_side(direction: int, vanishing_angle: float) -> RestoringSideFit:
        vanishing_x = vanishing_angle / escape
        candidates = [
            root
            for root in derivative_roots
            if direction * (root - equilibrium_x) > 0.0 and direction * (vanishing_x - root) > 0.0
        ]
        if not candidates:
            raise ValueError("no restoring peak lies between equilibrium and escape angle")
        peak_x = min(candidates, key=lambda root: abs(root - equilibrium_x))
        threshold_angle = peak_x * escape
        threshold_distance = abs(threshold_angle - equilibrium_angle)
        vanishing_distance = abs(vanishing_angle - equilibrium_angle)
        if not 0.0 < threshold_distance < vanishing_distance:
            raise ValueError("piecewise-linear threshold must lie inside escape angle")
        repeller_slope = threshold_distance / (vanishing_distance - threshold_distance)
        growth_rate = damping + math.sqrt(repeller_slope * central_omega**2 + damping**2)
        return RestoringSideFit(
            direction=direction,
            threshold_angle_rad=threshold_angle,
            vanishing_angle_rad=vanishing_angle,
            threshold_distance_rad=threshold_distance,
            vanishing_distance_rad=vanishing_distance,
            repeller_slope=repeller_slope,
            growth_rate_s=growth_rate,
        )

    return DangerMarginFit(
        equilibrium_angle_rad=equilibrium_angle,
        central_omega_rad_s=central_omega,
        damping_coefficient_s=damping,
        positive=fit_side(1, escape),
        negative=fit_side(-1, -negative_escape),
    )
