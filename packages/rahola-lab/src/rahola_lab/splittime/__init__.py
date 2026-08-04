"""Online split-time capsize-rate estimation from physical roll signals."""

from rahola_lab.splittime.crossings import (
    Crossing,
    decluster_crossings,
    detect_crossings,
    roll_decorrelation_time,
)
from rahola_lab.splittime.estimator import (
    RateEmission,
    RateTrajectory,
    SplitTimeConfig,
    estimate_rate_trajectory,
)
from rahola_lab.splittime.tail import (
    ExponentialTailEstimate,
    GammaRatePrior,
    estimate_exponential_tail,
    exponential_rate_mle,
)

__all__ = [
    "Crossing",
    "ExponentialTailEstimate",
    "GammaRatePrior",
    "RateEmission",
    "RateTrajectory",
    "SplitTimeConfig",
    "decluster_crossings",
    "detect_crossings",
    "estimate_exponential_tail",
    "estimate_rate_trajectory",
    "exponential_rate_mle",
    "roll_decorrelation_time",
]
