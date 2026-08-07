"""Synthetic nonlinear ship-roll dynamics with falsification-first validation."""

__version__ = "0.1.0"

from rahola.config import (
    Family,
    ForcingConfig,
    ParametricConfig,
    ProtocolConfig,
    ProtocolKind,
    SimulationConfig,
)
from rahola.dataset import SimulationDataset, TangentRollout
from rahola.simulate import simulate_batch, simulate_restarted_batch, simulate_tangent_batch

__all__ = [
    "Family",
    "ForcingConfig",
    "ParametricConfig",
    "ProtocolConfig",
    "ProtocolKind",
    "SimulationConfig",
    "SimulationDataset",
    "TangentRollout",
    "simulate_batch",
    "simulate_restarted_batch",
    "simulate_tangent_batch",
]
