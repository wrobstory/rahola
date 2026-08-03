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
from rahola.dataset import SimulationDataset
from rahola.simulate import simulate_batch

__all__ = [
    "Family",
    "ForcingConfig",
    "ParametricConfig",
    "ProtocolConfig",
    "ProtocolKind",
    "SimulationConfig",
    "SimulationDataset",
    "simulate_batch",
]
