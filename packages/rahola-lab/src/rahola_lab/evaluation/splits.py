"""Seed-block allocation that refuses access to the Prototype #2 reserve."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rahola_lab.constants import SEED_BLOCK_SIZE, SEED_BLOCK_START, SeedBlock


class ReserveBlockError(ValueError):
    """Raised when Phase 0.5/Prototype #1 attempts to touch reserve seeds."""


def seeds_for(block: SeedBlock | str, count: int, *, offset: int = 0) -> NDArray[np.uint64]:
    """Return seeds from a named split; raw seed boundaries stay centralized."""
    selected = SeedBlock(block)
    if selected == SeedBlock.RESERVE:
        raise ReserveBlockError("the reserve seed block belongs exclusively to Prototype #2")
    if count < 1 or offset < 0 or offset + count > SEED_BLOCK_SIZE:
        raise ValueError("requested seed slice must fit inside its frozen block")
    start = SEED_BLOCK_START[selected] + offset
    return np.arange(start, start + count, dtype=np.uint64)


def assert_seed_membership(seeds: NDArray[np.integer], block: SeedBlock | str) -> None:
    """Reject a dataset whose seeds do not all belong to the declared block."""
    selected = SeedBlock(block)
    if selected == SeedBlock.RESERVE:
        raise ReserveBlockError("reserve data may not be inspected in this task")
    values = np.asarray(seeds)
    start = SEED_BLOCK_START[selected]
    if np.any(values < start) or np.any(values >= start + SEED_BLOCK_SIZE):
        raise ValueError(f"dataset contains seeds outside the {selected} block")
