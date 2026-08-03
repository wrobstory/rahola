from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.constants import SEED_BLOCK_START, SeedBlock
from rahola_lab.evaluation.splits import ReserveBlockError, assert_seed_membership, seeds_for


def test_named_seed_blocks_are_disjoint() -> None:
    train = seeds_for(SeedBlock.TRAIN, 10, offset=17)
    calibration = seeds_for(SeedBlock.CALIBRATION, 10, offset=17)
    test = seeds_for(SeedBlock.TEST, 10, offset=17)
    assert not set(train) & set(calibration)
    assert not set(train) & set(test)
    assert not set(calibration) & set(test)


@pytest.mark.parametrize("block", [SeedBlock.RESERVE, SeedBlock.RESERVE2])
def test_reserve_blocks_cannot_be_materialized(block: SeedBlock) -> None:
    with pytest.raises(ReserveBlockError):
        seeds_for(block, 1)
    with pytest.raises(ReserveBlockError):
        assert_seed_membership(np.array([SEED_BLOCK_START[block]]), block)


def test_membership_rejects_cross_split_data() -> None:
    with pytest.raises(ValueError, match="outside"):
        assert_seed_membership(np.array([1, 200_001]), SeedBlock.TRAIN)
