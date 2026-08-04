"""Trajectory-block bootstrap intervals for dependent warning records."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rahola_lab.constants import TRAJECTORY_BOOTSTRAP_REPLICATES, TRAJECTORY_BOOTSTRAP_SEED


@dataclass(frozen=True)
class BootstrapEstimate:
    """Percentile interval conditional on a frozen evaluation policy."""

    estimate: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    valid_replicates: NDArray[np.int64]
    requested_replicates: int
    seed: int


def trajectory_block_bootstrap[Item](
    items: Sequence[Item],
    statistic: Callable[[list[Item]], float | NDArray[np.floating]],
    *,
    strata: Sequence[Hashable] | None = None,
    replicates: int = TRAJECTORY_BOOTSTRAP_REPLICATES,
    seed: int = TRAJECTORY_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> BootstrapEstimate:
    """Resample whole trajectories within campaign strata.

    Each replicate preserves the number of trajectories in every stratum, so
    pooled campaign weights remain fixed. ``statistic`` receives the complete
    resampled records and must recompute any window or episode logic.
    """
    if not items:
        raise ValueError("at least one trajectory is required")
    if replicates < 1_000:
        raise ValueError("trajectory bootstrap requires at least 1,000 replicates")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")
    labels = ["all"] * len(items) if strata is None else list(strata)
    if len(labels) != len(items):
        raise ValueError("strata must match the trajectory count")

    members: dict[Hashable, list[int]] = {}
    for index, label in enumerate(labels):
        members.setdefault(label, []).append(index)
    index_groups = tuple(np.asarray(indices, dtype=np.int64) for indices in members.values())

    estimate = np.atleast_1d(np.asarray(statistic(list(items)), dtype=np.float64))
    samples = np.full((replicates, len(estimate)), np.nan, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        sampled: list[Item] = []
        for indices in index_groups:
            chosen = rng.choice(indices, size=len(indices), replace=True)
            sampled.extend(items[int(index)] for index in chosen)
        value = np.atleast_1d(np.asarray(statistic(sampled), dtype=np.float64))
        if value.shape != estimate.shape:
            raise ValueError("bootstrap statistic changed shape across replicates")
        samples[replicate] = value

    valid = np.sum(np.isfinite(samples), axis=0).astype(np.int64)
    if np.any(valid == 0):
        raise ValueError("bootstrap statistic produced no finite replicates")
    tail = (1.0 - confidence_level) / 2.0
    return BootstrapEstimate(
        estimate=estimate,
        lower=np.nanquantile(samples, tail, axis=0),
        upper=np.nanquantile(samples, 1.0 - tail, axis=0),
        valid_replicates=valid,
        requested_replicates=replicates,
        seed=seed,
    )
