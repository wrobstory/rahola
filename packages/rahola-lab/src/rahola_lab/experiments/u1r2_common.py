"""Shared frozen controls and data routing for U1-r2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rahola.dataset import SimulationDataset
from rahola_lab.campaigns import load_campaign_split, u1r2_name
from rahola_lab.constants import (
    U1R2_EMISSION_POLICY,
    U1R2_PRIOR_STRENGTH,
    U1R2_TAIL_QUANTILE,
    U1R2_TRAILING_WINDOW_S,
    SeedBlock,
)
from rahola_lab.experiments.common import FAMILIES
from rahola_lab.experiments.u1_common import (
    ScoredRateTrajectory,
    calibration_tail_priors,
    campaign_family,
    load_split,
    score_dataset,
)
from rahola_lab.forecast import DangerMarginFit
from rahola_lab.splittime import SplitTimeConfig


def u1a_campaigns() -> list[str]:
    return [f"{family}_{role}" for family in FAMILIES for role in ("stationary", "evaluation")]


def calibration_datasets(data_root: Path) -> dict[str, SimulationDataset]:
    return {name: load_split(data_root, name, SeedBlock.CALIBRATION) for name in u1a_campaigns()}


def frozen_tail_priors(
    datasets: dict[str, SimulationDataset],
) -> dict[str, dict[str, float]]:
    return calibration_tail_priors(datasets, (U1R2_TAIL_QUANTILE,))[str(U1R2_TAIL_QUANTILE)]


def load_fresh_test(data_root: Path, base_name: str) -> SimulationDataset:
    return load_campaign_split(
        data_root / u1r2_name(base_name),
        SeedBlock.TEST,
    )


def score_selected(
    dataset: SimulationDataset,
    base_name: str,
    priors: dict[str, dict[str, float]],
    *,
    fit: DangerMarginFit | None = None,
    critical_rate_scales: list[dict[int, NDArray[np.float64]]] | None = None,
) -> list[ScoredRateTrajectory]:
    prior = priors[campaign_family(base_name)]
    return score_dataset(
        dataset,
        prior_mean=prior["mean_rate"],
        prior_strength=U1R2_PRIOR_STRENGTH,
        prior_threshold_w=prior["threshold_w"],
        prior_exceedance_probability=prior["exceedance_probability"],
        config=SplitTimeConfig(
            tail_quantile=U1R2_TAIL_QUANTILE,
            trailing_window_s=U1R2_TRAILING_WINDOW_S,
            emission_policy=U1R2_EMISSION_POLICY,
        ),
        fit=fit,
        critical_rate_scales=critical_rate_scales,
    )
