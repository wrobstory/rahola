"""Prototype #3 oracle, filtered-state ceiling, and fixed XGBoost ablation."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from rahola.config import ProtocolKind, SimulationConfig
from rahola.dataset import SimulationDataset
from rahola.simulate import simulate_restarted_batch
from rahola.windowing import binary_auc
from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    CEILING_AUC_GAP,
    CEILING_BOOTSTRAP_REPLICATES,
    CEILING_BOOTSTRAP_SEED,
    CEILING_WINDOWS_PER_CAMPAIGN,
    ORACLE_ROLLOUTS,
    PF_PARTICLES,
    SeedBlock,
)
from rahola_lab.detectors import (
    DetectorWindowDataset,
    engineered_features,
    extract_detector_windows,
)
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.detector_common import (
    campaign_dir,
    fit_frozen_suite,
    training_windows,
)
from rahola_lab.inference import bootstrap_particle_filter

CAMPAIGNS = (
    "softening_evaluation",
    "parametric_evaluation",
    "biased_evaluation",
    "softening_ramp",
    "parametric_ramp",
    "biased_ramp",
    "softening_bandwidth_gamma_1",
    "softening_bandwidth_gamma_3_3",
)


@dataclass(frozen=True)
class SampledCampaign:
    name: str
    dataset: SimulationDataset
    windows: DetectorWindowDataset


def _take(windows: DetectorWindowDataset, indices: NDArray[np.integer]) -> DetectorWindowDataset:
    return DetectorWindowDataset(
        **{
            field: getattr(windows, field)[indices]
            for field in DetectorWindowDataset.__dataclass_fields__
        }
    )


def _sample_stratified(
    windows: DetectorWindowDataset,
    duration_s: float,
    count: int,
    *,
    seed: int,
) -> DetectorWindowDataset:
    """Sample evenly by label and quartile of current protocol time."""
    rng = np.random.default_rng(seed)
    time_bin = np.minimum((4.0 * windows.end_times_s / duration_s).astype(int), 3)
    group = 4 * windows.labels.astype(int) + time_bin
    target = min(count, len(windows.labels))
    quota = target // 8
    selected: list[int] = []
    for value in range(8):
        candidates = np.flatnonzero(group == value)
        if len(candidates):
            selected.extend(rng.choice(candidates, min(quota, len(candidates)), replace=False))
    remaining = np.setdiff1d(np.arange(len(group)), np.asarray(selected), assume_unique=False)
    if len(selected) < target:
        selected.extend(rng.choice(remaining, target - len(selected), replace=False))
    return _take(windows, np.sort(np.asarray(selected, dtype=np.int64)))


def _true_stiffness(
    config: SimulationConfig, time_s: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if (
        config.protocol.kind == ProtocolKind.RAMPED
        and config.protocol.ramp_parameter == "stiffness"
    ):
        start = float(config.protocol.ramp_start)
        rate = (float(config.protocol.ramp_end) - start) / config.duration_s
        return start + rate * time_s, np.full(len(time_s), rate)
    return np.ones(len(time_s)), np.zeros(len(time_s))


def _rollout_scores(
    sample: SampledCampaign,
    *,
    filtered: bool,
    rollouts: int,
    seed_base: int,
    windows_per_batch: int = 16,
) -> NDArray[np.float64]:
    config = SimulationConfig.from_dict(sample.dataset.config)
    windows = sample.windows
    scores = np.empty(len(windows.labels), dtype=np.float64)
    dt_s = float(np.median(np.diff(sample.dataset.time_s)))
    true_stiffness, true_drift = _true_stiffness(config, windows.end_times_s)
    history_length = windows.features.shape[1]
    for start in range(0, len(windows.labels), windows_per_batch):
        stop = min(start + windows_per_batch, len(windows.labels))
        angles: list[NDArray[np.float64]] = []
        rates: list[NDArray[np.float64]] = []
        stiffness: list[NDArray[np.float64]] = []
        drift: list[NDArray[np.float64]] = []
        offsets: list[NDArray[np.float64]] = []
        for row in range(start, stop):
            if filtered:
                trajectory = int(windows.trajectory_indices[row])
                end = int(np.searchsorted(sample.dataset.time_s, windows.end_times_s[row]))
                first = end - history_length + 1
                posterior = bootstrap_particle_filter(
                    sample.dataset.angle_rad[trajectory, first : end + 1],
                    sample.dataset.rate_rad_s[trajectory, first : end + 1],
                    dt_s,
                    config,
                    particle_count=PF_PARTICLES,
                    seed=seed_base + row,
                    absolute_start_s=float(sample.dataset.time_s[first]),
                )
                rng = np.random.default_rng(seed_base + 10_000_000 + row)
                chosen = rng.choice(PF_PARTICLES, rollouts, replace=False)
                angles.append(posterior.roll_rad[chosen])
                rates.append(posterior.rate_rad_s[chosen])
                stiffness.append(posterior.stiffness_multiplier[chosen])
                drift.append(posterior.stiffness_rate_per_s[chosen])
            else:
                angles.append(np.full(rollouts, windows.raw_angle_rad[row]))
                rates.append(np.full(rollouts, windows.raw_rate_rad_s[row]))
                stiffness.append(np.full(rollouts, true_stiffness[row]))
                drift.append(np.full(rollouts, true_drift[row]))
            offsets.append(np.full(rollouts, windows.end_times_s[row]))
        count = (stop - start) * rollouts
        seed_start = seed_base * 100_000_000 + start * rollouts
        futures = simulate_restarted_batch(
            config,
            np.arange(seed_start, seed_start + count, dtype=np.uint64),
            duration_s=50.0 * config.natural_period_s,
            initial_angle_rad=np.concatenate(angles),
            initial_rate_rad_s=np.concatenate(rates),
            stiffness_multiplier=np.concatenate(stiffness),
            stiffness_rate_per_s=np.concatenate(drift),
            time_offset_s=np.concatenate(offsets),
        )
        scores[start:stop] = np.mean(futures.capsized.reshape(stop - start, rollouts), axis=1)
    return scores


def _auc_interval(
    labels: NDArray[np.int8],
    scores: NDArray[np.float64],
    blocks: NDArray[np.int64],
) -> dict[str, object]:
    rng = np.random.default_rng(CEILING_BOOTSTRAP_SEED)
    unique = np.unique(blocks)
    members = {block: np.flatnonzero(blocks == block) for block in unique}
    estimates = []
    for _ in range(CEILING_BOOTSTRAP_REPLICATES):
        chosen = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[block] for block in chosen])
        if len(np.unique(labels[indices])) == 2:
            estimates.append(binary_auc(labels[indices], scores[indices]))
    return {
        "auc": binary_auc(labels, scores),
        "interval": [float(value) for value in np.quantile(estimates, [0.025, 0.975])],
        "bootstrap_unit": "trajectory",
        "bootstrap_replicates": len(estimates),
    }


def _calibration(labels: NDArray[np.int8], scores: NDArray[np.float64]) -> dict[str, object]:
    bins = np.minimum((10 * scores).astype(int), 9)
    rows = []
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if np.any(selected):
            predicted = float(np.mean(scores[selected]))
            observed = float(np.mean(labels[selected]))
            ece += np.mean(selected) * abs(predicted - observed)
            rows.append(
                {"count": int(np.sum(selected)), "predicted": predicted, "observed": observed}
            )
    return {"brier": float(np.mean((scores - labels) ** 2)), "ece_10_bin": ece, "bins": rows}


def _fit_b0(
    data_root: Path,
    selected: dict[str, object],
) -> xgb.Booster:
    matrices = []
    labels = []
    for family in FAMILIES:
        for role in ("stationary", "ramp"):
            dataset = load_campaign_split(
                campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TRAIN
            )
            windows = extract_detector_windows(dataset, stride_s=20.0, max_windows_per_trajectory=3)
            matrices.append(
                engineered_features(
                    windows,
                    SimulationConfig.from_dict(dataset.config),
                    neighbor_radius=float(selected["neighbor_radius"]),
                )
            )
            labels.append(windows.labels)
    training = xgb.DMatrix(np.concatenate(matrices), label=np.concatenate(labels))
    return xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.05,
            "max_depth": 3,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 20_260_803,
            "nthread": 4,
        },
        training,
        num_boost_round=200,
    )


def run(
    data_root: Path,
    output_root: Path,
    *,
    windows_per_campaign: int = CEILING_WINDOWS_PER_CAMPAIGN,
    rollouts: int = ORACLE_ROLLOUTS,
    write: bool = True,
) -> dict[str, object]:
    """Run the fixed ceiling program and apply the architecture gate."""
    started = time.perf_counter()
    d1 = json.loads((output_root / "d1_operating_curves.json").read_text(encoding="utf-8"))
    training_data = [
        load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TRAIN)
        for family in FAMILIES
        for role in ("stationary", "ramp")
    ]
    suite = fit_frozen_suite(
        training_windows(training_data, max_windows_per_trajectory=3), d1["selected"]
    )
    del training_data
    gc.collect()
    samples = []
    for campaign_index, name in enumerate(CAMPAIGNS):
        dataset = load_campaign_split(campaign_dir(data_root, name), SeedBlock.TEST)
        all_windows = extract_detector_windows(dataset, stride_s=10.0)
        samples.append(
            SampledCampaign(
                name,
                dataset,
                _sample_stratified(
                    all_windows,
                    float(dataset.time_s[-1]),
                    windows_per_campaign,
                    seed=80_000 + campaign_index,
                ),
            )
        )
    b0 = _fit_b0(data_root, d1["selected"])
    labels = np.concatenate([sample.windows.labels for sample in samples])
    cnn_scores = np.concatenate(
        [suite.cnn.predict_scores(sample.windows.features) for sample in samples]
    )
    b0_scores = np.concatenate(
        [
            b0.predict(
                xgb.DMatrix(
                    engineered_features(
                        sample.windows,
                        SimulationConfig.from_dict(sample.dataset.config),
                        neighbor_radius=float(d1["selected"]["neighbor_radius"]),
                    )
                )
            )
            for sample in samples
        ]
    )
    c1_parts = []
    c2_parts = []
    campaign_rows = []
    for campaign_index, sample in enumerate(samples):
        campaign_started = time.perf_counter()
        c1 = _rollout_scores(
            sample,
            filtered=False,
            rollouts=rollouts,
            seed_base=1_000 + campaign_index * 20,
        )
        c2 = _rollout_scores(
            sample,
            filtered=True,
            rollouts=rollouts,
            seed_base=1_001 + campaign_index * 20,
        )
        c1_parts.append(c1)
        c2_parts.append(c2)
        campaign_rows.append(
            {
                "campaign": sample.name,
                "windows": len(sample.windows.labels),
                "positive_fraction": float(np.mean(sample.windows.labels)),
                "elapsed_s": time.perf_counter() - campaign_started,
            }
        )
    c1_scores = np.concatenate(c1_parts)
    c2_scores = np.concatenate(c2_parts)
    blocks = np.concatenate(
        [
            campaign_index * 10_000 + sample.windows.trajectory_indices
            for campaign_index, sample in enumerate(samples)
        ]
    ).astype(np.int64)
    methods = {
        "C1_oracle": _auc_interval(labels, c1_scores, blocks),
        "C2_filtered": _auc_interval(labels, c2_scores, blocks),
        "CNN": _auc_interval(labels, cnn_scores, blocks),
        "B0_XGBoost": _auc_interval(labels, b0_scores, blocks),
    }
    c1_auc = float(methods["C1_oracle"]["auc"])
    cnn_auc = float(methods["CNN"]["auc"])
    gate_closed = cnn_auc >= c1_auc - CEILING_AUC_GAP
    verdict = (
        "CNN AUC >= C1 AUC - 0.03: declare the motion-only problem information-limited; "
        "skip Part B's architectures entirely."
        if gate_closed
        else "CNN AUC < C1 AUC - 0.03: proceed to Part B and report where the deficit lives."
    )
    payload: dict[str, object] = {
        "experiment": "Prototype #3 ceiling",
        "sampling": {
            "requested_windows_per_campaign": windows_per_campaign,
            "strata": "label x quartile of protocol time",
            "campaigns": campaign_rows,
            "total_windows": len(labels),
        },
        "rollouts_per_window": rollouts,
        "particle_count": PF_PARTICLES,
        "methods": methods,
        "oracle_calibration": _calibration(labels, c1_scores),
        "gaps": {
            "C1_minus_C2": c1_auc - float(methods["C2_filtered"]["auc"]),
            "C2_minus_CNN": float(methods["C2_filtered"]["auc"]) - cnn_auc,
            "C1_minus_CNN": c1_auc - cnn_auc,
        },
        "gate_threshold_auc": CEILING_AUC_GAP,
        "gate_closed": gate_closed,
        "gate_verdict": verdict,
        "oracle_fairness": (
            "C1 knows exact roll state, current stiffness, deterministic remaining ramp, family, "
            "and sea-state specification, but never the realized future forcing. Every rollout "
            "uses a fresh independent forcing seed."
        ),
        "pf_judgment": (
            "Rao-Blackwellized bootstrap PF: observed roll/rate are pinned; 2,000 particles infer "
            "current stiffness and linear drift under a robust encounter-innovation likelihood. "
            "Family is known; no family-marginalized variant was run."
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    if write:
        write_result(output_root, "p3_ceiling", payload)
    return payload
