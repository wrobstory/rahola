"""B1 gate-open gray-box architecture experiment."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola.config import Family, ProtocolKind, SimulationConfig
from rahola.dataset import SimulationDataset
from rahola.windowing import binary_auc
from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    DETECTOR_MATCHED_SENSITIVITY,
    EWS_HORIZON_PERIODS,
    GRAYBOX_AUXILIARY_WEIGHT_GRID,
    GRAYBOX_STIFFNESS_MAE_LIMIT,
    GRAYBOX_TRANSFER_FPR_REDUCTION,
    GRAYBOX_TRANSFER_ROTATIONS_REQUIRED,
    SeedBlock,
)
from rahola_lab.detectors import DetectorWindowDataset, GrayBoxDetector, extract_detector_windows
from rahola_lab.evaluation import (
    EpisodeConfig,
    TrajectoryScores,
    estimate_decorrelation_time,
    operating_curve,
)
from rahola_lab.experiments.common import FAMILIES, load_result, subset_dataset, write_result
from rahola_lab.experiments.detector_common import (
    campaign_dir,
    common_natural_period_s,
    detector_risk_end_s,
    matched_point,
    point_payload,
    relative_fpr_reduction,
)


@dataclass(frozen=True)
class GrayWindows:
    windows: DetectorWindowDataset
    states: np.ndarray
    latents: np.ndarray


def _physical_arrays(
    windows: DetectorWindowDataset, config: SimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    states = np.column_stack(
        (
            windows.raw_angle_rad / config.escape_angle_rad,
            windows.raw_rate_rad_s / (config.escape_angle_rad * config.omega_n_rad_s),
        )
    )
    if (
        config.protocol.kind == ProtocolKind.RAMPED
        and config.protocol.ramp_parameter == "stiffness"
    ):
        start = float(config.protocol.ramp_start)
        drift = (float(config.protocol.ramp_end) - start) / config.duration_s
        stiffness = start + drift * windows.end_times_s
    else:
        stiffness = np.ones(len(windows.labels))
        drift = 0.0
    family_code = {Family.SOFTENING: 0.0, Family.PARAMETRIC: 0.5, Family.BIASED: 1.0}[config.family]
    latents = np.column_stack(
        (
            stiffness,
            np.full(len(stiffness), drift),
            np.full(len(stiffness), config.damping_ratio),
            np.full(len(stiffness), config.quadratic_damping),
            np.full(len(stiffness), config.bias_moment),
            np.full(len(stiffness), config.parametric.h0),
            np.full(len(stiffness), family_code),
        )
    )
    return states, latents


def _extract(
    dataset: SimulationDataset,
    *,
    stride_s: float,
    max_windows_per_trajectory: int | None = None,
    allow_censored_for_inference: bool = False,
) -> GrayWindows:
    windows = extract_detector_windows(
        dataset,
        stride_s=stride_s,
        max_windows_per_trajectory=max_windows_per_trajectory,
        allow_censored_for_inference=allow_censored_for_inference,
    )
    states, latents = _physical_arrays(windows, SimulationConfig.from_dict(dataset.config))
    return GrayWindows(windows, states, latents)


def _concatenate(parts: list[GrayWindows]) -> GrayWindows:
    return GrayWindows(
        DetectorWindowDataset(
            **{
                field: np.concatenate([getattr(part.windows, field) for part in parts])
                for field in DetectorWindowDataset.__dataclass_fields__
            }
        ),
        np.concatenate([part.states for part in parts]),
        np.concatenate([part.latents for part in parts]),
    )


def _fit(training_data: list[SimulationDataset], calibration_data: list[SimulationDataset]):
    training = _concatenate(
        [
            _extract(dataset, stride_s=20.0, max_windows_per_trajectory=3)
            for dataset in training_data
        ]
    )
    calibration = _concatenate(
        [
            _extract(dataset, stride_s=20.0, max_windows_per_trajectory=3)
            for dataset in calibration_data
        ]
    )
    rows = []
    best = None
    for weight in GRAYBOX_AUXILIARY_WEIGHT_GRID:
        model = GrayBoxDetector(auxiliary_weight=weight).fit(
            training.windows.features,
            training.states,
            training.windows.labels,
            training.latents,
        )
        scores = model.predict_scores(calibration.windows.features, calibration.states)
        auc = binary_auc(calibration.windows.labels, scores)
        row = {
            "auxiliary_weight": weight,
            "calibration_auc": auc,
            "parameter_count": model.parameter_count(),
        }
        rows.append(row)
        if best is None or auc > best[0]:
            best = (auc, model)
    assert best is not None
    return best[1], rows


def _score_dataset(dataset: SimulationDataset, model: GrayBoxDetector) -> list[TrajectoryScores]:
    output = []
    period = float(dataset.config["natural_period_s"])
    for start in range(0, dataset.batch_size, 128):
        chunk = subset_dataset(dataset, start, min(start + 128, dataset.batch_size))
        extracted = _extract(
            chunk, stride_s=10.0, allow_censored_for_inference=True
        )
        scores = model.predict_scores(extracted.windows.features, extracted.states)
        for trajectory in range(chunk.batch_size):
            selected = extracted.windows.trajectory_indices == trajectory
            capsize = float(chunk.t_capsize_s[trajectory])
            times = extracted.windows.end_times_s[selected]
            trajectory_scores = scores[selected]
            record_end_s = detector_risk_end_s(
                times,
                t_capsize_s=capsize,
                raw_record_end_s=float(chunk.time_s[-1]),
                horizon_s=EWS_HORIZON_PERIODS * period,
                record_start_s=60.0 * period,
            )
            output.append(
                TrajectoryScores(
                    times_s=times,
                    scores=trajectory_scores,
                    record_end_s=record_end_s,
                    t_capsize_s=capsize if np.isfinite(capsize) else None,
                    record_start_s=60.0 * period,
                )
            )
    return output


def _evaluate(
    calibration_data: list[SimulationDataset],
    test_data: list[SimulationDataset],
    model: GrayBoxDetector,
):
    horizon_s = EWS_HORIZON_PERIODS * common_natural_period_s(
        calibration_data + test_data
    )
    calibration = [item for dataset in calibration_data for item in _score_dataset(dataset, model)]
    values = np.concatenate([item.scores for item in calibration if len(item.scores)])
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("gray-box produced no finite calibration scores")
    thresholds = np.unique(
        np.append(
            np.quantile(values, np.linspace(0.0, 1.0, 41)),
            -np.finfo(np.float64).max,
        )
    )
    estimates = []
    for trajectory in calibration:
        if len(trajectory.scores) >= 4:
            dt_s = float(np.median(np.diff(trajectory.times_s)))
            estimates.append(estimate_decorrelation_time(trajectory.scores, dt_s))
    decorrelation_s = float(np.median(estimates)) if estimates else 10.0
    calibration_curve = operating_curve(
        calibration,
        EpisodeConfig(threshold=0.0, debounce_windows=3, refractory_windows=3),
        thresholds,
        horizon_s=horizon_s,
        decorrelation_time_s=decorrelation_s,
    )
    calibration_point = matched_point(calibration_curve)
    test = [item for dataset in test_data for item in _score_dataset(dataset, model)]
    test_point = operating_curve(
        test,
        EpisodeConfig(threshold=0.0, debounce_windows=3, refractory_windows=3),
        np.asarray([calibration_point.threshold], dtype=np.float64),
        horizon_s=horizon_s,
        decorrelation_time_s=decorrelation_s,
    )[0]
    return point_payload(test_point), point_payload(calibration_point), decorrelation_s


def _tracking(
    data_root: Path,
    output_root: Path,
    model: GrayBoxDetector,
) -> tuple[float, int, str]:
    figure_path = output_root / "p3_b1_stiffness_tracking.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    trajectory_errors = []
    for family in FAMILIES:
        dataset = load_campaign_split(campaign_dir(data_root, f"{family}_ramp"), SeedBlock.TEST)
        extracted = _extract(
            dataset, stride_s=10.0, allow_censored_for_inference=True
        )
        predicted = model.predict_latents(extracted.windows.features, extracted.states)[:, 0]
        true = extracted.latents[:, 0]
        survivor_indices = np.flatnonzero(~dataset.capsized)
        survivor_windows = np.isin(
            extracted.windows.trajectory_indices, survivor_indices
        )
        final = extracted.windows.end_times_s >= 2.0 * float(dataset.time_s[-1]) / 3.0
        for trajectory_index in survivor_indices:
            selected = (
                final & (extracted.windows.trajectory_indices == trajectory_index)
            )
            if np.any(selected):
                trajectory_errors.append(
                    float(np.mean(np.abs(predicted[selected] - true[selected])))
                )
        bins = np.linspace(240.0, 600.0, 19)
        centers = 0.5 * (bins[:-1] + bins[1:])
        means = []
        truth = []
        for left, right in pairwise(bins):
            selected = (
                survivor_windows
                & (extracted.windows.end_times_s >= left)
                & (extracted.windows.end_times_s < right)
            )
            means.append(float(np.mean(predicted[selected])))
            truth.append(float(np.mean(true[selected])))
        axis.plot(centers, means, label=f"{family} inferred")
        axis.plot(centers, truth, linestyle="--", alpha=0.65, label=f"{family} true")
    axis.set_xlabel("time in ramp (s)")
    axis.set_ylabel("stiffness multiplier")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return float(np.mean(trajectory_errors)), len(trajectory_errors), str(figure_path)


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    d1 = load_result(output_root, "d1_operating_curves")
    d2 = load_result(output_root, "d2_family_generalization")
    all_training = [
        load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TRAIN)
        for family in FAMILIES
        for role in ("stationary", "ramp")
    ]
    all_calibration = [
        load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.CALIBRATION)
        for family in FAMILIES
        for role in ("stationary", "ramp")
    ]
    model, selection = _fit(all_training, all_calibration)
    all_test = [
        load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TEST)
        for family in FAMILIES
        for role in ("evaluation", "ramp")
    ]
    d1_gray, d1_gray_calibration, d1_decorrelation = _evaluate(
        all_calibration, all_test, model
    )
    cnn_d1 = d1["headline_at_calibration_selected_threshold"]["cnn"]
    cnn_ci = cnn_d1["false_episodes_per_hour_interval"]
    parity_limit = float(cnn_d1["false_episodes_per_hour"]) + float(cnn_ci[1] - cnn_ci[0])
    parity_diagnostic = (
        float(d1_gray["sensitivity"]) < DETECTOR_MATCHED_SENSITIVITY
        or float(d1_gray["false_episodes_per_hour"]) > parity_limit
    )
    tracking_mae, tracking_trajectories, figure = _tracking(data_root, output_root, model)
    tracking_diagnostic = tracking_mae > GRAYBOX_STIFFNESS_MAE_LIMIT
    del all_training, all_calibration, all_test, model
    gc.collect()

    rotations = []
    improvements = 0
    d2_by_family = {row["held_out_family"]: row for row in d2["rotations"]}
    for held_out in FAMILIES:
        included = [family for family in FAMILIES if family != held_out]
        training = [
            load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TRAIN)
            for family in included
            for role in ("stationary", "ramp")
        ]
        calibration = [
            load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.CALIBRATION)
            for family in included
            for role in ("stationary", "ramp")
        ]
        rotation_model, rotation_selection = _fit(training, calibration)
        test = [
            load_campaign_split(campaign_dir(data_root, f"{held_out}_{role}"), SeedBlock.TEST)
            for role in ("evaluation", "ramp")
        ]
        gray, gray_calibration, decorrelation = _evaluate(calibration, test, rotation_model)
        cnn = d2_by_family[held_out]["headline_at_calibration_selected_threshold"]["cnn"]
        reduction = relative_fpr_reduction(
            float(gray["false_episodes_per_hour"]),
            float(cnn["false_episodes_per_hour"]),
        )
        earns = (
            float(gray["sensitivity"]) >= DETECTOR_MATCHED_SENSITIVITY
            and float(cnn["sensitivity"]) >= DETECTOR_MATCHED_SENSITIVITY
            and reduction is not None
            and reduction >= GRAYBOX_TRANSFER_FPR_REDUCTION
        )
        improvements += int(earns)
        rotations.append(
            {
                "held_out_family": held_out,
                "graybox": gray,
                "graybox_calibration_operating_point": gray_calibration,
                "cnn": cnn,
                "relative_fpr_reduction": reduction,
                "earns_15_percent_improvement": earns,
                "selection": rotation_selection,
                "decorrelation_time_s": decorrelation,
            }
        )
        del training, calibration, test, rotation_model
        gc.collect()
    transfer_kill = improvements < GRAYBOX_TRANSFER_ROTATIONS_REQUIRED
    survives = False
    payload: dict[str, object] = {
        "experiment": "B1 gray-box",
        "operating_point_policy": "Thresholds are selected on calibration and frozen for test.",
        "selection": selection,
        "parameter_count": GrayBoxDetector(
            auxiliary_weight=selection[0]["auxiliary_weight"]
        ).parameter_count(),
        "d1": {
            "graybox": d1_gray,
            "graybox_calibration_operating_point": d1_gray_calibration,
            "cnn": cnn_d1,
            "decorrelation_time_s": d1_decorrelation,
            "parity_limit_fpr_per_hour": parity_limit,
        },
        "d2_rotations": rotations,
        "stiffness_tracking": {
            "survivor_conditioned_final_third_trajectory_mean_mae": tracking_mae,
            "trajectory_count": tracking_trajectories,
            "limit": GRAYBOX_STIFFNESS_MAE_LIMIT,
            "figure": figure,
        },
        "kills": {
            "transfer": {
                "fired": transfer_kill,
                "verbatim": (
                    "Kill: fails to beat the from-scratch CNN's D2 FPR/h by >=15% in at least "
                    "two of three rotations."
                ),
                "rotations_earned": improvements,
            },
            "within_distribution_parity": {
                "fired": None,
                "evaluable_without_test_selection": False,
                "calibration_targeted_diagnostic_fired": parity_diagnostic,
                "verbatim": (
                    "Kill: worse than the CNN by more than its CI width at matched sensitivity."
                ),
                "reason": (
                    "The calibration-selected methods have different test sensitivities; matching "
                    "them on test would violate the frozen-threshold protocol."
                ),
            },
            "stiffness_tracking": {
                "fired": None,
                "evaluable_unconditionally": False,
                "survivor_conditioned_diagnostic_fired": tracking_diagnostic,
                "verbatim": (
                    "Kill: mean absolute stiffness error exceeds 10% over the final third of "
                    "the ramp."
                ),
                "reason": (
                    "Trajectories that capsize before the final third have no defined final-third "
                    "error; the reported diagnostic conditions on survival and weights each "
                    "trajectory equally."
                ),
            },
        },
        "survives_all_kills": survives,
    }
    write_result(
        output_root,
        "p3_b1_graybox",
        payload,
        upstream_results={
            "d1_operating_curves": d1,
            "d2_family_generalization": d2,
        },
    )
    return payload
