"""F1: margin closure, energy depletion, and common-forcing tangents."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from rahola.dataset import SimulationDataset
from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import DETECTOR_MATCHED_SENSITIVITY, SeedBlock
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.detector_common import (
    bootstrap_point_payload,
    bootstrap_window_auc,
    decorrelation_times,
    evaluate_suite_at_thresholds,
    select_operating_points,
    threshold_grids,
    window_auc,
)
from rahola_lab.experiments.f1_common import (
    FrozenLogistic,
    StatisticRows,
    finite_time_score,
    fit_logistic,
    reproduce_tangent,
    statistic_rows,
    trajectory_scores,
)
from rahola_lab.experiments.v02_common import point_payload_without_dependent_intervals

F1_FIRST_ENDPOINT_S = 540.0
F1_LAST_ENDPOINT_S = 700.0
F1_FTLE_PERIOD_GRID = (1, 2, 5)
F1_SIGNAL_AUC = 0.60
F1_LEAKAGE_AUDIT_AUC = 0.58
CAUSAL_NAMES = (
    "S1_margin",
    "S2_margin_closure",
    "S3_time_to_closure",
    "S4_energy_reserve",
    "S4_energy_depletion",
    "S5_margin_level_rate",
    "S5_energy_level_rate",
    "S7_instantaneous_normal_strain",
)


def _fit_models(rows: list[StatisticRows]) -> dict[str, FrozenLogistic]:
    labels = np.concatenate([row.labels for row in rows])
    return {
        name: fit_logistic(np.concatenate([row.features[name] for row in rows]), labels)
        for name in ("margin", "energy")
    }


def _with_models(rows: StatisticRows, models: dict[str, FrozenLogistic]) -> StatisticRows:
    scores = dict(rows.scores)
    scores["S5_margin_level_rate"] = models["margin"].predict(rows.features["margin"])
    scores["S5_energy_level_rate"] = models["energy"].predict(rows.features["energy"])
    return replace(rows, scores=scores)


def _score_map(
    datasets: list[SimulationDataset], rows: list[StatisticRows]
) -> dict[str, list]:
    return {
        name: [
            trajectory
            for dataset, row in zip(datasets, rows, strict=True)
            for trajectory in trajectory_scores(dataset, row, row.scores[name])
        ]
        for name in CAUSAL_NAMES
    }


def _frozen_policy(
    datasets: list[SimulationDataset],
    rows: list[StatisticRows],
) -> dict[str, object]:
    scores = _score_map(datasets, rows)
    grids = threshold_grids(scores)
    decorrelation = decorrelation_times(scores)
    points = select_operating_points(scores, grids, decorrelation)
    return {
        "thresholds": {name: point.threshold for name, point in points.items()},
        "decorrelation_s": decorrelation,
        "calibration_operating_points": {
            name: point_payload_without_dependent_intervals(point)
            for name, point in points.items()
        },
    }


def _serialize_models(models: dict[str, FrozenLogistic]) -> dict[str, object]:
    return {name: model.to_dict() for name, model in models.items()}


def _deserialize_models(payload: dict[str, object]) -> dict[str, FrozenLogistic]:
    return {
        name: FrozenLogistic.from_dict(model)
        for name, model in payload.items()
        if isinstance(model, dict)
    }


def calibrate(reference_root: Path, v02_root: Path, output_root: Path) -> dict[str, object]:
    """Freeze T choices, logistic fits, and F1b/F1c operating policies."""
    step_datasets = [
        load_campaign_split(v02_root / "softening_step_v02", block)
        for block in (SeedBlock.TRAIN, SeedBlock.CALIBRATION)
    ]
    step_rows: dict[str, list[StatisticRows]] = {"oracle": [], "operational": []}
    step_rollouts = []
    for dataset in step_datasets:
        rollout = reproduce_tangent(dataset)
        step_rollouts.append(rollout)
        for setting in step_rows:
            step_rows[setting].append(
                statistic_rows(
                    rollout,
                    setting=setting,
                    first_endpoint_s=F1_FIRST_ENDPOINT_S,
                    last_endpoint_s=F1_LAST_ENDPOINT_S,
                )
            )
    step_models = {setting: _fit_models(rows) for setting, rows in step_rows.items()}
    step_calibration = step_rows["oracle"][1]
    ftle_selection: dict[str, object] = {}
    for name, directed in (("S6_ftle", False), ("S7_escape_directed_ftle", True)):
        candidates = []
        for periods in F1_FTLE_PERIOD_GRID:
            values = finite_time_score(
                step_rollouts[1],
                step_calibration,
                periods=periods,
                escape_directed=directed,
            )
            auc = window_auc(trajectory_scores(step_datasets[1], step_calibration, values))
            candidates.append(
                {
                    "periods": periods,
                    "auc": auc,
                    "orientation_independent_auc": max(auc, 1.0 - auc),
                }
            )
        selected = max(
            candidates,
            key=lambda row: (row["orientation_independent_auc"], -row["periods"]),
        )
        ftle_selection[name] = {"candidates": candidates, "selected_periods": selected["periods"]}
    del step_rollouts
    gc.collect()

    datasets: dict[str, dict[str, SimulationDataset]] = {}
    base_rows: dict[str, dict[str, dict[str, StatisticRows]]] = {
        "oracle": {},
        "operational": {},
    }
    for family in FAMILIES:
        datasets[family] = {
            "train": load_campaign_split(
                reference_root / f"{family}_stationary", SeedBlock.TRAIN
            ),
            "calibration": load_campaign_split(
                reference_root / f"{family}_evaluation", SeedBlock.CALIBRATION
            ),
        }
        for role, dataset in datasets[family].items():
            rollout = reproduce_tangent(dataset, require_stored_match=False)
            datasets[family][role] = rollout.dataset
            for setting in base_rows:
                base_rows[setting].setdefault(family, {})[role] = statistic_rows(
                    rollout, setting=setting
                )
            del rollout
            gc.collect()

    policies: dict[str, object] = {}
    rotations: dict[str, object] = {}
    for setting in ("oracle", "operational"):
        fit_rows = [
            base_rows[setting][family][role]
            for family in FAMILIES
            for role in ("train", "calibration")
        ]
        models = _fit_models(fit_rows)
        calibration_datasets = [datasets[family]["calibration"] for family in FAMILIES]
        calibration_rows = [
            _with_models(base_rows[setting][family]["calibration"], models)
            for family in FAMILIES
        ]
        policies[setting] = {
            "models": _serialize_models(models),
            **_frozen_policy(calibration_datasets, calibration_rows),
        }
        rotations[setting] = {}
        for held_out in FAMILIES:
            included = [family for family in FAMILIES if family != held_out]
            rotation_models = _fit_models(
                [
                    base_rows[setting][family][role]
                    for family in included
                    for role in ("train", "calibration")
                ]
            )
            rotation_datasets = [datasets[family]["calibration"] for family in included]
            rotation_rows = [
                _with_models(base_rows[setting][family]["calibration"], rotation_models)
                for family in included
            ]
            rotations[setting][held_out] = {
                "models": _serialize_models(rotation_models),
                **_frozen_policy(rotation_datasets, rotation_rows),
            }

    payload: dict[str, object] = {
        "experiment": "F1 calibration and frozen policies",
        "controls": {
            "calibration_motion_policy": (
                "Re-integrate the declared TRAIN/CALIBRATION seeds under the current fixed-cutoff "
                "simulator; v0.1 stored motion is not mixed with current F1 tangents."
            ),
            "margin_closure_difference_periods": 1,
            "time_to_closure_epsilon": 1e-12,
            "ftle_period_grid": list(F1_FTLE_PERIOD_GRID),
            "ftle_selection": (
                "maximum calibration orientation-independent AUC; shortest T wins ties"
            ),
            "f1a_signal_auc": F1_SIGNAL_AUC,
            "leakage_audit_interval": [F1_LEAKAGE_AUDIT_AUC, F1_SIGNAL_AUC],
            "f1b_target_sensitivity": DETECTOR_MATCHED_SENSITIVITY,
            "f1b_success": (
                "candidate and S1 each retain at least 0.90 fresh sensitivity and the "
                "candidate bootstrap FPR upper bound is below S1's lower bound"
            ),
            "oracle_finite_time_statistics_are_acausal": True,
        },
        "predictions_preregistered_verbatim": [
            "(i) generic and escape-directed FTLE will be weak on F1a in both settings;",
            "(ii) margin and energy closure may improve vulnerability estimation on F1b "
            "but will not identify the terminal encounter on F1a;",
            "(iii) no motion-only statistic will reach the 0.60 bar on F1a.",
        ],
        "conditional_arm_trigger": (
            "If any S6/S7 oracle statistic shows F1a signal (>= 0.60), implement the thesis's "
            "motion-only estimator (Sano-Sawada-style local Jacobian from historical neighbors, "
            "thesis Chapter 3 conventions, causal normalization) and measure its fidelity against "
            "the true tangent quantity."
        ),
        "step_logistic_models": {
            setting: _serialize_models(models) for setting, models in step_models.items()
        },
        "ftle_selection": ftle_selection,
        "f1b_policies": policies,
        "f1c_rotation_policies": rotations,
    }
    write_result(output_root, "f1_calibration_f1", payload)
    return payload


def _auc_payload(
    dataset: SimulationDataset, rows: StatisticRows, name: str, values: np.ndarray
) -> dict[str, object]:
    scores = trajectory_scores(dataset, rows, values)
    result = bootstrap_window_auc(scores)
    auc = float(result["auc"])
    return result | {
        "orientation_independent_auc": max(auc, 1.0 - auc),
        "setting": "oracle" if name.startswith("oracle/") else "operational",
    }


def run_f1a(f1_root: Path, output_root: Path, calibration: dict[str, object]) -> dict[str, object]:
    dataset = load_campaign_split(f1_root / "softening_step_v02_f1", SeedBlock.TEST)
    rollout = reproduce_tangent(dataset)
    methods: dict[str, object] = {}
    for setting in ("oracle", "operational"):
        models = _deserialize_models(calibration["step_logistic_models"][setting])
        rows = statistic_rows(
            rollout,
            setting=setting,
            first_endpoint_s=F1_FIRST_ENDPOINT_S,
            last_endpoint_s=F1_LAST_ENDPOINT_S,
            logistic_models=models,
        )
        for statistic, values in rows.scores.items():
            methods[f"{setting}/{statistic}"] = _auc_payload(
                dataset, rows, f"{setting}/{statistic}", values
            )
        if setting == "oracle":
            for statistic, directed in (
                ("S6_ftle", False),
                ("S7_escape_directed_ftle", True),
            ):
                periods = int(calibration["ftle_selection"][statistic]["selected_periods"])
                values = finite_time_score(
                    rollout, rows, periods=periods, escape_directed=directed
                )
                methods[f"oracle/{statistic}"] = _auc_payload(
                    dataset, rows, f"oracle/{statistic}", values
                ) | {"periods": periods, "acausal": True}
    triggered = [
        name
        for name, result in methods.items()
        if name in {"oracle/S6_ftle", "oracle/S7_escape_directed_ftle"}
        and result["orientation_independent_auc"] >= F1_SIGNAL_AUC
    ]
    leakage = [
        name
        for name, result in methods.items()
        if F1_LEAKAGE_AUDIT_AUC < result["orientation_independent_auc"] <= F1_SIGNAL_AUC
    ]
    payload: dict[str, object] = {
        "experiment": "F1a timing kill",
        "window_geometry": {
            "first_endpoint_s": F1_FIRST_ENDPOINT_S,
            "last_endpoint_s": F1_LAST_ENDPOINT_S,
            "stride_s": 10.0,
            "history_periods": 60,
            "horizon_periods": 50,
        },
        "methods": methods,
        "leakage_audit_triggered": leakage,
        "conditional_arm_triggered_by": triggered,
        "conditional_arm_decision": "implement" if triggered else "skip",
        "conditional_arm_skip_reason": (
            None
            if triggered
            else (
                "No true oracle finite-time exponent reached 0.60; its "
                "timing-information ceiling was already at the floor."
            )
        ),
    }
    write_result(output_root, "f1a_timing_f1", payload)
    return payload


def _fresh_evaluation(f1_root: Path) -> dict[str, SimulationDataset]:
    return {
        family: load_campaign_split(f1_root / f"{family}_evaluation_f1", SeedBlock.TEST)
        for family in FAMILIES
    }


def run_f1b(
    f1_root: Path, output_root: Path, calibration: dict[str, object]
) -> tuple[dict[str, object], dict[str, dict[str, list]]]:
    datasets = _fresh_evaluation(f1_root)
    scored: dict[str, dict[str, list]] = {"oracle": {}, "operational": {}}
    rows_by_setting: dict[str, dict[str, StatisticRows]] = {
        "oracle": {},
        "operational": {},
    }
    for family, dataset in datasets.items():
        rollout = reproduce_tangent(dataset)
        for setting in scored:
            models = _deserialize_models(calibration["f1b_policies"][setting]["models"])
            rows_by_setting[setting][family] = statistic_rows(
                rollout, setting=setting, logistic_models=models
            )
        del rollout
        gc.collect()
    methods: dict[str, object] = {}
    qualifiers: list[str] = []
    for setting in scored:
        rows = [rows_by_setting[setting][family] for family in FAMILIES]
        data = [datasets[family] for family in FAMILIES]
        scores = _score_map(data, rows)
        policy = calibration["f1b_policies"][setting]
        points = evaluate_suite_at_thresholds(
            scores,
            {name: float(value) for name, value in policy["thresholds"].items()},
            {name: float(value) for name, value in policy["decorrelation_s"].items()},
        )
        baseline = None
        for name, point in points.items():
            payload = bootstrap_point_payload(
                point,
                scores[name],
                horizon_s=200.0,
                decorrelation_s=float(policy["decorrelation_s"][name]),
                campaign_strata=[
                    family for family in FAMILIES for _ in range(datasets[family].batch_size)
                ],
            )
            methods[f"{setting}/{name}"] = payload
            if name == "S1_margin":
                baseline = payload
        assert baseline is not None
        for name in CAUSAL_NAMES:
            candidate = methods[f"{setting}/{name}"]
            passes = (
                candidate["sensitivity"] >= DETECTOR_MATCHED_SENSITIVITY
                and baseline["sensitivity"] >= DETECTOR_MATCHED_SENSITIVITY
                and candidate["false_episodes_per_hour_trajectory_bootstrap_interval"][1]
                < baseline["false_episodes_per_hour_trajectory_bootstrap_interval"][0]
            )
            candidate["beats_s1_predeclared"] = passes
            if passes and name != "S1_margin":
                qualifiers.append(f"{setting}/{name}")
        scored[setting] = scores
    d1 = json.loads((output_root / "d1_operating_curves_v02.json").read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "experiment": "F1b vulnerability value",
        "methods": methods,
        "qualifiers_for_f1c": qualifiers,
        "frozen_d1_danger_margin_record": d1["headline_at_calibration_selected_threshold"][
            "danger_margin"
        ],
    }
    write_result(output_root, "f1b_value_f1", payload)
    return payload, scored


def run_f1c(
    f1_root: Path,
    output_root: Path,
    calibration: dict[str, object],
    qualifiers: list[str],
) -> dict[str, object]:
    if not qualifiers:
        payload: dict[str, object] = {
            "experiment": "F1c transfer",
            "status": "skipped",
            "reason": "No causal F1b statistic met the predeclared improvement rule.",
            "rotations": {},
        }
        write_result(output_root, "f1c_transfer_f1", payload)
        return payload
    datasets = _fresh_evaluation(f1_root)
    rotations: dict[str, object] = {}
    for qualified in qualifiers:
        setting, name = qualified.split("/", 1)
        rotations[qualified] = {}
        for held_out in FAMILIES:
            policy = calibration["f1c_rotation_policies"][setting][held_out]
            models = _deserialize_models(policy["models"])
            rollout = reproduce_tangent(datasets[held_out])
            rows = statistic_rows(rollout, setting=setting, logistic_models=models)
            scores = {name: trajectory_scores(datasets[held_out], rows, rows.scores[name])}
            point = evaluate_suite_at_thresholds(
                scores,
                {name: float(policy["thresholds"][name])},
                {name: float(policy["decorrelation_s"][name])},
            )[name]
            rotations[qualified][held_out] = bootstrap_point_payload(
                point,
                scores[name],
                horizon_s=200.0,
                decorrelation_s=float(policy["decorrelation_s"][name]),
            )
            del rollout
            gc.collect()
    payload = {"experiment": "F1c transfer", "status": "run", "rotations": rotations}
    write_result(output_root, "f1c_transfer_f1", payload)
    return payload


def write_provenance_manifest(
    repository_root: Path, predeclaration_commit: str, data_anchor_commit: str
) -> Path:
    artifacts = {}
    for path in sorted((repository_root / "results").glob("*_f1.json")):
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    campaign_manifests = {}
    for path in sorted((repository_root / "data" / "f1").glob("*/manifest.json")):
        campaign_manifests[str(path.relative_to(repository_root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    payload = {
        "experiment": "F1",
        "predeclaration_commit": predeclaration_commit,
        "data_anchor_commit": data_anchor_commit,
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "artifacts_sha256": artifacts,
        "campaign_manifests_sha256": campaign_manifests,
        "reserves_touched": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["_manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    path = repository_root / "results" / "provenance_manifest_f1.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
