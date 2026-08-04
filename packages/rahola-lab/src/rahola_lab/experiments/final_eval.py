"""One-time, guarded scoring on the current inaccessible reserve block."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rahola.simulate import simulate_batch
from rahola.storage import write_dataset
from rahola_lab.campaigns import load_campaign_definition, load_campaign_split
from rahola_lab.constants import (
    B2_TRAJECTORIES_PER_CAMPAIGN,
    EWS_HORIZON_PERIODS,
    SEED_BLOCK_SIZE,
    SEED_BLOCK_START,
    SeedBlock,
)
from rahola_lab.detectors import ChronosClassifier
from rahola_lab.evaluation import AlarmMetrics, EpisodeConfig, evaluate_alarms
from rahola_lab.experiments.b2_chronos import (
    _evaluate_at_policy as evaluate_foundation_policy,
)
from rahola_lab.experiments.b2_chronos import (
    _score_foundation as score_foundation,
)
from rahola_lab.experiments.b2_chronos import (
    _select_operating_policy as select_foundation_policy,
)
from rahola_lab.experiments.b2_chronos import (
    _training_data as foundation_training_data,
)
from rahola_lab.experiments.b2_chronos import (
    _training_windows as foundation_training_windows,
)
from rahola_lab.experiments.common import (
    FAMILIES,
    _write_result_locked,
    load_result,
    result_graph_lock,
)
from rahola_lab.experiments.detector_common import (
    DETECTOR_NAMES,
    campaign_dir,
    fit_frozen_suite,
    merge_scores,
    score_dataset,
    training_windows,
)


class FinalEvaluationError(RuntimeError):
    """Raised when the one-time reserve protocol cannot start safely."""


def _repository_root() -> Path:
    start = Path(__file__).resolve().parent
    output = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(output).resolve()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=_repository_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_canonical_path(actual: Path, expected: Path, *, name: str) -> None:
    if actual.resolve() != expected.resolve():
        raise FinalEvaluationError(f"{name} must be the canonical path {expected}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_clean_worktree() -> None:
    if _git_output("status", "--porcelain"):
        raise FinalEvaluationError("final-eval requires a clean committed working tree")


def _revalidate_final_inputs(
    output_root: Path, expected_digests: dict[str, str]
) -> dict[str, dict[str, object]]:
    """Recheck mutable inputs immediately before the exclusive reserve claim."""
    _require_clean_worktree()
    current: dict[str, dict[str, object]] = {}
    for name, expected_digest in expected_digests.items():
        try:
            artifact = load_result(output_root, name)
        except (OSError, ValueError, TypeError) as error:
            raise FinalEvaluationError(f"final-eval input {name} is no longer valid") from error
        if artifact.get("_artifact_sha256") != expected_digest:
            raise FinalEvaluationError(f"final-eval input {name} changed during preflight")
        current[name] = artifact
    return current


def _write_exclusive_json(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FinalEvaluationError(
            "reserve access was already started; refusing a repeat"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Durably replace an existing attestation without exposing partial JSON."""
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


def _reserve_seeds(block: SeedBlock, count: int, offset: int) -> np.ndarray:
    """Materialize reserve-2; this code path categorically refuses the spent first reserve."""
    if block != SeedBlock.RESERVE2:
        raise FinalEvaluationError("only the unspent reserve2 block may be materialized")
    if count < 1 or offset < 0 or offset + count > SEED_BLOCK_SIZE:
        raise ValueError("reserve slice must fit inside its frozen block")
    start = SEED_BLOCK_START[block] + offset
    return np.arange(start, start + count, dtype=np.uint64)


def _metrics_payload(metrics: AlarmMetrics, threshold: float) -> dict[str, object]:
    lead_quantiles = (
        [float(value) for value in np.quantile(metrics.lead_times_s, [0.1, 0.5, 0.9])]
        if len(metrics.lead_times_s)
        else [float("nan")] * 3
    )
    return {
        "threshold": threshold,
        "sensitivity": metrics.sensitivity,
        "sensitivity_interval": [
            metrics.sensitivity_interval.lower,
            metrics.sensitivity_interval.upper,
        ],
        "false_episodes_per_hour": metrics.false_positives_per_hour,
        "false_episodes_per_hour_interval": [
            metrics.false_positives_per_hour_interval.lower,
            metrics.false_positives_per_hour_interval.upper,
        ],
        "lead_time_quantiles_s": lead_quantiles,
        "capsize_count": metrics.capsize_count,
        "false_episode_count": metrics.false_episode_count,
        "exposure_hours": metrics.exposure_hours,
    }


def run_final_evaluation(
    *,
    data_root: Path,
    output_root: Path,
    config_root: Path,
    reserve_root: Path,
    chunk_size: int = 256,
    reserve_block: SeedBlock = SeedBlock.RESERVE2,
) -> dict[str, object]:
    """Score the frozen suite once under the repository-local attestation procedure."""
    if reserve_block != SeedBlock.RESERVE2:
        raise FinalEvaluationError("the spent Prototype #2 reserve may never be re-run")
    repository_root = _repository_root()
    _require_canonical_path(data_root, repository_root / "data" / "reference", name="data_root")
    _require_canonical_path(output_root, repository_root / "results", name="output_root")
    _require_canonical_path(
        config_root,
        repository_root
        / "packages"
        / "rahola-lab"
        / "src"
        / "rahola_lab"
        / "campaigns"
        / "configs",
        name="config_root",
    )
    _require_canonical_path(
        reserve_root, repository_root / "data" / "final-reserve2", name="reserve_root"
    )
    attestation_path = output_root / "final_reserve2_attestation.json"
    if attestation_path.exists() or reserve_root.exists():
        raise FinalEvaluationError("reserve access was already started; refusing a repeat")
    _require_clean_worktree()
    b2_path = output_root / "p3_b2_chronos.json"
    try:
        b2 = load_result(output_root, "p3_b2_chronos")
    except (OSError, ValueError, TypeError):
        b2 = {}
    if not b2.get("survives_kill", False):
        raise FinalEvaluationError("reserve-2 requires a frozen Part B survivor")
    relative_b2 = b2_path.relative_to(repository_root)
    try:
        _git_output("ls-files", "--error-unmatch", str(relative_b2))
    except subprocess.CalledProcessError as error:
        raise FinalEvaluationError("the Part B survivor must be committed") from error
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise FinalEvaluationError("chunk_size must be a positive integer")

    try:
        d1 = load_result(output_root, "d1_operating_curves")
        selected = d1["selected"]
        d1["headline_at_calibration_selected_threshold"]
        d1["decorrelation_time_s"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FinalEvaluationError("D1 artifact is missing or incompatible") from error

    reserve_campaigns = []
    for family in FAMILIES:
        for role in ("evaluation", "ramp"):
            name = f"{family}_{role}"
            definition = load_campaign_definition(config_root / f"{name}.yaml")
            test_split = next(split for split in definition.splits if split.block == SeedBlock.TEST)
            reserve_campaigns.append((name, definition, test_split))

    training_data = [
        load_campaign_split(campaign_dir(data_root, f"{family}_{role}"), SeedBlock.TRAIN)
        for family in FAMILIES
        for role in ("stationary", "ramp")
    ]
    suite = fit_frozen_suite(
        training_windows(training_data, max_windows_per_trajectory=3), selected
    )
    del training_data
    gc.collect()

    foundation_training = foundation_training_windows(
        foundation_training_data(data_root, list(FAMILIES))
    )
    foundation_models = {
        mode: ChronosClassifier(mode=mode, seed=72_001).fit(
            foundation_training.features, foundation_training.labels
        )
        for mode in ("frozen", "finetune")
    }
    foundation_calibration = [
        load_campaign_split(
            campaign_dir(data_root, f"{family}_{role}"),
            SeedBlock.CALIBRATION,
            limit=B2_TRAJECTORIES_PER_CAMPAIGN,
        )
        for family in FAMILIES
        for role in ("stationary", "ramp")
    ]
    foundation_policies = {}
    for mode, model in foundation_models.items():
        calibration_scores = [
            item for dataset in foundation_calibration for item in score_foundation(dataset, model)
        ]
        foundation_policies[mode] = select_foundation_policy(calibration_scores)
    del foundation_calibration
    del foundation_training
    gc.collect()

    expected_digests = {
        "p3_b2_chronos": str(b2["_artifact_sha256"]),
        "d1_operating_curves": str(d1["_artifact_sha256"]),
    }
    revalidated = _revalidate_final_inputs(output_root, expected_digests)
    b2 = revalidated["p3_b2_chronos"]
    d1 = revalidated["d1_operating_curves"]

    commit = _git_output("rev-parse", "HEAD")
    timestamp = datetime.now(UTC).isoformat()
    output_root.mkdir(parents=True, exist_ok=True)
    attestation: dict[str, object] = {
        "git_commit": commit,
        "survivor_artifact": str(relative_b2),
        "survivor_sha256": _sha256(b2_path),
        "started_at_utc": timestamp,
        "status": "started",
        "seed_block": str(reserve_block),
        "statement": "No reserve-2 seed was accessed before this invocation.",
        "repeat_policy": "This reserve-2 run will not be repeated after any outcome.",
    }
    _write_exclusive_json(attestation_path, attestation)

    try:
        score_parts = []
        reserve_datasets = []
        campaigns = []
        for name, definition, test_split in reserve_campaigns:
            reserve_count = min(B2_TRAJECTORIES_PER_CAMPAIGN, test_split.count)
            seeds = _reserve_seeds(reserve_block, reserve_count, test_split.offset)
            campaign_root = reserve_root / name
            chunk_records = []
            capsized = 0
            for chunk_number, start in enumerate(range(0, len(seeds), chunk_size)):
                chunk_seeds = seeds[start : start + chunk_size]
                dataset = simulate_batch(definition.simulation, chunk_seeds)
                reserve_datasets.append(dataset)
                capsized += int(np.sum(dataset.capsized))
                score_parts.append(score_dataset(dataset, suite))
                chunk_path = campaign_root / f"chunk-{chunk_number:05d}"
                manifest_path = write_dataset(dataset, chunk_path, shard_size=chunk_size)
                chunk_records.append(
                    {
                        "manifest": str(manifest_path.relative_to(campaign_root)),
                        "count": len(chunk_seeds),
                        "first_seed": int(chunk_seeds[0]),
                        "last_seed": int(chunk_seeds[-1]),
                    }
                )
            campaign_manifest = {
                "name": name,
                "git_commit": commit,
                "reserve_offset": test_split.offset,
                "count": reserve_count,
                "capsized": capsized,
                "capsize_fraction": capsized / reserve_count,
                "simulation": definition.simulation.to_dict(),
                "chunks": chunk_records,
            }
            campaign_root.mkdir(parents=True, exist_ok=True)
            (campaign_root / "manifest.json").write_text(
                json.dumps(campaign_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            campaigns.append(campaign_manifest | {"chunks": len(chunk_records)})

        scores = merge_scores(score_parts)
        horizon_s = EWS_HORIZON_PERIODS * 4.0
        methods = {}
        for name in DETECTOR_NAMES:
            threshold = float(d1["headline_at_calibration_selected_threshold"][name]["threshold"])
            metrics = evaluate_alarms(
                scores[name],
                EpisodeConfig(threshold=threshold, debounce_windows=3, refractory_windows=3),
                horizon_s=horizon_s,
                decorrelation_time_s=float(d1["decorrelation_time_s"][name]),
            )
            methods[name] = _metrics_payload(metrics, threshold)
        for mode, model in foundation_models.items():
            reserve_scores = [
                item for dataset in reserve_datasets for item in score_foundation(dataset, model)
            ]
            methods[f"chronos_{mode}"] = evaluate_foundation_policy(
                reserve_scores, foundation_policies[mode]
            )
        payload: dict[str, object] = {
            "experiment": "Final reserve-2 evaluation",
            "git_commit": commit,
            "reserve_seed_block": [400_000, 500_000],
            "reserve_protocol": (
                "128 trajectories per D1-mirroring campaign, matching the frozen B2 CPU probe; "
                "all methods use the same 768 trajectories."
            ),
            "campaigns": campaigns,
            "methods": methods,
        }
        with result_graph_lock(output_root):
            result_path = _write_result_locked(
                output_root,
                "final_reserve2",
                payload,
                upstream_results={
                    "d1_operating_curves": d1,
                    "p3_b2_chronos": b2,
                },
            )
            load_result(output_root, "final_reserve2")
            attestation.update(
                {
                    "status": "complete",
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "result": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "campaign_count": len(campaigns),
                    "trajectory_count": int(sum(item["count"] for item in campaigns)),
                }
            )
            _write_atomic_json(attestation_path, attestation)
        return payload
    except Exception as error:
        attestation.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _write_atomic_json(attestation_path, attestation)
        raise
