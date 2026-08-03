"""One-time, guarded scoring on the current inaccessible reserve block."""

from __future__ import annotations

import gc
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rahola.simulate import simulate_batch
from rahola.storage import write_dataset
from rahola_lab.campaigns import load_campaign_definition, load_campaign_split
from rahola_lab.constants import (
    EWS_HORIZON_PERIODS,
    SEED_BLOCK_SIZE,
    SEED_BLOCK_START,
    SeedBlock,
)
from rahola_lab.evaluation import AlarmMetrics, EpisodeConfig, evaluate_alarms
from rahola_lab.experiments.common import FAMILIES, write_result
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


def _git_output(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _reserve_seeds(block: SeedBlock, count: int, offset: int) -> np.ndarray:
    """Materialize reserve-2; the spent Prototype #2 reserve is permanently refused."""
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
    """Score the frozen suite exactly once and permanently attest the access."""
    if reserve_block != SeedBlock.RESERVE2:
        raise FinalEvaluationError("the spent Prototype #2 reserve may never be re-run")
    attestation_path = output_root / "final_reserve2_attestation.json"
    if attestation_path.exists() or reserve_root.exists():
        raise FinalEvaluationError("reserve access was already started; refusing a repeat")
    if _git_output("status", "--porcelain"):
        raise FinalEvaluationError("final-eval requires a clean committed working tree")
    commit = _git_output("rev-parse", "HEAD")
    timestamp = datetime.now(UTC).isoformat()
    output_root.mkdir(parents=True, exist_ok=True)
    attestation: dict[str, object] = {
        "git_commit": commit,
        "started_at_utc": timestamp,
        "status": "started",
        "seed_block": str(reserve_block),
        "statement": "No reserve-2 seed was accessed before this invocation.",
        "repeat_policy": "This reserve-2 run will not be repeated after any outcome.",
    }
    attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    try:
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

        score_parts = []
        campaigns = []
        for family in FAMILIES:
            for role in ("evaluation", "ramp"):
                name = f"{family}_{role}"
                definition = load_campaign_definition(config_root / f"{name}.yaml")
                test_split = next(
                    split for split in definition.splits if split.block == SeedBlock.TEST
                )
                seeds = _reserve_seeds(reserve_block, test_split.count, test_split.offset)
                campaign_root = reserve_root / name
                chunk_records = []
                capsized = 0
                for chunk_number, start in enumerate(range(0, len(seeds), chunk_size)):
                    chunk_seeds = seeds[start : start + chunk_size]
                    dataset = simulate_batch(definition.simulation, chunk_seeds)
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
                    "count": test_split.count,
                    "capsized": capsized,
                    "capsize_fraction": capsized / test_split.count,
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
            threshold = float(d1["headline_at_90_percent_sensitivity"][name]["threshold"])
            metrics = evaluate_alarms(
                scores[name],
                EpisodeConfig(threshold=threshold, debounce_windows=3, refractory_windows=3),
                horizon_s=horizon_s,
                decorrelation_time_s=float(d1["decorrelation_time_s"][name]),
            )
            methods[name] = _metrics_payload(metrics, threshold)
        payload: dict[str, object] = {
            "experiment": "Final reserve-2 evaluation",
            "git_commit": commit,
            "reserve_seed_block": [400_000, 500_000],
            "campaigns": campaigns,
            "methods": methods,
        }
        result_path = write_result(output_root, "final_reserve2", payload)
        attestation.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "result": str(result_path),
                "campaign_count": len(campaigns),
                "trajectory_count": int(sum(item["count"] for item in campaigns)),
            }
        )
        attestation_path.write_text(
            json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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
        attestation_path.write_text(
            json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
