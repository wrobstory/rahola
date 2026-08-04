"""B2 v0.2 orientation/leakage audit on the established-regime geometry."""

from __future__ import annotations

from pathlib import Path

from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import (
    B2_D5_LEAKAGE_AUC,
    CHRONOS_CHECKPOINT,
    CHRONOS_LICENSE,
    CHRONOS_REVISION,
    SeedBlock,
)
from rahola_lab.detectors import ChronosClassifier
from rahola_lab.experiments.b2_chronos import (
    _score_foundation,
    _training_data,
    _training_windows,
)
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.d5_v02 import fully_post_step
from rahola_lab.experiments.detector_common import bootstrap_window_auc


def run(
    historical_root: Path, versioned_root: Path, output_root: Path
) -> dict[str, object]:
    training = _training_windows(_training_data(historical_root, list(FAMILIES)))
    step = load_campaign_split(
        versioned_root / "softening_step_v02", SeedBlock.TEST, limit=128
    )
    modes = {}
    for offset, mode in enumerate(("frozen", "finetune")):
        model = ChronosClassifier(mode=mode, seed=72_001 + offset).fit(
            training.features, training.labels
        )
        scores = fully_post_step(
            {"chronos": _score_foundation(step, model)}
        )["chronos"]
        auc = bootstrap_window_auc(scores)
        raw = float(auc["auc"])
        orientation_independent = max(raw, 1.0 - raw)
        modes[mode] = auc | {
            "orientation_independent_auc": orientation_independent,
            "leakage_audit_triggered": orientation_independent > B2_D5_LEAKAGE_AUC,
            "scored_trajectories": step.batch_size,
        }
    payload: dict[str, object] = {
        "experiment": "B2 D5_v02 orientation/leakage audit",
        "checkpoint": CHRONOS_CHECKPOINT,
        "revision": CHRONOS_REVISION,
        "license": CHRONOS_LICENSE,
        "geometry": {
            "transition_s": 300.0,
            "first_endpoint_s": 540.0,
            "last_endpoint_s": 700.0,
            "complete_history_and_horizon": True,
        },
        "orientation_independent_trigger": B2_D5_LEAKAGE_AUC,
        "modes": modes,
        "any_leakage_audit_triggered": any(
            bool(result["leakage_audit_triggered"]) for result in modes.values()
        ),
    }
    write_result(output_root, "p3_b2_d5_audit_v02", payload)
    return payload
