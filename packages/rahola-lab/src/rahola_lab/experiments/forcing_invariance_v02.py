"""Paired audit of the v0.1 Nyquist and v0.2 fixed-cutoff forcing definitions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from rahola.simulate import simulate_batch
from rahola_lab.campaigns import load_campaign_definition
from rahola_lab.constants import (
    FORCING_PAIRED_TRAJECTORIES_PER_CAMPAIGN,
    FORCING_PREVALENCE_TOLERANCE,
)
from rahola_lab.evaluation import seeds_for
from rahola_lab.experiments.common import write_result


def _historical_prevalence(manifest: dict[str, object]) -> tuple[int, int]:
    splits = manifest["splits"]
    count = sum(int(split["count"]) for split in splits.values())
    capsized = sum(int(split["capsized"]) for split in splits.values())
    return capsized, count


def affected_campaigns(
    rows: list[dict[str, object]], *, tolerance: float = FORCING_PREVALENCE_TOLERANCE
) -> list[str]:
    """Apply the preregistered absolute-prevalence decision rule."""
    return [
        str(row["campaign"])
        for row in rows
        if abs(float(row["prevalence_shift"])) > tolerance
    ]


def run(
    config_root: Path,
    data_root: Path,
    output_root: Path,
    *,
    chunk_size: int = 128,
) -> dict[str, object]:
    definitions = [
        load_campaign_definition(path)
        for path in sorted(config_root.glob("*.yaml"))
        if path.stem != "softening_step_v02"
    ]
    rows: list[dict[str, object]] = []
    for definition in definitions:
        manifest = json.loads(
            (data_root / definition.name / "manifest.json").read_text(encoding="utf-8")
        )
        historical_capsized, total = _historical_prevalence(manifest)
        cutoff_capsized = 0
        paired_seen = 0
        paired_old_capsized = 0
        paired_new_capsized = 0
        paired_discordant = 0
        paired_time_differences: list[float] = []
        legacy = replace(
            definition.simulation,
            forcing=replace(
                definition.simulation.forcing,
                max_frequency_ratio=None,
            ),
        )
        for split in definition.splits:
            split_seeds = seeds_for(split.block, split.count, offset=split.offset)
            for start in range(0, len(split_seeds), chunk_size):
                chunk_seeds = split_seeds[start : start + chunk_size]
                cutoff = simulate_batch(definition.simulation, chunk_seeds)
                cutoff_capsized += int(np.sum(cutoff.capsized))
                remaining = FORCING_PAIRED_TRAJECTORIES_PER_CAMPAIGN - paired_seen
                if remaining <= 0:
                    continue
                pair_count = min(remaining, len(chunk_seeds))
                legacy_data = simulate_batch(legacy, chunk_seeds[:pair_count])
                new_capsized = cutoff.capsized[:pair_count]
                old_capsized = legacy_data.capsized
                paired_old_capsized += int(np.sum(old_capsized))
                paired_new_capsized += int(np.sum(new_capsized))
                paired_discordant += int(np.sum(old_capsized != new_capsized))
                both = old_capsized & new_capsized
                paired_time_differences.extend(
                    np.abs(
                        legacy_data.t_capsize_s[both]
                        - cutoff.t_capsize_s[:pair_count][both]
                    ).tolist()
                )
                paired_seen += pair_count
        historical_prevalence = historical_capsized / total
        cutoff_prevalence = cutoff_capsized / total
        rows.append(
            {
                "campaign": definition.name,
                "trajectories": total,
                "legacy_capsized": historical_capsized,
                "legacy_prevalence": historical_prevalence,
                "fixed_cutoff_capsized": cutoff_capsized,
                "fixed_cutoff_prevalence": cutoff_prevalence,
                "prevalence_shift": cutoff_prevalence - historical_prevalence,
                "paired_trajectories": paired_seen,
                "paired_legacy_capsized": paired_old_capsized,
                "paired_fixed_cutoff_capsized": paired_new_capsized,
                "paired_discordant_capsize_outcomes": paired_discordant,
                "paired_both_capsized": len(paired_time_differences),
                "paired_median_absolute_capsize_time_shift_s": (
                    float(np.median(paired_time_differences))
                    if paired_time_differences
                    else None
                ),
                "paired_maximum_absolute_capsize_time_shift_s": (
                    float(np.max(paired_time_differences))
                    if paired_time_differences
                    else None
                ),
            }
        )
    affected = affected_campaigns(rows)
    payload: dict[str, object] = {
        "experiment": "v0.2 forcing-definition invariance audit",
        "fixed_max_frequency_ratio": 4.0,
        "legacy_definition": "integration-half-step Nyquist cutoff",
        "predeclared_absolute_prevalence_tolerance": FORCING_PREVALENCE_TOLERANCE,
        "paired_trajectories_per_campaign": FORCING_PAIRED_TRAJECTORIES_PER_CAMPAIGN,
        "rows": rows,
        "affected_campaigns": affected,
        "decision": (
            "regenerate affected campaigns as _v02 data"
            if affected
            else "retain historical campaigns; adopt fixed cutoff for new data"
        ),
        "interpretation": (
            "The roll oscillator attenuates high-frequency forcing, which explains why the "
            "legacy solver-convergence tests passed despite their Nyquist-defined sea field. "
            "D3 used one common integration grid, so its internal bandwidth comparison remains "
            "well-defined under the historical forcing convention."
        ),
    }
    write_result(output_root, "forcing_invariance_v02", payload)
    return payload
