"""Predeclared fresh ordinary-TEST slices for the final F1 experiment."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from rahola_lab.campaigns.definition import (
    CampaignDefinition,
    SplitDefinition,
    load_campaign_definition,
)
from rahola_lab.campaigns.generate import GenerationResult, generate_campaign
from rahola_lab.constants import SEED_BLOCK_START, SeedBlock

F1_TEST_SLICES: dict[str, tuple[int, int]] = {
    "softening_evaluation": (1_500, 1_000),
    "parametric_evaluation": (3_200, 6_000),
    "biased_evaluation": (1_700, 21_000),
    "softening_step_v02": (3_000, 38_000),
}

F1_EXPECTED_CAPSIZE_FRACTIONS: dict[str, float] = {
    "softening_evaluation": 0.0200,
    "parametric_evaluation": 0.0095,
    "biased_evaluation": 0.01867,
    "softening_step_v02": 0.6290,
}


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def verify_f1_test_slices(manifest_roots: tuple[Path, ...]) -> dict[str, object]:
    """Prove every F1 slice is fresh, pairwise disjoint, and non-reserve."""
    occupied: list[tuple[str, int, int]] = []
    manifests = []
    for root in manifest_roots:
        for path in sorted(root.glob("*/manifest.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            test = payload.get("splits", {}).get(str(SeedBlock.TEST))
            if test is None:
                continue
            start = int(test["offset"])
            stop = start + int(test["count"])
            occupied.append((str(path), start, stop))
            manifests.append(str(path))
    proposed = {
        name: (offset, offset + count) for name, (count, offset) in F1_TEST_SLICES.items()
    }
    for name, interval in proposed.items():
        conflicts = [path for path, start, stop in occupied if _overlap(interval, (start, stop))]
        if conflicts:
            raise ValueError(f"F1 TEST slice {name} overlaps {conflicts}")
    names = list(proposed)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if _overlap(proposed[name], proposed[other]):
                raise ValueError(f"F1 TEST slices {name} and {other} overlap")
    block_start = SEED_BLOCK_START[SeedBlock.TEST]
    return {
        "verified_manifests": manifests,
        "slices": {
            name: {
                "count": count,
                "offset": offset,
                "absolute_half_open_range": [
                    block_start + offset,
                    block_start + offset + count,
                ],
                "expected_capsizes": count * F1_EXPECTED_CAPSIZE_FRACTIONS[name],
            }
            for name, (count, offset) in F1_TEST_SLICES.items()
        },
        "pairwise_disjoint": True,
        "disjoint_from_existing_manifests": True,
        "declared_count": sum(count for count, _ in F1_TEST_SLICES.values()),
    }


def f1_definitions(config_root: Path) -> tuple[CampaignDefinition, ...]:
    definitions = []
    for base_name, (count, offset) in F1_TEST_SLICES.items():
        source = load_campaign_definition(config_root / f"{base_name}.yaml")
        definitions.append(
            replace(
                source,
                name=f"{base_name}_f1",
                rationale=f"Fresh one-shot F1 TEST slice for {base_name}.",
                splits=(SplitDefinition(block=SeedBlock.TEST, count=count, offset=offset),),
            )
        )
    return tuple(definitions)


def generate_f1_campaigns(
    config_root: Path,
    output_root: Path,
    existing_roots: tuple[Path, ...],
    *,
    chunk_size: int = 256,
) -> tuple[GenerationResult, ...]:
    """Materialize the frozen F1 slices once, refusing overlap or overwrite."""
    verify_f1_test_slices(existing_roots)
    results = []
    for definition in f1_definitions(config_root):
        campaign_root = output_root / definition.name
        if campaign_root.exists():
            raise FileExistsError(f"refusing to overwrite F1 campaign {campaign_root}")
        results.append(generate_campaign(definition, output_root, chunk_size=chunk_size))
    return tuple(results)
