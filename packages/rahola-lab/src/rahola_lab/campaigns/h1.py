"""Predeclared fresh TEST slices for the H1 one-shot evaluation."""

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

H1_TEST_SLICES: dict[str, tuple[int, int]] = {
    "softening_stationary": (500, 92_000),
    "parametric_stationary": (500, 92_500),
    "biased_stationary": (500, 93_000),
    "softening_evaluation": (1_500, 93_500),
    "parametric_evaluation": (3_200, 95_000),
    "biased_evaluation": (1_700, 98_200),
}

# DATA.md's measured all-split fractions, frozen for the H1 power floor.
H1_MEASURED_CAPSIZE_FRACTIONS: dict[str, float] = {
    "softening_stationary": 0.0865,
    "parametric_stationary": 0.0845,
    "biased_stationary": 0.1330,
    "softening_evaluation": 0.0200,
    "parametric_evaluation": 0.0095,
    "biased_evaluation": 0.01867,
}


def h1_name(base_name: str) -> str:
    if base_name not in H1_TEST_SLICES:
        raise ValueError(f"unknown H1 campaign: {base_name}")
    return f"{base_name}_h1"


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def verify_h1_test_slices(manifest_roots: tuple[Path, ...]) -> dict[str, object]:
    """Prove every H1 TEST slice is fresh and pairwise disjoint."""
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

    proposed = {name: (offset, offset + count) for name, (count, offset) in H1_TEST_SLICES.items()}
    for name, interval in proposed.items():
        conflicts = [path for path, start, stop in occupied if _overlap(interval, (start, stop))]
        if conflicts:
            raise ValueError(f"H1 TEST slice {name} overlaps {conflicts}")
    names = list(proposed)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if _overlap(proposed[name], proposed[other]):
                raise ValueError(f"H1 TEST slices {name} and {other} overlap")

    block_start = SEED_BLOCK_START[SeedBlock.TEST]
    merged: list[list[int]] = []
    for start, stop in sorted((start, stop) for _, start, stop in occupied):
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    occupied_size = sum(stop - start for start, stop in merged)
    return {
        "verified_manifest_count": len(manifests),
        "verified_manifests": manifests,
        "slices": {
            name: {
                "count": count,
                "offset": offset,
                "absolute_half_open_range": [
                    block_start + offset,
                    block_start + offset + count,
                ],
            }
            for name, (count, offset) in H1_TEST_SLICES.items()
        },
        "pairwise_disjoint": True,
        "disjoint_from_existing_manifests": True,
        "declared_count": sum(count for count, _ in H1_TEST_SLICES.values()),
        "expected_capsizes": {
            name: count * H1_MEASURED_CAPSIZE_FRACTIONS[name]
            for name, (count, _) in H1_TEST_SLICES.items()
        },
        "occupied_interval_sum_before_h1": occupied_size,
        "untouched_test_room_before_h1": 100_000 - occupied_size,
    }


def h1_definitions(config_root: Path) -> tuple[CampaignDefinition, ...]:
    definitions = []
    for base_name, (count, offset) in H1_TEST_SLICES.items():
        source = load_campaign_definition(config_root / f"{base_name}.yaml")
        definitions.append(
            replace(
                source,
                name=h1_name(base_name),
                rationale=(
                    f"Fresh H1 one-shot TEST slice for {base_name}; "
                    "the hybrid design was frozen before materialization."
                ),
                splits=(SplitDefinition(block=SeedBlock.TEST, count=count, offset=offset),),
            )
        )
    return tuple(definitions)


def generate_h1_campaigns(
    config_root: Path,
    output_root: Path,
    existing_roots: tuple[Path, ...],
    *,
    chunk_size: int = 256,
) -> tuple[GenerationResult, ...]:
    """Generate each predeclared H1 slice exactly once."""
    verify_h1_test_slices(existing_roots)
    results = []
    for definition in h1_definitions(config_root):
        campaign_root = output_root / definition.name
        if campaign_root.exists():
            raise FileExistsError(f"refusing to overwrite H1 campaign {campaign_root}")
        results.append(generate_campaign(definition, output_root, chunk_size=chunk_size))
    return tuple(results)
