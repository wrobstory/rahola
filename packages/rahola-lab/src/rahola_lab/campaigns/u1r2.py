"""Predeclared fresh TEST slices for the U1-r2 one-shot evaluation."""

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

U1R2_TEST_SLICES: dict[str, tuple[int, int]] = {
    "softening_stationary": (1_000, 11_000),
    "parametric_stationary": (1_000, 12_000),
    "biased_stationary": (1_000, 13_000),
    "softening_ramp": (1_000, 14_000),
    "parametric_ramp": (1_000, 15_000),
    "biased_ramp": (1_000, 16_000),
    "softening_step": (3_000, 17_000),
    "softening_step_v02": (3_000, 44_000),
    "softening_evaluation": (5_000, 77_000),
    "parametric_evaluation": (5_000, 82_000),
    "biased_evaluation": (5_000, 87_000),
}


def u1r2_name(base_name: str) -> str:
    if base_name not in U1R2_TEST_SLICES:
        raise ValueError(f"unknown U1-r2 campaign: {base_name}")
    return f"{base_name}_u1r2"


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def verify_u1r2_test_slices(
    manifest_roots: tuple[Path, ...],
) -> dict[str, object]:
    """Prove every proposed TEST slice is fresh and pairwise disjoint."""
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
        name: (offset, offset + count) for name, (count, offset) in U1R2_TEST_SLICES.items()
    }
    for name, interval in proposed.items():
        conflicts = [path for path, start, stop in occupied if _overlap(interval, (start, stop))]
        if conflicts:
            raise ValueError(f"U1-r2 TEST slice {name} overlaps {conflicts}")
    names = list(proposed)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            if _overlap(proposed[name], proposed[other]):
                raise ValueError(f"U1-r2 TEST slices {name} and {other} overlap")
    block_start = SEED_BLOCK_START[SeedBlock.TEST]
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
            for name, (count, offset) in U1R2_TEST_SLICES.items()
        },
        "pairwise_disjoint": True,
        "disjoint_from_existing_manifests": True,
    }


def u1r2_definitions(config_root: Path) -> tuple[CampaignDefinition, ...]:
    definitions = []
    for base_name, (count, offset) in U1R2_TEST_SLICES.items():
        source = load_campaign_definition(config_root / f"{base_name}.yaml")
        definitions.append(
            replace(
                source,
                name=u1r2_name(base_name),
                rationale=(
                    f"Fresh U1-r2 one-shot TEST slice for {base_name}; "
                    "calibration controls were frozen before materialization."
                ),
                splits=(
                    SplitDefinition(
                        block=SeedBlock.TEST,
                        count=count,
                        offset=offset,
                    ),
                ),
            )
        )
    return tuple(definitions)


def generate_u1r2_campaigns(
    config_root: Path,
    output_root: Path,
    existing_roots: tuple[Path, ...],
    *,
    chunk_size: int = 256,
) -> tuple[GenerationResult, ...]:
    """Generate each predeclared fresh slice exactly once."""
    verify_u1r2_test_slices(existing_roots)
    results = []
    for definition in u1r2_definitions(config_root):
        campaign_root = output_root / definition.name
        if campaign_root.exists():
            raise FileExistsError(f"refusing to overwrite U1-r2 campaign {campaign_root}")
        results.append(generate_campaign(definition, output_root, chunk_size=chunk_size))
    return tuple(results)
