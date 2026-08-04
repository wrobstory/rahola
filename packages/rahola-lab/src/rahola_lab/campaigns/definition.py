"""Self-contained YAML definitions for frozen reference campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import yaml

from rahola.config import SimulationConfig
from rahola_lab.constants import SEED_BLOCK_SIZE, SeedBlock


@dataclass(frozen=True)
class SplitDefinition:
    block: SeedBlock
    count: int
    offset: int


@dataclass(frozen=True)
class CampaignDefinition:
    name: str
    role: str
    rationale: str
    simulation: SimulationConfig
    splits: tuple[SplitDefinition, ...]


def _integer_control(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-boolean integer")
    return int(value)


def load_campaign_definition(path: str | Path) -> CampaignDefinition:
    with Path(path).open("rb") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)
    splits = tuple(
        SplitDefinition(
            block=SeedBlock(name),
            count=_integer_control(values["count"], name=f"splits.{name}.count"),
            offset=_integer_control(values.get("offset", 0), name=f"splits.{name}.offset"),
        )
        for name, values in raw["splits"].items()
    )
    if any(split.block in {SeedBlock.RESERVE, SeedBlock.RESERVE2} for split in splits):
        raise ValueError("campaign definitions may not allocate reserve seeds")
    if any(split.count < 1 or split.offset < 0 for split in splits):
        raise ValueError("campaign split counts must be positive and offsets nonnegative")
    if any(split.offset + split.count > SEED_BLOCK_SIZE for split in splits):
        raise ValueError("campaign split must fit inside its frozen seed block")
    return CampaignDefinition(
        name=str(raw["name"]),
        role=str(raw["role"]),
        rationale=str(raw["rationale"]),
        simulation=SimulationConfig.from_dict(raw["simulation"]),
        splits=splits,
    )
