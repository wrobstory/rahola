"""Self-contained YAML definitions for frozen reference campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rahola.config import SimulationConfig
from rahola_lab.constants import SeedBlock


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


def load_campaign_definition(path: str | Path) -> CampaignDefinition:
    with Path(path).open("rb") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)
    splits = tuple(
        SplitDefinition(
            block=SeedBlock(name),
            count=int(values["count"]),
            offset=int(values.get("offset", 0)),
        )
        for name, values in raw["splits"].items()
    )
    if any(split.block == SeedBlock.RESERVE for split in splits):
        raise ValueError("campaign definitions may not allocate reserve seeds")
    return CampaignDefinition(
        name=str(raw["name"]),
        role=str(raw["role"]),
        rationale=str(raw["rationale"]),
        simulation=SimulationConfig.from_dict(raw["simulation"]),
        splits=splits,
    )
