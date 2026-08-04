"""Versioned regeneration selected by the v0.2 forcing audit."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from rahola_lab.campaigns.definition import CampaignDefinition, load_campaign_definition
from rahola_lab.campaigns.generate import GenerationResult, generate_campaign


def versioned_definitions(
    config_root: Path, affected_campaigns: list[str]
) -> list[CampaignDefinition]:
    """Build `_v02` definitions without changing frozen v0.1 YAML files."""
    definitions = []
    for name in affected_campaigns:
        path = config_root / f"{name}.yaml"
        if name == "softening_step":
            path = config_root / "softening_step_v02.yaml"
        definition = load_campaign_definition(path)
        if not definition.name.endswith("_v02"):
            definition = replace(
                definition,
                name=f"{definition.name}_v02",
                rationale=(
                    f"v0.2 fixed-cutoff regeneration of {definition.name}; "
                    f"historical rationale: {definition.rationale}"
                ),
            )
        definitions.append(definition)
    return definitions


def generate_selected_v02(
    forcing_result_path: Path,
    config_root: Path,
    output_root: Path,
    *,
    chunk_size: int = 256,
) -> list[GenerationResult]:
    """Generate only campaigns named by the committed forcing decision."""
    decision = json.loads(forcing_result_path.read_text(encoding="utf-8"))
    affected = [str(name) for name in decision["affected_campaigns"]]
    definitions = versioned_definitions(config_root, affected)
    collisions = [
        definition.name
        for definition in definitions
        if (output_root / definition.name).exists()
    ]
    if collisions:
        raise FileExistsError(f"refusing to overwrite v0.2 campaigns: {collisions}")
    return [
        generate_campaign(definition, output_root, chunk_size=chunk_size)
        for definition in definitions
    ]
