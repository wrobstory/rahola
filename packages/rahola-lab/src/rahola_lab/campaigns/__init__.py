"""Reference campaign definitions, generation, and loading."""

from rahola_lab.campaigns.definition import CampaignDefinition, load_campaign_definition
from rahola_lab.campaigns.generate import GenerationResult, generate_campaign
from rahola_lab.campaigns.generate_v02 import generate_selected_v02, versioned_definitions
from rahola_lab.campaigns.load import load_campaign_split
from rahola_lab.campaigns.u1r2 import (
    U1R2_TEST_SLICES,
    generate_u1r2_campaigns,
    u1r2_definitions,
    u1r2_name,
    verify_u1r2_test_slices,
)

__all__ = [
    "U1R2_TEST_SLICES",
    "CampaignDefinition",
    "GenerationResult",
    "generate_campaign",
    "generate_selected_v02",
    "generate_u1r2_campaigns",
    "load_campaign_definition",
    "load_campaign_split",
    "u1r2_definitions",
    "u1r2_name",
    "verify_u1r2_test_slices",
    "versioned_definitions",
]
