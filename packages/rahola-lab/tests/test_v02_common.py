from __future__ import annotations

from pathlib import Path

from rahola_lab.experiments.v02_common import campaign_path_v02


def test_v02_campaign_router_uses_only_anchored_regenerations() -> None:
    historical = Path("historical")
    versioned = Path("versioned")
    assert campaign_path_v02(
        historical, versioned, "softening_bandwidth_gamma_7"
    ) == versioned / "softening_bandwidth_gamma_7_v02"
    assert campaign_path_v02(
        historical, versioned, "softening_bandwidth_gamma_3_3"
    ) == historical / "softening_bandwidth_gamma_3_3"
