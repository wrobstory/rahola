from __future__ import annotations

from pathlib import Path

from rahola_lab.experiments.common import _artifact_digest
from rahola_lab.experiments.v02_common import (
    campaign_path_v02,
    load_frozen_v02_result,
)


def test_v02_campaign_router_uses_only_anchored_regenerations() -> None:
    historical = Path("historical")
    versioned = Path("versioned")
    assert campaign_path_v02(
        historical, versioned, "softening_bandwidth_gamma_7"
    ) == versioned / "softening_bandwidth_gamma_7_v02"
    assert campaign_path_v02(
        historical, versioned, "softening_bandwidth_gamma_3_3"
    ) == historical / "softening_bandwidth_gamma_3_3"


def test_frozen_v02_result_loader_checks_content_digest(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = {"value": 3}
    payload["_artifact_sha256"] = _artifact_digest(payload)
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    assert load_frozen_v02_result(path)["value"] == 3
    path.write_text('{"value":4,"_artifact_sha256":"bad"}', encoding="utf-8")
    try:
        load_frozen_v02_result(path)
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("corrupt result was accepted")
