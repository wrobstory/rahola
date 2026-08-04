from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rahola_lab.experiments.common import _artifact_digest
from rahola_lab.experiments.v02_common import (
    campaign_path_v02,
    load_frozen_v02_result,
    provenance_manifest_digest,
)


def _write_manifest(root: Path, artifacts: dict[str, object]) -> None:
    campaign_root = Path(__file__).parents[1] / "src" / "rahola_lab" / "campaigns"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "reference_anchors_sha256": {
            name: hashlib.sha256((campaign_root / name).read_bytes()).hexdigest()
            for name in ("reference_checksums.json", "reference_checksums_v02.json")
        },
        "artifacts": artifacts,
    }
    manifest["_manifest_sha256"] = provenance_manifest_digest(manifest)
    (root / "provenance_manifest_v02.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_v02_campaign_router_uses_only_anchored_regenerations() -> None:
    historical = Path("historical")
    versioned = Path("versioned")
    assert (
        campaign_path_v02(historical, versioned, "softening_bandwidth_gamma_7")
        == versioned / "softening_bandwidth_gamma_7_v02"
    )
    assert (
        campaign_path_v02(historical, versioned, "softening_bandwidth_gamma_3_3")
        == historical / "softening_bandwidth_gamma_3_3"
    )


def test_frozen_v02_result_loader_checks_content_digest(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = {"value": 3}
    payload["_artifact_sha256"] = _artifact_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _write_manifest(
        tmp_path,
        {"result.json": {"artifact_sha256": payload["_artifact_sha256"], "upstream_artifacts": {}}},
    )
    assert load_frozen_v02_result(path)["value"] == 3
    path.write_text('{"value":4,"_artifact_sha256":"bad"}', encoding="utf-8")
    try:
        load_frozen_v02_result(path)
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("corrupt result was accepted")


def test_frozen_v02_result_loader_rejects_mutated_upstream(tmp_path: Path) -> None:
    upstream_path = tmp_path / "upstream.json"
    upstream = {"value": 1}
    upstream["_artifact_sha256"] = _artifact_digest(upstream)
    upstream_path.write_text(json.dumps(upstream), encoding="utf-8")
    result_path = tmp_path / "result.json"
    result = {"value": 2}
    result["_artifact_sha256"] = _artifact_digest(result)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "result.json": {
                "artifact_sha256": result["_artifact_sha256"],
                "upstream_artifacts": {"upstream.json": upstream["_artifact_sha256"]},
            }
        },
    )
    assert load_frozen_v02_result(result_path)["value"] == 2
    upstream_path.write_text('{"value":9,"_artifact_sha256":"bad"}', encoding="utf-8")
    try:
        load_frozen_v02_result(result_path)
    except ValueError as error:
        assert "upstream artifact mismatch" in str(error)
    else:
        raise AssertionError("mutated upstream was accepted")


def test_frozen_v02_result_loader_rejects_mutated_manifest(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = {"value": 3}
    payload["_artifact_sha256"] = _artifact_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _write_manifest(
        tmp_path,
        {"result.json": {"artifact_sha256": payload["_artifact_sha256"], "upstream_artifacts": {}}},
    )
    manifest_path = tmp_path / "provenance_manifest_v02.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["result.json"]["artifact_sha256"] = "bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        load_frozen_v02_result(path)
    except ValueError as error:
        assert "manifest digest mismatch" in str(error)
    else:
        raise AssertionError("mutated provenance manifest was accepted")


def test_frozen_v02_result_loader_rejects_wrong_reference_anchor(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = {"value": 3}
    payload["_artifact_sha256"] = _artifact_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _write_manifest(
        tmp_path,
        {"result.json": {"artifact_sha256": payload["_artifact_sha256"], "upstream_artifacts": {}}},
    )
    manifest_path = tmp_path / "provenance_manifest_v02.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reference_anchors_sha256"]["reference_checksums_v02.json"] = "bad"
    manifest["_manifest_sha256"] = provenance_manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        load_frozen_v02_result(path)
    except ValueError as error:
        assert "reference anchor digest mismatch" in str(error)
    else:
        raise AssertionError("incorrect v0.2 reference anchor was accepted")
