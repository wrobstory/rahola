"""Shared data routing and serialization for non-destructive v0.2 reruns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rahola.dataset import SimulationDataset
from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import SeedBlock
from rahola_lab.evaluation import OperatingPoint
from rahola_lab.experiments.common import _artifact_digest
from rahola_lab.experiments.detector_common import point_payload

V02_PROVENANCE_MANIFEST = "provenance_manifest_v02.json"


def provenance_manifest_digest(payload: dict[str, object]) -> str:
    """Hash a v0.2 provenance manifest independently of its self-digest."""
    document = dict(payload)
    document.pop("_manifest_sha256", None)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def v02_campaign_names() -> frozenset[str]:
    path = Path(__file__).parents[1] / "campaigns" / "reference_checksums_v02.json"
    anchored = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(name.removesuffix("_v02") for name in anchored)


def campaign_path_v02(historical_root: Path, versioned_root: Path, name: str) -> Path:
    return (
        versioned_root / f"{name}_v02" if name in v02_campaign_names() else historical_root / name
    )


def load_campaign_split_v02(
    historical_root: Path,
    versioned_root: Path,
    name: str,
    block: SeedBlock,
) -> SimulationDataset:
    return load_campaign_split(campaign_path_v02(historical_root, versioned_root, name), block)


def campaign_strata(names: list[str], datasets: list[SimulationDataset]) -> list[str]:
    if len(names) != len(datasets):
        raise ValueError("campaign names and datasets must align")
    return [
        name
        for name, dataset in zip(names, datasets, strict=True)
        for _ in range(dataset.batch_size)
    ]


def point_payload_without_dependent_intervals(point: OperatingPoint) -> dict[str, object]:
    """Serialize estimates without the historical window-binomial interval."""
    payload = point_payload(point)
    payload["sensitivity_exact_capsize_event_interval"] = payload.pop("sensitivity_interval")
    payload.pop("false_episodes_per_hour_interval")
    return payload


def load_frozen_v02_result(path: Path) -> dict[str, object]:
    """Verify v0.2 content, campaign anchors, and declared upstream artifacts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("_artifact_sha256") != _artifact_digest(payload):
        raise ValueError(f"result artifact digest mismatch: {path}")
    manifest_path = path.parent / V02_PROVENANCE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("_manifest_sha256") != provenance_manifest_digest(manifest):
        raise ValueError(f"v0.2 provenance manifest digest mismatch: {manifest_path}")
    campaign_root = Path(__file__).parents[1] / "campaigns"
    for name, expected in manifest["reference_anchors_sha256"].items():
        actual = hashlib.sha256((campaign_root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"v0.2 reference anchor digest mismatch: {name}")
    entry = manifest["artifacts"].get(path.name)
    if entry is None or entry.get("artifact_sha256") != payload["_artifact_sha256"]:
        raise ValueError(f"v0.2 artifact is not bound by provenance manifest: {path}")
    for upstream_name, expected in entry.get("upstream_artifacts", {}).items():
        upstream_path = path.parent / upstream_name
        upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
        if (
            upstream.get("_artifact_sha256") != _artifact_digest(upstream)
            or upstream.get("_artifact_sha256") != expected
        ):
            raise ValueError(f"v0.2 upstream artifact mismatch: {upstream_path}")
    return payload
