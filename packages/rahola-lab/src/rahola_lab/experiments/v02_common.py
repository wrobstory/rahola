"""Shared data routing and serialization for non-destructive v0.2 reruns."""

from __future__ import annotations

import json
from pathlib import Path

from rahola.dataset import SimulationDataset
from rahola_lab.campaigns import load_campaign_split
from rahola_lab.constants import SeedBlock
from rahola_lab.evaluation import OperatingPoint
from rahola_lab.experiments.detector_common import point_payload


def v02_campaign_names() -> frozenset[str]:
    path = (
        Path(__file__).parents[1]
        / "campaigns"
        / "reference_checksums_v02.json"
    )
    anchored = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(name.removesuffix("_v02") for name in anchored)


def campaign_path_v02(
    historical_root: Path, versioned_root: Path, name: str
) -> Path:
    return (
        versioned_root / f"{name}_v02"
        if name in v02_campaign_names()
        else historical_root / name
    )


def load_campaign_split_v02(
    historical_root: Path,
    versioned_root: Path,
    name: str,
    block: SeedBlock,
) -> SimulationDataset:
    return load_campaign_split(
        campaign_path_v02(historical_root, versioned_root, name), block
    )


def campaign_strata(
    names: list[str], datasets: list[SimulationDataset]
) -> list[str]:
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
    payload["sensitivity_exact_capsize_event_interval"] = payload.pop(
        "sensitivity_interval"
    )
    payload.pop("false_episodes_per_hour_interval")
    return payload
