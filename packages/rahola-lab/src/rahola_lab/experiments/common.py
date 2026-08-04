"""Shared, deliberately small experiment plumbing."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rahola.dataset import SimulationDataset
from rahola_lab.constants import FORECAST_HISTORY_S
from rahola_lab.forecast import (
    EnvelopePersistenceForecaster,
    JaxLSTMQuantileForecaster,
    LinearQuantileForecaster,
    extract_forecast_dataset,
)

FloatArray = NDArray[np.float64]
FAMILIES = ("softening", "parametric", "biased")
MODEL_NAMES = ("envelope", "linear", "lstm")


def _artifact_path(output_root: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).is_absolute()
        or Path(name).name != name
    ):
        raise ValueError(f"invalid result artifact name: {name!r}")
    return output_root / f"{name}.json"


@contextmanager
def result_graph_lock(output_root: Path) -> Iterator[None]:
    """Serialize result-graph publication across processes for one output root."""
    lock_root = Path(tempfile.gettempdir()) / "rahola-result-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(output_root.resolve()).encode("utf-8")).hexdigest()
    descriptor = os.open(lock_root / f"{key}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class Forecaster(Protocol):
    def fit(self, histories: FloatArray, targets: FloatArray) -> Forecaster: ...

    def predict(self, histories: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class TrajectoryForecast:
    times_s: FloatArray
    targets_rad: FloatArray
    angle_rad: FloatArray
    rate_rad_s: FloatArray
    raw_upper_rad: dict[str, FloatArray]
    record_end_s: float
    t_capsize_s: float | None


def campaign_path(data_root: Path, family: str, role: str) -> Path:
    suffix = "evaluation" if role == "evaluation" else "stationary"
    return data_root / f"{family}_{suffix}"


def subset_dataset(dataset: SimulationDataset, start: int, stop: int) -> SimulationDataset:
    selected = slice(start, stop)
    return SimulationDataset(
        time_s=dataset.time_s,
        angle_rad=dataset.angle_rad[selected],
        rate_rad_s=dataset.rate_rad_s[selected],
        seeds=dataset.seeds[selected],
        capsized=dataset.capsized[selected],
        t_capsize_s=dataset.t_capsize_s[selected],
        metadata=dataset.metadata[selected],
        config=dataset.config,
    )


def fit_forecasters(dataset: SimulationDataset, horizon_s: float) -> dict[str, Forecaster]:
    """Fit the frozen three-tier grid; there is no test-driven model selection."""
    training = extract_forecast_dataset(
        dataset,
        history_s=FORECAST_HISTORY_S,
        horizons_s=(horizon_s,),
        stride_s=90.0,
        max_samples_per_trajectory=4,
        first_history_end_s=180.0,
    )
    if len(training.targets_rad) == 0:
        raise ValueError("training campaign produced no forecast samples")
    targets = training.targets_rad[:, 0]
    models: dict[str, Forecaster] = {
        "envelope": EnvelopePersistenceForecaster(),
        "linear": LinearQuantileForecaster(iterations=750),
        "lstm": JaxLSTMQuantileForecaster(epochs=6, batch_size=128),
    }
    for model in models.values():
        model.fit(training.histories, targets)
    return models


def predict_upper(model: Forecaster, histories: FloatArray, batch_size: int = 4096) -> FloatArray:
    pieces = [
        model.predict(histories[start : start + batch_size])[:, -1]
        for start in range(0, len(histories), batch_size)
    ]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)


def snapshot(
    dataset: SimulationDataset,
    models: dict[str, Forecaster],
    horizon_s: float,
    *,
    history_end_s: float,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    samples = extract_forecast_dataset(
        dataset,
        history_s=FORECAST_HISTORY_S,
        horizons_s=(horizon_s,),
        stride_s=600.0,
        max_samples_per_trajectory=1,
        first_history_end_s=history_end_s,
    )
    predictions = {name: predict_upper(model, samples.histories) for name, model in models.items()}
    return samples.targets_rad[:, 0], predictions


def trajectory_forecasts(
    dataset: SimulationDataset,
    models: dict[str, Forecaster],
    horizon_s: float,
    *,
    stride_s: float = 10.0,
    first_history_end_s: float = FORECAST_HISTORY_S,
    trajectory_batch_size: int = 128,
) -> list[TrajectoryForecast]:
    """Predict dense trajectories in bounded-memory batches."""
    output: list[TrajectoryForecast] = []
    for start in range(0, dataset.batch_size, trajectory_batch_size):
        stop = min(start + trajectory_batch_size, dataset.batch_size)
        chunk = subset_dataset(dataset, start, stop)
        samples = extract_forecast_dataset(
            chunk,
            history_s=FORECAST_HISTORY_S,
            horizons_s=(horizon_s,),
            stride_s=stride_s,
            first_history_end_s=first_history_end_s,
        )
        predictions = {
            name: predict_upper(model, samples.histories) for name, model in models.items()
        }
        for local_index in range(chunk.batch_size):
            selected = samples.trajectory_indices == local_index
            selected_times = samples.history_end_s[selected]
            cap_time = float(chunk.t_capsize_s[local_index])
            if not np.isfinite(cap_time):
                cap_time = None
            output.append(
                TrajectoryForecast(
                    times_s=selected_times,
                    targets_rad=samples.targets_rad[selected, 0],
                    angle_rad=samples.histories[selected, -1, 0],
                    rate_rad_s=samples.histories[selected, -1, 1],
                    raw_upper_rad={name: values[selected] for name, values in predictions.items()},
                    record_end_s=(
                        float(selected_times[-1])
                        if len(selected_times)
                        else float(first_history_end_s)
                    ),
                    t_capsize_s=cap_time,
                )
            )
    return output


def _artifact_digest(document: dict[str, object]) -> str:
    content = dict(document)
    content.pop("_artifact_sha256", None)
    serialized = json.dumps(
        _json_safe(content), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_result(
    output_root: Path,
    name: str,
    payload: dict[str, object],
    *,
    upstream_results: dict[str, dict[str, object]] | None = None,
) -> Path:
    with result_graph_lock(output_root):
        return _write_result_locked(output_root, name, payload, upstream_results=upstream_results)


def _write_result_locked(
    output_root: Path,
    name: str,
    payload: dict[str, object],
    *,
    upstream_results: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Write one result while the caller holds the output-root graph lock."""
    output_root.mkdir(parents=True, exist_ok=True)
    path = _artifact_path(output_root, name)
    document = dict(payload)
    upstream_digests: dict[str, str] = {}
    for upstream_name, upstream in (upstream_results or {}).items():
        current = _load_result(output_root, upstream_name, ancestors=frozenset())
        if _result_depends_on(output_root, upstream_name, name):
            raise ValueError(f"writing result {name} would create a cyclic upstream dependency")
        digest = upstream.get("_artifact_sha256")
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"upstream result {upstream_name} has no verified artifact digest")
        if current.get("_artifact_sha256") != digest:
            raise ValueError(f"upstream result {upstream_name} is no longer current")
        upstream_digests[upstream_name] = digest
    document["_provenance"] = _current_provenance() | {"upstream_artifacts": upstream_digests}
    document["_artifact_sha256"] = _artifact_digest(document)
    serialized = json.dumps(_json_safe(document), indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=output_root
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_result(output_root: Path, name: str) -> dict[str, object]:
    """Load a current, content-intact development result."""
    return _load_result(output_root, name, ancestors=frozenset())


def _load_result(output_root: Path, name: str, *, ancestors: frozenset[str]) -> dict[str, object]:
    if name in ancestors:
        chain = " -> ".join((*sorted(ancestors), name))
        raise ValueError(f"cyclic upstream artifact dependency: {chain}")
    path = _artifact_path(output_root, name)
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json)
    expected = _current_provenance()
    provenance = payload.get("_provenance")
    if not isinstance(provenance, dict) or any(
        provenance.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"{path} was not produced from the current source and campaign anchor")
    upstream_artifacts = provenance.get("upstream_artifacts")
    if not isinstance(upstream_artifacts, dict):
        raise ValueError(f"{path} does not record exact upstream artifact dependencies")
    if payload.get("_artifact_sha256") != _artifact_digest(payload):
        raise ValueError(f"{path} content does not match its artifact digest")
    for upstream_name, recorded_digest in upstream_artifacts.items():
        if not isinstance(upstream_name, str) or not isinstance(recorded_digest, str):
            raise ValueError(f"{path} has an invalid upstream artifact dependency")
        upstream = _load_result(output_root, upstream_name, ancestors=ancestors | {name})
        if upstream.get("_artifact_sha256") != recorded_digest:
            raise ValueError(f"{path} does not match current upstream artifact {upstream_name}")
    return payload


def _result_depends_on(output_root: Path, name: str, target: str) -> bool:
    if name == target:
        return True
    payload = _load_result(output_root, name, ancestors=frozenset())
    upstream_artifacts = payload["_provenance"]["upstream_artifacts"]
    return any(
        _result_depends_on(output_root, upstream_name, target)
        for upstream_name in upstream_artifacts
    )


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _current_provenance() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[5]
    anchor = (
        repository_root
        / "packages"
        / "rahola-lab"
        / "src"
        / "rahola_lab"
        / "campaigns"
        / "reference_checksums.json"
    )
    versioned_anchor = anchor.with_name("reference_checksums_v02.json")
    u1r2_anchor = anchor.with_name("reference_checksums_u1r2.json")
    digest = hashlib.sha256()
    source_roots = (
        repository_root / "src",
        repository_root / "packages" / "rahola-lab" / "src",
        repository_root / "examples",
    )
    files = [
        path
        for root in source_roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml"}
    ]
    files.extend(
        path
        for path in (
            repository_root / "pyproject.toml",
            repository_root / "packages" / "rahola-lab" / "pyproject.toml",
            repository_root / "uv.lock",
        )
        if path.exists()
    )
    for path in sorted(files):
        digest.update(str(path.relative_to(repository_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "schema_version": 1,
        "source_sha256": digest.hexdigest(),
        "reference_anchor_sha256": hashlib.sha256(anchor.read_bytes()).hexdigest(),
        "reference_v02_anchor_sha256": hashlib.sha256(versioned_anchor.read_bytes()).hexdigest(),
        "reference_u1r2_anchor_sha256": hashlib.sha256(u1r2_anchor.read_bytes()).hexdigest(),
    }


def _json_safe(value: object) -> object:
    """Convert non-finite numeric results to explicit JSON null values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
