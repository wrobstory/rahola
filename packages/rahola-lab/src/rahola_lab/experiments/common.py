"""Shared, deliberately small experiment plumbing."""

from __future__ import annotations

import json
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
            cap_time = float(chunk.t_capsize_s[local_index])
            if not np.isfinite(cap_time):
                cap_time = None
            output.append(
                TrajectoryForecast(
                    times_s=samples.history_end_s[selected],
                    targets_rad=samples.targets_rad[selected, 0],
                    angle_rad=samples.histories[selected, -1, 0],
                    rate_rad_s=samples.histories[selected, -1, 1],
                    raw_upper_rad={name: values[selected] for name, values in predictions.items()},
                    record_end_s=float(chunk.time_s[-1]),
                    t_capsize_s=cap_time,
                )
            )
    return output


def write_result(output_root: Path, name: str, payload: dict[str, object]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
