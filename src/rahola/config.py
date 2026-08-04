"""Typed configuration and dimensional-boundary conversion.

Angles are radians and time is seconds at this public boundary. The integrator
uses x = phi / phi_v and tau = omega_n * t internally.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral
from pathlib import Path
from typing import Any

import yaml


def _require_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")


class Family(StrEnum):
    SOFTENING = "softening"
    PARAMETRIC = "parametric"
    BIASED = "biased"


class ProtocolKind(StrEnum):
    STATIONARY = "stationary"
    RAMPED = "ramped"
    STEP = "step"


class ParametricMode(StrEnum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


@dataclass(frozen=True)
class SeaState:
    hs_m: float = 4.0
    tp_s: float = 10.0
    gamma: float = 3.3

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.hs_m, self.tp_s, self.gamma)):
            raise ValueError("SeaState values must be finite")
        if self.hs_m <= 0 or self.tp_s <= 0 or self.gamma < 1:
            raise ValueError("SeaState requires hs_m > 0, tp_s > 0, and gamma >= 1")


@dataclass(frozen=True)
class ForcingConfig:
    sea_state: SeaState = field(default_factory=SeaState)
    effective_wave_slope: float = 0.08
    min_components: int = 200
    deterministic_amplitudes: bool = True
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        _require_integer(self.min_components, "min_components")
        if not math.isfinite(self.effective_wave_slope) or not math.isfinite(
            self.gravity_m_s2
        ):
            raise ValueError("forcing values must be finite")
        if self.effective_wave_slope < 0:
            raise ValueError("effective_wave_slope must be nonnegative")
        if self.gravity_m_s2 <= 0:
            raise ValueError("gravity_m_s2 must be positive")
        if self.min_components < 200:
            raise ValueError("min_components must be at least 200")
        if not self.deterministic_amplitudes:
            raise ValueError("Phase 0 implements deterministic amplitudes with random phases")


@dataclass(frozen=True)
class ParametricConfig:
    mode: ParametricMode = ParametricMode.DETERMINISTIC
    h0: float = 0.0
    excitation_ratio: float = 2.0
    stochastic_std: float = 0.1

    def __post_init__(self) -> None:
        values = (self.h0, self.excitation_ratio, self.stochastic_std)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("parametric values must be finite")
        if self.excitation_ratio <= 0 or self.stochastic_std < 0:
            raise ValueError("excitation_ratio must be positive and stochastic_std nonnegative")


@dataclass(frozen=True)
class SeaStateStep:
    time_s: float
    sea_state: SeaState

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("step time must be finite and nonnegative")


@dataclass(frozen=True)
class ProtocolConfig:
    kind: ProtocolKind = ProtocolKind.STATIONARY
    ramp_parameter: str | None = None
    ramp_start: float | None = None
    ramp_end: float | None = None
    steps: tuple[SeaStateStep, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (("ramp_start", self.ramp_start), ("ramp_end", self.ramp_end)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when supplied")
        if self.kind == ProtocolKind.RAMPED:
            if self.ramp_parameter not in {"stiffness", "forcing_scale"}:
                raise ValueError("ramp_parameter must be 'stiffness' or 'forcing_scale'")
            if self.ramp_start is None or self.ramp_end is None:
                raise ValueError("ramped protocols require ramp_start and ramp_end")
        if self.kind == ProtocolKind.STEP and not self.steps:
            raise ValueError("step protocols require at least one transition")
        if any(b.time_s <= a.time_s for a, b in zip(self.steps, self.steps[1:], strict=False)):
            raise ValueError("step transition times must increase strictly")


@dataclass(frozen=True)
class SimulationConfig:
    family: Family = Family.SOFTENING
    duration_s: float = 600.0
    natural_period_s: float = 10.0
    escape_angle_rad: float = math.radians(35.0)
    negative_escape_angle_rad: float | None = None
    damping_ratio: float = 0.05
    quadratic_damping: float = 0.02
    bias_moment: float = 0.0
    quintic_coefficient: float = 0.0
    integration_steps_per_period: int = 40
    output_rate_hz: float = 10.0
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    parametric: ParametricConfig = field(default_factory=ParametricConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    initial_angle_rad: float = 0.0
    initial_rate_rad_s: float = 0.0
    linear_restoring: bool = False

    def __post_init__(self) -> None:
        _require_integer(
            self.integration_steps_per_period, "integration_steps_per_period"
        )
        numeric = (
            self.duration_s,
            self.natural_period_s,
            self.escape_angle_rad,
            self.damping_ratio,
            self.quadratic_damping,
            self.bias_moment,
            self.quintic_coefficient,
            self.output_rate_hz,
            self.initial_angle_rad,
            self.initial_rate_rad_s,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("simulation values must be finite")
        if self.negative_escape_angle_rad is not None and not math.isfinite(
            self.negative_escape_angle_rad
        ):
            raise ValueError("negative_escape_angle_rad must be finite when supplied")
        if self.duration_s <= 0 or self.natural_period_s <= 0:
            raise ValueError("duration_s and natural_period_s must be positive")
        if self.escape_angle_rad <= 0:
            raise ValueError("escape_angle_rad must be positive")
        if self.integration_steps_per_period < 40:
            raise ValueError("integration_steps_per_period must be at least 40")
        if self.output_rate_hz <= 0 or self.damping_ratio < 0 or self.quadratic_damping < 0:
            raise ValueError("rates and damping values must be nonnegative")
        if self.negative_escape_angle_rad is not None and self.negative_escape_angle_rad <= 0:
            raise ValueError("negative_escape_angle_rad must be positive")
        if self.family != Family.BIASED and self.negative_escape_angle_rad is not None:
            raise ValueError("asymmetric escape angles are only valid for the biased family")
        output_intervals = self.duration_s * self.output_rate_hz
        if not math.isclose(
            output_intervals, round(output_intervals), rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError("duration_s must contain an integer number of output intervals")
        if any(step.time_s > self.duration_s for step in self.protocol.steps):
            raise ValueError("step times must not exceed duration_s")

    @property
    def omega_n_rad_s(self) -> float:
        return 2.0 * math.pi / self.natural_period_s

    @property
    def integration_dt_s(self) -> float:
        maximum_dt = self.natural_period_s / self.integration_steps_per_period
        output_dt = 1.0 / self.output_rate_hz
        integration_steps_per_output = math.ceil(output_dt / maximum_dt)
        return output_dt / integration_steps_per_output

    @property
    def negative_escape_rad(self) -> float:
        return self.negative_escape_angle_rad or self.escape_angle_rad

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimulationConfig:
        data = dict(raw)
        data["family"] = Family(data.get("family", Family.SOFTENING))
        forcing_raw = dict(data.pop("forcing", {}))
        forcing_raw["sea_state"] = SeaState(**forcing_raw.get("sea_state", {}))
        data["forcing"] = ForcingConfig(**forcing_raw)
        parametric_raw = dict(data.pop("parametric", {}))
        parametric_raw["mode"] = ParametricMode(
            parametric_raw.get("mode", ParametricMode.DETERMINISTIC)
        )
        data["parametric"] = ParametricConfig(**parametric_raw)
        protocol_raw = dict(data.pop("protocol", {}))
        protocol_raw["kind"] = ProtocolKind(protocol_raw.get("kind", ProtocolKind.STATIONARY))
        protocol_raw["steps"] = tuple(
            SeaStateStep(time_s=item["time_s"], sea_state=SeaState(**item["sea_state"]))
            for item in protocol_raw.get("steps", ())
        )
        data["protocol"] = ProtocolConfig(**protocol_raw)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimulationConfig:
        with Path(path).open("rb") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.from_dict(raw)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    return value
