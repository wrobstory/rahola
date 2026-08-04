"""U1-r2 Phase A: calibration-only crossing and bias diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import genpareto

from rahola.dataset import SimulationDataset
from rahola_lab.constants import U1_PRIOR_STRENGTHS, U1_TAIL_QUANTILES, SeedBlock
from rahola_lab.experiments.common import FAMILIES, write_result
from rahola_lab.experiments.u1_common import (
    campaign_family,
    load_split,
    restoring_fit,
    score_dataset,
    terminal_severities,
)
from rahola_lab.splittime import (
    Crossing,
    GammaRatePrior,
    SplitTimeConfig,
    decluster_crossings,
    detect_crossings,
    estimate_exponential_tail,
    exponential_rate_mle,
    roll_decorrelation_time,
)

R1_QUANTILE = 0.75
R1_PRIOR_STRENGTH = 10.0


@dataclass(frozen=True)
class CrossingSample:
    retained: tuple[Crossing, ...]
    capsized: bool
    capsize_time_s: float | None
    exposure_s: float


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "maximum": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": len(array),
        "minimum": float(np.min(array)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _finite_stop(angle: NDArray[np.float64], rate: NDArray[np.float64]) -> int:
    finite = np.isfinite(angle) & np.isfinite(rate)
    invalid = np.flatnonzero(~finite)
    return int(invalid[0]) if len(invalid) else len(angle)


def _causal_moment_correction(
    dataset: SimulationDataset,
    angle: NDArray[np.float64],
    rate: NDArray[np.float64],
    stop: int,
) -> tuple[dict[int, NDArray[np.float64]], dict[int, NDArray[np.float64]]]:
    """Estimate Eq. 15's particular solution from a causal local moment.

    The observed angular acceleration and configured damping/restoring terms
    reconstruct the excitation moment per unit inertia. A trailing half-period
    mean removes wave-frequency variation. The diagnostic uses the quasi-static
    repeller response ``p=-m/(k1*omega^2)`` and its backward derivative.
    """
    config = dataset.config
    fit = restoring_fit(dataset)
    dt = float(np.median(np.diff(dataset.time_s)))
    omega = 2.0 * np.pi / float(config["natural_period_s"])
    escape = float(config["escape_angle_rad"])
    damping_ratio = float(config["damping_ratio"])
    quadratic = float(config["quadratic_damping"])
    quintic = float(config.get("quintic_coefficient", 0.0))
    bias = float(config.get("bias_moment", 0.0))

    acceleration = np.zeros(stop, dtype=np.float64)
    acceleration[1:] = np.diff(rate[:stop]) / dt
    acceleration[0] = acceleration[1] if stop > 1 else 0.0
    x = angle[:stop] / escape
    restoring = escape * omega**2 * (x - x**3 + quintic * x**5 - bias)
    linear_damping = 2.0 * damping_ratio * omega * rate[:stop]
    quadratic_damping = quadratic * rate[:stop] * np.abs(rate[:stop]) / escape
    moment = acceleration + linear_damping + quadratic_damping + restoring

    width = max(1, int(np.ceil(0.5 * float(config["natural_period_s"]) / dt)))
    cumulative = np.concatenate(([0.0], np.cumsum(moment)))
    indices = np.arange(stop)
    starts = np.maximum(0, indices - width + 1)
    local_moment = (cumulative[indices + 1] - cumulative[starts]) / (indices - starts + 1)
    local_moment_rate = np.zeros(stop, dtype=np.float64)
    local_moment_rate[1:] = np.diff(local_moment) / dt

    particular: dict[int, NDArray[np.float64]] = {}
    particular_rate: dict[int, NDArray[np.float64]] = {}
    for side, side_fit in ((1, fit.positive), (-1, fit.negative)):
        repeller = side_fit.repeller_slope * fit.central_omega_rad_s**2
        particular[side] = -local_moment / repeller
        particular_rate[side] = -local_moment_rate / repeller
    return particular, particular_rate


def _samples(
    dataset: SimulationDataset,
    *,
    decorrelation_scale: float = 1.0,
    forced_correction: bool = False,
) -> list[CrossingSample]:
    fit = restoring_fit(dataset)
    dt = float(np.median(np.diff(dataset.time_s)))
    output: list[CrossingSample] = []
    for row, (angle, rate) in enumerate(zip(dataset.angle_rad, dataset.rate_rad_s, strict=True)):
        stop = _finite_stop(angle, rate)
        capsize_time = float(dataset.t_capsize_s[row])
        if not np.isfinite(capsize_time):
            capsize_time = None
        if stop < 3:
            output.append(CrossingSample((), bool(dataset.capsized[row]), capsize_time, 0.0))
            continue
        crossings = detect_crossings(dataset.time_s, angle, rate, fit)
        if forced_correction:
            particular, particular_rate = _causal_moment_correction(dataset, angle, rate, stop)
            corrected = []
            for event in crossings:
                side_fit = fit.positive if event.side == 1 else fit.negative
                index = event.detection_index
                critical = side_fit.critical_rate_at_threshold(
                    particular[event.side][index],
                    particular_rate[event.side][index],
                )
                critical = max(float(np.finfo(np.float64).tiny), critical)
                corrected.append(
                    replace(
                        event,
                        critical_rate_rad_s=critical,
                        severity_u=event.outward_rate_rad_s / critical,
                    )
                )
            crossings = tuple(corrected)
        decorrelation = decorrelation_scale * roll_decorrelation_time(angle[:stop], dt)
        retained = decluster_crossings(crossings, decorrelation)
        output.append(
            CrossingSample(
                retained=retained,
                capsized=bool(dataset.capsized[row]),
                capsize_time_s=capsize_time,
                exposure_s=float(dataset.time_s[stop - 1] - dataset.time_s[0]),
            )
        )
    return output


def _crossing_structure(samples: list[CrossingSample]) -> dict[str, object]:
    first_terminal = 0
    eligible_capsizes = 0
    capsize_without_crossing = 0
    first_to_capsize: list[float] = []
    for sample in samples:
        if not sample.capsized:
            continue
        if not sample.retained or sample.capsize_time_s is None:
            capsize_without_crossing += 1
            continue
        terminal = sample.retained[-1]
        same_side = [event for event in sample.retained if event.side == terminal.side]
        eligible_capsizes += 1
        first_terminal += terminal == same_side[0]
        first_to_capsize.append(sample.capsize_time_s - same_side[0].time_s)
    return {
        "declustered_crossings_per_trajectory": _summary(
            [float(len(sample.retained)) for sample in samples]
        ),
        "capsizing_trajectories": sum(sample.capsized for sample in samples),
        "capsizes_without_retained_crossing": capsize_without_crossing,
        "terminal_is_first_on_side": {
            "numerator": first_terminal,
            "denominator": eligible_capsizes,
            "fraction": first_terminal / eligible_capsizes if eligible_capsizes else None,
        },
        "first_retained_crossing_to_capsize_s": _summary(first_to_capsize),
    }


def _pooled_values(samples: list[CrossingSample]) -> NDArray[np.float64]:
    return np.asarray(
        [event.severity_u for sample in samples for event in sample.retained],
        dtype=np.float64,
    )


def _critical_ground_truth(samples: list[CrossingSample]) -> tuple[int, int, float]:
    crossings = sum(len(sample.retained) for sample in samples)
    terminal = sum(sample.capsized and bool(sample.retained) for sample in samples)
    return terminal, crossings, terminal / crossings


def _tail_row(
    samples: list[CrossingSample], *, quantile: float, strength: float
) -> dict[str, float | int | bool]:
    values = _pooled_values(samples)
    prior_mean = exponential_rate_mle(values, quantile=quantile)
    tail = estimate_exponential_tail(
        values,
        quantile=quantile,
        prior=GammaRatePrior.from_mean(prior_mean, strength),
    )
    counts = np.asarray([len(sample.retained) for sample in samples], dtype=np.float64)
    integrated = counts * tail.critical_probability
    return {
        "tail_quantile": quantile,
        "prior_strength": strength,
        "retained_crossings": int(np.sum(counts)),
        "tail_threshold_w": tail.threshold_w,
        "exponential_probability_critical_given_crossing": tail.critical_probability,
        "crossing_count_prediction": float(np.sum(integrated)),
        "absorbing_event_prediction": float(np.sum(-np.expm1(-integrated))),
        "threshold_clipped": tail.threshold_clipped,
    }


def _attribution(dataset: SimulationDataset) -> dict[str, object]:
    nominal = _samples(dataset)
    shorter = _samples(dataset, decorrelation_scale=0.5)
    longer = _samples(dataset, decorrelation_scale=1.5)
    forced = _samples(dataset, forced_correction=True)
    realized = int(np.sum(dataset.capsized))
    terminal, crossing_count, empirical_probability = _critical_ground_truth(nominal)
    candidates = [
        _tail_row(nominal, quantile=quantile, strength=strength)
        for quantile in U1_TAIL_QUANTILES
        for strength in U1_PRIOR_STRENGTHS
    ]
    selected = next(
        row
        for row in candidates
        if row["tail_quantile"] == R1_QUANTILE and row["prior_strength"] == R1_PRIOR_STRENGTH
    )
    forced_selected = _tail_row(forced, quantile=R1_QUANTILE, strength=R1_PRIOR_STRENGTH)
    shorter_selected = _tail_row(shorter, quantile=R1_QUANTILE, strength=R1_PRIOR_STRENGTH)
    longer_selected = _tail_row(longer, quantile=R1_QUANTILE, strength=R1_PRIOR_STRENGTH)
    nominal_counts = np.asarray([len(sample.retained) for sample in nominal], dtype=np.float64)
    empirical_integrated = nominal_counts * empirical_probability
    empirical_event_prediction = float(np.sum(-np.expm1(-empirical_integrated)))

    values = _pooled_values(nominal)
    threshold = float(np.quantile(values, R1_QUANTILE))
    threshold = min(threshold, float(np.nextafter(1.0, -np.inf)))
    exceedances = values[values > threshold] - threshold
    shape, _, scale = genpareto.fit(exceedances, floc=0.0)
    gpd_probability = float(
        len(exceedances) / len(values) * genpareto.sf(1.0 - threshold, shape, loc=0.0, scale=scale)
    )

    baseline = float(selected["crossing_count_prediction"])

    def row(cause: str, predicted: float) -> dict[str, object]:
        shift = predicted - baseline
        return {
            "cause": cause,
            "predicted_count": predicted,
            "direction_vs_baseline": "up" if shift > 0.0 else "down" if shift < 0.0 else "none",
            "magnitude_vs_baseline": abs(shift),
            "residual_vs_realized": predicted - realized,
        }

    table = [
        row("baseline: nominal-declustered exponential critical crossings", baseline),
        row(
            "event versus crossing accounting: absorbing transform",
            float(selected["absorbing_event_prediction"]),
        ),
        row("tail form: empirical terminal-crossing probability", empirical_event_prediction),
        row(
            "tail form: diagnostic GPD under absorbing accounting",
            float(np.sum(-np.expm1(-(nominal_counts * gpd_probability)))),
        ),
        row(
            "Eq. 15 motion-derived forced correction",
            float(forced_selected["crossing_count_prediction"]),
        ),
        row(
            "declustering: 0.5x decorrelation time",
            float(shorter_selected["crossing_count_prediction"]),
        ),
        row(
            "declustering: 1.5x decorrelation time",
            float(longer_selected["crossing_count_prediction"]),
        ),
    ]
    return {
        "campaign": "softening_stationary calibration",
        "realized_capsizes": realized,
        "empirical_terminal_critical_crossings": terminal,
        "retained_crossings": crossing_count,
        "empirical_probability_critical_given_crossing": empirical_probability,
        "exponential_candidates": candidates,
        "diagnostic_gpd": {
            "threshold_w": threshold,
            "shape": float(shape),
            "scale": float(scale),
            "probability_critical_given_crossing": gpd_probability,
        },
        "forced_correction": {
            "unforced_probability_critical_given_crossing": selected[
                "exponential_probability_critical_given_crossing"
            ],
            "forced_probability_critical_given_crossing": forced_selected[
                "exponential_probability_critical_given_crossing"
            ],
            "unforced_absolute_error_from_empirical": abs(
                float(selected["exponential_probability_critical_given_crossing"])
                - empirical_probability
            ),
            "forced_absolute_error_from_empirical": abs(
                float(forced_selected["exponential_probability_critical_given_crossing"])
                - empirical_probability
            ),
            "known_configuration_diagnostic": True,
            "moment_estimator": (
                "backward acceleration plus configured linear/quadratic damping and "
                "restoring; trailing half-period mean; quasi-static repeller response"
            ),
        },
        "attribution_table": table,
    }


def _gated_coverage(
    datasets: dict[str, SimulationDataset],
) -> dict[str, dict[str, object]]:
    prior_means: dict[str, float] = {}
    for family in FAMILIES:
        values = np.concatenate(
            [
                terminal_severities(dataset)
                for name, dataset in datasets.items()
                if campaign_family(name) == family
            ]
        )
        prior_means[family] = exponential_rate_mle(values, quantile=R1_QUANTILE)
    output: dict[str, dict[str, object]] = {}
    for name, dataset in datasets.items():
        scores = score_dataset(
            dataset,
            prior_mean=prior_means[campaign_family(name)],
            prior_strength=R1_PRIOR_STRENGTH,
            config=SplitTimeConfig(
                tail_quantile=R1_QUANTILE,
                trailing_window_s=None,
            ),
        )
        total_exposure = sum(item.exposure_end_s for item in scores)
        covered_exposure = sum(
            max(0.0, item.exposure_end_s - item.rate.emissions[0].time_s)
            for item in scores
            if item.rate.emissions
        )
        realized_covered = sum(
            item.capsized
            and item.t_capsize_s is not None
            and bool(item.rate.emissions)
            and item.t_capsize_s >= item.rate.emissions[0].time_s
            for item in scores
        )
        predicted = float(sum(item.rate.integrated_count for item in scores))
        output[name] = {
            "total_exposure_s": total_exposure,
            "valid_emission_exposure_s": covered_exposure,
            "valid_emission_exposure_fraction": (
                covered_exposure / total_exposure if total_exposure else 0.0
            ),
            "trajectories_with_valid_emission": sum(bool(item.rate.emissions) for item in scores),
            "predicted_count_on_covered_slices": predicted,
            "realized_capsizes_on_covered_slices": realized_covered,
            "covered_slice_count_bias": predicted - realized_covered,
        }
    return output


def run(data_root: Path, versioned_root: Path, output_root: Path) -> dict[str, object]:
    base_names = [
        f"{family}_{role}" for family in FAMILIES for role in ("stationary", "evaluation", "ramp")
    ] + ["softening_step"]
    datasets = {name: load_split(data_root, name, SeedBlock.CALIBRATION) for name in base_names}
    datasets["softening_step_v02"] = load_split(
        versioned_root, "softening_step_v02", SeedBlock.CALIBRATION
    )
    samples = {name: _samples(dataset) for name, dataset in datasets.items()}
    u1a_names = [f"{family}_{role}" for family in FAMILIES for role in ("stationary", "evaluation")]
    payload: dict[str, object] = {
        "experiment": "U1-r2 Phase A",
        "information_boundary": "calibration blocks only",
        "crossing_structure": {name: _crossing_structure(rows) for name, rows in samples.items()},
        "softening_overshoot_attribution": _attribution(datasets["softening_stationary"]),
        "r1_gated_coverage": _gated_coverage({name: datasets[name] for name in u1a_names}),
    }
    write_result(output_root, "u1_phase_a_u1r2", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--versioned-root", type=Path, default=Path("data/reference_v02"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    payload = run(arguments.data_root, arguments.versioned_root, arguments.output_root)
    attribution = payload["softening_overshoot_attribution"]
    print(
        f"Phase A realized={attribution['realized_capsizes']} "
        f"baseline={attribution['attribution_table'][0]['predicted_count']:.3f}"
    )


if __name__ == "__main__":
    main()
