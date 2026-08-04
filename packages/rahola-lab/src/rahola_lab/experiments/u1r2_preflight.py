"""Calibration-only preflight for the frozen U1-r2 one-shot runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rahola_lab.experiments.common import write_result
from rahola_lab.experiments.u1c import _point_with_bootstrap_or_unevaluable
from rahola_lab.experiments.u1c_r2 import calibration_operating_point
from rahola_lab.experiments.u1r2_common import (
    calibration_datasets,
    frozen_tail_priors,
    score_selected,
)


def run(data_root: Path, output_root: Path) -> dict[str, object]:
    datasets = calibration_datasets(data_root)
    priors = frozen_tail_priors(datasets)
    sample_name = "softening_stationary"
    sample_scores = score_selected(datasets[sample_name], sample_name, priors)
    if not sample_scores or any(not item.rate.emissions for item in sample_scores):
        raise AssertionError("prior-from-start must emit for every calibration trajectory")
    if any(item.rate.emissions[0].time_s != 0.0 for item in sample_scores):
        raise AssertionError("the first prior-from-start emission must occur at t=0")
    if not any(
        "prior_dominated" in emission.flags
        for item in sample_scores
        for emission in item.rate.emissions
    ):
        raise AssertionError("sparse calibration emissions must retain provenance flags")

    selected_point, calibration, horizon_s, decorrelation = calibration_operating_point(
        data_root, priors
    )
    bootstrap = _point_with_bootstrap_or_unevaluable(
        selected_point,
        calibration,
        horizon_s=horizon_s,
        decorrelation_s=decorrelation,
    )
    payload: dict[str, object] = {
        "experiment": "U1-r2 calibration-only preflight",
        "test_data_accessed": np.bool_(False),
        "prior_from_start": {
            "trajectory_count": len(sample_scores),
            "all_first_emissions_at_t0": np.bool_(True),
            "prior_dominated_flag_observed": np.bool_(True),
        },
        "u1c_calibration_operating_point": bootstrap,
        "numpy_boolean_serialization_probe": np.bool_(True),
    }
    path = write_result(output_root, "u1r2_preflight_u1r2", payload)
    serialized = json.loads(path.read_text())
    if serialized["numpy_boolean_serialization_probe"] is not True:
        raise AssertionError("NumPy boolean serialization preflight failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/reference"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    payload = run(arguments.data_root, arguments.output_root)
    print(
        "U1-r2 preflight calibration-only "
        f"threshold={payload['u1c_calibration_operating_point']['threshold']}"
    )


if __name__ == "__main__":
    main()
