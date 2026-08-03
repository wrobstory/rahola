"""Reproduce E1 stationary coverage validation."""

from pathlib import Path

from rahola_lab.experiments.e1 import run

if __name__ == "__main__":
    result = run(Path("data/reference"), Path("results"))
    print(
        "E1 mean/max absolute coverage delta: "
        f"{result['mean_absolute_coverage_delta_pp']:.2f}/"
        f"{result['max_absolute_coverage_delta_pp']:.2f} pp"
    )
