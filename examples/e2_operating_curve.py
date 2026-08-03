"""Reproduce E2 alarm operating curves."""

from pathlib import Path

from rahola_lab.experiments.e2 import run

if __name__ == "__main__":
    result = run(Path("data/reference"), Path("results"))
    print("E2 headline:", result["at_90_percent_sensitivity"])
