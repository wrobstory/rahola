"""Reproduce E3b DtACI and sliding-recalibration comparison."""

from pathlib import Path

from rahola_lab.experiments.e3b import run

if __name__ == "__main__":
    result = run(Path("data/reference"), Path("results"))
    print(
        "E3b pass flags:",
        {name: result[name]["passed"] for name in ("dtaci", "sliding_aci")},
    )
