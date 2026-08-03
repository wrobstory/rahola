"""Reproduce Prototype #2 D4 wave-group stratification."""

from pathlib import Path

from rahola_lab.experiments.d4 import run

if __name__ == "__main__":
    run(Path("data/reference"), Path("results"))
