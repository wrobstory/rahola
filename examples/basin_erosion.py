"""Flagship Family-1 basin-erosion severity sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rahola.validation import harmonic_capsize_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("basin-erosion.png"))
    args = parser.parse_args()
    severity = np.linspace(0.02, 0.30, 29)
    fractions = [harmonic_capsize_fraction(value, 1.0, 0.02, phases=64) for value in severity]
    plt.plot(severity, fractions, "o-")
    plt.xlabel("harmonic forcing severity")
    plt.ylabel("capsize fraction over forcing phase")
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
