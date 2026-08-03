"""Render generated and target JONSWAP spectra."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from rahola.config import SeaState
from rahola.spectrum import jonswap_spectrum, synthesize_jonswap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("spectral-check.png"))
    args = parser.parse_args()
    state = SeaState(hs_m=4.0, tp_s=10.0, gamma=3.3)
    sea = synthesize_jonswap(state, 4096.0, 0.25, seed=814)
    frequency, estimate = welch(sea.elevation_m, fs=4.0, nperseg=4096)
    target = 2 * np.pi * jonswap_spectrum(2 * np.pi * frequency, state)
    plt.loglog(frequency[1:], estimate[1:], label="FFT realization")
    plt.loglog(frequency[1:], target[1:], "--", label="JONSWAP target")
    plt.xlabel("frequency [Hz]")
    plt.ylabel(r"$S_\eta(f)$ [$m^2$/Hz]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
