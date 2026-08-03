"""Reproduce E4 cross-sea-state calibration stress test."""

from pathlib import Path

from rahola_lab.experiments.e4 import run

if __name__ == "__main__":
    result = run(Path("data/reference"), Path("results"))
    print(
        "E4 raw/CQR/ACI coverage: "
        f"{result['raw_lstm_snapshot_coverage']:.3f}/"
        f"{result['split_cqr_snapshot_coverage']:.3f}/"
        f"{result['aci_dense_post_coverage']:.3f}"
    )
