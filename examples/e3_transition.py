"""Reproduce E3 adaptive coverage through a sea-state transition."""

from pathlib import Path

from rahola_lab.experiments.e3 import run

if __name__ == "__main__":
    result = run(Path("data/reference"), Path("results"))
    print(
        f"E3 gamma={result['selected_gamma']:g}, recovery="
        f"{result['aci_recovery_time_s']} s, kill={result['kill_criterion']}"
    )
