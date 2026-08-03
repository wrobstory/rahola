"""Run the gate-open B1 gray-box experiment."""

from pathlib import Path

from rahola_lab.experiments.b1_graybox import run

if __name__ == "__main__":
    result = run(Path("data/reference"), Path("results"))
    print(f"survives all kills: {result['survives_all_kills']}")
    print(result["kills"])
