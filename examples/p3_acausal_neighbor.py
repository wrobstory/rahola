"""Run the deliberately acausal neighbor-normalization appendix."""

from pathlib import Path

from rahola_lab.experiments.acausal_neighbor import run

if __name__ == "__main__":
    result = run(Path("data/reference"), Path("results"))
    print(f"held-out acausal neighbor AUC: {result['heldout_auc']:.4f}")
