# Rahola reference-data contract

This file freezes the data choices shared by Rahola prototypes. Generated data live under
`data/reference/` and are intentionally ignored by Git; the manifests make regeneration auditable.

## Frozen constants

| Choice | Frozen value | Rationale |
|---|---:|---|
| Forecast horizons | 30 s, 60 s | Operationally actionable and several roll cycles long |
| Forecast history | 120 s | Captures slow envelope modulation without future motion |
| Alarm angle | 0.60 × escape angle | Leaves escape margin while rejecting ordinary moderate roll |
| Prototype #2 EWS window | 60 roll periods | Long baseline for slow statistics |
| Prototype #2 EWS horizon | 50 roll periods | Planned bifurcation-warning horizon |
| Near-miss exclusion | 5 roll periods | Removes ambiguous negatives near an event |
| ACI γ grid | 0.001, 0.005, 0.01, 0.02, 0.05 | Slow through aggressive adaptation, declared before E3 |
| ACI episode explosion | >2 episodes/h **and** >4× fixed CQR | Absolute and relative operational guardrails |

Every campaign is 600 seconds at 2 Hz with a 4-second natural period. This accommodates the
110-period EWS history-plus-horizon and its transient. The biased-family campaigns start at the
stable static equilibrium; otherwise the imposed bias creates an artificial zero-state transient.

## Frozen campaigns

Rates are measured across every declared split, not pilot intentions. SHA values are the first 12
hex digits of the top-level manifest's SHA-256.

| Campaign | Role | Trajectories | Duration | Capsize fraction | Manifest SHA |
|---|---|---:|---:|---:|---|
| `softening_stationary` | stationary training | 4,000 | 600 s | 8.650% | `bfc73be2c9ec` |
| `parametric_stationary` | stationary training | 4,000 | 600 s | 8.450% | `cbe939395daf` |
| `biased_stationary` | stationary training | 4,000 | 600 s | 13.300% | `6540e85019db` |
| `softening_ramp` | Prototype #2 ramp | 3,500 | 600 s | 47.943% | `da75672f2c89` |
| `parametric_ramp` | Prototype #2 ramp | 3,500 | 600 s | 28.200% | `5ac54c72ee14` |
| `biased_ramp` | Prototype #2 ramp | 3,500 | 600 s | 24.200% | `55820347af4e` |
| `softening_step` | sea-state transition | 6,000 | 600 s | 38.417% | `7402eb0a3547` |
| `softening_evaluation` | rare-event evaluation | 6,000 | 600 s | 2.000% | `b02bd4debcbe` |
| `parametric_evaluation` | rare-event evaluation | 6,000 | 600 s | 0.950% | `b0d022af64fa` |
| `biased_evaluation` | rare-event evaluation | 6,000 | 600 s | 1.867% | `c416ddc02233` |
| `softening_bandwidth_gamma_1` | D3 bandwidth sweep | 2,400 | 600 s | 41.708% | `f973c5814438` |
| `softening_bandwidth_gamma_3_3` | D3 bandwidth sweep | 2,400 | 600 s | 37.958% | `9e36a8b217e0` |
| `softening_bandwidth_gamma_7` | D3 bandwidth sweep | 2,400 | 600 s | 40.958% | `76ce921118cc` |
| `softening_bandwidth_gamma_15` | D3 bandwidth sweep | 2,400 | 600 s | 42.417% | `f48de367e516` |
| `softening_bandwidth_gamma_30` | D3 bandwidth sweep | 2,400 | 600 s | 43.208% | `29aeb7d5b6db` |

Total: 58,500 trajectories and 1.629 GiB of file content (1.644 GiB allocated on disk). The final
full generation took 94.480 seconds. Prototype #2 added 16,500 trajectories and 0.459 GiB, versus
its limits of 30 minutes and 15 GiB additional.

Split-specific counts and rates remain in each manifest. Stationary campaigns contain
2,000 train / 1,000 calibration / 1,000 test trajectories. Evaluation campaigns contain
1,000 calibration / 5,000 test trajectories. The step campaign contains 2,000 train / 1,000
calibration / 3,000 test trajectories. Ramps contain 2,000 train / 500 calibration / 1,000 test
trajectories. Each bandwidth campaign contains 1,000 train / 400 calibration / 1,000 test.

The bandwidth sweep changes JONSWAP peak enhancement while retaining one softening-family ramp.
Values γ=15 and 30 are controlled narrow-band stress cases, not oceanographically typical seas.
Terminal stiffness was tuned on train-block pilots, before calibration/test scoring, to keep total
capsize fraction inside 20–60%: ramp ends were −0.047, −0.048, −0.050, −0.052, and −0.054 for γ=1,
3.3, 7, 15, and 30. The resulting test fractions were 40.5%, 38.6%, 38.8%, 40.6%, and 44.6%. This
fixed-outcome-band rule prevents severity from becoming the bandwidth axis.

## Seed allocation

| Block | Half-open range | Permitted use here |
|---|---|---|
| train | `[0, 100000)` | Campaign pilots, model fitting, fixed model grid |
| calibration | `[100000, 200000)` | CQR scores and ACI γ selection |
| test | `[200000, 300000)` | Final experiment scoring |
| reserve | `[300000, 400000)` | One guarded Prototype #2 final evaluation only |

Campaign offsets in YAML allocate disjoint subranges within each block. Public split utilities take
a block name and reject `reserve`; they do not accept arbitrary seed vectors. Only the guarded
`rahola-lab final-eval` path can construct reserve seeds. It refuses a dirty tree and refuses any
second invocation once access has begun.

The one-time access completed on commit `843b24a25437c5386208bc66ee0b79776ad207dc`. It materialized
18,000 trajectories (514 MiB allocated): 5,000 evaluation plus 1,000 ramp trajectories per family.
Reserve capsize fractions were 2.38%/48.9% for softening evaluation/ramp, 1.08%/30.7% for
parametric, and 1.90%/24.5% for biased. The timestamped attestation and headline result are checked
in under `results/`; public reserve guards remain in force.

## Exact regeneration

From the repository root:

```bash
uv sync --all-packages
uv run rahola-lab generate --all --out data/reference --chunk-size 256
```

The generator writes deterministic Parquet shards, chunk manifests and the top-level manifests used
above. On this machine the final per-campaign timings sum to 94.480 seconds; filesystem and first-run
JAX compilation differences may change wall time without changing content.
