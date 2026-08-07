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
| reserve2 | `[400000, 500000)` | Prototype #3 final holdout and standing automated-search holdout |

Campaign offsets in YAML allocate disjoint subranges within each block. Public split utilities take
a block name and reject both reserves; they do not accept arbitrary seed vectors. The guarded
`rahola-lab final-eval` path can construct reserve-2 seeds only. It refuses the spent Prototype #2
reserve, requires canonical paths and a clean tree, and atomically refuses a second reserve-2
invocation once access has begun. This repository-local mechanism supports the audit; the no-rerun
rule remains procedural rather than external access control.

The one-time access completed on commit `843b24a25437c5386208bc66ee0b79776ad207dc`. It materialized
18,000 trajectories (514 MiB allocated): 5,000 evaluation plus 1,000 ramp trajectories per family.
Reserve capsize fractions were 2.38%/48.9% for softening evaluation/ramp, 1.08%/30.7% for
parametric, and 1.90%/24.5% for biased. The timestamped attestation and headline result are checked
in under `results/`; public reserve guards remain in force.

Prototype #3 adds no campaigns. Its restart reference draws independent 200-second futures from
per-trajectory roll, roll-rate, current stiffness, and linear stiffness drift while preserving the
absolute phase of deterministic parametric modulation. Fresh rollout seeds are unique and outside
all campaign seed ranges. The restart discards the realized stochastic-forcing phase and is a
comparator, not a Bayes motion-history ceiling. Restarted stationary-ensemble variance must match
the corresponding full-run segment within the predeclared 15%, with capsize fraction within five
percentage points.
Chronos B2 appeared to survive its predeclared kill under the historical test-selected-threshold
procedure, so reserve-2 was materialized once on commit `5d4c6be`:
128 trajectories from each of the six D1-mirroring campaigns (768 total), matching the CPU probe's
frozen campaign limit. It contains 131 total capsizes, 129 observable under the historical scoring
rule, and 22 MiB on disk. The CNN, physics floor, and both
Chronos modes use the same holdout. This is an explicit reduction from Prototype #2's
18,000-trajectory final audit, not a new campaign. The spent reserve-2 result remains an immutable
historical audit and will not be repeated.

## Exact regeneration

From the repository root:

```bash
uv sync --all-packages --all-extras
uv run rahola-lab generate --all --out data/reference --chunk-size 256
```

The tracked `rahola_lab/campaigns/reference_checksums.json` anchors the checked-in reference
manifests. An intentional regeneration changes provenance fields and therefore requires reviewing
and updating those anchors before the replacement data can be treated as the new reference set.

The generator writes deterministic Parquet shards, chunk manifests and the top-level manifests used
above. On this machine the final per-campaign timings sum to 94.480 seconds; filesystem and first-run
JAX compilation differences may change wall time without changing content.

## v0.2 selective data addendum — August 2026

The v0.1 reference tree is immutable. A fixed spectral cutoff audit triggered three replacement
bandwidth campaigns, and the established-regime test required one longer step campaign. These
13,200 trajectories live under `data/reference_v02/`; loaders route only the affected v0.2
experiments to them.

| Campaign | Trajectories | Duration | Capsize fraction | Manifest SHA-256 prefix |
| --- | ---: | ---: | ---: | --- |
| `softening_bandwidth_gamma_7_v02` | 2,400 | 600 s | 42.375% | `f51c794fd6df` |
| `softening_bandwidth_gamma_15_v02` | 2,400 | 600 s | 43.542% | `286c24c894b7` |
| `softening_bandwidth_gamma_30_v02` | 2,400 | 600 s | 42.125% | `2d30254a5038` |
| `softening_step_v02` | 6,000 | 900 s | 62.900% | `218219a9643a` |

The step campaign transitions at 300 seconds and retains the historical train/calibration/test
counts of 2,000/1,000/3,000. Its corresponding capsize fractions are 62.45%, 62.00%, and 63.50%.
The full-history D5 endpoint must lie from 540 through 700 seconds: 60 natural periods of history
begin entirely after the step, and the 50-period outcome horizon remains complete.

`packages/rahola-lab/src/rahola_lab/campaigns/reference_checksums_v02.json` anchors these manifests.
The earlier ratio-4 cutoff stress data remain a local audit under
`data/reference_v02_ratio4_sensitivity/`; they are not reference data and no result loader selects
them. Regenerate the declared v0.2 campaigns into a fresh directory; these commands preserve the
`_v02` names and refuse to overwrite the three audit-selected bandwidth campaigns:

```bash
uv run python -c 'from pathlib import Path; from rahola_lab.campaigns import generate_selected_v02; generate_selected_v02(Path("results/forcing_invariance_final_v02.json"), Path("packages/rahola-lab/src/rahola_lab/campaigns/configs"), Path("/tmp/rahola-reference-v02-reproduction"))'
uv run python -c 'from pathlib import Path; from rahola_lab.campaigns import generate_campaign, versioned_definitions; configs=Path("packages/rahola-lab/src/rahola_lab/campaigns/configs"); generate_campaign(versioned_definitions(configs, ["softening_step"])[0], Path("/tmp/rahola-reference-v02-reproduction"))'
```

No reserve or reserve-2 data were opened or regenerated for v0.2.

## U1-r2 fresh TEST slices — predeclared 2026-08-04

U1-r2 uses 27,000 new trajectories in `data/u1r2/`. The offsets below were frozen before
materialization. A repository check compared each half-open interval with every TEST interval in
the v0.1 and v0.2 manifests and with every other proposed U1-r2 interval. All are pairwise disjoint
and lie inside the ordinary `[200000, 300000)` TEST block.

| Base campaign | Count | TEST offset | Absolute seed range |
| --- | ---: | ---: | --- |
| `softening_stationary` | 1,000 | 11,000 | `[211000, 212000)` |
| `parametric_stationary` | 1,000 | 12,000 | `[212000, 213000)` |
| `biased_stationary` | 1,000 | 13,000 | `[213000, 214000)` |
| `softening_ramp` | 1,000 | 14,000 | `[214000, 215000)` |
| `parametric_ramp` | 1,000 | 15,000 | `[215000, 216000)` |
| `biased_ramp` | 1,000 | 16,000 | `[216000, 217000)` |
| `softening_step` | 3,000 | 17,000 | `[217000, 220000)` |
| `softening_step_v02` | 3,000 | 44,000 | `[244000, 247000)` |
| `softening_evaluation` | 5,000 | 77,000 | `[277000, 282000)` |
| `parametric_evaluation` | 5,000 | 82,000 | `[282000, 287000)` |
| `biased_evaluation` | 5,000 | 87,000 | `[287000, 292000)` |

The new campaign names append `_u1r2` to the base names. The tracked
`reference_checksums_u1r2.json` anchors their manifests. The one-shot materialization produced:

| Campaign | Capsizes | Capsize fraction | Manifest SHA-256 prefix |
| --- | ---: | ---: | --- |
| `softening_stationary_u1r2` | 82 / 1,000 | 8.200% | `6b9d465727c2` |
| `parametric_stationary_u1r2` | 96 / 1,000 | 9.600% | `70e8a6595d12` |
| `biased_stationary_u1r2` | 144 / 1,000 | 14.400% | `3d13a74941eb` |
| `softening_ramp_u1r2` | 497 / 1,000 | 49.700% | `2a9f5a39ebc2` |
| `parametric_ramp_u1r2` | 295 / 1,000 | 29.500% | `3558011099a7` |
| `biased_ramp_u1r2` | 223 / 1,000 | 22.300% | `086a4125e702` |
| `softening_step_u1r2` | 1,169 / 3,000 | 38.967% | `e94b37c4eb35` |
| `softening_step_v02_u1r2` | 1,888 / 3,000 | 62.933% | `5890da592a0c` |
| `softening_evaluation_u1r2` | 111 / 5,000 | 2.220% | `abbd243aa4d7` |
| `parametric_evaluation_u1r2` | 46 / 5,000 | 0.920% | `2b2285dedced` |
| `biased_evaluation_u1r2` | 105 / 5,000 | 2.100% | `a28439eb18d7` |

Neither reserve block was opened or generated.

## H1 fresh TEST slices — predeclared 2026-08-04

H1 uses 7,900 new trajectories in `data/h1/`. Before H1, the union of ordinary TEST intervals
anchored by v0.1, v0.2, and U1-r2 occupies 56,000 of 100,000 seeds, leaving 44,000 untouched. The
six H1 slices consume offsets `[92000, 99900)`, leaving 36,100 ordinary TEST seeds untouched after
H1. A repository check compares these ranges with every existing manifest before generation and
refuses an overlap or overwrite.

The power floor uses the measured all-split capsize fractions in this file. Every campaign expects
at least 30 realized capsizes without using either reserve block.

| Base campaign | Count | Measured fraction | Expected capsizes | TEST offset | Absolute seed range |
| --- | ---: | ---: | ---: | ---: | --- |
| `softening_stationary` | 500 | 8.650% | 43.25 | 92,000 | `[292000, 292500)` |
| `parametric_stationary` | 500 | 8.450% | 42.25 | 92,500 | `[292500, 293000)` |
| `biased_stationary` | 500 | 13.300% | 66.50 | 93,000 | `[293000, 293500)` |
| `softening_evaluation` | 1,500 | 2.000% | 30.00 | 93,500 | `[293500, 295000)` |
| `parametric_evaluation` | 3,200 | 0.950% | 30.40 | 95,000 | `[295000, 298200)` |
| `biased_evaluation` | 1,700 | 1.867% | 31.74 | 98,200 | `[298200, 299900)` |

The generated names append `_h1`. The tracked `reference_checksums_h1.json` anchors their
manifests. The one-shot materialization produced:

| Campaign | Capsizes | Capsize fraction | Manifest SHA-256 prefix |
| --- | ---: | ---: | --- |
| `softening_stationary_h1` | 31 / 500 | 6.200% | `be6ab7be934b` |
| `parametric_stationary_h1` | 58 / 500 | 11.600% | `cd774914ab58` |
| `biased_stationary_h1` | 69 / 500 | 13.800% | `feb44abb190e` |
| `softening_evaluation_h1` | 34 / 1,500 | 2.267% | `c29dcdc8d4fe` |
| `parametric_evaluation_h1` | 24 / 3,200 | 0.750% | `424dcd432087` |
| `biased_evaluation_h1` | 32 / 1,700 | 1.882% | `8018de7e22c7` |

Five of six campaigns realized at least 30 capsizes. `parametric_evaluation_h1` realized 24; this
is a power shortfall in the frozen one-shot sample, not grounds to redraw it. Neither reserve block
was opened or generated.

## F1 fresh TEST slices — predeclared 2026-08-06

F1 freezes 9,400 ordinary TEST seeds before materialization. The repository overlap check compares
these half-open ranges with the v0.1, v0.2, U1-r2, and H1 top-level manifests and refuses an
overlap or overwrite. The slices occupy only previously untouched gaps and do not use either
reserve block. Expected counts use the measured fractions already recorded above; all four scored
campaigns meet the expected-event floor of 30.

| Base campaign | Count | Measured fraction | Expected capsizes | TEST offset | Absolute seed range |
| --- | ---: | ---: | ---: | ---: | --- |
| `softening_evaluation` | 1,500 | 2.000% | 30.00 | 1,000 | `[201000, 202500)` |
| `parametric_evaluation` | 3,200 | 0.950% | 30.40 | 6,000 | `[206000, 209200)` |
| `biased_evaluation` | 1,700 | 1.867% | 31.739 | 21,000 | `[221000, 222700)` |
| `softening_step_v02` | 3,000 | 62.900% | 1,887.00 | 38,000 | `[238000, 241000)` |

Generated campaign names append `_f1`. F1a uses only `softening_step_v02_f1`; F1b and any earned
F1c rotations use the three evaluation slices. The predeclaration commit was `ae986c2`. The
one-shot materialization then produced:

| Campaign | Capsizes | Capsize fraction | Manifest SHA-256 prefix |
| --- | ---: | ---: | --- |
| `softening_evaluation_f1` | 22 / 1,500 | 1.467% | `6afce71925f0` |
| `parametric_evaluation_f1` | 24 / 3,200 | 0.750% | `d5875316152f` |
| `biased_evaluation_f1` | 33 / 1,700 | 1.941% | `98d93c43f4ae` |
| `softening_step_v02_f1` | 1,908 / 3,000 | 63.600% | `86ee20871189` |

Softening and parametric evaluation realized fewer than 30 capsizes. They were neither redrawn nor
supplemented; the pooled F1b campaign retains 79 events, while any earned family-specific F1c row
must carry the corresponding power shortfall. Neither reserve block was opened or generated.
