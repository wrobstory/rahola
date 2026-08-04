# Rahola project summary

This document records what Rahola has built, what the experiments established, and which claims
were revised after the August 2026 methodology audit. `RESULTS.md` and the JSON files under
`results/` remain the numeric record.

## 1. Phase 0: synthetic roll-dynamics library

Rahola began as a deterministic, high-throughput 1-DOF nonlinear roll simulator for three failure
families: softening restoring, parametric excitation, and biased/asymmetric restoring. It includes
JONSWAP forcing, stationary/ramped/step protocols, nondimensional dynamics, inverse-FFT forcing
synthesis, a compiled RK4 hot path, capsize termination, causal window extraction, exclusion-band
labels, Parquet shards, manifests, and a command-line interface.

The physics validation suite covers the linear transfer-function limit, Mathieu instability,
Melnikov's necessary threshold, restoring equilibria and barriers, forcing spectra, determinism,
causal leakage, and throughput. A review found the original Melnikov test bracket made its headline
lower-bound assertion true by construction; the bracket and sub-threshold check were corrected.

## 2. Phase 0.5 and Prototype #1: campaigns, forecasting, and conformal alarms

The project then created versioned reference campaigns with disjoint train, calibration,
development-test, and guarded reserve seed blocks. The evaluation layer added trajectory-level
splits, forecast datasets, causal forecasters, conformalized quantile regression, alarm episodes,
Clopper–Pearson intervals, and reproducible experiment artifacts.

The main experiments were:

- E1: stationary marginal coverage for envelope, linear, and small recurrent forecasters.
- E2: operating cost for conformal alarms and a split-time critical-roll-rate physics baseline.
- E3: abrupt sea-state transition with scalar adaptive conformal inference.
- E3b: deterministic DtACI and sliding-score recalibration successors.
- E4: cross-sea-state stress testing and deployment-distribution recalibration.

The result was nuanced. Stationary split conformal behaved close to nominal. Scalar ACI with a
frozen score set could not meet a rolling coverage-and-alarm-cost requirement through an abrupt
shift. With outcomes delayed until the 60-second target is observable, neither DtACI nor sliding
recalibration repaired the rolling-coverage criterion, and both failed the alarm-cost kill. The
physics margin proved a strong zero-training baseline, while the audit-corrected E2 run showed that
calibration-selected controls do not necessarily retain the target sensitivity on test.

## 3. Prototype #2: motion-history detectors

Prototype #2 put five methods behind one causal scoring and episode harness:

- a compact temporal CNN;
- classical early-warning trends;
- a roll-power adaptation of Galeazzi's GLRT;
- the split-time danger margin;
- Story's phase-space neighbor-loss score.

D1 measured pooled within-distribution skill, D2 held out one failure family at a time, D3 varied
forcing bandwidth, D4 stratified events by critical wave groups, and D5 tested discrimination after
entry into a harsh established regime. A deliberately acausal appendix quantified the effect of
whole-record normalization.

The corrected record says the CNN has the best D1 calibration-fixed alarm cost while preserving
about 90% sensitivity, but it does not transfer a valid operating point across held-out families.
In D3 the CNN retains strong window-level ranking across bandwidths but misses the predeclared
broadband FPR improvement, so the stored verdict is inconclusive rather than collapse. In D5 all methods remain
near chance by AUC, and high sensitivity costs roughly 23–24 alarm episodes/hour. In D4, 75–88% of
nominal false episodes coincide with evaluator-defined critical wave groups when overlap begins at
the debounce-confirming window, but the high group prevalence and absence of a matched null make
this a descriptive overlap rather than evidence of encounter identification.

## 4. Prototype #3: restart comparisons and architecture probes

Rahola gained mid-run restart support so independent future-forcing ensembles can be launched from
a recorded state. The restart-comparison experiment compares, on identical stratified windows:

- C1, exact current state plus independent future-forcing restarts;
- C2, a particle-filtered state plus independent future-forcing restarts;
- the frozen CNN;
- an engineered-feature XGBoost baseline.

C1 and C2 are restart-comparison scores, not Bayes-optimal motion-only ceilings: they change the
future forcing realization and do not condition on the realized forcing history. The audit also
introduced post-stratification weights for the probability-calibration analysis, so Brier score and
ECE reconstruct the source-window population rather than the capped-equal stratified sample.

On the corrected 16,000-window run, capped-equal sample AUC is 0.851 for C1, 0.485 for C2, 0.627 for
the frozen CNN, and 0.762 for XGBoost. A clock-only protocol-quartile comparator reaches 0.656,
above the CNN and C2, so the table is materially confounded by absolute protocol time. The values
do not define an information ordering because C1/C2 discard forcing-history information. The run
took 31.6 minutes with 200 restarts per window,
comfortably inside the two-hour budget.

The restart-equivalence regression currently covers stationary softening only. Sampling uses
capped-equal allocation because some label/time strata cannot fill an equal quota, and the reported
AUC bootstrap conditions on the realized sample and rollout draws rather than propagating either
the unequal-probability sampling design or rollout Monte Carlo uncertainty. These are explicit
limits on the restart evidence.

The architecture branch added a small gray-box temporal encoder with physical latent supervision
and a split-time-inspired hazard head, plus a pinned Chronos-T5-tiny transfer probe in frozen and
one-epoch fine-tuned modes. Under the corrected protocol the gray-box model fails transfer. Its
verbatim matched-sensitivity parity gate is unevaluable without selecting on test, and its
unconditional final-third stiffness gate is unevaluable because early capsizes have no final-third
outcome; a survivor-conditioned trajectory-level diagnostic still exceeds the stated limit. The
Chronos probe
produced one apparent development survivor under the original protocol and triggered the
one-time reserve-2 run, but the audit found that its threshold was selected on reserve labels. The
published reserve-2 number is therefore descriptive historical evidence, not an operational
validation. With calibration-only thresholds evaluated at the same dense ten-second cadence as
test, corrected B2 produces one qualifying development rotation and survives its probe kill. Its
corrected D5 orientation audit remains below the leakage trigger. The reserve is spent and will not
be rerun, so this creates no valid final-holdout claim.

## 5. August 2026 audit and hardening

The audit corrected both scientific evaluation and implementation robustness:

- operating controls and thresholds are selected on calibration only, frozen, and evaluated at a
  single test point;
- every supervised label requires a fully observed forecast horizon, giving both outcomes common
  protocol-time support; operational inference retains every causal pre-capsize score endpoint;
- detector debounce uses that uninterrupted score stream and timestamps an episode at the window
  that confirms the required run, while event risk sets and exposure for both outcomes stop at the
  last horizon-complete endpoint;
- online conformal feedback is delayed until each forecast target becomes observable;
- forecast alarm exposure ends at the last horizon-complete scored endpoint, eliminating an
  unscored 60-second tail from E2/E3/E3b rate denominators;
- sensitivity excludes capsizes for which debounce was impossible or no scored endpoint entered
  the warning horizon;
- always-on endpoints are finite, empty score streams are supported, non-finite AUC inputs are
  rejected, and JSON output contains no NaN or infinity literals;
- the restart sampler uses capped-equal stratum allocation, explicit population weights, and
  reports weighted Brier/ECE;
- campaign loading verifies the tracked reference anchor, chunk manifests, shard hashes, row
  counts, and containment of every manifest path;
- simulation configuration rejects non-finite values and guarantees an exact regular output grid;
- reserve execution is pinned to canonical repository paths, performs safe preflight before
  claiming access, and requires a committed survivor. The current runner directory-syncs the
  access claim and holds a cross-process result-graph lock from completed-result publication
  through recursive verification and atomic terminal attestation; the immutable historical
  attestations predate result-digest binding;
- development artifacts carry source and reference-anchor provenance, bind their own serialized
  content, record upstream artifact digests, and reject stale or mutated dependencies;
- Chronos imports are lazy in both the detector package and CLI, preventing a duplicate OpenMP
  runtime merely from importing ordinary campaign tooling.

The reserve guards are strong repository-local procedural controls, not an external security
boundary. The simulator still accepts arbitrary public seeds, and anyone with filesystem control
could bypass local policy. Documentation now states that limitation directly.

## 6. Current scientific position

Rahola supports four durable conclusions.

First, several motion-derived scores rank risk, but protocol time can explain or exceed that ranking
in time-varying campaigns, and a useful score does not automatically yield a threshold that
transfers across dynamics or sea states. Second, in the tested
softening-step regime, the tested motion-only scores do not discriminate reliably within the danger
regime; the experiment does not identify whether encounter timing, omitted state, or model
misspecification causes that failure.
Third, outcome-based false episodes frequently overlap non-capsizing group encounters in
this campaign, but that overlap alone does not establish useful encounter detection. Fourth, the
strongest next information gain is likely to come
from exogenous encounter sensing or action-aware decision policies rather than unconstrained
motion-only architecture search.

All conclusions remain limited by the synthetic 1-DOF setting. Rahola is a method-development
platform, not evidence of shipboard readiness; stability variation in waves, coupled heave/pitch,
stern-quartering behavior, sensor error, and human operational response require higher-fidelity and
real-world validation.

## 7. Audit trail and reproducibility

The repository history contains the phase implementations, campaign and harness additions,
Prototype #1 follow-on work, Prototype #2 detectors and reserve evaluation, and Prototype #3
restart/architecture/reserve-2 work as separate commits. Corrected development artifacts are
regenerated by the commands in `RESULTS.md`. The original reserve and reserve-2 results and their
attestations are intentionally immutable and are labeled historical where their old method differs
from the corrected harness.
