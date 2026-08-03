# Prototype #1 results — conformal and physics alarm layers

The JSON files under `results/` are the numeric record; PNG files are the figures. Unless stated
otherwise, brackets are two-sided 95% Clopper–Pearson intervals computed by the shared evaluation
harness. Sensitivity trials are observable capsize events. False-episode intervals treat scorable
windows as alarm-opening opportunities, then rescale the probability interval by opportunities per
exposure hour. Debounce and refractory logic decluster episodes, but residual serial dependence
remains; full decorrelation-time intervals are deferred to Prototype #2.

Per-time E3/E3b coverage is computed over the surviving, non-capsized population. This conditioning
and the dependence among dense windows are not covered by marginal conformal guarantees, so their
narrow window-level intervals should not be read as trajectory-independent uncertainty.

## E1 — stationary coverage

Across 72 cells (three families, two horizons, three forecasters and four alpha values), mean
absolute coverage error was a descriptive **0.75 percentage points**. The worst cell was the linear
forecaster on the 60-second softening target at nominal 80%: coverage was **76.62% [73.83, 79.25]**,
a **−3.38-point delta [−6.17, −0.75]**. Exact intervals for every cell are in the JSON and the figure
retains binomial acceptance bands. This validates the implementation broadly but does not turn
marginal coverage into conditional or survivor-conditioned coverage.

![E1 stationary coverage](results/e1_coverage.png)

## E2 — alarm operating cost and the physics floor

At the lowest-FPR sampled point at or above 90% sensitivity on the pooled 15,000-trajectory
rare-event test set (60-second horizon):

| Method | Control | Sensitivity | False episodes / exposure h | Median lead |
|---|---:|---:|---:|---:|
| Envelope persistence | alpha=0.010 | 91.83% [87.24, 95.17] | 6.396 [6.286, 6.507] | 192.4 s |
| Linear quantile | alpha=0.005 | 92.79% [88.38, 95.91] | 6.459 [6.349, 6.571] | 199.5 s |
| 4.6k-parameter JAX LSTM | alpha=0.020 | 92.79% [88.38, 95.91] | **6.097 [5.990, 6.205]** | 199.5 s |
| Split-time danger margin | threshold=−1.75 rad/s | 96.15% [92.56, 98.33] | 7.444 [7.326, 7.564] | 278.8 s |

The danger-margin baseline fits a three-range local piecewise-linear restoring model separately on
each side: translate to the stable equilibrium, match its slope and the first smooth-restoring peak,
then force the repeller branch to vanish at the configured escape angle. The resulting repeller
slope enters Belenky et al.'s damped critical-rate formula (Eq. 13). At arbitrary scoring instants we
use the nearer intermediate threshold and extrapolate the separatrix line; the alarm score is
measured outward rate minus critical rate. Eq. 15's particular-solution correction is implemented,
but it is zero in E2 because wave-field inputs are prohibited.

This quantity was developed as an offline rare-event extrapolation metric. Our documented search
found no prior use as a continuously evaluated online alarm score; that novelty claim is therefore
provisional, not proof of absence. Here it is a useful physics floor, but at the ≥90% operating point
it costs 1.35 more false episodes/hour than the conformal LSTM.

Exposure begins when a complete 120-second history makes the trajectory scorable. Every episode
overlapping a pre-capsize horizon is event-associated, so repeated alarms inside the horizon are not
charged as false. Lead time begins at the earliest associated episode and can exceed 60 seconds.

![E2 operating curve](results/e2_operating_curve.png)

## E3 — scalar ACI through the sea-state transition

Nominal coverage was 90%. Correctly restricting “pre-step” to horizon-complete windows ending no
later than 240 seconds, fixed CQR covered **89.37% [89.06, 89.68]**. Windows ending at 250–290
seconds already contain post-step targets and covered only **17.68% [17.07, 18.30]**. Fixed CQR's
pooled post-step coverage was **0.74% [0.68, 0.81]**.

The calibration-selected scalar ACI setting remained gamma=0.05. It covered **94.18%
[93.94, 94.41]** on horizon-complete pre-step windows and **86.97% [86.71, 87.24]** post-step, but
never held the trailing-60-second curve inside ±3 points. It produced **5.599 [5.355, 5.851]** false
episodes/hour versus **0 [0, 0.011]** for fixed CQR.

The frozen kill criterion therefore still triggers. This is a rolling/conditional demand;
Gibbs–Candès Proposition 4.1 guarantees long-run average error frequency, not rolling-window
coverage. The figure's large-gamma sawtooth is the scalar ACI limit cycle: a frozen score set drives
the working alpha into exterior levels to inflate bounds, which also inflates alarm episodes. The
precise finding is: **scalar ACI over a frozen score set cannot track this abrupt shift at the frozen
operational alarm cost**. It is not an indictment of every online adapter.

![E3 transition](results/e3_transition.png)

## E3b — DtACI and sliding score recalibration

E3b retained the same test bed and kill thresholds. Deterministic DtACI uses Gibbs and Candès'
published eight-expert gamma grid and target interval 500. Sliding ACI was selected only on the held
out calibration half from gamma `{0.001, 0.005, 0.01, 0.02, 0.05}` and score windows `{25, 50, 100}`;
the selected pair was gamma=0.05, window=100.

| Adapter | Post-step coverage | Recovery to ±3 points | False episodes / h | Verdict |
|---|---:|---:|---:|---|
| DtACI | 75.27% [74.93, 75.61] | not attained | 5.322 [5.084, 5.569] | kill |
| Sliding-score ACI | 91.96% [91.75, 92.18] | 50 s | 5.400 [5.160, 5.648] | kill on episode cost |

Recent-score recalibration repairs rolling coverage, so the broader architecture survives
statistically. It still fails operationally: neither successor stays under the predeclared alarm-cost
threshold. Sliding recalibration is explicitly nonexchangeable and does not inherit ordinary
split-conformal coverage; its motivation is recent weighting under distribution drift, not a claim
of exact exchangeable validity.

![E3b adapters](results/e3b_adapters.png)

## E4 — cross-sea-state stress test

Training at Hs=4 m and deploying after the Hs=5 m step dropped raw nominal-95% LSTM snapshot
coverage to **69.48% [67.75, 71.17]**, a **25.52-point shortfall [23.83, 27.25]**. Deployment-
distribution calibration restored split-CQR snapshot coverage to **94.34% [93.43, 95.17]**, a
**−0.66-point delta [−1.57, 0.17]**. On dense post-step windows, fixed CQR covered **93.64%
[93.45, 93.83]** and ACI covered **96.76% [96.62, 96.90]**.

![E4 stress test](results/e4_stress_test.png)

# Prototype #2 results — deep motion-history warning

Prototype #2 uses 60-period, causally normalized roll/roll-rate windows, a 50-period event horizon,
the same three-window debounce/refractory policy for every method, and score-specific decorrelation
times estimated from calibration autocorrelation envelopes. Brackets remain 95% Clopper–Pearson
intervals. The numeric JSON is authoritative where a compact table omits secondary methods.

The following criteria were frozen in `rahola_lab.constants` before development-test scoring:

1. Stop CNN iteration after the two-model grid if it cannot beat classical EWS at matched ≥90%
   sensitivity in D1.
2. Apply the D3 bandwidth interpretation verbatim, whichever branch the data select.
3. Call the CNN an overall win only if it beats the danger-margin physics floor in D1 and is at
   least 10% lower-FPR than B1 in every held-out-family D2 rotation.

## D1 — within-distribution skill

The selected 4,021-parameter CNN uses 16/32 channels, kernel 7, and the family-classification head
at weight 0.10. Calibration selected AC1 trend over a 50%-window subwindow and neighbor radius
0.35. The physics-regression head was implemented but its predeclared weight stayed zero in both
grid entries, so it was not used.

| Detector | Sensitivity | False episodes / exposure h | Lead q10 / median / q90 |
|---|---:|---:|---:|
| Temporal CNN | 90.46% [88.63, 92.09] | **6.288 [6.173, 6.404]** | 279.5 / 328.3 / 357.5 s |
| Classical EWS (AC1 trend) | 91.67% [89.93, 93.19] | 9.258 [9.119, 9.398] | 199.8 / 358.0 / 360.2 s |
| Galeazzi roll-power GLRT | 90.21% [88.35, 91.85] | 8.305 [8.173, 8.438] | 177.6 / 310.1 / 358.6 s |
| Split-time danger margin | 95.27% [93.89, 96.42] | 9.128 [8.990, 9.268] | 29.4 / 335.6 / 359.8 s |
| Story (2009) neighbor loss | 97.85% [96.85, 98.61] | 9.320 [9.181, 9.461] | 291.5 / 358.8 / 360.2 s |

The first kill criterion does not fire: the frozen CNN grid beats B1 by 32.1% FPR/h. It also beats
the physics floor by 31.1% FPR/h, although its sensitivity is lower at the nearest available curve
point. This is a within-distribution result, not yet an overall win.

![D1 detector operating curves](results/d1_operating_curves.png)

## D2 — family generalization

| Held-out family | CNN sensitivity | CNN FPR/h | B1 sensitivity | B1 FPR/h | CNN ≥10% better? |
|---|---:|---:|---:|---:|---:|
| Softening | 93.03% [90.56, 95.02] | 9.033 [8.795, 9.275] | 90.09% [87.27, 92.47] | 8.910 [8.674, 9.150] | No |
| Parametric | 90.51% [86.72, 93.50] | 6.604 [6.401, 6.811] | 92.41% [88.91, 95.07] | 5.642 [5.455, 5.835] | No |
| Biased | 91.42% [87.68, 94.32] | 8.702 [8.470, 8.940] | 90.76% [86.92, 93.77] | 9.493 [9.250, 9.741] | No (8.3%) |

The CNN fails all three predeclared materiality tests. It is worse than B1 on softening and
parametric transfer, and its 8.3% biased-family improvement misses the 10% threshold. Criterion 3
therefore fires: despite the D1 win, the CNN is not an overall winner because its learned skill does
not transfer materially beyond the trained equations. The full five-detector rotation table is in
`results/d2_family_generalization.json`.

## D3 — skill versus forcing bandwidth

Severity was separately tuned to a 20–60% capsize band at every γ. JONSWAP γ=1 is the broadband
end; γ=15 and 30 are deliberately non-oceanographic narrow-band controls.

| γ | CNN sensitivity | CNN FPR/h | CNN AUC | B1 FPR/h | B1 AUC |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 90.62% [87.35, 93.27] | 5.250 [4.813, 5.716] | 0.774 | 5.961 [5.496, 6.456] | 0.322 |
| 3.3 | 91.97% [88.79, 94.48] | 5.820 [5.360, 6.309] | 0.749 | 6.151 [5.678, 6.653] | 0.347 |
| 7.0 | 90.21% [86.81, 92.98] | 5.480 [5.034, 5.955] | 0.754 | 6.132 [5.659, 6.632] | 0.355 |
| 15.0 | 92.36% [89.34, 94.75] | 5.521 [5.072, 5.997] | 0.764 | 5.951 [5.486, 6.445] | 0.365 |
| 30.0 | 92.15% [89.25, 94.47] | 5.140 [4.708, 5.601] | 0.763 | 5.551 [5.102, 6.029] | 0.371 |

The applied predeclared verdict is: **“If skill survives at gamma=1.0 materially above the B1
floor, motion history contains precursor information beyond critical slowing down.”** At γ=1 the
CNN's AUC exceeds B1 by 0.452 and its matched FPR is 11.9% lower. Both the per-γ and pooled
cross-γ CNNs retain similar skill, so bandwidth collapse is falsified in this synthetic system.

![D3 detector skill by bandwidth](results/d3_bandwidth_skill.png)

## D4 — wave-group stratification

A critical group is an evaluator-only run with instantaneous wave-height proxy
`2 × |Hilbert(elevation)| ≥ 0.75 Hs` for at least 1.5 peak periods. A capsize is group-preceded when
such a run overlaps its 200-second warning horizon. This labels 1,125/1,164 observable capsizes,
96.65% [95.45, 97.61%]. No forcing value enters a detector.

| Detector | Sensitivity: group (median lead) | Sensitivity: no group (median lead) | False episodes coincident with noncapsizing group |
|---|---:|---:|---:|
| CNN | 90.22% [88.34, 91.90] (328.0 s) | 97.44% [86.52, 99.94] (338.8 s) | 86.02% [85.36, 86.65] |
| Classical EWS | 91.82% [90.06, 93.36] (358.0 s) | 87.18% [72.57, 95.70] (358.4 s) | 90.58% [90.13, 91.02] |
| Galeazzi GLRT | 89.96% [88.05, 91.65] (309.8 s) | 97.44% [86.52, 99.94] (329.2 s) | 83.41% [82.80, 84.00] |
| Danger margin | 95.11% [93.68, 96.30] (330.4 s) | 100% [90.97, 100] (347.3 s) | 96.39% [96.09, 96.67] |
| Neighbor loss | 98.04% [97.05, 98.77] (358.8 s) | 92.31% [79.13, 98.38] (359.1 s) | 99.01% [98.85, 99.15] |

The no-group stratum has only 39 capsizes, hence its wide intervals. More importantly, 83–99% of
episodes charged as false coincide with a noncapsizing critical-group encounter. This supports the
E3b concern: outcome-based FPR often charges a detector for identifying a real hazardous encounter
that happens not to end in capsize.

## D5 — within-regime discrimination

| Detector | Post-step window AUC | Maximum episode sensitivity | Fallback FPR/h |
|---|---:|---:|---:|
| CNN | 0.463 | 87.11% [85.05, 88.97] | 9.266 [8.851, 9.694] |
| Classical EWS | 0.500 | 87.11% [85.05, 88.97] | 9.266 [8.851, 9.694] |
| Galeazzi GLRT | **0.529** | 87.11% [85.05, 88.97] | 9.266 [8.851, 9.694] |
| Danger margin | 0.499 | 87.11% [85.05, 88.97] | 9.266 [8.851, 9.694] |
| Neighbor loss | 0.518 | 87.11% [85.05, 88.97] | 9.266 [8.851, 9.694] |

No method discriminates imminent capsize well inside the harsh post-step regime. The ≥90% target is
unattainable for every curve because some post-transition capsizes occur before three consecutive
10-second flags can satisfy the common debounce. The identical row is therefore an always-on
fallback, not a meaningful tie; the near-chance AUCs are the substantive D5 result.

## Final reserve evaluation

The reserve has not yet been accessed in this frozen-development record. The guarded final command
will replace this paragraph with the single reserve headline table and commit attestation; it will
not be rerun after any outcome.

## What the thesis rematch showed

The 2009 neighbor-loss idea is sensitive but indiscriminate here. Its nearest ≥90% D1 point catches
97.85% of capsizes, yet costs 9.320 false episodes/h—the highest of all five methods, versus 6.288
for the CNN and 9.128 for the physics floor. It also has only 0.518 within-regime AUC and 99.01% of
its nominally false D1 episodes overlap noncapsizing critical wave groups. The core observation—new
phase-space regions precede extreme motion—survives, but neighbor count alone is not a competitive
alarm policy under a common operating-cost harness.

## Prototype #2 implementation judgments and deviations

- Story's Chapter 3 normalized each complete roll/pitch record by its mean and standard deviation,
  which uses future data. Rahola instead applies strict prior-only normalization to roll/rate. It
  replaces pitch with roll rate because the simulator is 1-DOF (Story 2009, Sec. 3.2.2, pp. 48–52,
  Figs. 38–41; Sec. 4.2.2, pp. 64–66).
- Chapter 3 set the binary warning at fewer than 50 neighbors and accumulated one flag per low-count
  sample. Rahola preserves score `−neighbor_count` and includes threshold −50 in the curve, but
  resolves persistence through the common three-window episode policy. Calibration selected radius
  0.35; one natural period is excluded so adjacent samples cannot count themselves.
- Chapter 4 searched the full history only on newly encountered roll regions, then reused the prior
  count. Rahola searches the exact trailing 60-period normalized phase plane at every score time;
  this preserves the novelty meaning but replaces the thesis's computational cache and pooled
  cross-run historical database. The thesis itself reports abandoning its real-time cumulative flag
  after corrupted output (Story 2009, pp. 65–67, Fig. 51).
- Galeazzi's W2-GLRT uses the transformed roll/pitch signal `d=roll²×pitch`. With no pitch and no
  permitted wave input, B2 applies the published fixed-shape scale-change likelihood ratio to
  natural-frequency-band roll, retaining the four-roll-period detection segment. It is a documented
  roll-power adaptation, not a literal reproduction of the multichannel statistic.
- B3 takes the dimensional roll/rate endpoint from the same causal window rather than its normalized
  representation; the closed-form danger margin requires physical units and remains unchanged.
- D4's 0.75 Hs/1.5 Tp Hilbert-envelope group is a stated run-length proxy inspired by critical wave
  groups, not Anastopoulos and Spyrou's full Markov-chain construction. Hilbert edge effects and the
  broad definition make groups common.
- Decorrelating alarm episodes merges gaps no longer than each calibration score's autocorrelation-
  envelope crossing time, following the Belenky-school convention. Clopper–Pearson FPR intervals
  still use scorable windows as the binomial opportunity convention; declustering reduces repeated
  episodes but does not make dense windows independent.
- Development test blocks were rerun while correcting the explicit thesis curve point and D4 risk-
  set denominator. Hyperparameters and thresholds came only from train/calibration blocks, but the
  literal test-touched-once process rule is not claimed. The untouched reserve is the one-time audit.

## Relation to prior work

Our documented search found no conformal-prediction application to motion-based ship-stability or
capsize early warning. Nearby maritime uses concern AIS trajectory and collision regions, including
a recent conformal collision-boundary study. The established motion-based false-alarm benchmark is
Galeazzi et al.'s parametric-roll monitoring: Weibull/phase signal tests, designed false-alarm
discipline, and year-long full-scale validation. Rahola is a ROM-first synthetic step toward adding
distribution-free uncertainty to that operational standard, not a replacement for its sea trials.

The 1-DOF approach follows the field's ROM-first methodology: Belenky et al. explicitly describe
1-DOF models as guides for multiple-DOF numerical methods. The caveat is substantive. Stability
variation in waves and stern-quartering roll/rate dependence require body-nonlinear models with at
least heave, roll, and pitch; this library cannot validate those effects.

## Reproduce

```bash
uv run python examples/e1_coverage.py
uv run python examples/e2_operating_curve.py
uv run python examples/e3_transition.py
uv run python examples/e3b_adapters.py
uv run python examples/e4_stress_test.py
uv run python examples/d1_detectors.py
uv run python examples/d2_family_generalization.py
uv run python examples/d3_bandwidth.py
uv run python examples/d4_wave_groups.py
uv run python examples/d5_within_regime.py
```

`uv run rahola-lab final-eval` is deliberately absent from the reproducible command list: it is a
one-time protocol, and the completed attestation permanently prevents a repeat.

## Frozen judgment calls and deviations

- The alarm angle is 0.60× the relevant escape angle; episode debounce and refractory are three
  10-second scoring windows. All episodes overlapping the event horizon are excluded from false
  counts; this recommended choice lowers re-alarming penalties relative to the original E3 report.
- False-episode confidence intervals use scorable windows as binomial opportunities. This satisfies
  the requested common Clopper–Pearson calculation but is only an interval convention under serial
  dependence, which is stated rather than hidden.
- The danger statistic uses the nearer side's threshold and an instantaneous separatrix
  extrapolation between exact upcrossings. The forced correction is zero because E2 is motion-only.
- The initial physics threshold grid ended at −1.0 rad/s and failed to include its analytic always-on
  endpoint. Before final scoring it was extended to −1.75 using fitted config values only; no test
  labels selected the endpoint.
- For asymmetric biased runs, maximum-absolute-roll targets use the smaller escape magnitude in both
  directions. This is conservative but can score a positive excursion against the negative margin.
- DtACI uses deterministic Algorithm 2 rather than randomized Algorithm 1. Sliding ACI tunes only on
  the held-out calibration half and carries no exchangeable finite-sample claim.
- CQR calibration retains one independent snapshot per calibration trajectory. Dense windows are
  operational units, not independent conformal units. Per-time E3 curves also condition on survival.
- Pure JAX remains the only neural runtime. E2 reports the operational 60-second horizon; E1 covers
  both frozen horizons.
- Provisional earlier work accessed original test splits more than once while correcting the risk-set
  and initialization defects. No test labels selected model hyperparameters, but the literal
  test-touched-once process rule remains unmet and fresh-offset replication was not performed.

## Method references

- Story, [Predicting Ship Capsize Using Lyapunov Exponents](https://vtechworks.lib.vt.edu/items/7eee36dd-055b-4aec-b49d-b173c2232278),
  Virginia Tech M.S. thesis (2009), especially Secs. 3.2.2, 3.3, and 4.2.2–4.3.
- Dakos et al., [Methods for Detecting Early Warnings of Critical Transitions in Time Series](https://dash.harvard.edu/handle/1/9637972),
  *PLoS ONE* 7 (2012), rolling variance/AC1 and Kendall trend.
- Bury et al., [Deep learning for early warning signals of tipping points](https://pubmed.ncbi.nlm.nih.gov/34544867/),
  *PNAS* 118 (2021), compact sequence-model comparison.
- Galeazzi, Blanke & Poulsen, [Early Detection of Parametric Roll Resonance on Container Ships](https://backend.orbit.dtu.dk/ws/files/7633739/Early_Detection.pdf),
  IEEE CDC (2012), W2-GLRT Eqs. 37 and 42 and the four-period segment.
- Belenky et al., [Estimation of probability of capsizing with split-time method](https://sandlab.mit.edu/wp-content/uploads/24_OEJ.pdf),
  *Ocean Engineering* 292 (2024) 116452, especially Eqs. 11–15 and 26–36.
- Anastopoulos & Spyrou, [Ship dynamic stability assessment based on realistic wave groups](https://doi.org/10.1016/j.oceaneng.2016.10.042),
  *Ocean Engineering* 134 (2017), critical-wave-group motivation.
- Romano, Patterson & Candès, [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222)
  (2019), one-sided upper-tail Theorem 2.
- Gibbs & Candès, [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170)
  (2021), equation (2) and Proposition 4.1.
- Gibbs & Candès, [Conformal Inference for Online Prediction with Arbitrary Distribution Shifts](https://jmlr.org/papers/v25/22-1218.html)
  (2024 journal version; 2022 preprint), deterministic DtACI Algorithm 2.
- Foygel Barber et al., [Conformal Prediction Beyond Exchangeability](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.full),
  *Annals of Statistics* 51 (2023), recent weighting under drift.
- Galeazzi et al., [Parametric roll resonance monitoring using signal-based detection](https://orbit.dtu.dk/en/publications/parametric-roll-resonance-monitoring-using-signal-based-detection/),
  *Ocean Engineering* 109 (2015), 355–371.
- Alba et al., [Enhancing Maritime Safety](https://www.mdpi.com/1424-8220/25/5/1365),
  *Sensors* 25 (2025), AIS collision boundaries with conformal prediction regions.
