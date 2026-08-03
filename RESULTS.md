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
```

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

- Belenky et al., [Estimation of probability of capsizing with split-time method](https://sandlab.mit.edu/wp-content/uploads/24_OEJ.pdf),
  *Ocean Engineering* 292 (2024) 116452, especially Eqs. 11–15 and 26–36.
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
