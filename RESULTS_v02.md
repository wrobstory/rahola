# Rahola v0.2 results addendum

This addendum records the v0.2 methodology freeze and selective regeneration. It does not alter
the v0.1 audit in `RESULTS.md`; each final table below will identify the historical quantity it
supersedes. Unless noted for Prototype #3 below, uncertainty intervals resample complete
trajectories within campaigns and hold campaign weights fixed. They are conditional on the
calibration-selected policy frozen before test scoring.

## Preregistration record

The following prediction was committed before the 900-second step campaign was generated or
scored:

> Prediction (preregistered): fully post-step within-regime discrimination will remain near chance (AUC below 0.58) for every motion-only method, consistent with the immediate-post-transition result it replaces.

The D5_v02 scoring rule requires the full 60-period motion history to start at or after the
300-second transition and the full 50-period outcome horizon to remain in the record. Scored
endpoints therefore run from 540 through 700 seconds. The comparison includes a protocol-clock
baseline.

## Frozen preprocessing policy

Physical preprocessing is primary for classical early-warning statistics, the neighbor score,
the roll-band scale-increase statistic, and other physics-adjacent features. It uses
$x=\phi/\phi_v$ and $v=\dot\phi/(\omega_n\phi_v)$. The CNN and XGBoost models report both a
fixed-window past-only fit and the historical cumulative-online transform; the fixed-window mode
is primary. Cumulative-online results estimate the performance of the normalization-plus-detector
system, whose state contains the full observed motion history.

## Forcing audit decision rule

The v0.2 forcing grid uses a cutoff of 40 times the natural frequency: the Nyquist limit of the
validated reference solver's half-step grid. It preserves the reference cutoff while adopting a
fixed interval grid; the paired audit below measures the resulting field change. The cutoff no
longer changes when the integration step is refined. If the absolute
capsize-prevalence shift exceeds one percentage point in any reference campaign, the affected
campaign will be regenerated under an `_v02` name.

An earlier ratio-4 sensitivity run is retained as an audit artifact. It removed all D3 events and
therefore exposed an ill-posed cutoff choice before detector scoring; it is not the forcing
invariance decision run.

## Forcing-invariance result

The final ratio-40 audit compared the fixed field with the legacy, step-dependent Nyquist field on
the full reference campaigns and paired 512-trajectory subsets. Three campaigns crossed the
predeclared one-percentage-point threshold.

| Campaign | Legacy prevalence | Fixed-cutoff prevalence | Shift | Decision |
| --- | ---: | ---: | ---: | --- |
| bandwidth $`\gamma=7`$ | 40.958% | 42.375% | +1.417 pp | regenerate |
| bandwidth $`\gamma=15`$ | 42.417% | 43.542% | +1.125 pp | regenerate |
| bandwidth $`\gamma=30`$ | 43.208% | 42.125% | -1.083 pp | regenerate |

All other campaigns moved by at most 0.743 percentage points. The affected bandwidth campaigns
were regenerated as `_v02`; the new 900-second step campaign was generated directly under the
fixed definition. The roll oscillator's low-pass response explains why the earlier step-halving
tests passed despite the moving high-frequency boundary. D3's historical internal comparison
also used one common grid, so it was coherent; v0.2 nevertheless uses the resolution-independent
definition.

## Normalization ablation

The frozen primary is fixed-window causal for learned motion models and physical scaling for
physics-adjacent scores. Cumulative-online values describe a normalization-plus-detector system
with full-history state.

| Experiment | Fixed-window CNN (primary) | Cumulative-online CNN | Interpretation |
| --- | ---: | ---: | --- |
| D1 pooled | 91.00% at 13.409 false episodes/h | 93.83% at 16.451/h | both reach the calibration target; the primary costs less |
| D2 hold out softening | 99.15% at 20.661/h | 64.53% at 20.528/h | primary transfers sensitivity only at near-always-on cost |
| D2 hold out parametric | 90.83% at 12.503/h | 90.42% at 13.556/h | both reach 90%; primary costs less |
| D2 hold out biased | 76.21% at 3.139/h | 76.21% at 3.533/h | neither reaches 90% |
| D3 AUC range | 0.883–0.913 | 0.866–0.920 | bandwidth ranking survives in both modes |
| D5 orientation-independent AUC | 0.556 | 0.538 | neither approaches the 0.58 trigger |

All three preprocessing modes pass the future-only leakage probe. Physical scaling is used for
classical EWS, the one-sided roll-band statistic, the neighbor score, and physics-adjacent
engineered features.

## Established-regime result

The preregistered prediction is retained. The largest orientation-independent motion-only AUC was
0.556 for the fixed-window CNN; cumulative-online CNN reached 0.538, classical EWS 0.513, the
neighbor score 0.509, the one-sided roll-band statistic 0.502, and the two-sided danger margin
0.500. The protocol-clock comparator reached 0.515. Raw AUC intervals resample trajectories and
keep every window belonging to a sampled trajectory. Orientation-independent AUCs are point
transforms of the raw estimates; no separate transformed intervals are claimed.

The B2 audit did not trigger: frozen Chronos reached orientation-independent AUC 0.518 and the
one-epoch fine-tuned mode 0.513. D5_v02 was run once. Neither result was used to reopen a model or
threshold choice.

## Forecast repair results

Only affected rows were regenerated. Envelope and linear E1 rows stand because epoch shuffling
changes only the LSTM. E2 was rerun in full because asymmetric alarm scaling affects conformal
alarms on the biased family and the corrected danger margin affects its physics baseline.

| Result | v0.1 | v0.2 |
| --- | ---: | ---: |
| E1 LSTM mean absolute coverage error | 0.84 pp | 0.89 pp |
| E1 LSTM worst absolute coverage error | 2.66 pp | 2.56 pp |
| E2 envelope sensitivity / false episodes per hour | 95.5% / 7.304 | 91.0% / 6.763 |
| E2 linear sensitivity / false episodes per hour | 85.0% / 6.383 | 85.0% / 6.173 |
| E2 LSTM sensitivity / false episodes per hour | 96.5% / 6.963 | 89.0% / 6.370 |
| E2 danger sensitivity / false episodes per hour | 82.0% / 8.228 | 90.5% / 7.069 |
| E3 selected ACI $`\gamma`$ | 0.05 | 0.05 |
| E3 fixed / ACI false episodes per hour | 0.000 / 6.130 | 0.000 / 6.092 |
| E4 raw / split-CQR snapshot coverage | 69.48% / 94.34% | 73.41% / 94.31% |
| E4 dense ACI post-step coverage | 92.12% | 91.84% |

E3's ACI still fails both the recovery and alarm-explosion guards. The unchanged numeric rows are
new artifacts because their uncertainty labels and conditioning statements changed.

## Detector supersession table

This table lists the headline v0.1 quantities replaced by the methodology freeze. Unlisted rows
stand, as recorded in `RESULTS.md`.

| Quantity | v0.1 | v0.2 |
| --- | ---: | ---: |
| D1 CNN sensitivity / false episodes per hour | 92.36% / 15.548 | 91.00% / 13.409 (fixed-window primary) |
| D1 CNN cumulative-online | 92.36% / 15.548 | 93.83% / 16.451 |
| D1 classical EWS sensitivity / false episodes per hour | 100.00% / 21.391 | 100.00% / 21.391 |
| D1 danger sensitivity / false episodes per hour | 99.16% / 21.368 | 100.00% / 21.391 |
| D1 roll-band statistic sensitivity / false episodes per hour | 100.00% / 21.391 | 100.00% / 21.391 |
| D1 neighbor sensitivity / false episodes per hour | 96.34% / 20.959 | 100.00% / 21.391 |
| D2 CNN, held-out softening | 64.53% / 20.528 | 99.15% / 20.661 (primary); 64.53% / 20.528 (cumulative) |
| D2 CNN, held-out parametric | 88.33% / 11.764 | 90.83% / 12.503 (primary); 90.42% / 13.556 (cumulative) |
| D2 CNN, held-out biased | 76.21% / 3.533 | 76.21% / 3.139 (primary); 76.21% / 3.533 (cumulative) |
| D2 classical EWS, held-out softening | 100.00% / 20.828 | 62.61% / 20.816 |
| D2 danger, held-out softening | 81.62% / 20.995 | 100.00% / 20.828 |
| D2 roll-band statistic, held-out softening | 100.00% / 20.828 | 100.00% / 20.828 |
| D2 neighbor, held-out softening | 87.39% / 21.051 | 100.00% / 20.828 |
| D2 classical EWS, held-out parametric | 100.00% / 21.685 | 100.00% / 21.685 |
| D2 danger, held-out parametric | 99.17% / 21.685 | 100.00% / 21.685 |
| D2 roll-band statistic, held-out parametric | 100.00% / 21.685 | 100.00% / 21.685 |
| D2 neighbor, held-out parametric | 100.00% / 21.685 | 100.00% / 21.685 |
| D2 classical EWS, held-out biased | 100.00% / 21.657 | 100.00% / 21.657 |
| D2 danger, held-out biased | 98.39% / 21.657 | 100.00% / 21.657 |
| D2 roll-band statistic, held-out biased | 100.00% / 21.657 | 100.00% / 21.657 |
| D2 neighbor, held-out biased | 99.60% / 21.660 | 100.00% / 21.657 |
| D3 CNN AUC, $`\gamma=1,3.3,7,15,30`$ | 0.920, 0.891, 0.871, 0.879, 0.862 | 0.913, 0.897, 0.897, 0.883, 0.883 |
| D3 cumulative-online CNN AUC, same order | historical CNN row | 0.920, 0.888, 0.868, 0.869, 0.866 |
| D3 cross-gamma CNN AUC, same order | 0.909, 0.892, 0.882, 0.883, 0.867 | 0.919, 0.906, 0.904, 0.876, 0.880 |
| D3 classical EWS raw AUC, same order | 0.350, 0.441, 0.468, 0.486, 0.534 | 0.206, 0.216, 0.209, 0.204, 0.226 |
| D3 danger raw AUC, same order | 0.425, 0.438, 0.416, 0.435, 0.429 | 0.395, 0.402, 0.384, 0.378, 0.368 |
| D3 roll-band statistic raw AUC, same order | 0.547, 0.462, 0.439, 0.444, 0.420 | 0.466, 0.499, 0.503, 0.507, 0.517 |
| D3 neighbor raw AUC, same order | 0.283, 0.273, 0.260, 0.255, 0.251 | 0.500, 0.500, 0.500, 0.500, 0.500 |
| D5 CNN AUC | 0.474 immediate-post-transition | 0.444 raw / 0.556 orientation-independent after full wash-in |
| D5 classical EWS AUC | 0.509 | 0.513 |
| D5 danger-margin AUC | 0.499 | 0.500 |
| D5 roll-band statistic AUC | 0.493 | 0.498 raw / 0.502 orientation-independent |
| D5 neighbor AUC | 0.498 | 0.491 raw / 0.509 orientation-independent |
| D4 danger sensitivity, capsize preceded by group | 99.13% | 100.00% |
| D4 danger sensitivity, capsize not preceded by group | 100.00% | 100.00% |
| D4 danger false-episode group coincidence | 85.44% | 88.45% |

## Prototype #3 normalization and restart result

The 16,000-window run used 200 futures per window and completed in 1,878.4 seconds. C1 and C2 now
receive the same rollout seeds, so their paired difference is not inflated by independent future
draws. The restart semantics are unchanged: both replace the realized correlated forcing future.

| Quantity | v0.1 | v0.2 |
| --- | ---: | ---: |
| C1 exact-state restart AUC | 0.851 | 0.850 [0.835, 0.867] |
| C2 filtered-state restart AUC | 0.485 | 0.486 [0.461, 0.515] |
| CNN AUC | 0.627 cumulative-online | 0.652 [0.623, 0.684] fixed-window primary; 0.622 [0.591, 0.658] cumulative-online |
| XGBoost AUC | 0.762 cumulative-online | 0.723 [0.703, 0.744] fixed-window primary; 0.768 [0.750, 0.786] cumulative-online |
| Protocol-clock quartile AUC | 0.656 | 0.656 [0.644, 0.671] |

Intervals are 2,000-replicate global trajectory bootstraps conditional on the stratified sampled
windows and realized rollout draws. Campaign mixture can vary across replicates; the interval
estimates therefore do not implement the fixed-campaign-weight convention used elsewhere in v0.2.
The point AUCs retain the declared stratified campaign weights. The clock comparator slightly
exceeds both CNN modes. XGBoost receives configuration-assisted physics features, so neither its
advantage nor C1's advantage identifies a motion-architecture ceiling.

## Information sets

| Method or feature | Motion window | Full history | True vessel configuration | Protocol time | Sea state or wave field | Future forcing |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fixed-window CNN | yes | no | no | no | no | no |
| Cumulative-online CNN | yes | through normalization | no | no | no | no |
| Classical EWS / GLRT / neighbor | yes | no | scale constants only | no | no | no |
| Danger margin | endpoint state | no | yes; known-configuration comparator | no | no | no |
| XGBoost engineered features | yes | mode dependent | yes; configuration-assisted | no | no | no |
| Protocol-clock comparator | no | no | no | yes | no | no |
| C1/C2 restart comparators | endpoint or filtered state | C2 filter only | yes | ramp state | configured sea | independent replacement future |
| D4 evaluator | detector dependent | detector dependent | detector dependent | no | reconstructed wave field after scoring | no |

## Judgment calls

- The cutoff is $40\omega_n$, not the initially explored $4\omega_n$, because 40 preserves the
  validated reference cutoff while separating the physical field definition from solver resolution.
- A prevalence change strictly greater than one percentage point triggers regeneration. The rule
  was applied campaign by campaign; only $`\gamma=7,15,30`$ crossed it.
- Exact capsize-event intervals remain descriptive. Episode sensitivity, false-episode rates, and
  AUC use at least 1,000 trajectory-block replicates with fixed seeds and fixed campaign weights.
- C1 and C2 receive common rollout seeds in v0.2. Their futures remain independent replacements,
  so neither is a Bayes ceiling for a detector that observes correlated motion history.
- The ratio-4 forcing run and all v0.1 results remain immutable audit records, not alternative
  selections from which the most favorable outcome was chosen.
