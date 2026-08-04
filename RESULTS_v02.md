# Rahola v0.2 results addendum

This addendum records the v0.2 methodology freeze and selective regeneration. It does not alter
the v0.1 audit in `RESULTS.md`; each final table below will identify the historical quantity it
supersedes. All uncertainty intervals resample complete trajectories within campaigns and hold
campaign weights fixed. They are conditional on the calibration-selected policy frozen before
test scoring.

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
validated reference solver's half-step grid. The cutoff therefore preserves the reference sea
definition but no longer changes when the integration step is refined. If the absolute
capsize-prevalence shift exceeds one percentage point in any reference campaign, the affected
campaign will be regenerated under an `_v02` name.

An earlier ratio-4 sensitivity run is retained as an audit artifact. It removed all D3 events and
therefore exposed an ill-posed cutoff choice before detector scoring; it is not the forcing
invariance decision run.

## Results pending execution

The forcing-invariance table, normalization ablation, D1–D5 selective reruns, Prototype #3 rows,
and the side-by-side supersession table will be added only after their frozen runs finish.
