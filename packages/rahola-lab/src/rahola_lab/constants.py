"""Frozen experimental constants shared by every Rahola prototype."""

from __future__ import annotations

from enum import StrEnum

# Operational alarm horizons: short and actionable, while spanning several rolls.
FORECAST_HORIZONS_S = (30.0, 60.0)
# Two minutes captures slow envelope modulation without using future motion.
FORECAST_HISTORY_S = 120.0
# 0.6*phi_v leaves escape margin while rejecting ordinary moderate roll.
ALARM_THRESHOLD_ESCAPE_FRACTION = 0.60
# Prototype #2 needs a long baseline for slow early-warning statistics.
EWS_WINDOW_PERIODS = 60.0
# Roughly 50 cycles preserves the planned bifurcation-warning horizon.
EWS_HORIZON_PERIODS = 50.0
# Five periods remove ambiguous near-misses from the negative class.
EXCLUSION_BUFFER_PERIODS = 5.0

# ACI is operationally unacceptable only if it exceeds both this absolute rate...
ACI_EXPLOSION_FPR_PER_HOUR = 2.0
# ...and four times the corresponding fixed-CQR episode rate.
ACI_EXPLOSION_FACTOR = 4.0
# Small predeclared grid spans slow adaptation through deliberately aggressive updates.
ACI_GAMMA_GRID = (0.001, 0.005, 0.01, 0.02, 0.05)
# Gibbs--Candes' published DtACI expert grid; aggregation removes manual gamma selection.
DTACI_GAMMA_EXPERTS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)
# Calibration-only E3b grid for recent-score recalibration memory.
SLIDING_RECALIBRATION_WINDOWS = (25, 50, 100)
# Fitted-config bound brackets always-on through selective physics alarms.
DANGER_SCORE_THRESHOLDS_RAD_S = tuple(-1.75 + 0.025 * index for index in range(91))

# Disjoint 100k-wide ranges prevent phase reuse across statistical roles.
SEED_BLOCK_SIZE = 100_000


class SeedBlock(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"
    RESERVE = "reserve"


SEED_BLOCK_START = {
    SeedBlock.TRAIN: 0,
    SeedBlock.CALIBRATION: 100_000,
    SeedBlock.TEST: 200_000,
    # Frozen for Prototype #2 final evaluation; this task must never materialize it.
    SeedBlock.RESERVE: 300_000,
}
