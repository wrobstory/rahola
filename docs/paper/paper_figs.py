"""Generate the paper's two purpose-built figures (model overview, audit effect)."""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rahola import SimulationConfig, simulate_batch
from rahola.config import ForcingConfig, SeaState

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)
INK, RED, GRAY = "#1a1a1a", "#b03a2e", "#8a8a8a"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7})

# --- Figure 1: restoring families and two lives of one ship -------------------
config = SimulationConfig(
    duration_s=600.0, natural_period_s=4.0, output_rate_hz=2.0,
    forcing=ForcingConfig(sea_state=SeaState(hs_m=3.5, tp_s=5.0), effective_wave_slope=0.05),
)
dataset = None
for slope in (0.05, 0.07, 0.09, 0.11, 0.14, 0.17, 0.20, 0.24):
    cfg = replace(config, forcing=replace(config.forcing, effective_wave_slope=slope))
    ds = simulate_batch(cfg, seeds=range(60))
    print(f"slope={slope}: {int(ds.capsized.sum())}/60 capsized")
    if ds.capsized.any() and (~ds.capsized).any():
        dataset = ds
        break
assert dataset is not None, "no mixed-outcome severity found"
cap = int(np.argmax(dataset.capsized))
srv_candidates = np.flatnonzero(~dataset.capsized)
srv = int(srv_candidates[np.argmax(
    np.nanmax(np.abs(dataset.angle_rad[srv_candidates]), axis=1))])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4),
                               gridspec_kw={"width_ratios": [1, 2]})
x = np.linspace(-1.4, 1.4, 300)
ax1.plot(x, x - x**3, color=INK, lw=1.4, label="softening")
ax1.plot(x, 1.25 * (x - x**3), color=INK, lw=0.9, ls=":")
ax1.plot(x, 0.75 * (x - x**3), color=INK, lw=0.9, ls=":")
ax1.axhline(0.2, color=GRAY, lw=0.9, ls="--")
ax1.text(1.35, 0.26, "bias $b$", fontsize=7.5, color=GRAY, ha="right")
ax1.text(-1.3, 0.62, r"parametric: $(1\pm h)$", fontsize=7.5, color=INK)
for s in (1, -1):
    ax1.axvline(s, color=RED, lw=0.8, ls="--")
ax1.set_xlabel(r"$x=\phi/\phi_v$")
ax1.set_ylabel(r"restoring $R(x)$")
ax1.set_title("(a) the three families", fontsize=9)

t = dataset.time_s
deg = 180 / np.pi
ax2.plot(t, dataset.angle_rad[srv] * deg, color=GRAY, lw=0.8, label="survives")
ax2.plot(t, dataset.angle_rad[cap] * deg, color=INK, lw=0.9, label="capsizes")
esc = float(dataset.config["escape_angle_rad"]) * deg
for s in (1, -1):
    ax2.axhline(s * esc, color=RED, lw=0.8, ls="--")
tc = dataset.t_capsize_s[cap]
y_cap = dataset.angle_rad[cap][np.nanargmax(np.abs(dataset.angle_rad[cap]))] * deg
ax2.plot([tc], [y_cap], "o", color=RED, ms=4)
ax2.annotate("capsize", (tc, y_cap), textcoords="offset points",
             xytext=(10, 4), color=RED, fontsize=7.5)
ax2.set_xlabel("time (s)")
ax2.set_ylabel("roll (deg)")
ax2.set_title("(b) two lives of one ship, same sea state", fontsize=9)
ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_model.png", dpi=300)

# --- Figure 2: the audit's effect on the same detector ------------------------
fig, ax = plt.subplots(figsize=(4.6, 1.9))
vals = [6.3, 15.5]
labels = ["threshold selected\non test data (as first reported)",
          "threshold frozen on\ncalibration data (corrected)"]
bars = ax.barh([1, 0], vals, height=0.55, color=[RED, INK])
for y, v in zip([1, 0], vals, strict=True):
    ax.text(v + 0.3, y, f"{v:.1f} / h", va="center", fontsize=9)
ax.set_yticks([1, 0], labels, fontsize=8)
ax.set_xlabel("false episodes per exposure hour")
ax.set_xlim(0, 19)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_audit.png", dpi=300)
print("wrote", sorted(p.name for p in OUT.iterdir()))
