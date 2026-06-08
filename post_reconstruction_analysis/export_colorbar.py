"""Export misorientation colorbars (inferno, 0–30°) as transparent PNGs."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import numpy as np

RESULTS_DIR = Path('/work3/msaca/textomo_run_022/texture_visualization')

cmap = plt.get_cmap("inferno")
norm = mcolors.Normalize(vmin=0, vmax=30)

# ── Vertical ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(1.0, 4.0))
fig.subplots_adjust(left=0.05, right=0.4, top=0.97, bottom=0.03)

ticks = [0, 15, 30]

cb = mcolorbar.ColorbarBase(
    ax, cmap=cmap, norm=norm,
    orientation="vertical",
    ticks=ticks,
)
cb.ax.set_yticklabels([f"{int(t)}°" for t in ticks])
cb.ax.tick_params(labelsize=14)

fig.savefig(
    RESULTS_DIR / "misorientation_colorbar_vertical_KAM.png",
    dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True,
)
plt.close(fig)
print(f"Saved → {RESULTS_DIR / 'misorientation_colorbar_vertical_KAM.png'}")

# ── Horizontal ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.0, 1.0))
fig.subplots_adjust(left=0.03, right=0.97, top=0.55, bottom=0.05)

cb = mcolorbar.ColorbarBase(
    ax, cmap=cmap, norm=norm,
    orientation="horizontal",
    ticks=ticks,
)
cb.ax.set_xticklabels([f"{int(t)}°" for t in ticks])
cb.ax.tick_params(labelsize=14)

fig.savefig(
    RESULTS_DIR / "misorientation_colorbar_horizontal_KAM.png",
    dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True,
)
plt.close(fig)
print(f"Saved → {RESULTS_DIR / 'misorientation_colorbar_horizontal_KAM.png'}")




















"""Export misorientation colorbars (cividis, 0–1) as transparent PNGs."""

cmap = plt.get_cmap("cividis")
norm = mcolors.Normalize(vmin=0, vmax=1)

# ── Vertical ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(1.0, 4.0))
fig.subplots_adjust(left=0.05, right=0.4, top=0.97, bottom=0.03)

ticks = [0, 0.5, 1]

cb = mcolorbar.ColorbarBase(
    ax, cmap=cmap, norm=norm,
    orientation="vertical",
    ticks=ticks,
)
cb.ax.set_yticklabels([f"{t:g}" for t in ticks])
cb.ax.tick_params(labelsize=14)

fig.savefig(
    RESULTS_DIR / "misorientation_colorbar_vertical_sumcoeff.png",
    dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True,
)
plt.close(fig)
print(f"Saved → {RESULTS_DIR / 'misorientation_colorbar_vertical_sumcoeff.png'}")

# ── Horizontal ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.0, 1.0))
fig.subplots_adjust(left=0.03, right=0.97, top=0.55, bottom=0.05)

cb = mcolorbar.ColorbarBase(
    ax, cmap=cmap, norm=norm,
    orientation="horizontal",
    ticks=ticks,
)
cb.ax.set_xticklabels([f"{t:g}" for t in ticks])
cb.ax.tick_params(labelsize=14)

fig.savefig(
    RESULTS_DIR / "misorientation_colorbar_horizontal_sumcoeff.png",
    dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True,
)
plt.close(fig)
print(f"Saved → {RESULTS_DIR / 'misorientation_colorbar_horizontal_sumcoeff.png'}")