"""eta-omega figures of every ring from the fits of fit_rings.py (run that first), summed over all translations
and at the translation through the rotation axis, PNG (300 dpi) and SVG (text as text):

    python visualization/ring_holdout/make_figures.py

visualization/figures/eta_omega_rings/
    all_rings_fitted/eta_omega_ring<r>_<hkl>_{summed,center}   Data | TT-Adaptive | TT-Uniform, fitted to all rings
    ring_left_out/eta_omega_ring<r>_<hkl>_{summed,center}      Data | TT-Adaptive (and TT-Uniform for 220 and
                                                               333+511), fitted without the ring shown
    additional_model_fits                                      appendix figure of the article: 220 (a-c) and
                                                               333+511 (d-f) at the centre translation, Data |
                                                               TT-Adaptive | TT-Uniform, each fitted without the
                                                               ring shown; no titles
    summary.md                                                 correlation and residual of every ring

Panel style as data_misfit.ipynb: power-law colour scale (gamma 0.5) from 0 to the data's 99.9th percentile,
one scale per figure (per row in the appendix figure), eta bins with zero weight grey.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_rings import N_RINGS, OUT, UNIFORM_HOLDOUT, load_problem  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_ROOT = os.path.join(REPO, "visualization", "figures", "eta_omega_rings")
APPENDIX_RINGS = [2, 9]  # 220, 333+511
DPI = 300
GAMMA = 0.5
LABELS = {"adaptive": "TT-Adaptive", "uniform": "TT-Uniform"}
VIEWS = {"summed": "summed over all translations", "center": "translation through the rotation axis"}

# Times New Roman as in the article (registered explicitly where matplotlib's font cache does not list it)
for _f in ("Times.TTF", "Timesi.TTF", "Timesbd.TTF", "Timesbi.TTF"):
    if os.path.isfile(os.path.join("/usr/share/fonts/ttf", _f)):
        font_manager.fontManager.addfont(os.path.join("/usr/share/fonts/ttf", _f))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino", "Georgia"],
    "font.size": 12,
    "svg.fonttype": "none",
})

arr, weights, cfg, mat, meta, i_center = load_problem(mmap=True)
fitted = weights > 0
eta = np.array(meta["eta_deg"])
omega_bin_deg = np.array(meta["omega_bin_deg"])
labels = meta["ring_labels"]
EXTENT = [eta[0] - 0.5, eta[-1] + 0.5, omega_bin_deg[-1], omega_bin_deg[0]]
data = {"summed": np.asarray(arr.sum(axis=1, dtype=np.float64)), "center": np.asarray(arr[:, i_center], dtype=np.float64)}
pred, info = {}, {}
for t in sorted(f[:-4] for f in os.listdir(OUT) if f.endswith(".npz")):
    z = np.load(os.path.join(OUT, t + ".npz"))
    assert int(z["i_center"]) == i_center
    pred[t] = {"summed": z["summed"].astype(np.float64), "center": z["center"].astype(np.float64)}
    with open(os.path.join(OUT, t + ".json")) as fh:
        info[t] = json.load(fh)
needed = ["adaptive_all", "uniform_all"] + [f"adaptive_wo{r:02d}" for r in range(N_RINGS)] + \
    [f"uniform_wo{r:02d}" for r in UNIFORM_HOLDOUT]
missing = [t for t in needed if t not in pred]
if missing:
    raise SystemExit(f"fits missing in {OUT}: {missing} (run fit_rings.py)")
CMAP = plt.get_cmap("magma").copy()
CMAP.set_bad("0.75")


def image(a, ring):
    return np.where(fitted[:, ring][None, :], a[:, :, ring], np.nan)


def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def show(ax, img, norm):
    return ax.imshow(img, extent=EXTENT, aspect="auto", interpolation="nearest", cmap=CMAP, norm=norm)


def save(fig, path):
    for ext in ("svg", "png"):
        fig.savefig(f"{path}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def ring_figure(ring, view, panels, title, path):
    d = image(data[view], ring)
    norm = mcolors.PowerNorm(GAMMA, vmin=0, vmax=np.nanpercentile(d, 99.9))
    n = len(panels) + 1
    fig, axes = plt.subplots(1, n, figsize=(0.4 + 3.87 * n, 4.5), sharey=True, constrained_layout=True)
    for k, (ax, (name, img)) in enumerate(zip(axes, [("Data", d)] + panels)):
        im = show(ax, img, norm)
        ax.set_title(name)
        ax.set_xlabel("Azimuth η (°)")
        ax.set_xticks([-90, 0, 90])
        if k == 0:
            ax.set_ylabel("Rotation ω (°)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Intensity")
    fig.suptitle(title)
    save(fig, path)


def appendix_figure(path):
    fig = plt.figure(figsize=(13.0, 9.6))
    outer = fig.add_gridspec(2, 2, width_ratios=[1, 0.025], wspace=0.04, hspace=0.32,
                             left=0.07, right=0.94, bottom=0.07, top=0.97)
    letters = iter("abcdef")
    for row, ring in enumerate(APPENDIX_RINGS):
        d = image(data["center"], ring)
        imgs = [d] + [image(pred[f"{g}_wo{ring:02d}"]["center"], ring) for g in ("adaptive", "uniform")]
        norm = mcolors.PowerNorm(GAMMA, vmin=0, vmax=np.nanpercentile(d, 99.9))
        inner = outer[row, 0].subgridspec(1, 3, wspace=0.22)
        for col, img in enumerate(imgs):
            ax = fig.add_subplot(inner[0, col])
            im = show(ax, img, norm)
            ax.set_xticks([-90, 0, 90])
            ax.set_xlabel("Azimuth η (°)")
            if col == 0:
                ax.set_ylabel("Rotation ω (°)")
            else:
                ax.tick_params(labelleft=False)
            ax.text(-0.02, 1.015, f"({next(letters)})", transform=ax.transAxes, ha="right", va="bottom")
        fig.colorbar(im, cax=fig.add_subplot(outer[row, 1]), label="Intensity")
    save(fig, path)


for folder in ("all_rings_fitted", "ring_left_out"):
    os.makedirs(os.path.join(FIG_ROOT, folder), exist_ok=True)
rows = []
for ring in range(N_RINGS):
    tag = f"ring{ring}_{labels[ring]}"
    row = {"ring": labels[ring]}
    for view, vtext in VIEWS.items():
        d = image(data[view], ring)
        full = [(LABELS[g], image(pred[f"{g}_all"][view], ring)) for g in ("adaptive", "uniform")]
        ring_figure(ring, view, full, f"{labels[ring]}, {vtext}",
                    os.path.join(FIG_ROOT, "all_rings_fitted", f"eta_omega_{tag}_{view}"))
        held = [("adaptive", f"adaptive_wo{ring:02d}")]
        if ring in UNIFORM_HOLDOUT:
            held.append(("uniform", f"uniform_wo{ring:02d}"))
        out = [(LABELS[g], image(pred[t][view], ring)) for g, t in held]
        ring_figure(ring, view, out, f"{labels[ring]} left out of the fit, {vtext}",
                    os.path.join(FIG_ROOT, "ring_left_out", f"eta_omega_{tag}_{view}"))
        for g, (_, img) in zip(("adaptive", "uniform"), full):
            row[f"{g}_all_{view}"] = corr(img, d)
        for (g, _), (_, img) in zip(held, out):
            row[f"{g}_wo_{view}"] = corr(img, d)
    for key, t in (("adaptive", "adaptive_all"), ("adaptive_wo", f"adaptive_wo{ring:02d}"), ("uniform", "uniform_all")):
        row[f"res_{key}"] = info[t]["rings"][ring]["rel_residual"]
    if ring in UNIFORM_HOLDOUT:
        row["res_uniform_wo"] = info[f"uniform_wo{ring:02d}"]["rings"][ring]["rel_residual"]
    rows.append(row)
    print(f"ring {labels[ring]}: figures written", flush=True)
appendix_figure(os.path.join(FIG_ROOT, "additional_model_fits"))
print("appendix figure written", flush=True)


def fmt(v):
    return "" if v is None else f"{v:.3f}"


versions = sorted({info[t]["diffractom"] for t in needed})
lines = [
    "# eta-omega images of every ring: all rings fitted vs. the ring left out",
    "",
    "Al1050, the reconstructions of texture_tomography/textomo_{adaptive,uniform}.ipynb (adaptive basis, K 48 017,",
    "sigma 0.4 deg; uniform grid, K 82 198, sigma 2 deg; FISTA-Huber, 200 iterations) fitted to all rings and with",
    "the ring shown left out (its weights zero): adaptive for every ring, uniform also for 220 and 333+511.",
    f"diffractom: {', '.join(versions)}. Made by visualization/ring_holdout/fit_rings.py and make_figures.py.",
    "",
    "Correlation of the predicted eta-omega image with the data (fitted eta bins only), and the relative residual",
    "||d - p|| / ||d|| of the ring over every bin (all translations, omega, fitted eta bins).",
    "",
    "| ring | summed: adaptive all / left out | centre: adaptive all / left out | residual adaptive all / left out "
    "| summed: uniform all / left out | centre: uniform all / left out | residual uniform all / left out |",
    "|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append(
        f"| {r['ring']} | {fmt(r['adaptive_all_summed'])} / {fmt(r['adaptive_wo_summed'])} "
        f"| {fmt(r['adaptive_all_center'])} / {fmt(r['adaptive_wo_center'])} "
        f"| {fmt(r['res_adaptive'])} / {fmt(r['res_adaptive_wo'])} "
        f"| {fmt(r['uniform_all_summed'])} / {fmt(r.get('uniform_wo_summed'))} "
        f"| {fmt(r['uniform_all_center'])} / {fmt(r.get('uniform_wo_center'))} "
        f"| {fmt(r['res_uniform'])} / {fmt(r.get('res_uniform_wo'))} |")
with open(os.path.join(FIG_ROOT, "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
