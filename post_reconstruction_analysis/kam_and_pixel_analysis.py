"""
KAM map + single-pixel analysis plots.

Produces:
  1. KAM (Kernel Average Misorientation) map  (same as in plot_odf_density.py)
  2. Copy of the KAM map with two highlighted pixels (red & green)
  3. IPF scatter plot for the GREEN pixel  (weighted by ODF coefficients)
  4. Misorientation histogram for the RED pixel  (deviation from weighted medoid)

All outputs saved to  OUT_DIR / 'misorientation_plots/'
"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import time
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import map_coordinates

from orix.quaternion import Rotation, Orientation
from orix.quaternion.symmetry import Oh
from orix.quaternion import symmetry
from orix.vector import Vector3d
from orix import plot as orix_plot

# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS  — edit these to taste
# ═══════════════════════════════════════════════════════════════════════════
RECON_PATH  = Path('/work3/msaca/textomo_run_026/reconstruction.h5')

SPATIAL_MASK_THRESHOLD = 0.015
NORMALIZE_COEFFS_BY_MAX_SUM = False

OUT_DIR = Path('/work3/msaca/textomo_run_026/texture_visualization')
PLOT_DIR = OUT_DIR / 'misorientation_plots'

# Medoid computation
TOP_K = 24   # number of top orientations per pixel for medoid search

# KAM / grain segmentation
KAM_MISORI_THRESHOLD_DEG = 5.0

# ── KAM map visualisation ─────────────────────────────────────────────────
KAM_VMAX_DEG = 50.0        # vmax for KAM colorbar in degrees (None → 99th pctl)
KAM_LOGSCALE = True         # True → log(ε + x) colour scale
KAM_LOG_EPS  = 4.0          # ε for log scale

# ── Highlighted pixels (row, col) in the ORIGINAL (pre-rotation) grid ────
#    These are indices into x_full *after* the [5:-5, 5:-5] crop.
GREEN_PIXEL = (39, 80)      # IPF scatter plot
RED_PIXEL   = (67, 50)      # misorientation histogram

# ── IPF scatter options ───────────────────────────────────────────────────
IPF_POINT_SIZE_MIN  = 10     # min marker size (smallest-weight orientation)
IPF_POINT_SIZE_MAX  = 80    # max marker size (largest-weight orientation)
IPF_ALPHA_MIN       = 0.25  # min transparency
IPF_ALPHA_MAX       = 1.0   # max transparency
IPF_BG_COLOR        = 'white'   # background colour of the IPF triangle
IPF_EDGE_COLOR      = 'black'   # colour of the triangle boundary line

# ── Histogram options ─────────────────────────────────────────────────────
HIST_N_BINS          = 40       # number of bins
HIST_XMAX_DEG        = 15    # max misorientation on x-axis (62.8° ≈ Oh max)
HIST_LOG_XAXIS       = False    # True → logarithmic x-axis
HIST_COLOR           = '#d62728'  # bar colour
HIST_EDGECOLOR       = 'white'

# ── Font ──────────────────────────────────────────────────────────────────
FONT_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Georgia'],
}

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading {RECON_PATH} …")
with h5py.File(RECON_PATH, 'r') as f:
    x_full_raw = f['x'][()].astype(np.float64)[5:-5, 5:-5]
    coeff_scale = float(x_full_raw.sum(axis=2).max())
    if coeff_scale <= 0:
        raise ValueError("Coefficient scale is non-positive.")
    x_full = (x_full_raw / coeff_scale) if NORMALIZE_COEFFS_BY_MAX_SUM else x_full_raw
    rot_mats = f['orientations'][()]               # (K, 3, 3)
    sigma_rad = float(f.attrs.get('sigma', 0.007))

Ny, Nx, K = x_full.shape
sum_coeffs = x_full.sum(axis=2)
spatial_mask_threshold_eff = SPATIAL_MASK_THRESHOLD
spatial_mask = sum_coeffs > spatial_mask_threshold_eff

print(f"  Volume shape: ({Ny}, {Nx}, {K}),  σ = {sigma_rad:.4f} rad")
print(f"  Active pixels: {int(spatial_mask.sum())}")

grid_mats = rot_mats
SYM_MATS = np.array(Rotation(Oh).to_matrix())  # (48, 3, 3)
n_sym = SYM_MATS.shape[0]


# ═══════════════════════════════════════════════════════════════════════════
# MEDOID ORIENTATION MAP
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing weighted medoid orientation for each pixel …")

rec_medoid = np.full((Ny, Nx, 3, 3), np.nan)
n_active_pixels = int(spatial_mask.sum())
count = 0
t0 = time.time()

for iy in range(Ny):
    for ix in range(Nx):
        if not spatial_mask[iy, ix]:
            continue
        count += 1
        if count % 2000 == 0:
            elapsed = time.time() - t0
            rate = count / elapsed if elapsed > 0 else 0
            eta_s = (n_active_pixels - count) / rate if rate > 0 else 0
            print(f"  {count}/{n_active_pixels}  ({elapsed:.0f}s, ~{eta_s:.0f}s remaining)")

        w = x_full[iy, ix]
        if w.max() <= 0:
            continue

        top_idx = np.argsort(w)[-TOP_K:]
        top_idx = top_idx[w[top_idx] > 0]
        n_top = len(top_idx)

        if n_top <= 1:
            rec_medoid[iy, ix] = grid_mats[np.argmax(w)]
            continue

        wt = w[top_idx]
        top_mats = grid_mats[top_idx]
        top_equivs = np.einsum('sij,kjl->ksil', SYM_MATS, top_mats)
        traces = np.einsum('iab,jsab->ijs', top_mats, top_equivs)
        angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
        min_angles = angles.min(axis=-1)
        costs = min_angles @ wt
        rec_medoid[iy, ix] = top_mats[np.argmin(costs)]

elapsed = time.time() - t0
print(f"  Done ({count} pixels, {elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# KAM MAP
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing KAM map …")

medoid_valid = np.isfinite(rec_medoid[..., 0, 0])
valid_iy, valid_ix = np.where(medoid_valid)
valid_mats = rec_medoid[medoid_valid]
n_valid = len(valid_iy)

idx_map = np.full((Ny, Nx), -1, dtype=np.int64)
for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    idx_map[iy, ix] = k


def misorientation_angle_batch(mats, idx_from, idx_to, sym_mats):
    R1 = mats[idx_from]
    R2 = mats[idx_to]
    dR = np.einsum('mji,mjk->mik', R1, R2)
    sym_dR = np.einsum('sij,mjk->msik', sym_mats, dR)
    traces = np.trace(sym_dR, axis1=2, axis2=3)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angles.min(axis=1))


right_from, right_to = [], []
down_from, down_to = [], []
for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    if ix + 1 < Nx and idx_map[iy, ix + 1] >= 0:
        right_from.append(k)
        right_to.append(idx_map[iy, ix + 1])
    if iy + 1 < Ny and idx_map[iy + 1, ix] >= 0:
        down_from.append(k)
        down_to.append(idx_map[iy + 1, ix])
right_from = np.array(right_from, dtype=np.int64)
right_to   = np.array(right_to, dtype=np.int64)
down_from  = np.array(down_from, dtype=np.int64)
down_to    = np.array(down_to, dtype=np.int64)

BATCH = 2000
misori_right_vals = np.zeros(len(right_from))
for b in range(0, len(right_from), BATCH):
    e = min(b + BATCH, len(right_from))
    misori_right_vals[b:e] = misorientation_angle_batch(
        valid_mats, right_from[b:e], right_to[b:e], SYM_MATS)

misori_down_vals = np.zeros(len(down_from))
for b in range(0, len(down_from), BATCH):
    e = min(b + BATCH, len(down_from))
    misori_down_vals[b:e] = misorientation_angle_batch(
        valid_mats, down_from[b:e], down_to[b:e], SYM_MATS)

kam_sum   = np.zeros(n_valid, dtype=np.float64)
kam_count = np.zeros(n_valid, dtype=np.int32)
for i in range(len(right_from)):
    a, b = right_from[i], right_to[i]
    v = misori_right_vals[i]
    kam_sum[a] += v; kam_count[a] += 1
    kam_sum[b] += v; kam_count[b] += 1
for i in range(len(down_from)):
    a, b = down_from[i], down_to[i]
    v = misori_down_vals[i]
    kam_sum[a] += v; kam_count[a] += 1
    kam_sum[b] += v; kam_count[b] += 1

kam_map = np.full((Ny, Nx), np.nan)
for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    if kam_count[k] > 0:
        kam_map[iy, ix] = kam_sum[k] / kam_count[k]

print(f"  KAM range: [{np.nanmin(kam_map):.2f}°, {np.nanmax(kam_map):.2f}°]")


# ═══════════════════════════════════════════════════════════════════════════
# ROTATE / CROP  (re-use the same helper as the main script)
# ═══════════════════════════════════════════════════════════════════════════

def rotate_map_45(arr_2d, fill_value=np.nan, interp_order=0):
    upsample = 2
    theta = 0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    h, w = arr_2d.shape[:2]
    extra_dims = arr_2d.shape[2:] if arr_2d.ndim > 2 else ()
    out_dtype = np.result_type(arr_2d.dtype, np.float32, type(fill_value))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    corners = np.array([
        [-cx - 0.5, -cy - 0.5],
        [w - 1 - cx + 0.5, -cy - 0.5],
        [-cx - 0.5, h - 1 - cy + 0.5],
        [w - 1 - cx + 0.5, h - 1 - cy + 0.5],
    ], dtype=np.float64)
    x_c, y_c = corners[:, 0], corners[:, 1]
    x_rot = cos_t * x_c - sin_t * y_c
    y_rot = sin_t * x_c + cos_t * y_c
    x_min, x_max = x_rot.min(), x_rot.max()
    y_min, y_max = y_rot.min(), y_rot.max()
    step = 1.0 / upsample
    out_w = int(np.ceil((x_max - x_min) / step)) + 1
    out_h = int(np.ceil((y_max - y_min) / step)) + 1
    x_out = x_min + np.arange(out_w) * step
    y_out = y_min + np.arange(out_h) * step
    X_out, Y_out = np.meshgrid(x_out, y_out)
    X_in = cos_t * X_out + sin_t * Y_out + cx
    Y_in = -sin_t * X_out + cos_t * Y_out + cy
    valid = (Y_in >= 0) & (Y_in <= h - 1) & (X_in >= 0) & (X_in <= w - 1)
    coords = np.vstack([Y_in.ravel(), X_in.ravel()])
    if extra_dims:
        out = np.empty((out_h, out_w) + extra_dims, dtype=out_dtype)
        for idx in np.ndindex(extra_dims):
            sampled = map_coordinates(
                arr_2d[(slice(None), slice(None)) + idx].astype(out_dtype, copy=False),
                coords, order=interp_order, mode="constant", cval=fill_value)
            out[(slice(None), slice(None)) + idx] = sampled.reshape(out_h, out_w)
    else:
        sampled = map_coordinates(
            arr_2d.astype(out_dtype, copy=False),
            coords, order=interp_order, mode="constant", cval=fill_value)
        out = sampled.reshape(out_h, out_w)
    return out, valid


def crop_to_valid(arr, valid_mask):
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return arr[rmin:rmax+1, cmin:cmax+1], valid_mask[rmin:rmax+1, cmin:cmax+1]


print("  Rotating KAM map …")
kam_rot, valid_rot = rotate_map_45(kam_map, fill_value=np.nan)
sample_mask_rot, _ = rotate_map_45(medoid_valid.astype(np.float64), fill_value=0, interp_order=0)
sample_mask_rot    = sample_mask_rot > 0.5

kam_rot, valid_crop    = crop_to_valid(kam_rot, valid_rot)
sample_mask_rot, _     = crop_to_valid(sample_mask_rot, valid_rot)
display_mask           = valid_crop


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS: KAM colourbar logic  (shared by both KAM plots)
# ═══════════════════════════════════════════════════════════════════════════
kam_display = kam_rot.copy()
kam_display[~display_mask] = np.nan

_vmax_deg = KAM_VMAX_DEG if KAM_VMAX_DEG is not None else float(np.nanpercentile(kam_display, 99))

if KAM_LOGSCALE:
    _kam_plot_data = np.log(KAM_LOG_EPS + np.clip(kam_display, 0.0, _vmax_deg))
    _imshow_vmin   = float(np.log(KAM_LOG_EPS))
    _imshow_vmax   = float(np.log(KAM_LOG_EPS + _vmax_deg))
else:
    _kam_plot_data = np.clip(kam_display, 0.0, _vmax_deg)
    _imshow_vmin, _imshow_vmax = 0.0, _vmax_deg


def _nice_round(v):
    if v <= 0:
        return 0.0
    exp = np.floor(np.log10(v))
    frac = v / 10.0**exp
    nf = 1.0 if frac < 1.5 else (2.0 if frac < 3.5 else (5.0 if frac < 7.5 else 10.0))
    return nf * 10.0**exp

def _fmt_deg(v):
    return f"{v:.3g}"

if KAM_LOGSCALE:
    _log_min = float(np.log(KAM_LOG_EPS))
    _log_max = float(np.log(KAM_LOG_EPS + _vmax_deg))
    _raw = np.clip(np.exp(np.linspace(_log_min, _log_max, 5)) - KAM_LOG_EPS,
                   0.0, _vmax_deg)
    _tick_misoris = np.array(
        [0.0] + [_nice_round(v) for v in _raw[1:-1]] + [_vmax_deg])
    _tick_positions = np.log(KAM_LOG_EPS + _tick_misoris)
else:
    _tick_positions = np.linspace(0.0, _vmax_deg, 5)
    _tick_misoris   = _tick_positions

_tick_labels = [_fmt_deg(v) for v in _tick_misoris]


def _add_kam_colorbar(fig, ax, im):
    """Attach a left-side colorbar with the pre-computed ticks."""
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("left", size="5%", pad=0.1)
    cb      = fig.colorbar(im, cax=cax)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.set_ticks(_tick_positions)
    cb.set_ticklabels(_tick_labels)
    cb.set_label('Misorientation (°)')
    return cb


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: plain KAM map
# ═══════════════════════════════════════════════════════════════════════════
print("\nSaving KAM map …")
with matplotlib.rc_context({**FONT_RC, 'font.size': 16}):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(_kam_plot_data, cmap='inferno',
                   vmin=_imshow_vmin, vmax=_imshow_vmax, interpolation='nearest')
    ax.axis('off')
    _add_kam_colorbar(fig, ax, im)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'kam_map.png', dpi=500, transparent=True)
    plt.close(fig)
print(f"  Saved → {PLOT_DIR / 'kam_map.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: KAM map with two highlighted pixels
# ═══════════════════════════════════════════════════════════════════════════
print("Saving KAM map with highlighted pixels …")

# Map original-grid pixel coords into the rotated/cropped/upsampled frame
def _original_to_rotated(iy, ix):
    """Convert (row, col) in the original grid to (row, col) in the
    rotated+cropped KAM display array (approximate nearest pixel)."""
    upsample = 2
    theta = 0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cy_orig, cx_orig = (Ny - 1) / 2.0, (Nx - 1) / 2.0
    # relative to centre
    dx = ix - cx_orig
    dy = iy - cy_orig
    # forward rotation
    xr = cos_t * dx - sin_t * dy
    yr = sin_t * dx + cos_t * dy
    # the rotate_map_45 function uses specific x_min/y_min for the output grid
    h, w = Ny, Nx
    corners = np.array([
        [-cx_orig - 0.5, -cy_orig - 0.5],
        [w - 1 - cx_orig + 0.5, -cy_orig - 0.5],
        [-cx_orig - 0.5, h - 1 - cy_orig + 0.5],
        [w - 1 - cx_orig + 0.5, h - 1 - cy_orig + 0.5],
    ], dtype=np.float64)
    x_c, y_c = corners[:, 0], corners[:, 1]
    x_rot_c = cos_t * x_c - sin_t * y_c
    y_rot_c = sin_t * x_c + cos_t * y_c
    x_min, y_min = x_rot_c.min(), y_rot_c.min()
    step = 1.0 / upsample
    col_out = int(round((xr - x_min) / step))
    row_out = int(round((yr - y_min) / step))
    # adjust for crop
    rows_any = np.any(valid_rot, axis=1)
    cols_any = np.any(valid_rot, axis=0)
    rmin = np.where(rows_any)[0][0]
    cmin = np.where(cols_any)[0][0]
    return row_out - rmin, col_out - cmin


green_r, green_c = _original_to_rotated(*GREEN_PIXEL)
red_r, red_c     = _original_to_rotated(*RED_PIXEL)

with matplotlib.rc_context({**FONT_RC, 'font.size': 16}):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(_kam_plot_data, cmap='inferno',
                   vmin=_imshow_vmin, vmax=_imshow_vmax, interpolation='nearest')
    # marker size in display units — large enough to see
    ms = 1
    ax.plot(green_c, green_r, 'o', color='#00ff00', markersize=ms,
            markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.plot(red_c, red_r, 'o', color='red', markersize=ms,
            markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.axis('off')
    _add_kam_colorbar(fig, ax, im)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'kam_map_highlighted.png', dpi=500, transparent=True)
    plt.close(fig)
print(f"  Saved → {PLOT_DIR / 'kam_map_highlighted.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: IPF scatter for GREEN pixel
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing IPF scatter for green pixel …")
gy, gx = GREEN_PIXEL
coeffs_green = x_full[gy, gx]
w = coeffs_green.astype(float)

# keep only orientations with positive weight
pos_mask = w > 0
w_pos = w[pos_mask]
mats_pos = rot_mats[pos_mask]

w_norm = w_pos / w_pos.max()

O = Orientation.from_matrix(mats_pos, symmetry=Oh)

dirs    = [Vector3d([0, 0, 1]), Vector3d([0, 1, 0]), Vector3d([1, 0, 0])]
titles  = ["IPF-Z", "IPF-Y", "IPF-X"]

with matplotlib.rc_context({**FONT_RC, 'font.size': 12}):
    fig, dummy_axes = plt.subplots(3, 1, figsize=(5, 14))

    for i, (ax0, v, title) in enumerate(zip(dummy_axes, dirs, titles)):
        ax0.remove()
        ax = fig.add_subplot(3, 1, i + 1,
                             projection='ipf',
                             symmetry=Oh,
                             direction=v)
        ax.set_facecolor(IPF_BG_COLOR)

        ipfkey = orix_plot.IPFColorKeyTSL(Oh, direction=v)
        colors = np.asarray(ipfkey.orientation2color(O))
        if colors.ndim == 1:
            colors = colors.reshape(-1, 3)

        sizes = IPF_POINT_SIZE_MIN + (IPF_POINT_SIZE_MAX - IPF_POINT_SIZE_MIN) * w_norm
        alphas = IPF_ALPHA_MIN + (IPF_ALPHA_MAX - IPF_ALPHA_MIN) * w_norm

        sort_idx = np.argsort(alphas)
        ax.scatter(
            O[sort_idx],
            c=colors[sort_idx],
            s=sizes[sort_idx],
            alpha=alphas[sort_idx],
            edgecolors='none',
            zorder=2,
        )

        ax._edge_patch.set_edgecolor(IPF_EDGE_COLOR)
        ax._edge_patch.set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'ipf_scatter_green.png', dpi=500, transparent=True)
    plt.close(fig)
print(f"  Saved → {PLOT_DIR / 'ipf_scatter_green.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3b: IPF scatter for GREEN pixel — monochrome (dark grey)
# ═══════════════════════════════════════════════════════════════════════════
print("Saving monochrome IPF scatter for green pixel …")
GREY_COLOR = np.array([0.25, 0.25, 0.25])   # dark grey RGB

with matplotlib.rc_context({**FONT_RC, 'font.size': 12}):
    fig, dummy_axes = plt.subplots(3, 1, figsize=(5, 14))

    for i, (ax0, v, title) in enumerate(zip(dummy_axes, dirs, titles)):
        ax0.remove()
        ax = fig.add_subplot(3, 1, i + 1,
                             projection='ipf',
                             symmetry=Oh,
                             direction=v)
        ax.set_facecolor(IPF_BG_COLOR)

        sizes  = IPF_POINT_SIZE_MIN + (IPF_POINT_SIZE_MAX - IPF_POINT_SIZE_MIN) * w_norm
        alphas = IPF_ALPHA_MIN + (IPF_ALPHA_MAX - IPF_ALPHA_MIN) * w_norm

        sort_idx = np.argsort(alphas)
        grey_colors = np.tile(GREY_COLOR, (len(sort_idx), 1))
        rgba_colors = np.column_stack([grey_colors, alphas[sort_idx]])

        ax.scatter(
            O[sort_idx],
            c=rgba_colors,
            s=sizes[sort_idx],
            edgecolors='none',
            zorder=2,
        )

        ax._edge_patch.set_edgecolor(IPF_EDGE_COLOR)
        ax._edge_patch.set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'ipf_scatter_green_grey.png', dpi=500, transparent=True)
    plt.close(fig)
print(f"  Saved → {PLOT_DIR / 'ipf_scatter_green_grey.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Misorientation histogram for RED pixel
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing misorientation histogram for red pixel …")
ry, rx = RED_PIXEL
coeffs_red = x_full[ry, rx].astype(float)

# weighted medoid for this pixel
pos_mask_r = coeffs_red > 0
w_r = coeffs_red[pos_mask_r]
mats_r = rot_mats[pos_mask_r]

if len(w_r) > 1:
    # use top-K for medoid
    top_k_r = min(TOP_K, len(w_r))
    top_idx_r = np.argsort(w_r)[-top_k_r:]
    wt_r = w_r[top_idx_r]
    top_mats_r = mats_r[top_idx_r]
    top_equivs_r = np.einsum('sij,kjl->ksil', SYM_MATS, top_mats_r)
    traces_r = np.einsum('iab,jsab->ijs', top_mats_r, top_equivs_r)
    angles_r = np.arccos(np.clip((traces_r - 1.0) / 2.0, -1.0, 1.0))
    min_angles_r = angles_r.min(axis=-1)
    medoid_mat = top_mats_r[np.argmin(min_angles_r @ wt_r)]
else:
    medoid_mat = mats_r[0] if len(w_r) == 1 else rot_mats[np.argmax(coeffs_red)]

# compute symmetry-reduced misorientation angle from medoid for ALL positive orientations
dR_all = np.einsum('ji,kjl->kil', medoid_mat, mats_r)  # medoid^T @ each
sym_dR_all = np.einsum('sij,mjk->msik', SYM_MATS, dR_all)
traces_all = np.trace(sym_dR_all, axis1=2, axis2=3)
angles_all = np.degrees(np.arccos(np.clip((traces_all - 1.0) / 2.0, -1.0, 1.0)))
misori_from_medoid = angles_all.min(axis=1)  # (n_pos,)

# build weighted histogram
if HIST_LOG_XAXIS:
    bin_edges = np.geomspace(max(misori_from_medoid[misori_from_medoid > 0].min() * 0.5, 0.01),
                             HIST_XMAX_DEG, HIST_N_BINS + 1)
else:
    bin_edges = np.linspace(0, HIST_XMAX_DEG, HIST_N_BINS + 1)

hist_weights, _ = np.histogram(misori_from_medoid, bins=bin_edges, weights=w_r)

# normalise so area sums to 1
bin_widths = np.diff(bin_edges)
hist_density = hist_weights / (hist_weights.sum() * bin_widths + 1e-30)

with matplotlib.rc_context({**FONT_RC, 'font.size': 12}):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(bin_edges[:-1], hist_density, width=bin_widths, align='edge',
           color=HIST_COLOR, edgecolor=HIST_EDGECOLOR, linewidth=0.4)
    if HIST_LOG_XAXIS:
        ax.set_xscale('log')
    ax.set_xlim(bin_edges[0], HIST_XMAX_DEG)
    ax.set_xlabel('Misorientation from medoid (°)')
    ax.set_ylabel('Density')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'misori_histogram_red.png', dpi=500, transparent=True)
    plt.close(fig)
print(f"  Saved → {PLOT_DIR / 'misori_histogram_red.png'}")


# ═══════════════════════════════════════════════════════════════════════════
print(f"\n✓ All plots saved to {PLOT_DIR}")
for p in sorted(PLOT_DIR.glob('*.png')):
    print(f"  {p.name}")
