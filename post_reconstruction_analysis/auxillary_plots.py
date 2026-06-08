# This file should contain contain the following plots

# Sum of coefficients, rotated with pi/4 + 0.09, upscaled x 8
# KAM, rotated with pi/4 + 0.09, upscaled x 8

import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import label as ndlabel
from scipy.spatial.transform import Rotation as ScipyR

from orix.quaternion import Rotation, Orientation
from orix.quaternion import von_mises
from orix.quaternion.symmetry import Oh
from orix.quaternion import symmetry
from orix.vector import Vector3d
from orix import plot as orix_plot
from scipy.ndimage import map_coordinates

RECON_PATH  = Path('/work3/msaca/textomo_run_022/reconstruction.h5')
PIXEL_PATH  = Path('/work3/msaca/textomo_run_022/pixel_data.h5')
OUT_DIR = Path('/work3/msaca/textomo_run_022/texture_visualization')
OUT_DIR.mkdir(parents=True, exist_ok=True)

COEFF_THRESHOLD_BULK  = 3e-7
KAM_MISORI_THRESHOLD_DEG = 5.0   # threshold for grain boundary detection
GRAIN_MIN_SIZE           = 10     # minimum grain area in pixels
TOP_K_MEDOID             = 24     # top orientations kept for weighted medoid search


print(f"Loading {RECON_PATH} …")
with h5py.File(RECON_PATH, 'r') as f:
    x_full = f['x'][()].astype(np.float64)[5:-5, 5:-5]        # (H, W, K)
    rot_mats = f['orientations'][()]               # (K, 3, 3)
    sigma_rad = float(f.attrs.get('sigma', 0.007))


Ny, Nx, K = x_full.shape
sum_coeffs = x_full.sum(axis=2)                    # (H, W)
spatial_mask = sum_coeffs > 0.002                   # (H, W)











def rotate_map_45(arr_2d, fill_value=np.nan, interp_order=0):
    """
    Rotate a 2D array by 45 degrees and resample it onto a grid that is
    6x finer than the original grid.

    Parameters
    ----------
    arr_2d : ndarray
        Input array of shape (H, W) or (H, W, ...).
    fill_value : scalar, optional8
        Value used outside the original image domain.
    interp_order : int, optional
        Interpolation order (0=nearest, 1=bilinear). Use 0 for
        categorical data like grain labels or boolean masks.

    Returns
    -------
    out : ndarray8
        Rotated, reinterpolated array on a 6x finer grid.
    valid : ndarray of bool
        Mask of shape out.shape[:2], True where the output pixel maps
        inside the original image domain.
    """
    upsample = 2#8
    theta = 0#np.pi / 4.0 + 0.09
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    h, w = arr_2d.shape[:2]
    extra_dims = arr_2d.shape[2:] if arr_2d.ndim > 2 else ()

    # Use floating output because interpolation is involved.
    # This also avoids problems when fill_value=np.nan and input is integer.
    out_dtype = np.result_type(arr_2d.dtype, np.float32, type(fill_value))

    # Original image center in pixel-center coordinates
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    # Use image outer corners (pixel edges) to determine required canvas size.
    # Coordinates are expressed relative to the image center.
    corners = np.array([
        [-cx - 0.5, -cy - 0.5],   # left, top
        [ w - 1 - cx + 0.5, -cy - 0.5],   # right, top
        [-cx - 0.5,  h - 1 - cy + 0.5],   # left, bottom
        [ w - 1 - cx + 0.5,  h - 1 - cy + 0.5],   # right, bottom
    ], dtype=np.float64)

    # Rotate corners forward to get output extent
    x_c = corners[:, 0]
    y_c = corners[:, 1]
    x_rot = cos_t * x_c - sin_t * y_c
    y_rot = sin_t * x_c + cos_t * y_c

    x_min, x_max = x_rot.min(), x_rot.max()
    y_min, y_max = y_rot.min(), y_rot.max()

    # Pixel spacing in the output grid is 1/upsample of an input pixel
    step = 1.0 / upsample

    out_w = int(np.ceil((x_max - x_min) / step)) + 1
    out_h = int(np.ceil((y_max - y_min) / step)) + 1

    # Coordinates of output pixel centers in rotated space
    x_out = x_min + np.arange(out_w) * step
    y_out = y_min + np.arange(out_h) * step
    X_out, Y_out = np.meshgrid(x_out, y_out)

    # Inverse rotation: output -> input-relative coordinates
    # Since R^{-1}(theta) = R(-theta):
    # x_in =  cos(theta) * x_out + sin(theta) * y_out
    # y_in = -sin(theta) * x_out + cos(theta) * y_out
    X_in_rel = cos_t * X_out + sin_t * Y_out
    Y_in_rel = -sin_t * X_out + cos_t * Y_out

    # Shift back to image coordinates
    X_in = X_in_rel + cx
    Y_in = Y_in_rel + cy

    # Valid mask: points whose source coordinate lies inside the original image
    valid = (
        (Y_in >= 0.0) & (Y_in <= h - 1) &
        (X_in >= 0.0) & (X_in <= w - 1)
    )

    # Interpolate
    coords = np.vstack([Y_in.ravel(), X_in.ravel()])

    if extra_dims:
        out = np.empty((out_h, out_w) + extra_dims, dtype=out_dtype)
        for idx in np.ndindex(extra_dims):
            sampled = map_coordinates(
                arr_2d[(slice(None), slice(None)) + idx].astype(out_dtype, copy=False),
                coords,
                order=interp_order,
                mode="constant",
                cval=fill_value
            )
            out[(slice(None), slice(None)) + idx] = sampled.reshape(out_h, out_w)
    else:
        sampled = map_coordinates(
            arr_2d.astype(out_dtype, copy=False),
            coords,
            order=interp_order,
            mode="constant",
            cval=fill_value
        )
        out = sampled.reshape(out_h, out_w)

    return out, valid

def crop_to_valid(arr, valid_mask):
    """Crop array and mask to the bounding box of valid_mask."""
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return arr[rmin:rmax+1, cmin:cmax+1], valid_mask[rmin:rmax+1, cmin:cmax+1]


def make_corner_mask(valid_mask):
    """
    From the valid_mask of rotated pixels, create a display mask:
    the original square becomes a diamond after 45° rotation.
    We want to cut away the corners of the bounding box that were NOT
    part of the original image.  valid_mask already encodes this.
    """
    return valid_mask


def weighted_medoid_from_coeffs(coeffs, mats, sym_mats, top_k=TOP_K_MEDOID):
    """Return weighted medoid rotation matrix from one coefficient vector."""
    w = np.asarray(coeffs, dtype=np.float64)
    if w.size == 0:
        return None

    w_pos = np.maximum(w, 0.0)
    nz = np.flatnonzero(w_pos > 0)
    if nz.size == 0:
        return None

    if top_k is not None and nz.size > top_k:
        sel = nz[np.argsort(w_pos[nz])[-top_k:]]
    else:
        sel = nz

    w_sel = w_pos[sel]
    mats_sel = mats[sel]  # (n_sel, 3, 3)
    n_sel = mats_sel.shape[0]
    if n_sel == 1:
        return mats_sel[0]

    # Pairwise symmetry-reduced geodesic distance matrix.
    equivs = np.einsum('sij,kjl->ksil', sym_mats, mats_sel)      # (n_sel, n_sym, 3, 3)
    traces = np.einsum('iab,jsab->ijs', mats_sel, equivs)        # (n_sel, n_sel, n_sym)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    dmin = angles.min(axis=-1)                                   # (n_sel, n_sel)

    costs = dmin @ w_sel
    return mats_sel[int(np.argmin(costs))]


def export_colorbar_pair(out_dir, filename_base, cmap_name, vmin, vmax, ticks, formatter, unit=''):
    """Export vertical and horizontal transparent colorbars."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # Vertical
    fig, ax = plt.subplots(figsize=(1.0, 4.0))
    fig.subplots_adjust(left=0.05, right=0.4, top=0.97, bottom=0.03)
    cb = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(cb, cax=ax, orientation='vertical', ticks=ticks)
    cbar.ax.set_yticklabels([formatter(t) for t in ticks])
    cbar.ax.tick_params(labelsize=14)
    v_path = out_dir / f"{filename_base}_colorbar_vertical.png"
    fig.savefig(v_path, dpi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close(fig)
    print(f"    Saved → {v_path}")

    # Horizontal
    fig, ax = plt.subplots(figsize=(4.0, 1.0))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.55, bottom=0.05)
    cb = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(cb, cax=ax, orientation='horizontal', ticks=ticks)
    cbar.ax.set_xticklabels([formatter(t) for t in ticks])
    cbar.ax.tick_params(labelsize=14)
    h_path = out_dir / f"{filename_base}_colorbar_horizontal.png"
    fig.savefig(h_path, dpi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close(fig)
    print(f"    Saved → {h_path}")







# Compute weighted-medoid orientation per pixel
print("  Computing weighted-medoid orientations per pixel …")
SYM_MATS = np.array(Rotation(Oh).to_matrix())  # (48, 3, 3)
rec_medoid = np.full((Ny, Nx, 3, 3), np.nan)
iy_all, ix_all = np.where(spatial_mask)
n_active_pixels = len(iy_all)
for i, (iy, ix) in enumerate(zip(iy_all, ix_all), start=1):
    if i % 2000 == 0:
        print(f"    {i}/{n_active_pixels}")
    m = weighted_medoid_from_coeffs(x_full[iy, ix], rot_mats, SYM_MATS)
    if m is not None:
        rec_medoid[iy, ix] = m
medoid_valid = np.isfinite(rec_medoid[..., 0, 0])

# Rotate and crop all maps consistently
sum_coeffs_rot, valid_rot  = rotate_map_45(sum_coeffs, fill_value=0)
sample_mask_rot, _         = rotate_map_45(medoid_valid.astype(np.float64), fill_value=0, interp_order=0)
sample_mask_rot            = sample_mask_rot > 0.5

# Crop all to the bounding box of valid_rot (must all use the same valid_rot)
sum_coeffs_rot, valid_crop  = crop_to_valid(sum_coeffs_rot, valid_rot)
sample_mask_rot, _          = crop_to_valid(sample_mask_rot, valid_rot)
display_mask                = make_corner_mask(valid_crop)
name = 'sum_coeffs_rot.png'

# Apply cividis colormap — show all pixels inside the rotated diamond,
# regardless of the spatial mask (no threshold transparency here)
active = display_mask
s_norm = np.zeros_like(sum_coeffs_rot)
if active.any():
    s_max = np.nanmax(sum_coeffs_rot[active])
    if s_max > 0:
        s_norm[active] = np.clip(sum_coeffs_rot[active] / s_max, 0, 1)
rgba_sum = plt.get_cmap('cividis')(s_norm)  # (H, W, 4)
rgba_sum[~active, 3] = 0.0

fig, ax = plt.subplots(figsize=(7, 7))
h, w = rgba_sum.shape[:2]

ax.imshow(rgba_sum, interpolation='nearest')
#ax.set_xlim(350, w - 350)
#ax.set_ylim(h - 350, 350)   # y-axis is inverted for images

ax.axis('off')
fig.tight_layout()
fig.savefig(OUT_DIR / name, dpi=400, transparent=True)
plt.close(fig)
print(f"    Saved → {OUT_DIR / name}")










# ── Save KAM map ────────────────────────────────────────────────────────
# KAM: average misorientation with all 4-connected neighbours

def misorientation_angle_batch_neighbours(mats, idx_from, idx_to, sym_mats):
    """Compute misorientation angles for pairs (idx_from[i], idx_to[i]).
    mats: (N, 3, 3). Returns angles in degrees, shape (len(idx_from),)."""
    R1 = mats[idx_from]   # (M, 3, 3)
    R2 = mats[idx_to]     # (M, 3, 3)
    # dR = R1^T @ R2 for each pair
    dR = np.einsum('mji,mjk->mik', R1, R2)      # (M, 3, 3)
    # Apply symmetry: S @ dR  →  (M, n_sym, 3, 3)
    sym_dR = np.einsum('sij,mjk->msik', sym_mats, dR)
    traces = np.trace(sym_dR, axis1=2, axis2=3)  # (M, n_sym)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angles.min(axis=1))         # (M,)

# Build lookup and arrays for valid medoid pixels
valid_iy, valid_ix = np.where(medoid_valid)
valid_mats = rec_medoid[medoid_valid]               # (N_valid, 3, 3)
n_valid = len(valid_iy)

idx_map = np.full((Ny, Nx), -1, dtype=np.int64)
for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    idx_map[iy, ix] = k

# Build neighbour pair lists for right and down
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

print(f"  Computing misorientations: {len(right_from)} right pairs, {len(down_from)} down pairs …")

# Compute in batches to avoid huge temporaries
BATCH = 2000
misori_right_vals = np.zeros(len(right_from))
for b in range(0, len(right_from), BATCH):
    e = min(b + BATCH, len(right_from))
    misori_right_vals[b:e] = misorientation_angle_batch_neighbours(
        valid_mats, right_from[b:e], right_to[b:e], SYM_MATS)

misori_down_vals = np.zeros(len(down_from))
for b in range(0, len(down_from), BATCH):
    e = min(b + BATCH, len(down_from))
    misori_down_vals[b:e] = misorientation_angle_batch_neighbours(
        valid_mats, down_from[b:e], down_to[b:e], SYM_MATS)

# Store in maps
misori_right = np.full((Ny, Nx), np.nan)
misori_down  = np.full((Ny, Nx), np.nan)
for i, (k, _) in enumerate(zip(right_from, right_to)):
    iy, ix = valid_iy[k], valid_ix[k]
    misori_right[iy, ix] = misori_right_vals[i]
for i, (k, _) in enumerate(zip(down_from, down_to)):
    iy, ix = valid_iy[k], valid_ix[k]
    misori_down[iy, ix] = misori_down_vals[i]

# KAM: average misorientation with all 4-connected neighbours
print("  Computing KAM map …")
kam_map = np.full((Ny, Nx), np.nan)
# Accumulate: for each edge, add the misorientation to both endpoints
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

for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    if kam_count[k] > 0:
        kam_map[iy, ix] = kam_sum[k] / kam_count[k]

print(f"  KAM range: [{np.nanmin(kam_map):.2f}°, {np.nanmax(kam_map):.2f}°]")

kam_rot, _  = rotate_map_45(kam_map, fill_value=np.nan)
kam_rot, _  = crop_to_valid(kam_rot, valid_rot)
print("  Saving rotated KAM map …")
# Apply inferno colormap with transparent background
kam_active = display_mask & sample_mask_rot & np.isfinite(kam_rot)
vmax_kam = 30.0
kam_norm = np.zeros_like(kam_rot)
kam_norm[kam_active] = np.clip(kam_rot[kam_active] / vmax_kam, 0, 1)
rgba_kam = plt.get_cmap('inferno')(kam_norm)  # (H, W, 4)
rgba_kam[~kam_active, 3] = 0.0

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(rgba_kam, interpolation='nearest')
#ax.set_xlim(350, w - 350)
#ax.set_ylim(h - 350, 350)   # y-axis is inverted for images
ax.axis('off')
fig.tight_layout()
fig.savefig(OUT_DIR / 'kam_map_rot.png', dpi=400, transparent=True)
plt.close(fig)
print(f"    Saved → {OUT_DIR / 'kam_map_rot.png'}")





# ═══════════════════════════════════════════════════════════════════════════
# PART D: GRAIN SEGMENTATION
#
# Strategy: build a grain-boundary map from the KAM + pairwise
# misorientation. Two adjacent pixels belong to the same grain if their
# misorientation is below KAM_MISORI_THRESHOLD_DEG.  Connected components
# give the grain labels.
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("PART D: Grain segmentation")
print("="*72)

# Use the misorientation maps already computed in PART C for grain segmentation.

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

n_valid = len(valid_iy)
uf = UnionFind(n_valid)

for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    # Right
    if ix + 1 < Nx:
        nidx = idx_map[iy, ix + 1]
        if nidx >= 0 and misori_right[iy, ix] < KAM_MISORI_THRESHOLD_DEG:
            uf.union(k, nidx)
    # Down
    if iy + 1 < Ny:
        nidx = idx_map[iy + 1, ix]
        if nidx >= 0 and misori_down[iy, ix] < KAM_MISORI_THRESHOLD_DEG:
            uf.union(k, nidx)

# Build grain label map
grain_labels = np.full((Ny, Nx), -1, dtype=np.int32)
root_to_label = {}
next_label = 0
for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    root = uf.find(k)
    if root not in root_to_label:
        root_to_label[root] = next_label
        next_label += 1
    grain_labels[iy, ix] = root_to_label[root]

n_grains_raw = next_label
print(f"  Raw grains: {n_grains_raw}")

# Remove grains smaller than GRAIN_MIN_SIZE
grain_sizes = np.bincount(grain_labels[grain_labels >= 0], minlength=n_grains_raw)
small_mask = grain_sizes < GRAIN_MIN_SIZE
for iy in range(Ny):
    for ix in range(Nx):
        if grain_labels[iy, ix] >= 0 and small_mask[grain_labels[iy, ix]]:
            grain_labels[iy, ix] = -1

# Re-label contiguously
unique_labels = np.unique(grain_labels[grain_labels >= 0])
relabel_map = {old: new for new, old in enumerate(unique_labels)}
for iy in range(Ny):
    for ix in range(Nx):
        if grain_labels[iy, ix] >= 0:
            grain_labels[iy, ix] = relabel_map[grain_labels[iy, ix]]

n_grains = len(unique_labels)
print(f"  Grains after removing small ones (< {GRAIN_MIN_SIZE} px): {n_grains}")


# ═══════════════════════════════════════════════════════════════════════════
# PART E: GROD + GOS USING GRAINWISE WEIGHTED-MEDOID REFERENCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART E: GROD and GOS maps")
print("=" * 72)

grain_ref_mats = np.full((n_grains, 3, 3), np.nan)
for gid in range(n_grains):
    gmask = grain_labels == gid
    if not np.any(gmask):
        continue
    grain_mean_coeffs = x_full[gmask].mean(axis=0)
    m_ref = weighted_medoid_from_coeffs(grain_mean_coeffs, rot_mats, SYM_MATS)
    if m_ref is not None:
        grain_ref_mats[gid] = m_ref

grod_map = np.full((Ny, Nx), np.nan)
for gid in range(n_grains):
    gmask = grain_labels == gid
    if not np.any(gmask):
        continue
    if not np.isfinite(grain_ref_mats[gid, 0, 0]):
        continue
    ref = grain_ref_mats[gid]
    pix_idx = idx_map[gmask]
    valid_pix = pix_idx >= 0
    if not np.any(valid_pix):
        continue
    pix_idx = pix_idx[valid_pix]
    R_pix = valid_mats[pix_idx]
    dR = np.einsum('ji,mjk->mik', ref, R_pix)             # ref^T @ R_pix
    sym_dR = np.einsum('sij,mjk->msik', SYM_MATS, dR)
    traces = np.trace(sym_dR, axis1=2, axis2=3)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    grod_vals = np.degrees(angles.min(axis=1))

    iy_gid, ix_gid = np.where(gmask)
    iy_sel = iy_gid[valid_pix]
    ix_sel = ix_gid[valid_pix]
    grod_map[iy_sel, ix_sel] = grod_vals

gos_per_grain = np.full(n_grains, np.nan)
for gid in range(n_grains):
    vals = grod_map[grain_labels == gid]
    vals = vals[np.isfinite(vals)]
    if vals.size > 0:
        gos_per_grain[gid] = vals.mean()

gos_map = np.full((Ny, Nx), np.nan)
for gid in range(n_grains):
    if np.isfinite(gos_per_grain[gid]):
        gos_map[grain_labels == gid] = gos_per_grain[gid]

print(f"  GROD range: [{np.nanmin(grod_map):.2f}°, {np.nanmax(grod_map):.2f}°]")
print(f"  GOS range:  [{np.nanmin(gos_map):.2f}°, {np.nanmax(gos_map):.2f}°]")

# Rotate and crop with the same geometry and masks used above
grod_rot, _ = rotate_map_45(grod_map, fill_value=np.nan)
gos_rot, _ = rotate_map_45(gos_map, fill_value=np.nan)
grain_labels_rot, _ = rotate_map_45(grain_labels.astype(np.float64), fill_value=np.nan, interp_order=0)
grod_rot, _ = crop_to_valid(grod_rot, valid_rot)
gos_rot, _ = crop_to_valid(gos_rot, valid_rot)
grain_labels_rot, _ = crop_to_valid(grain_labels_rot, valid_rot)

# Build thin, exact boundaries from neighbouring grain-label changes
# on the rotated high-resolution label field.
graiN_Omega_i = np.where(np.isfinite(grain_labels_rot), grain_labels_rot, -1).astype(np.int32)
valid_graiN_Omega = graiN_Omega_i >= 0
grain_boundary_rot = np.zeros_like(valid_graiN_Omega, dtype=bool)

right_diff = (
    valid_graiN_Omega[:, :-1]
    & valid_graiN_Omega[:, 1:]
    & (graiN_Omega_i[:, :-1] != graiN_Omega_i[:, 1:])
)
down_diff = (
    valid_graiN_Omega[:-1, :]
    & valid_graiN_Omega[1:, :]
    & (graiN_Omega_i[:-1, :] != graiN_Omega_i[1:, :])
)

# Mark only one side of each boundary pair to keep lines thin.
grain_boundary_rot[:, :-1] |= right_diff
grain_boundary_rot[:-1, :] |= down_diff

grod_active = display_mask & sample_mask_rot & np.isfinite(grod_rot)
gos_active = display_mask & sample_mask_rot & np.isfinite(gos_rot)

vmax_grod = 8##float(np.nanpercentile(grod_map[np.isfinite(grod_map)], 99)) if np.isfinite(grod_map).any() else 1.0
vmax_gos = 4#float(np.nanpercentile(gos_map[np.isfinite(gos_map)], 99)) if np.isfinite(gos_map).any() else 1.0
if vmax_grod <= 0:
    vmax_grod = 1.0
if vmax_gos <= 0:
    vmax_gos = 1.0

grod_norm = np.zeros_like(grod_rot)
gos_norm = np.zeros_like(gos_rot)
grod_norm[grod_active] = np.clip(grod_rot[grod_active] / vmax_grod, 0, 1)
gos_norm[gos_active] = np.clip(gos_rot[gos_active] / vmax_gos, 0, 1)

rgba_grod = plt.get_cmap('Reds')(grod_norm)
rgba_gos = plt.get_cmap('Reds')(gos_norm)
rgba_grod[~grod_active, 3] = 0.0
rgba_gos[~gos_active, 3] = 0.0

# Draw grain boundaries in black using KAM-derived boundary mask.
grod_boundaries = grain_boundary_rot & grod_active
gos_boundaries = grain_boundary_rot & gos_active
rgba_grod[grod_boundaries, :3] = 0.0
rgba_grod[grod_boundaries, 3] = 1.0
rgba_gos[gos_boundaries, :3] = 0.0
rgba_gos[gos_boundaries, 3] = 1.0

print("  Saving rotated GROD/GOS maps …")
fig, ax = plt.subplots(figsize=(7, 7))
h, w = rgba_grod.shape[:2]
ax.imshow(rgba_grod, interpolation='nearest')
#ax.set_xlim(350, w - 350)
#ax.set_ylim(h - 350, 350)
ax.axis('off')
fig.tight_layout()
fig.savefig(OUT_DIR / 'grod_map_rot.png', dpi=400, transparent=True)
plt.close(fig)
print(f"    Saved → {OUT_DIR / 'grod_map_rot.png'}")

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(rgba_gos, interpolation='nearest')
#ax.set_xlim(350, w - 350)
#ax.set_ylim(h - 350, 350)
ax.axis('off')
fig.tight_layout()
fig.savefig(OUT_DIR / 'gos_map_rot.png', dpi=400, transparent=True)
plt.close(fig)
print(f"    Saved → {OUT_DIR / 'gos_map_rot.png'}")

# Export colorbars separately (vertical + horizontal), no in-plot colorbar
print("  Exporting GROD/GOS colorbars …")
ticks_grod = [0.0, 0.5 * vmax_grod, vmax_grod]
ticks_gos = [0.0, 0.5 * vmax_gos, vmax_gos]
export_colorbar_pair(
    OUT_DIR,
    filename_base='misorientation_grod',
    cmap_name='Reds',
    vmin=0.0,
    vmax=vmax_grod,
    ticks=ticks_grod,
    formatter=lambda t: f"{t:.1f}°",
)
export_colorbar_pair(
    OUT_DIR,
    filename_base='misorientation_gos',
    cmap_name='Reds',
    vmin=0.0,
    vmax=vmax_gos,
    ticks=ticks_gos,
    formatter=lambda t: f"{t:.1f}°",
)


