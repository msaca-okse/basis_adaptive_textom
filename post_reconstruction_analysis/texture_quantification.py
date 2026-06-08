"""
Texture quantification script.

Extracts quantitative statistics from the reconstructed texture-tomography
data and produces both text summaries and publication-ready plots.

Outputs (all saved to OUT_DIR):
  1. GOS statistics: histogram, mean/median, GOS vs grain size
  2. Texture strength: ODF maximum, texture index J
  3. Intragranular orientation spread: max misorientation, 95th %, std dev
  4. Grain size statistics: distribution, mean, equivalent diameter
  5. Fiber fraction: % of grains with <001> within 5°/10° of rod axis

Reuses data-loading and grain-segmentation logic from auxillary_plots.py
and plot_odf_density.py.
"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.spatial.transform import Rotation as ScipyR

from orix.quaternion import Rotation, Orientation, von_mises
from orix.quaternion.symmetry import Oh
from orix.vector import Vector3d

# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
RECON_PATH  = Path('/work3/msaca/textomo_run_022/reconstruction.h5')
PIXEL_PATH  = Path('/work3/msaca/textomo_run_022/pixel_data.h5')
OUT_DIR     = Path('/work3/msaca/textomo_run_022/texture_quantification')
OUT_DIR.mkdir(parents=True, exist_ok=True)

KAM_MISORI_THRESHOLD_DEG = 5.0
GRAIN_MIN_SIZE           = 10
TOP_K_MEDOID             = 24
SPATIAL_MASK_THRESHOLD   = 0.002

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print(f"Loading {RECON_PATH} …")
with h5py.File(RECON_PATH, 'r') as f:
    x_full = f['x'][()].astype(np.float64)[5:-5, 5:-5]
    rot_mats = f['orientations'][()]               # (K, 3, 3)
    sigma_rad = float(f.attrs.get('sigma', 0.007))

Ny, Nx, K = x_full.shape
sum_coeffs = x_full.sum(axis=2)
spatial_mask = sum_coeffs > SPATIAL_MASK_THRESHOLD

print(f"  Volume shape: ({Ny}, {Nx}, {K}),  σ = {sigma_rad:.4f} rad")
print(f"  Active pixels: {int(spatial_mask.sum())}")

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
SYM_MATS = np.array(Rotation(Oh).to_matrix())  # (48, 3, 3)


def weighted_medoid_from_coeffs(coeffs, mats, sym_mats, top_k=TOP_K_MEDOID):
    """Return weighted medoid rotation matrix from one coefficient vector."""
    w = np.asarray(coeffs, dtype=np.float64)
    w_pos = np.maximum(w, 0.0)
    nz = np.flatnonzero(w_pos > 0)
    if nz.size == 0:
        return None
    if top_k is not None and nz.size > top_k:
        sel = nz[np.argsort(w_pos[nz])[-top_k:]]
    else:
        sel = nz
    w_sel = w_pos[sel]
    mats_sel = mats[sel]
    n_sel = mats_sel.shape[0]
    if n_sel == 1:
        return mats_sel[0]
    equivs = np.einsum('sij,kjl->ksil', sym_mats, mats_sel)
    traces = np.einsum('iab,jsab->ijs', mats_sel, equivs)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    dmin = angles.min(axis=-1)
    costs = dmin @ w_sel
    return mats_sel[int(np.argmin(costs))]


def misorientation_angle_batch(mats, idx_from, idx_to, sym_mats):
    """Misorientation angles in degrees for pairs (idx_from[i], idx_to[i])."""
    R1 = mats[idx_from]
    R2 = mats[idx_to]
    dR = np.einsum('mji,mjk->mik', R1, R2)
    sym_dR = np.einsum('sij,mjk->msik', sym_mats, dR)
    traces = np.trace(sym_dR, axis1=2, axis2=3)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angles.min(axis=1))


# ═══════════════════════════════════════════════════════════════════════════
# MEDOID ORIENTATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing weighted-medoid orientations per pixel …")
rec_medoid = np.full((Ny, Nx, 3, 3), np.nan)
iy_all, ix_all = np.where(spatial_mask)
n_active = len(iy_all)
for i, (iy, ix) in enumerate(zip(iy_all, ix_all), start=1):
    if i % 2000 == 0:
        print(f"  {i}/{n_active}")
    m = weighted_medoid_from_coeffs(x_full[iy, ix], rot_mats, SYM_MATS)
    if m is not None:
        rec_medoid[iy, ix] = m
medoid_valid = np.isfinite(rec_medoid[..., 0, 0])

# ═══════════════════════════════════════════════════════════════════════════
# KAM + GRAIN SEGMENTATION  (same logic as auxillary_plots.py)
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing misorientations and KAM …")
valid_iy, valid_ix = np.where(medoid_valid)
valid_mats = rec_medoid[medoid_valid]
n_valid = len(valid_iy)

idx_map = np.full((Ny, Nx), -1, dtype=np.int64)
for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    idx_map[iy, ix] = k

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

misori_right = np.full((Ny, Nx), np.nan)
misori_down  = np.full((Ny, Nx), np.nan)
for i, (k, _) in enumerate(zip(right_from, right_to)):
    misori_right[valid_iy[k], valid_ix[k]] = misori_right_vals[i]
for i, (k, _) in enumerate(zip(down_from, down_to)):
    misori_down[valid_iy[k], valid_ix[k]] = misori_down_vals[i]

# KAM
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

# ── Grain segmentation (Union-Find) ──────────────────────────────────────
print("\nGrain segmentation …")


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


uf = UnionFind(n_valid)
for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    if ix + 1 < Nx:
        nidx = idx_map[iy, ix + 1]
        if nidx >= 0 and misori_right[iy, ix] < KAM_MISORI_THRESHOLD_DEG:
            uf.union(k, nidx)
    if iy + 1 < Ny:
        nidx = idx_map[iy + 1, ix]
        if nidx >= 0 and misori_down[iy, ix] < KAM_MISORI_THRESHOLD_DEG:
            uf.union(k, nidx)

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
grain_sizes_raw = np.bincount(grain_labels[grain_labels >= 0], minlength=n_grains_raw)
small_mask = grain_sizes_raw < GRAIN_MIN_SIZE
for iy in range(Ny):
    for ix in range(Nx):
        if grain_labels[iy, ix] >= 0 and small_mask[grain_labels[iy, ix]]:
            grain_labels[iy, ix] = -1

unique_labels = np.unique(grain_labels[grain_labels >= 0])
relabel_map = {old: new for new, old in enumerate(unique_labels)}
for iy in range(Ny):
    for ix in range(Nx):
        if grain_labels[iy, ix] >= 0:
            grain_labels[iy, ix] = relabel_map[grain_labels[iy, ix]]

n_grains = len(unique_labels)
grain_sizes = np.bincount(grain_labels[grain_labels >= 0], minlength=n_grains)
print(f"  Raw grains: {n_grains_raw}")
print(f"  Grains after removing small ones (< {GRAIN_MIN_SIZE} px): {n_grains}")

# ═══════════════════════════════════════════════════════════════════════════
# GROD / GOS  (same as auxillary_plots.py Part E)
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing GROD and GOS …")
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
    dR = np.einsum('ji,mjk->mik', ref, R_pix)
    sym_dR = np.einsum('sij,mjk->msik', SYM_MATS, dR)
    traces = np.trace(sym_dR, axis1=2, axis2=3)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    grod_vals = np.degrees(angles.min(axis=1))

    iy_gid, ix_gid = np.where(gmask)
    iy_sel = iy_gid[valid_pix]
    ix_sel = ix_gid[valid_pix]
    grod_map[iy_sel, ix_sel] = grod_vals

# GOS per grain + intragranular spread statistics
gos_per_grain         = np.full(n_grains, np.nan)
max_misori_per_grain  = np.full(n_grains, np.nan)
p95_misori_per_grain  = np.full(n_grains, np.nan)
std_misori_per_grain  = np.full(n_grains, np.nan)

for gid in range(n_grains):
    vals = grod_map[grain_labels == gid]
    vals = vals[np.isfinite(vals)]
    if vals.size > 0:
        gos_per_grain[gid]        = vals.mean()
        max_misori_per_grain[gid] = vals.max()
        p95_misori_per_grain[gid] = np.percentile(vals, 95)
        std_misori_per_grain[gid] = vals.std()

print(f"  GOS range: [{np.nanmin(gos_per_grain):.2f}°, {np.nanmax(gos_per_grain):.2f}°]")

# ═══════════════════════════════════════════════════════════════════════════
# FIBER FRACTION: <001> within 5° / 10° of rod axis (Z)
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing <001> fiber fraction …")
# For each grain, get the reference orientation and compute the minimum
# angle between <001> crystal direction and the sample Z-axis.
rod_axis = Vector3d([0, 0, 1])
crystal_001 = Vector3d([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [-1, 0, 0], [0, -1, 0], [0, 0, -1]])

grain_fiber_angle   = np.full(n_grains, np.nan)
grain_fiber_angle_y = np.full(n_grains, np.nan)
grain_fiber_angle_x = np.full(n_grains, np.nan)
for gid in range(n_grains):
    if not np.isfinite(grain_ref_mats[gid, 0, 0]):
        continue
    ori = Orientation.from_matrix(grain_ref_mats[gid].reshape(1, 3, 3), symmetry=Oh)
    # Map crystal <001> directions to sample frame
    sample_dirs = (~ori).outer(crystal_001)  # shape: (1, 6)
    #sample_dirs = ori.outer(crystal_001)
    dirs_flat = sample_dirs.flatten()
    # Minimum angle to each sample axis (best alignment)
    grain_fiber_angle[gid]   = np.degrees(np.arccos(np.clip(
        np.abs(dirs_flat.dot(rod_axis)).max(), -1, 1)))
    grain_fiber_angle_y[gid] = np.degrees(np.arccos(np.clip(
        np.abs(dirs_flat.dot(Vector3d([0, 1, 0]))).max(), -1, 1)))
    grain_fiber_angle_x[gid] = np.degrees(np.arccos(np.clip(
        np.abs(dirs_flat.dot(Vector3d([1, 0, 0]))).max(), -1, 1)))

valid_fiber = np.isfinite(grain_fiber_angle)
n_valid_fiber = int(valid_fiber.sum())
frac_5  = float((grain_fiber_angle[valid_fiber] <= 5.0).sum()) / n_valid_fiber * 100
frac_10 = float((grain_fiber_angle[valid_fiber] <= 10.0).sum()) / n_valid_fiber * 100
frac_15 = float((grain_fiber_angle[valid_fiber] <= 15.0).sum()) / n_valid_fiber * 100

valid_fiber_y = np.isfinite(grain_fiber_angle_y)
n_valid_fiber_y = int(valid_fiber_y.sum())
frac_5_y  = float((grain_fiber_angle_y[valid_fiber_y] <= 5.0).sum()) / n_valid_fiber_y * 100
frac_10_y = float((grain_fiber_angle_y[valid_fiber_y] <= 10.0).sum()) / n_valid_fiber_y * 100
frac_15_y = float((grain_fiber_angle_y[valid_fiber_y] <= 15.0).sum()) / n_valid_fiber_y * 100

valid_fiber_x = np.isfinite(grain_fiber_angle_x)
n_valid_fiber_x = int(valid_fiber_x.sum())
frac_5_x  = float((grain_fiber_angle_x[valid_fiber_x] <= 5.0).sum()) / n_valid_fiber_x * 100
frac_10_x = float((grain_fiber_angle_x[valid_fiber_x] <= 10.0).sum()) / n_valid_fiber_x * 100
frac_15_x = float((grain_fiber_angle_x[valid_fiber_x] <= 15.0).sum()) / n_valid_fiber_x * 100

print(f"  Grains with <001> within  5° of Z: {frac_5:.1f}%")
print(f"  Grains with <001> within 10° of Z: {frac_10:.1f}%")
print(f"  Grains with <001> within 15° of Z: {frac_15:.1f}%")
print(f"  Grains with <001> within  5° of Y: {frac_5_y:.1f}%")
print(f"  Grains with <001> within 10° of Y: {frac_10_y:.1f}%")
print(f"  Grains with <001> within 15° of Y: {frac_15_y:.1f}%")
print(f"  Grains with <001> within  5° of X: {frac_5_x:.1f}%")
print(f"  Grains with <001> within 10° of X: {frac_10_x:.1f}%")
print(f"  Grains with <001> within 15° of X: {frac_15_x:.1f}%")

# ── Uniform (random) reference for <001>||Z alignment ────────────────────
print("\nComputing uniform random reference for <001> fiber …")
_rng = np.random.default_rng(0)
_v = _rng.normal(size=(1_000_000, 3))
_v /= np.linalg.norm(_v, axis=1, keepdims=True)
angle_uniform = np.degrees(np.arccos(np.clip(np.max(np.abs(_v), axis=1), -1, 1)))
uniform_frac_5  = 100 * float(np.mean(angle_uniform <= 5.0))
uniform_frac_10 = 100 * float(np.mean(angle_uniform <= 10.0))
uniform_frac_15 = 100 * float(np.mean(angle_uniform <= 15.0))
uniform_mean    = float(angle_uniform.mean())
uniform_median  = float(np.median(angle_uniform))
print(f"  Uniform: <001> within  5° of Z: {uniform_frac_5:.1f}%")
print(f"  Uniform: <001> within 10° of Z: {uniform_frac_10:.1f}%")
print(f"  Uniform: <001> within 15° of Z: {uniform_frac_15:.1f}%")
print(f"  Uniform: mean deviation = {uniform_mean:.2f}°, median = {uniform_median:.2f}°")


# ═══════════════════════════════════════════════════════════════════════════
# GRAIN SIZE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\nGrain size statistics …")
grain_eq_diameter = 2.0 * np.sqrt(grain_sizes / np.pi)  # equiv. circular diameter in pixels
mean_size    = float(grain_sizes.mean())
median_size  = float(np.median(grain_sizes))
std_size     = float(grain_sizes.std())
mean_eq_diam = float(grain_eq_diameter.mean())
median_eq_diam = float(np.median(grain_eq_diameter))
print(f"  Number of grains: {n_grains}")
print(f"  Mean grain area:   {mean_size:.1f} px")
print(f"  Median grain area: {median_size:.1f} px")
print(f"  Std grain area:    {std_size:.1f} px")
print(f"  Mean equiv. diameter:   {mean_eq_diam:.2f} px")
print(f"  Median equiv. diameter: {median_eq_diam:.2f} px")

# ═══════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════
report_path = OUT_DIR / 'texture_quantification_report.txt'
print(f"\nWriting text report to {report_path} …")

valid_gos = gos_per_grain[np.isfinite(gos_per_grain)]
valid_max_misori = max_misori_per_grain[np.isfinite(max_misori_per_grain)]
valid_p95 = p95_misori_per_grain[np.isfinite(p95_misori_per_grain)]
valid_std = std_misori_per_grain[np.isfinite(std_misori_per_grain)]

lines = []
lines.append("=" * 72)
lines.append("TEXTURE QUANTIFICATION REPORT")
lines.append("=" * 72)
lines.append(f"Data: {RECON_PATH}")
lines.append(f"Volume shape: ({Ny}, {Nx}, {K})")
lines.append(f"Sigma: {sigma_rad:.4f} rad ({np.degrees(sigma_rad):.2f} deg)")
lines.append(f"Active pixels: {int(spatial_mask.sum())}")
lines.append(f"Misorientation threshold for grain boundaries: {KAM_MISORI_THRESHOLD_DEG}°")
lines.append(f"Minimum grain size: {GRAIN_MIN_SIZE} px")
lines.append("")

lines.append("-" * 72)
lines.append("1. GRAIN SIZE STATISTICS")
lines.append("-" * 72)
lines.append(f"  Number of grains:       {n_grains}")
lines.append(f"  Mean grain area:        {mean_size:.1f} px")
lines.append(f"  Median grain area:      {median_size:.1f} px")
lines.append(f"  Std grain area:         {std_size:.1f} px")
lines.append(f"  Min grain area:         {int(grain_sizes.min())} px")
lines.append(f"  Max grain area:         {int(grain_sizes.max())} px")
lines.append(f"  Mean equiv. diameter:   {mean_eq_diam:.2f} px")
lines.append(f"  Median equiv. diameter: {median_eq_diam:.2f} px")
lines.append("")

lines.append("-" * 72)
lines.append("2. GRAIN ORIENTATION SPREAD (GOS)")
lines.append("-" * 72)
lines.append(f"  Mean GOS:   {float(valid_gos.mean()):.2f}°")
lines.append(f"  Median GOS: {float(np.median(valid_gos)):.2f}°")
lines.append(f"  Std GOS:    {float(valid_gos.std()):.2f}°")
lines.append(f"  Min GOS:    {float(valid_gos.min()):.2f}°")
lines.append(f"  Max GOS:    {float(valid_gos.max()):.2f}°")
lines.append("")

lines.append("-" * 72)
lines.append("3. INTRAGRANULAR ORIENTATION SPREAD")
lines.append("-" * 72)
lines.append(f"  Max misorientation per grain:  mean={float(valid_max_misori.mean()):.2f}°, "
             f"median={float(np.median(valid_max_misori)):.2f}°")
lines.append(f"  95th %ile misorientation:      mean={float(valid_p95.mean()):.2f}°, "
             f"median={float(np.median(valid_p95)):.2f}°")
lines.append(f"  Std of misorientation:         mean={float(valid_std.mean()):.2f}°, "
             f"median={float(np.median(valid_std)):.2f}°")
lines.append("")

lines.append("-" * 72)
lines.append("4. <001> FIBER TEXTURE (alignment with sample axes)")
lines.append("-" * 72)
lines.append(f"  {'':30s}  {'Measured':>12s}  {'Uniform random':>14s}")
lines.append("")
lines.append("  -- <001> || Z (rod axis) --")
lines.append(f"  {'<001> within  5° of Z':30s}  {frac_5:>11.1f}%  {uniform_frac_5:>13.1f}%")
lines.append(f"  {'<001> within 10° of Z':30s}  {frac_10:>11.1f}%  {uniform_frac_10:>13.1f}%")
lines.append(f"  {'<001> within 15° of Z':30s}  {frac_15:>11.1f}%  {uniform_frac_15:>13.1f}%")
lines.append(f"  {'Mean deviation from Z':30s}  {float(grain_fiber_angle[valid_fiber].mean()):>11.2f}°  {uniform_mean:>13.2f}°")
lines.append(f"  {'Median deviation from Z':30s}  {float(np.median(grain_fiber_angle[valid_fiber])):>11.2f}°  {uniform_median:>13.2f}°")
lines.append("")
lines.append("  -- <001> || Y --")
lines.append(f"  {'<001> within  5° of Y':30s}  {frac_5_y:>11.1f}%  {uniform_frac_5:>13.1f}%")
lines.append(f"  {'<001> within 10° of Y':30s}  {frac_10_y:>11.1f}%  {uniform_frac_10:>13.1f}%")
lines.append(f"  {'<001> within 15° of Y':30s}  {frac_15_y:>11.1f}%  {uniform_frac_15:>13.1f}%")
lines.append(f"  {'Mean deviation from Y':30s}  {float(grain_fiber_angle_y[valid_fiber_y].mean()):>11.2f}°  {uniform_mean:>13.2f}°")
lines.append(f"  {'Median deviation from Y':30s}  {float(np.median(grain_fiber_angle_y[valid_fiber_y])):>11.2f}°  {uniform_median:>13.2f}°")
lines.append("")
lines.append("  -- <001> || X --")
lines.append(f"  {'<001> within  5° of X':30s}  {frac_5_x:>11.1f}%  {uniform_frac_5:>13.1f}%")
lines.append(f"  {'<001> within 10° of X':30s}  {frac_10_x:>11.1f}%  {uniform_frac_10:>13.1f}%")
lines.append(f"  {'<001> within 15° of X':30s}  {frac_15_x:>11.1f}%  {uniform_frac_15:>13.1f}%")
lines.append(f"  {'Mean deviation from X':30s}  {float(grain_fiber_angle_x[valid_fiber_x].mean()):>11.2f}°  {uniform_mean:>13.2f}°")
lines.append(f"  {'Median deviation from X':30s}  {float(np.median(grain_fiber_angle_x[valid_fiber_x])):>11.2f}°  {uniform_median:>13.2f}°")
lines.append(f"  (Grain count: {n_valid_fiber}; uniform reference: 1 000 000 Monte Carlo samples)")
lines.append("")

lines.append("-" * 72)
lines.append("5. KAM STATISTICS")
lines.append("-" * 72)
kam_vals = kam_map[np.isfinite(kam_map)]
lines.append(f"  Mean KAM:   {float(kam_vals.mean()):.2f}°")
lines.append(f"  Median KAM: {float(np.median(kam_vals)):.2f}°")
lines.append(f"  Std KAM:    {float(kam_vals.std()):.2f}°")
lines.append(f"  Min KAM:    {float(kam_vals.min()):.2f}°")
lines.append(f"  Max KAM:    {float(kam_vals.max()):.2f}°")
lines.append("")

lines.append("=" * 72)
report_text = "\n".join(lines)
print(report_text)

with open(report_path, 'w') as f:
    f.write(report_text + "\n")
print(f"  Saved → {report_path}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

# ── 1. GOS histogram ─────────────────────────────────────────────────────
print("\nPlotting GOS histogram …")
fig, ax = plt.subplots(figsize=(7, 5))
bins_gos = np.arange(0, float(valid_gos.max()) + 0.5, 0.25)
ax.hist(valid_gos, bins=bins_gos, edgecolor='black', linewidth=0.5,
        color='steelblue', alpha=0.85)
ax.axvline(valid_gos.mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean = {valid_gos.mean():.2f}°')
ax.axvline(np.median(valid_gos), color='orange', linestyle='--', linewidth=1.5,
           label=f'Median = {np.median(valid_gos):.2f}°')
ax.set_xlabel('Grain Orientation Spread (°)')
ax.set_ylabel('Number of grains')
ax.set_title('GOS distribution')
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / 'gos_histogram.png', dpi=300)
plt.close(fig)
print(f"  Saved → {OUT_DIR / 'gos_histogram.png'}")

# ── 2. GOS vs grain size ─────────────────────────────────────────────────
print("Plotting GOS vs grain size …")
fig, ax = plt.subplots(figsize=(7, 5))
valid_both = np.isfinite(gos_per_grain)
ax.scatter(grain_sizes[valid_both], gos_per_grain[valid_both],
           s=15, alpha=0.6, edgecolors='none', c='steelblue')
ax.set_xlabel('Grain area (px)')
ax.set_ylabel('GOS (°)')
ax.set_title('GOS vs grain size')
ax.set_xscale('log')
fig.tight_layout()
fig.savefig(OUT_DIR / 'gos_vs_grain_size.png', dpi=300)
plt.close(fig)
print(f"  Saved → {OUT_DIR / 'gos_vs_grain_size.png'}")

# ── 3. Grain size distribution ───────────────────────────────────────────
print("Plotting grain size distribution …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Area histogram
ax = axes[0]
max_area = int(np.percentile(grain_sizes, 99))
bins_area = np.linspace(0, max_area, 40)
ax.hist(grain_sizes, bins=bins_area, edgecolor='black', linewidth=0.5,
        color='teal', alpha=0.85)
ax.axvline(mean_size, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean = {mean_size:.1f} px')
ax.axvline(median_size, color='orange', linestyle='--', linewidth=1.5,
           label=f'Median = {median_size:.1f} px')
ax.set_xlabel('Grain area (px)')
ax.set_ylabel('Number of grains')
ax.set_title('Grain area distribution')
ax.legend()

# Equivalent diameter histogram
ax = axes[1]
max_diam = float(np.percentile(grain_eq_diameter, 99))
bins_diam = np.linspace(0, max_diam, 40)
ax.hist(grain_eq_diameter, bins=bins_diam, edgecolor='black', linewidth=0.5,
        color='darkorange', alpha=0.85)
ax.axvline(mean_eq_diam, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean = {mean_eq_diam:.2f} px')
ax.axvline(median_eq_diam, color='blue', linestyle='--', linewidth=1.5,
           label=f'Median = {median_eq_diam:.2f} px')
ax.set_xlabel('Equivalent diameter (px)')
ax.set_ylabel('Number of grains')
ax.set_title('Equivalent diameter distribution')
ax.legend()

fig.tight_layout()
fig.savefig(OUT_DIR / 'grain_size_distribution.png', dpi=300)
plt.close(fig)
print(f"  Saved → {OUT_DIR / 'grain_size_distribution.png'}")

# ── 4. Intragranular spread boxplots ─────────────────────────────────────
print("Plotting intragranular orientation spread …")
fig, ax = plt.subplots(figsize=(7, 5))
data_bp = [valid_gos, valid_max_misori, valid_p95, valid_std]
labels_bp = ['GOS (mean)', 'Max misori.', '95th %ile', 'Std misori.']
bp = ax.boxplot(data_bp, tick_labels=labels_bp, patch_artist=True,
                medianprops=dict(color='red', linewidth=1.5))
colors_bp = ['steelblue', 'coral', 'goldenrod', 'mediumseagreen']
for patch, c in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel('Misorientation (°)')
ax.set_title('Intragranular orientation spread statistics')
fig.tight_layout()
fig.savefig(OUT_DIR / 'intragranular_spread_boxplot.png', dpi=300)
plt.close(fig)
print(f"  Saved → {OUT_DIR / 'intragranular_spread_boxplot.png'}")

# ── 5. Fiber angle distributions (Z, Y, X) ───────────────────────────────
print("Plotting <001> fiber angle distributions …")
_fiber_axes = [
    ('Z', grain_fiber_angle,   valid_fiber,   frac_5,   frac_10,   frac_15,   n_valid_fiber),
    ('Y', grain_fiber_angle_y, valid_fiber_y, frac_5_y, frac_10_y, frac_15_y, n_valid_fiber_y),
    ('X', grain_fiber_angle_x, valid_fiber_x, frac_5_x, frac_10_x, frac_15_x, n_valid_fiber_x),
]
for _axis_name, _fiber_arr, _valid, _f5, _f10, _f15, _n_valid in _fiber_axes:
    _fiber_vals = _fiber_arr[_valid]
    fig, ax = plt.subplots(figsize=(7, 5))
    bins_fib = np.arange(0, float(_fiber_vals.max()) + 1, 1)
    bin_width = bins_fib[1] - bins_fib[0]

    ax.hist(_fiber_vals, bins=bins_fib, edgecolor='black', linewidth=0.5,
            color='mediumpurple', alpha=0.85, label='Measured')

    hist_uni, _ = np.histogram(angle_uniform, bins=bins_fib, density=True)
    uniform_scaled = hist_uni * _n_valid * bin_width
    ax.stairs(uniform_scaled, bins_fib, color='gray', linewidth=1.8,
              linestyle='--', label='Uniform random')

    ax.axvline(5.0, color='red', linestyle=':', linewidth=1.5,
               label=f'5°  meas={_f5:.1f}%  uni={uniform_frac_5:.1f}%')
    ax.axvline(10.0, color='orange', linestyle=':', linewidth=1.5,
               label=f'10° meas={_f10:.1f}%  uni={uniform_frac_10:.1f}%')
    ax.axvline(15.0, color='green', linestyle=':', linewidth=1.5,
               label=f'15° meas={_f15:.1f}%  uni={uniform_frac_15:.1f}%')
    ax.set_xlabel(f'Misorientation of <001> from {_axis_name} axis (°)')
    ax.set_ylabel('Number of grains')
    ax.set_title(f'<001>\u2225{_axis_name} fiber texture: deviation from {_axis_name} axis')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _fname = f'fiber_angle_distribution_{_axis_name}.png'
    fig.savefig(OUT_DIR / _fname, dpi=300)
    plt.close(fig)
    print(f"  Saved \u2192 {OUT_DIR / _fname}")

# ── 6. KAM histogram ─────────────────────────────────────────────────────
print("Plotting KAM histogram …")
fig, ax = plt.subplots(figsize=(7, 5))
bins_kam = np.arange(0, min(float(np.nanpercentile(kam_vals, 99.5)), 30) + 0.5, 0.5)
ax.hist(kam_vals, bins=bins_kam, edgecolor='black', linewidth=0.5,
        color='indianred', alpha=0.85)
ax.axvline(kam_vals.mean(), color='blue', linestyle='--', linewidth=1.5,
           label=f'Mean = {kam_vals.mean():.2f}°')
ax.axvline(np.median(kam_vals), color='green', linestyle='--', linewidth=1.5,
           label=f'Median = {np.median(kam_vals):.2f}°')
ax.set_xlabel('KAM (°)')
ax.set_ylabel('Number of pixels')
ax.set_title('KAM distribution')
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / 'kam_histogram.png', dpi=300)
plt.close(fig)
print(f"  Saved → {OUT_DIR / 'kam_histogram.png'}")

# ── 7. Fiber angle vs grain size ─────────────────────────────────────────
print("Plotting fiber angle vs grain size …")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(grain_sizes[valid_fiber], grain_fiber_angle[valid_fiber],
           s=15, alpha=0.6, edgecolors='none', c='mediumpurple')
ax.axhline(10.0, color='orange', linestyle='--', linewidth=1, alpha=0.7,
           label='10° threshold')
ax.set_xlabel('Grain area (px)')
ax.set_ylabel('<001>–Z deviation (°)')
ax.set_title('<001> fiber deviation vs grain size')
ax.set_xscale('log')
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / 'fiber_angle_vs_grain_size.png', dpi=300)
plt.close(fig)
print(f"  Saved → {OUT_DIR / 'fiber_angle_vs_grain_size.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print(f"All outputs saved to {OUT_DIR}")
print("=" * 72)
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")
