"""
Plot ODF density for reconstructed texture-tomography data using orix.

Produces both single-pixel and bulk-average ODF plots, plus:
  - Medoid orientation IPF maps (rotated 45°, corners masked)
  - KAM (Kernel Average Misorientation) map
  - Grain segmentation map

Data model
----------
Each grid orientation g_k carries a coefficient c_k that represents its
contribution to the ODF.  The full ODF is the weighted sum of von-Mises-
Fisher (vMF) kernels:

    f(g) = sum_k  c_k * vMF(g; g_k, alpha)

where alpha = 1 / (2 * sigma^2)  (Gaussian half-width sigma in radians).
"""

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

# ═══════════════════════════════════════════════════════════════════════════
# SETTINGS  — edit these to taste
# ═══════════════════════════════════════════════════════════════════════════
RECON_PATH  = Path('/work3/msaca/textomo_run_026/reconstruction.h5')

COEFF_THRESHOLD_BULK  = 7.5e-6
COEFF_THRESHOLD_PIXEL = 2e-5
SPATIAL_MASK_THRESHOLD = 0.015
NORMALIZE_COEFFS_BY_MAX_SUM = False

OUT_DIR = Path('/work3/msaca/textomo_run_026/texture_visualization')
OUT_DIR.mkdir(exist_ok=True)

# ── Smoothing parameters ──────────────────────────────────────────────────
SIGMA_VIZ_RODRIGUES_DEG = 5.0
PF_SIGMA_DEG            = 5.0
PF_RESOLUTION_DEG       = 3.0
IPF_SIGMA_DEG           = 3.0
IPF_RESOLUTION_DEG      = 0.55
EULER_SIGMA_DEG         = 5.0
EULER_RESOLUTION_DEG    = 5.0
EULER_PHI2_SECTIONS     = np.arange(0, 91, 5)

USE_SIGMA_FOR_ALPHA = True
ALPHA_OVERRIDE      = 500.0

FZ_GRID_RESOLUTION  = 7.0
FZ_DENSITY_THRESH   = 0.15

# Medoid computation
TOP_K = 24   # number of top orientations per pixel for medoid search

# KAM / grain segmentation
KAM_MISORI_THRESHOLD_DEG = 5.0   # threshold for grain boundary detection
GRAIN_MIN_SIZE           = 10     # minimum grain area in pixels

# ── KAM map visualisation ─────────────────────────────────────────────────
KAM_VMAX_DEG = 50.0      # vmax for KAM colorbar in degrees (set None for 99th-percentile)
KAM_LOGSCALE = True    # True  → colour scale is  log(KAM_LOG_EPS + x)
KAM_LOG_EPS  = 4.0      # ε in  log(ε + x);  only used when KAM_LOGSCALE = True


# ═══════════════════════════════════════════════════════════════════════════
# LOAD FULL-MAP DATA (always needed for medoid / IPF / KAM / grains)
# ═══════════════════════════════════════════════════════════════════════════
print(f"Loading {RECON_PATH} …")
with h5py.File(RECON_PATH, 'r') as f:
    x_full_raw = f['x'][()].astype(np.float64)[5:-5, 5:-5]
    coeff_scale = float(x_full_raw.sum(axis=2).max())
    if coeff_scale <= 0:
        raise ValueError("Coefficient scale is non-positive; cannot normalize.")
    x_full = (x_full_raw / coeff_scale) if NORMALIZE_COEFFS_BY_MAX_SUM else x_full_raw
    rot_mats = f['orientations'][()]               # (K, 3, 3)
    sigma_rad = float(f.attrs.get('sigma', 0.007))

Ny, Nx, K = x_full.shape
sum_coeffs = x_full.sum(axis=2)                    # (H, W)

filepath = OUT_DIR / f'sum_coeffs_histogram.png'
data = sum_coeffs.ravel()  # or arr.flatten()

# Optional: remove NaNs/Infs if present
data = data[np.isfinite(data)]

plt.figure()
plt.hist(data, bins=100)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(filepath, dpi=300)
plt.close()



if NORMALIZE_COEFFS_BY_MAX_SUM:
    coeff_threshold_bulk_eff = COEFF_THRESHOLD_BULK / coeff_scale
    coeff_threshold_pixel_eff = COEFF_THRESHOLD_PIXEL / coeff_scale
    spatial_mask_threshold_eff = SPATIAL_MASK_THRESHOLD / coeff_scale
else:
    coeff_threshold_bulk_eff = COEFF_THRESHOLD_BULK
    coeff_threshold_pixel_eff = COEFF_THRESHOLD_PIXEL
    spatial_mask_threshold_eff = SPATIAL_MASK_THRESHOLD

spatial_mask = sum_coeffs > spatial_mask_threshold_eff         # (H, W)

print(f"  Volume shape: ({Ny}, {Nx}, {K}),  σ = {sigma_rad:.4f} rad")
print(f"  Active pixels: {int(spatial_mask.sum())}")
print(f"  Coefficient scale: {coeff_scale:.6e}  (normalize={NORMALIZE_COEFFS_BY_MAX_SUM})")
print(f"  Effective thresholds: bulk={coeff_threshold_bulk_eff:.3e}, "
      f"pixel={coeff_threshold_pixel_eff:.3e}, mask={spatial_mask_threshold_eff:.3e}")

alpha = 1.0 / (2.0 * sigma_rad**2) if USE_SIGMA_FOR_ALPHA else ALPHA_OVERRIDE


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: run ODF plots for a set of coefficients
# ═══════════════════════════════════════════════════════════════════════════
def run_odf_plots(coeffs, rot_mats, sigma_rad, analysis_label,
                  coeff_threshold, out_dir, tag):
    """Generate Rodrigues, pole-figure, IPF, Euler, and FZ-density plots."""
    all_rots = Rotation.from_matrix(rot_mats)
    alpha_loc = 1.0 / (2.0 * sigma_rad**2) if USE_SIGMA_FOR_ALPHA else ALPHA_OVERRIDE

    # Filter
    mask = coeffs > coeff_threshold
    nz_idx  = np.where(mask)[0]
    nz_c    = coeffs[nz_idx]
    nz_rots = all_rots[nz_idx]

    if len(nz_idx) == 0:
        print(f"  [{tag}] No orientations above threshold – skipping.")
        return

    print(f"  [{tag}] {len(nz_idx)} orientations above {coeff_threshold:.1e}")
    odf_density = nz_c.copy()

    # ── FIG 1: 3-D Rodrigues scatter ─────────────────────────────────────
    nz_oris_fz = Orientation(nz_rots.data, symmetry=Oh).reduce()
    s_min, s_max = 3, 120
    sizes_3d = s_min + (s_max - s_min) * (odf_density / odf_density.max())

    fig3d = nz_oris_fz.scatter(
        projection='rodrigues', c=odf_density, cmap='inferno',
        s=sizes_3d, alpha=0.8, linewidths=0, return_figure=True,
        figure_kwargs={'figsize': (8, 7)},
    )
    ax3d = fig3d.axes[0]
    sm3d = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=odf_density.min(), vmax=odf_density.max()),
        cmap='inferno')
    sm3d.set_array([])
    fig3d.colorbar(sm3d, ax=ax3d, shrink=0.6, pad=0.05).set_label('ODF coefficient $c_k$')
    #ax3d.set_title(f'{analysis_label} — Rodrigues FZ scatter\n'
    #               f'σ = {sigma_rad:.4f} rad, {len(nz_idx)} orientations')
    fig3d.tight_layout()
    fig3d.savefig(out_dir / f'odf_rodrigues3d_{tag}.png', dpi=250)
    plt.close(fig3d)

    # ── FIG 2: Pole figures ──────────────────────────────────────────────
    nz_oris = Orientation(nz_rots.data, symmetry=Oh)
    hkl_labels  = ['{001}', '{011}', '{111}']
    hkl_vectors = [
        Vector3d([[1,0,0],[0,1,0],[0,0,1]]),
        Vector3d([[1,1,0],[1,0,1],[0,1,1],[-1,1,0],[-1,0,1],[0,-1,1]]),
        Vector3d([[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]]),
    ]
    hkl_vectors = [v.unit for v in hkl_vectors]

    fig_pf, axes_pf = plt.subplots(1, 3, figsize=(16, 5),
                                   subplot_kw={'projection': 'stereographic'})
    for ax, lab, hvec in zip(axes_pf, hkl_labels, hkl_vectors):
        v_sample = (~nz_oris).outer(hvec)
        v_flat   = v_sample.flatten()
        c_flat   = np.tile(nz_c, hvec.size) / hvec.size
        ax.pole_density_function(v_flat, sigma=PF_SIGMA_DEG,
                                 resolution=PF_RESOLUTION_DEG,
                                 weights=c_flat, colorbar=False, cmap='plasma')
        ax.set_title(f'${lab}$', fontsize=13)
        ax.set_labels('RD', 'TD')
    fig_pf.subplots_adjust(right=0.88)
    cax = fig_pf.add_axes([0.90, 0.18, 0.015, 0.64])
    sm = plt.cm.ScalarMappable(cmap='plasma'); sm.set_array([])
    fig_pf.colorbar(sm, cax=cax).set_label('Pole density (a.u.)')
    fig_pf.suptitle(f'{analysis_label} — Pole figures (σ={PF_SIGMA_DEG}°)', y=0.99)
    fig_pf.savefig(out_dir / f'odf_polefigure_{tag}.png', dpi=250, bbox_inches='tight')
    plt.close(fig_pf)

    # ── FIG 3: Inverse pole figures ──────────────────────────────────────
    sample_dirs = [Vector3d.xvector(), Vector3d.yvector(), Vector3d.zvector()]
    dir_labels  = ['X (RD)', 'Y (TD)', 'Z (ND)']
    fig_ipf, axes_ipf = plt.subplots(1, 3, figsize=(16, 5),
                                     subplot_kw={'projection': 'ipf', 'symmetry': Oh})
    for ax, d, lab in zip(axes_ipf, sample_dirs, dir_labels):
        ipf_dirs = (~nz_oris) * d
        ax.pole_density_function(ipf_dirs, sigma=IPF_SIGMA_DEG,
                                 resolution=IPF_RESOLUTION_DEG, weights=nz_c,
                                 colorbar=False, cmap='magma')
        ax.set_title(lab, fontsize=13)
    fig_ipf.subplots_adjust(right=0.88)
    cax = fig_ipf.add_axes([0.90, 0.18, 0.015, 0.64])
    sm = plt.cm.ScalarMappable(cmap='magma'); sm.set_array([])
    fig_ipf.colorbar(sm, cax=cax).set_label('Inverse pole density (a.u.)')
    fig_ipf.suptitle(f'{analysis_label} — IPFs (σ={IPF_SIGMA_DEG}°)', y=0.99)
    fig_ipf.savefig(out_dir / f'odf_ipf_{tag}.png', dpi=250, bbox_inches='tight')
    plt.close(fig_ipf)

    # ── FIG 4: Euler-angle ODF sections (Bunge) ─────────────────────────
    phi1_edges = np.arange(0, 90 + EULER_RESOLUTION_DEG, EULER_RESOLUTION_DEG)
    PHI_edges  = np.arange(0, 90 + EULER_RESOLUTION_DEG, EULER_RESOLUTION_DEG)
    phi2_edges = np.arange(0, 90 + EULER_RESOLUTION_DEG, EULER_RESOLUTION_DEG)
    phi1_c = 0.5*(phi1_edges[:-1]+phi1_edges[1:])
    PHI_c  = 0.5*(PHI_edges[:-1]+PHI_edges[1:])
    phi2_c = 0.5*(phi2_edges[:-1]+phi2_edges[1:])
    N1, N2, N3 = len(phi1_c), len(PHI_c), len(phi2_c)
    phi1_g, PHI_g, phi2_g = np.meshgrid(phi1_c, PHI_c, phi2_c, indexing='ij')
    euler_flat = np.column_stack([phi1_g.ravel(), PHI_g.ravel(), phi2_g.ravel()])
    grid_oris_euler = Rotation.from_euler(np.deg2rad(euler_flat))

    sigma_euler = np.radians(EULER_SIGMA_DEG)
    alpha_euler = 1.0 / (2.0 * sigma_euler**2)
    N_euler = grid_oris_euler.size
    REL_CHUNK = 500
    odf_euler = np.zeros(N_euler, dtype=np.float64)
    for start in range(0, N_euler, REL_CHUNK):
        end = min(start + REL_CHUNK, N_euler)
        g_chunk = grid_oris_euler[start:end]
        rel     = (~nz_rots).outer(g_chunk)
        vm_vals = von_mises(rel.flatten(), alpha_euler).reshape(nz_rots.size, end-start)
        odf_euler[start:end] = nz_c @ vm_vals
    odf_vol = odf_euler.reshape(N1, N2, N3)
    odf_vol /= odf_vol.max() + 1e-30

    phi2_sec_indices = [np.argmin(np.abs(phi2_c - p2)) for p2 in EULER_PHI2_SECTIONS]
    n_sections = len(phi2_sec_indices)
    n_cols = min(5, n_sections)
    n_rows = int(np.ceil(n_sections / n_cols))
    fig_eu, axes_eu = plt.subplots(n_rows, n_cols, figsize=(3.2*n_cols, 3*n_rows), squeeze=False)
    for i, (ax_flat, p2_idx) in enumerate(zip(axes_eu.flat, phi2_sec_indices)):
        section = odf_vol[:, :, p2_idx].T
        im = ax_flat.pcolormesh(phi1_edges, PHI_edges, section, cmap='hot',
                                vmin=0, vmax=1, shading='flat')
        ax_flat.set_aspect('equal')
        ax_flat.set_title(f'φ₂ = {phi2_c[p2_idx]:.0f}°', fontsize=9)
        ax_flat.set_xlabel('φ₁ (°)', fontsize=8)
        ax_flat.set_ylabel('Φ (°)', fontsize=8)
    for j in range(n_sections, n_rows*n_cols):
        axes_eu.flat[j].set_visible(False)
    fig_eu.subplots_adjust(right=0.88, hspace=0.45, wspace=0.35)
    cax = fig_eu.add_axes([0.90, 0.15, 0.015, 0.7])
    fig_eu.colorbar(im, cax=cax).set_label('Normalised ODF density')
    fig_eu.suptitle(f'{analysis_label} — Euler sections (σ={EULER_SIGMA_DEG}°)', y=1.01)
    fig_eu.savefig(out_dir / f'odf_euler_sections_{tag}.png', dpi=250, bbox_inches='tight')
    plt.close(fig_eu)

    # ── FIG 5: Smoothed FZ density ───────────────────────────────────────
    # from orix.sampling import get_sample_fundamental

    # sigma_viz = np.radians(SIGMA_VIZ_RODRIGUES_DEG)
    # alpha_viz = 1.0 / (2.0 * sigma_viz**2)

    # grid_rots = get_sample_fundamental(
    #     resolution=FZ_GRID_RESOLUTION,
    #     point_group=Oh
    # )

    # N_nz = nz_rots.size
    # N_grid = grid_rots.size
    # odf_grid = np.zeros(N_grid, dtype=np.float64)

    # for start in range(0, N_grid, REL_CHUNK):
    #     end = min(start + REL_CHUNK, N_grid)
    #     g_chunk = grid_rots[start:end]
    #     rel = (~nz_rots).outer(g_chunk)
    #     vm_vals = von_mises(rel.flatten(), alpha_viz).reshape(N_nz, end - start)
    #     odf_grid[start:end] = nz_c @ vm_vals

    # odf_grid /= odf_grid.max() + 1e-30

    # sig_mask = odf_grid > FZ_DENSITY_THRESH
    # dens_grid = odf_grid[sig_mask]
    # sizes_d = 3 + 57 * dens_grid

    # grid_oris_fz = Orientation(grid_rots[sig_mask].data, symmetry=Oh)

    # fig_vol = grid_oris_fz.scatter(
    #     projection='rodrigues',
    #     c=dens_grid,
    #     cmap='hot',
    #     s=sizes_d,
    #     alpha=0.7,
    #     linewidths=0,
    #     return_figure=True,
    #     figure_kwargs={'figsize': (8, 7)}
    # )

    # ax_vol = fig_vol.axes[0]
    # sm_vol = plt.cm.ScalarMappable(
    #     norm=mcolors.Normalize(0, 1),
    #     cmap='hot'
    # )
    # sm_vol.set_array([])

    # fig_vol.colorbar(sm_vol, ax=ax_vol, shrink=0.6, pad=0.05).set_label(
    #     'Normalised ODF density'
    # )
    # ax_vol.set_title(
    #     f'{analysis_label} — FZ density (σ_viz={SIGMA_VIZ_RODRIGUES_DEG}°)'
    # )
    # fig_vol.tight_layout()
    # fig_vol.savefig(out_dir / f'odf_rodrigues3d_density_{tag}.png', dpi=250)
    # plt.close(fig_vol)
    # print(f"  [{tag}] ODF plots saved.")



# ═══════════════════════════════════════════════════════════════════════════
# PART A.2: ODF PLOTS — BULK AVERAGE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("PART A.2: Bulk-average ODF plots")
print("="*72)
bulk_coeffs = x_full[spatial_mask].mean(axis=0)
bulk_sigma = 0.1   # empirical smoothing for bulk

fig_mask, ax_mask = plt.subplots(figsize=(5, 5))
ax_mask.imshow(spatial_mask, origin='lower', cmap='gray',interpolation='nearest')
#ax_mask.set_title(f'Spatial mask (sum > 0.002)\n{int(spatial_mask.sum())} pixels')
ax_mask.set_xlabel('col'); ax_mask.set_ylabel('row')
fig_mask.tight_layout()
fig_mask.savefig(OUT_DIR / 'sum_coeffs_mask.png', dpi=250)
plt.close(fig_mask)

run_odf_plots(bulk_coeffs, rot_mats, bulk_sigma,
              "Bulk average", coeff_threshold_bulk_eff/3, OUT_DIR, "bulk")

# ═══════════════════════════════════════════════════════════════════════════
# PART B: MEDOID ORIENTATION MAP (full volume)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("PART B: Computing weighted medoid orientation for each pixel")
print("="*72)

grid_mats = rot_mats   # (K, 3, 3)

# Symmetry matrices for Oh (48 elements)
SYM_MATS = np.array(Rotation(Oh).to_matrix())  # (48, 3, 3)
n_sym = SYM_MATS.shape[0]

rec_medoid = np.full((Ny, Nx, 3, 3), np.nan)
n_active_pixels = int(spatial_mask.sum())
print(f"  Computing medoid for {n_active_pixels} active pixels …")
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
        w_max = w.max()
        if w_max <= 0:
            continue

        # Select top-K orientations by weight
        top_idx = np.argsort(w)[-TOP_K:]
        top_idx = top_idx[w[top_idx] > 0]
        n_top = len(top_idx)

        if n_top <= 1:
            rec_medoid[iy, ix] = grid_mats[np.argmax(w)]
            continue

        wt = w[top_idx]
        top_mats = grid_mats[top_idx]                   # (n_top, 3, 3)
        # Symmetry equivalents only for the top-K (not all K)
        top_equivs = np.einsum('sij,kjl->ksil', SYM_MATS, top_mats)  # (n_top, n_sym, 3, 3)

        # Pairwise symmetry-reduced geodesic distance
        traces = np.einsum('iab,jsab->ijs', top_mats, top_equivs)
        angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
        min_angles = angles.min(axis=-1)

        # Weighted medoid
        costs = min_angles @ wt
        rec_medoid[iy, ix] = top_mats[np.argmin(costs)]

elapsed = time.time() - t0
print(f"  Done ({count} pixels, {elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# PART C: KAM (Kernel Average Misorientation) MAP
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("PART C: Computing KAM map")
print("="*72)

medoid_valid = np.isfinite(rec_medoid[..., 0, 0])

# Build lookup and arrays for valid medoid pixels
valid_iy, valid_ix = np.where(medoid_valid)
valid_mats = rec_medoid[medoid_valid]               # (N_valid, 3, 3)
n_valid = len(valid_iy)

idx_map = np.full((Ny, Nx), -1, dtype=np.int64)
for k, (iy, ix) in enumerate(zip(valid_iy, valid_ix)):
    idx_map[iy, ix] = k


def misorientation_angle_sym(R1, R2, sym_mats):
    """Compute symmetry-reduced misorientation angle between two rotation matrices.
    R1, R2: (3,3) rotation matrices. sym_mats: (n_sym, 3, 3).
    Returns angle in degrees."""
    dR = R1.T @ R2                                # (3, 3)
    # Apply all symmetry operators: S @ dR
    sym_dR = np.einsum('sij,jk->sik', sym_mats, dR)  # (n_sym, 3, 3)
    traces = np.trace(sym_dR, axis1=1, axis2=2)       # (n_sym,)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angles.min())


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
# PART E: 45° ROTATION + CORNER MASKING + VISUALISATION
#
# Rotate the pixel grid by 45° WITHOUT reinterpolation (just remap indices).
# Then mask out corners that were outside the original square image.
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("PART E: Rotated IPF maps, KAM map, grain map")
print("="*72)

import numpy as np
from scipy.ndimage import map_coordinates


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


# ── Build IPF color maps from medoid orientations ────────────────────────
# Vectorised IPF colouring: convert all valid medoid rotations at once
valid_oris_orix = Orientation.from_matrix(valid_mats, symmetry=Oh)

def compute_medoid_ipf_rgb(direction):
    """Compute IPF-coloured RGB image from medoid map (vectorised)."""
    ipfkey = orix_plot.IPFColorKeyTSL(symmetry.Oh, direction=Vector3d(direction))
    colors = ipfkey.orientation2color(valid_oris_orix)  # (n_valid, 3)
    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = colors.reshape(-1, 3)
    rgb = np.zeros((Ny, Nx, 3), dtype=np.float64)
    rgb[medoid_valid] = colors[:, :3]
    return rgb


print("  Computing IPF colours …")
ipf_x_rgb = compute_medoid_ipf_rgb([1, 0, 0])
ipf_y_rgb = compute_medoid_ipf_rgb([0, 1, 0])
ipf_z_rgb = compute_medoid_ipf_rgb([0, 0, 1])

# ── Rotate everything by 45° ────────────────────────────────────────────
print("  Rotating maps by 45° …")
ipf_x_rot, valid_rot = rotate_map_45(ipf_x_rgb, fill_value=0)
ipf_y_rot, _         = rotate_map_45(ipf_y_rgb, fill_value=0)
ipf_z_rot, _         = rotate_map_45(ipf_z_rgb, fill_value=0)
kam_rot, _            = rotate_map_45(kam_map, fill_value=np.nan)
# Use order=0 (nearest-neighbour) for grain labels to avoid fractional IDs
graiN_Omega, _          = rotate_map_45(grain_labels.astype(np.float64), fill_value=np.nan, interp_order=0)
# Rotate the sample mask so we know which rotated pixels are actual sample
sample_mask_rot, _    = rotate_map_45(medoid_valid.astype(np.float64), fill_value=0, interp_order=0)
sample_mask_rot       = sample_mask_rot > 0.5

# Crop to bounding box
ipf_x_rot, valid_crop = crop_to_valid(ipf_x_rot, valid_rot)
ipf_y_rot, _          = crop_to_valid(ipf_y_rot, valid_rot)
ipf_z_rot, _          = crop_to_valid(ipf_z_rot, valid_rot)
kam_rot, _             = crop_to_valid(kam_rot, valid_rot)
graiN_Omega, _           = crop_to_valid(graiN_Omega, valid_rot)
sample_mask_rot, _     = crop_to_valid(sample_mask_rot, valid_rot)
display_mask           = make_corner_mask(valid_crop)

# ── Save IPF maps ────────────────────────────────────────────────────────
print("  Saving rotated IPF maps …")
for rgb_rot, name, label in [
    (ipf_x_rot, 'medoid_ipf_x_rot45.png', 'IPF-X'),
    (ipf_y_rot, 'medoid_ipf_y_rot45.png', 'IPF-Y'),
    (ipf_z_rot, 'medoid_ipf_z_rot45.png', 'IPF-Z'),
]:
    # RGBA with transparent background for both corners AND inactive pixels
    rgba = np.ones(rgb_rot.shape[:2] + (4,), dtype=np.float64)
    rgba[..., :3] = rgb_rot
    # Make transparent: pixels outside the rotated diamond OR not part of the sample
    transparent = ~display_mask | ~sample_mask_rot
    rgba[transparent, 3] = 0

    fig, ax = plt.subplots(figsize=(7, 7))
    h, w = rgba.shape[:2]
    #ax.set_xlim(350, w - 350)
    #ax.set_ylim(h - 350, 350)   # y-axis is inverted for images
    ax.imshow(rgba, interpolation='nearest')
    #ax.set_title(f'Medoid {label} (45° rotated)', fontsize=12)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=400, transparent=True)
    plt.close(fig)
    print(f"    Saved → {OUT_DIR / name}")

# ── Save KAM map ────────────────────────────────────────────────────────
print("  Saving rotated KAM map …")
kam_display = kam_rot.copy()
kam_display[~display_mask] = np.nan

_vmax_deg = KAM_VMAX_DEG if KAM_VMAX_DEG is not None else float(np.nanpercentile(kam_display, 99))

if KAM_LOGSCALE:
    _kam_plot_data  = np.log(KAM_LOG_EPS + np.clip(kam_display, 0.0, _vmax_deg))
    _imshow_vmin    = float(np.log(KAM_LOG_EPS))
    _imshow_vmax    = float(np.log(KAM_LOG_EPS + _vmax_deg))
else:
    _kam_plot_data  = np.clip(kam_display, 0.0, _vmax_deg)
    _imshow_vmin, _imshow_vmax = 0.0, _vmax_deg

def _nice_round(v):
    """Round v to the nearest 1 / 2 / 5 × 10^n  ('nice' number)."""
    if v <= 0:
        return 0.0
    exp = np.floor(np.log10(v))
    frac = v / 10.0**exp
    nf = 1.0 if frac < 1.5 else (2.0 if frac < 3.5 else (5.0 if frac < 7.5 else 10.0))
    return nf * 10.0**exp

def _fmt_deg(v):
    """Format a (possibly fractional) degree value without ugly long decimals."""
    return f"{v:.3g}"

if KAM_LOGSCALE:
    _log_min = float(np.log(KAM_LOG_EPS))               # log(ε + 0)
    _log_max = float(np.log(KAM_LOG_EPS + _vmax_deg))   # log(ε + vmax)
    # raw degree values at 5 equally-spaced log positions
    _raw = np.clip(np.exp(np.linspace(_log_min, _log_max, 5)) - KAM_LOG_EPS,
                   0.0, _vmax_deg)
    # snap interior ticks to nice round numbers; keep 0 and vmax exact
    _tick_misoris = np.array(
        [0.0] + [_nice_round(v) for v in _raw[1:-1]] + [_vmax_deg]
    )
    _tick_positions = np.log(KAM_LOG_EPS + _tick_misoris)
else:
    _tick_positions = np.linspace(0.0, _vmax_deg, 5)
    _tick_misoris   = _tick_positions

_tick_labels = [_fmt_deg(v) for v in _tick_misoris]

import matplotlib as _mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable

with _mpl.rc_context({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Georgia'],
    'font.size': 16,            # base font size
}):
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(_kam_plot_data, cmap='inferno',
                   vmin=_imshow_vmin, vmax=_imshow_vmax, interpolation='nearest')
    ax.axis('off')

    _divider = _make_axes_locatable(ax)
    _cax     = _divider.append_axes("left", size="5%", pad=0.1)
    _cb      = fig.colorbar(im, cax=_cax)
    _cb.ax.yaxis.set_ticks_position('left')
    _cb.ax.yaxis.set_label_position('left')
    _cb.set_ticks(_tick_positions)
    _cb.set_ticklabels(_tick_labels)
    _cb.set_label('Misorientation (°)')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'kam_map_rot45.png', dpi=400, transparent=True)
    plt.close(fig)
print(f"    Saved → {OUT_DIR / 'kam_map_rot45.png'}")

# ── Save grain map ───────────────────────────────────────────────────────
print("  Saving rotated grain map …")
grain_display = graiN_Omega.copy()

# Create a random colour mapping for grains
rng = np.random.default_rng(42)
grain_colors = rng.random((n_grains, 3))

grain_rgb = np.zeros(grain_display.shape[:2] + (4,), dtype=np.float64)
valid_grain = display_mask & ~np.isnan(grain_display) & (grain_display >= 0)
gids = grain_display[valid_grain].astype(int)
grain_rgb[valid_grain, :3] = grain_colors[gids]
grain_rgb[valid_grain, 3] = 1.0

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(grain_rgb,interpolation='nearest')
#ax.set_xlim(350, w - 350)
#ax.set_ylim(h - 350, 350)   # y-axis is inverted for images
#ax.set_title(f'Grain segmentation ({n_grains} grains, 45° rotated)', fontsize=12)
ax.axis('off')
fig.tight_layout()
fig.savefig(OUT_DIR / 'grain_map_rot45.png', dpi=400, transparent=True)
plt.close(fig)
print(f"    Saved → {OUT_DIR / 'grain_map_rot45.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# PART F: Per-grain ODF plots (top 20 largest grains)
#
# For each grain: create a subdirectory with the full suite of ODF plots
# (Rodrigues scatter, pole figures, IPFs, Euler sections, FZ density)
# plus a grain-highlight image showing which grain is selected.
# Directories are numbered 01–20  (01 = largest grain).
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART F: Per-grain ODF plots (top 20 largest)")
print("=" * 72)

# Rank grains by pixel count (from NON-rotated grain_labels)
grain_pixel_counts = np.bincount(grain_labels[grain_labels >= 0], minlength=n_grains)
n_top_grains = min(5, n_grains)
top_grain_ids = np.argsort(grain_pixel_counts)[::-1][:n_top_grains]

# Deterministic gray levels for every grain: medium-to-dark, never white
rng_gray = np.random.default_rng(123)
gray_levels = rng_gray.uniform(0.25, 0.65, size=n_grains)

# Precompute per-pixel info on the rotated grain map
valid_px_rot = display_mask & np.isfinite(graiN_Omega) & (graiN_Omega >= 0)
gids_rot_all = np.full(graiN_Omega.shape, -1, dtype=int)
gids_rot_all[valid_px_rot] = graiN_Omega[valid_px_rot].astype(int)
grays_all = np.zeros(graiN_Omega.shape, dtype=np.float64)
grays_all[valid_px_rot] = gray_levels[gids_rot_all[valid_px_rot]]

# Precompute the highlight base image (all grains in gray, transparent bg)
highlight_base = np.zeros(graiN_Omega.shape[:2] + (4,), dtype=np.float64)
highlight_base[valid_px_rot, 0] = grays_all[valid_px_rot]
highlight_base[valid_px_rot, 1] = grays_all[valid_px_rot]
highlight_base[valid_px_rot, 2] = grays_all[valid_px_rot]
highlight_base[valid_px_rot, 3] = 1.0

t0_f = time.time()

for rank, grain_id in enumerate(top_grain_ids):
    rank_label = f"{rank + 1:02d}"
    grain_size = int(grain_pixel_counts[grain_id])
    print(f"\n  [{rank_label}/{n_top_grains:02d}] Grain #{grain_id}, {grain_size} pixels")

    # Create per-grain output directory
    grain_dir = OUT_DIR / f'grain_{rank_label}'
    grain_dir.mkdir(exist_ok=True)

    # ── Grain highlight image (rotated, transparent bg) ───────────────
    highlight = highlight_base.copy()
    sel = valid_px_rot & (gids_rot_all == grain_id)
    highlight[sel, 0] = 1.0
    highlight[sel, 1] = 0.0
    highlight[sel, 2] = 0.0

    fig_hl, ax_hl = plt.subplots(figsize=(7, 7))
    ax_hl.imshow(highlight, interpolation='nearest')
    #ax_hl.set_title(f'Grain #{grain_id}  (rank {rank + 1}, {grain_size} px)',
    #                fontsize=12)
    #ax_hl.set_xlim(350, w - 350)
    #ax_hl.set_ylim(h - 350, 350)   # y-axis is inverted for images
    ax_hl.axis('off')
    fig_hl.tight_layout()
    fig_hl.savefig(grain_dir / 'grain_highlight.png', dpi=250, transparent=True)
    plt.close(fig_hl)
    print(f"    Saved → grain_{rank_label}/grain_highlight.png")

    # ── Mean ODF coefficients for this grain (non-rotated) ────────────
    grain_pixel_mask = grain_labels == grain_id
    grain_coeffs = x_full[grain_pixel_mask].mean(axis=0)

    # ── Full ODF plot suite via run_odf_plots ─────────────────────────
    run_odf_plots(
        grain_coeffs, rot_mats, sigma_rad,
        f'Grain {rank_label} (#{grain_id}, {grain_size} px)',
        coeff_threshold_bulk_eff,
        grain_dir,
        'grain'
    )

elapsed_f = time.time() - t0_f
print(f"\n  Per-grain ODF plots done ({elapsed_f:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n✓ All plots saved to", OUT_DIR)
for p in sorted(OUT_DIR.glob('*.png')):
    print(f"  {p.name}")
# print(f"Medoid misorientation:  mean={mean_medoid:.2f}°,  median={median_medoid:.2f}°")