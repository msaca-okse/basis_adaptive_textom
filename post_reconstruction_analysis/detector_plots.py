import os
import sys
import sys
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(project_root))

import h5py
import hdf5plugin
import numpy as np
from pathlib import Path
from diffractom import SinglePhaseForwardOperator
import yaml

import pyopencl.array as clarray
from diffractom import Grid
from diffractom import Material
import matplotlib.pyplot as plt

# ---------------------------
# CREATE RUN FOLDER
# ---------------------------

RECON_PATH  = Path('/work3/msaca/textomo_run_022/reconstruction.h5')
PIXEL_PATH  = Path('/work3/msaca/textomo_run_022/pixel_data.h5')
OUT_DIR = Path('/work3/msaca/textomo_run_022/detector_plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)


print("Saving results to:", OUT_DIR)

# ------------------    ---------
# CONFIG
# ---------------------------
config_path = project_root / "configs" / "aluminum_config_single_large.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

N_eta = cfg['N_eta']
N_Omega = cfg['N_Omega']
N_theta = cfg['N_theta']
Nx = cfg['Nx']
Ny = cfg['Ny']
My = cfg['My']
datapath = cfg["datapath"]
filename_integrated = cfg["integrated_file"]
cif_path = cfg["cif_path"]
cor_offset = cfg["cor_offset"]
wavelength_kev = cfg['wavelength']
min_two_theta = cfg['min_two_theta']
max_two_theta = cfg['max_two_theta']
reflection_cutoff = cfg['reflection_cutoff']

filename = datapath + filename_integrated





######### Make detector plot


with h5py.File(datapath + "scan-0053_pilatus.h5", 'r') as f:
    dataset = f['entry/instrument/pilatus/data']
    detector_image = dataset[3003*6:3003*6+4].sum(axis=0)


outpath = OUT_DIR / 'detector_frame_measured_transparent.png'
"""Plot one polar detector frame with transparent background and no colorbar."""
vmax = 40

fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(
    detector_image,
    interpolation='nearest',
    aspect='auto',
    cmap='cividis',
    vmin=0.0,
    vmax=vmax,
    origin='lower',
)


# Axis labels with requested symbols and units

ax.axis('off')

fig.savefig(outpath, bbox_inches='tight', dpi=500, transparent=True)
plt.close(fig)
print(f"Saved -> {outpath}")








# ---------------------------
# LOAD DATA
# ---------------------------
with h5py.File(filename, "r") as f:
    dset = f["I"]
    arr = np.empty(dset.shape, dtype=dset.dtype)
    dset.read_direct(arr)

arr = arr/arr.sum(axis=(0,1,2), keepdims=True)*arr.sum()/500
all_data_reshaped1 = arr.reshape((3000, My, N_eta*N_theta))
all_data_reshaped = all_data_reshaped1[::3000//N_Omega]

for i in range(1,3000//N_Omega):
    all_data_reshaped += all_data_reshaped1[i::3000//N_Omega]

# ---------------------------
# MATERIAL
# ---------------------------
cif_folder = Path(cif_path)
filenames = [p.name for p in cif_folder.glob("*.cif")]
path = cif_folder / filenames[0]

material = Material.from_cif(
    cif_path=path,
    wavelength_kev=wavelength_kev,
    min_two_theta=min_two_theta,
    max_two_theta=max_two_theta,
    intensity_cutoff_fraction=reflection_cutoff,
)

# print(f"Loading {RECON_PATH} …")
# with h5py.File(RECON_PATH, 'r') as f:
#     x_full = f['x'][()].astype(np.float64)      # (H, W, K)
#     rot_mats = f['orientations'][()]               # (K, 3, 3)
#     sigma_rad = float(f.attrs.get('sigma', 0.007))

# sigma_rad = 0.007
# grid = Grid.from_rotation_matrices(rot_mats, sigma=sigma_rad)
# print('Generated grid')

# ---------------------------
# OPERATOR
# ---------------------------
# op = SinglePhaseForwardOperator(cfg=cfg, material=material, grid=grid, max_gb=1.0,
#                 verbose=True, normalized=True)

# queue = op.queue
# Nx, Ny, My, K = op.Nx, op.Ny, op.My, op.K

# # ---------------------------
# # FORWARD PROJECTION FROM RECONSTRUCTION (OpenCL, Fortran order)
# # ---------------------------
# x_recon = np.asfortranarray(x_full[::-1, ::-1], dtype=np.float32)
# if x_recon.shape == (op.Nx, op.Ny, op.K):
#     pass
# else:
#     raise ValueError(
#         f"Unexpected reconstruction shape {x_recon.shape}, expected "
#         f"(Nx, Ny, My, K)=({op.Nx}, {op.Ny}, {op.My}, {op.K}) or (Ny, Nx, K)."
#     )

# x_gpu = clarray.to_device(queue, x_recon)

# prediction = op.direct(x_gpu)
# prediction_cpu = prediction.get().reshape((N_Omega, My, N_eta, N_theta))

# Actual detector data in polar bins
measured_cpu = all_data_reshaped.reshape((N_Omega, My, N_eta, N_theta))


def plot_polar_detector_frame(frame, outpath, theta_deg, vmax):
    """Plot one polar detector frame with transparent background and no colorbar."""
    vmax = max(float(vmax), 1e-12)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ax.imshow(
        frame,
        interpolation='nearest',
        aspect='auto',
        cmap='cividis',
        vmin=0.0,
        vmax=vmax,
        origin='lower',
    )

    nx_plot = frame.shape[1]
    for x in np.arange(0.5, nx_plot - 0.5, 1):
        ax.axvline(x, color='white', linewidth=0.8, alpha=0.4)

    # Axis labels with requested symbols and units
    ax.set_ylabel(r'$\eta$ (°)', fontsize=16)
    ax.set_xlabel(r'$2\theta$ (°)', fontsize=16)

    ny_plot = frame.shape[0]
    ax.set_yticks(np.linspace(0, ny_plot - 1, 9))
    ax.set_yticklabels([f"{v:.0f}" for v in np.linspace(0, 360, 9)])

    ax.set_xticks(np.arange(nx_plot))
    ax.set_xticklabels([f"{v:.1f}" for v in theta_deg[:nx_plot]])

    fig.savefig(outpath, bbox_inches='tight', dpi=500, transparent=True)
    plt.close(fig)
    print(f"Saved -> {outpath}")


# Requested scattering-angle labels
theta_rad = np.array([0.15253486, 0.17618918, 0.24949357, 0.29284424, 0.3858888])
theta_deg = np.degrees(theta_rad)
if theta_deg.size != N_theta:
    theta_deg = np.linspace(min_two_theta, max_two_theta, N_theta)

ix_show = 50
irot_show = 50

# pred_frame = prediction_cpu[irot_show-5:irot_show+5, ix_show-5:ix_show+5].mean(axis=(0,1))
# meas_frame = measured_cpu[irot_show-5:irot_show+5, ix_show-5:ix_show+5].mean(axis=(0,1))
#pred_frame = prediction_cpu[:,ix_show].mean(axis=0)
#meas_frame = measured_cpu[:,ix_show].mean(axis=0)
#pred_frame = prediction_cpu[0,90]
meas_frame = measured_cpu[0,90]
#shared_vmax = float(np.nanpercentile(np.concatenate([pred_frame.ravel(), meas_frame.ravel()]), 99.95))
shared_vmax = 40

# plot_polar_detector_frame(
#     pred_frame,
#     OUT_DIR / 'detector_frame_predicted_polar_transparent.png',
#     theta_deg,
#     shared_vmax,
# )
plot_polar_detector_frame(
    meas_frame,
    OUT_DIR / 'detector_frame_measured_polar_transparent.png',
    theta_deg,
    shared_vmax,
)

