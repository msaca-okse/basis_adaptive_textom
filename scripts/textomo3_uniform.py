import os
import sys
import sys
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(project_root))


import h5py
import numpy as np
from pathlib import Path
from diffractom import SinglePhaseForwardOperator
from diffractom.operators.single_phase_forward_operator import estimate_L_power
import yaml
from diffractom import FISTAHuber
import pyopencl.array as clarray
from diffractom import Grid
from diffractom import Material
import matplotlib.pyplot as plt
from orix import plot
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree

# ---------------------------
# CREATE RUN FOLDER
# ---------------------------
base_run_dir = Path("/work3/msaca")

existing = sorted([p for p in base_run_dir.glob("textomo_run_*") if p.is_dir()])
if len(existing) == 0:
    run_id = 0
else:
    run_id = max(int(p.name.split("_")[-1]) for p in existing) + 1

run_dir = base_run_dir / f"textomo_run_{run_id:03d}"
run_dir.mkdir(parents=True, exist_ok=False)


print("Saving results to:", run_dir)

# ------------------    ---------
# CONFIG
# ---------------------------
config_path = project_root / "configs" / "aluminum_config_uniform.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

niter = 150
niter_L_estimate = 10
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

# ---------------------------
# LOAD DATA
# ---------------------------
all_data_reshaped = np.load('/work3/msaca/integrated_data_omega=500.npy')
all_data_reshaped = all_data_reshaped.reshape((N_Omega,500//N_Omega, My, N_eta, 540//N_eta, N_theta)).sum(axis=(1,4))
arr = all_data_reshaped
all_data_reshaped = all_data_reshaped.reshape(N_Omega, My, N_eta*N_theta)

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

# ---------------------------
# GRID
# ---------------------------

sigma_deg = 1.3
sigma_rad = sigma_deg/180*np.pi
grid = Grid.from_random_fundamental_zone(500000, "cubic", sigma_rad)
grid.prune_close_orientations(theta_deg=1.3, target=150000)
grid_mats = R.concatenate(grid.rotations_at_level(0)).as_matrix()  # (K, 3, 3)

print('Generated grid')

# ---------------------------
# OPERATOR
# ---------------------------
op = SinglePhaseForwardOperator(cfg=cfg, material=material, grid=grid, max_gb=0.2,
                verbose=True, normalized=True)


queue = op.queue
Nx, Ny, My, K = op.Nx, op.Ny, op.My, op.K

x_gpu = clarray.zeros(queue, (Nx, Ny, K), dtype=np.float32, order="F")

out_cpu = np.ascontiguousarray(all_data_reshaped, dtype=np.float32)
out_gpu = clarray.to_device(queue, out_cpu)

# ---------------------------
# SOLVE
# ---------------------------
norm_sq = estimate_L_power(op, niter=niter_L_estimate, seed=0, eps=1e-30, verbose=1)
print('Estimated operator norm')
solver = FISTAHuber(op, prox_kind="nonneg", lam=2.5e5, L=1.1*norm_sq, huber_delta=4000.0)

solver.run(x_gpu, out_gpu, niter=niter, verbose=1, diagnostics_interval=1)

prediction = op.direct(x_gpu)
prediction_cpu = prediction.get().reshape((N_Omega, My, N_eta, N_theta))


coeffs = x_gpu.get().transpose((0,1,2))[::-1,::-1]

# ---------------------------
# SAVE RECONSTRUCTION
# ---------------------------
reconstruction_path = run_dir / "reconstruction.h5"
with h5py.File(reconstruction_path, "w") as f:
    dset = f.create_dataset("x", data=coeffs, compression=None)
    f.create_dataset("orientations", data = grid_mats, compression=None)
    dset.attrs["units"] = "Arbitrary units"
    f.attrs["sigma"] = sigma_rad
    f.attrs["sigma_unit"] = 'Radians'
    f.attrs['Regularization+constraints'] = 'Nonneg'
    f.attrs['Optimizer'] = 'FISTA'
    f.attrs['N_iter'] = niter
    f.attrs['Datafit'] = 'Huber loss, delta = 4000'
    f.attrs['Normalized_rings'] = True
    f.attrs['N_eta'] = N_eta
    f.attrs['N_Omega'] = N_Omega




    #f.create_dataset("y", data=prediction_cpu, compression=None)

# ---------------------------
# PLOTS
# ---------------------------

plt.figure(figsize=(8,6))
plt.imshow(prediction_cpu[0,90], aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"prediction_00.png", dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.imshow(arr[0,90], aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"measured_00.png", dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.imshow(prediction_cpu[200,70], aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"prediction_01.png", dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.imshow(arr[200,70], aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"measured_01.png", dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.imshow(prediction_cpu[:,:].sum(axis=(0,1)), aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"prediction_sum_00.png", dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.imshow(arr[:,:].sum(axis=(0,1)), aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"measured_sum_00.png", dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.imshow(prediction_cpu[0,:].sum(axis=(0)), aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"prediction_sumtrans_00.png", dpi=300)
plt.close()

plt.figure(figsize=(8,6))
plt.imshow(arr[0,:].sum(axis=(0)), aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"measured_sumtrans_00.png", dpi=300)
plt.close()

plt.figure(figsize=(7,6))
plt.imshow(coeffs.sum(axis=-1))
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"coeff_sum.png", dpi=300)
plt.close()

grid_sp = R.concatenate(grid.rotations_at_level(0))

stepsize = 0.004
imshow_opts = {'extent':(0, coeffs.shape[0]*stepsize, coeffs.shape[1]*stepsize, 0)}

plt.figure(figsize=(7,6))
plt.imshow(coeffs.sum(axis=2), **imshow_opts)
plt.colorbar()
plt.tight_layout()
plt.savefig(run_dir/"coeff_sum_spatial.png", dpi=300)
plt.close()

sums = coeffs.sum(axis=-1)

def save_ipf(direction, name):
    ipfkey = plot.IPFColorKeyTSL(symmetry.Oh, direction=Vector3d(direction))
    orientations = Orientation.from_scipy_rotation(grid_sp)
    orientations.symmetry = ipfkey.symmetry
    rgb = ipfkey.orientation2color(orientations)[np.argmax(coeffs, axis=-1)]
    mask = coeffs.sum(axis=2)>0
    rgb[~mask] *= 0

    plt.figure(figsize=(7,6))
    plt.imshow(rgb[::-1, ::-1], **imshow_opts)
    plt.tight_layout()
    plt.savefig(run_dir/name, dpi=300)
    plt.close()

save_ipf([1,0,0], "ipf_x.png")
save_ipf([0,1,0], "ipf_y.png")
save_ipf([0,0,1], "ipf_z.png")

print("All outputs saved in:", run_dir)
