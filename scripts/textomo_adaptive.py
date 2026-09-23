import os
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
config_path = project_root / "configs" / "aluminum_config.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

niter = 200
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

# ---------------------------
# GRID
# ---------------------------
with h5py.File("/work3/msaca/basis_set.h5", "r") as f:
    numpy_orien = f["numpy_orien"][...][::15]
    scores = f["scores"][...]



def prune_rotations(Rmats, theta_deg, target=5000):
    # Convert to quaternions
    q = R.from_matrix(Rmats).as_quat()  # (x,y,z,w)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    # Antipodal equivalence: q and -q represent same rotation
    q_full = np.vstack([q, -q])

    tree = KDTree(q_full)

    theta = np.deg2rad(theta_deg)

    used = np.zeros(len(q_full), dtype=bool)
    keep = []

    for i in range(len(q)):
        if used[i]:
            continue

        keep.append(i)

        # mark neighbors as used
        idx = tree.query_radius(q[i:i+1], r=theta)[0]
        used[idx] = True

        if len(keep) >= target:
            break

    return Rmats[keep]


pruned_orient = prune_rotations(numpy_orien, theta_deg=0.15, target=50000)

pruned_orient = np.load('/work3/msaca/basis_3/numpy_matrix_flat_thin_2mrad.npy')
pruned_orient = pruned_orient.transpose((0,2,1))

def rotate_orientations_about_z(orientations: np.ndarray, degrees: float) -> np.ndarray:
    orientations = np.asarray(orientations, dtype=np.float64)

    if orientations.ndim != 3 or orientations.shape[1:] != (3, 3):
        raise ValueError(
            f"`orientations` must have shape (N, 3, 3), got {orientations.shape}"
        )

    theta = np.deg2rad(degrees)
    c = np.cos(theta)
    s = np.sin(theta)

    Rz = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return Rz[None, :, :] @ orientations

pruned_orient = rotate_orientations_about_z(pruned_orient, 50.0)

sigma = 0.007
grid = Grid.from_rotation_matrices(pruned_orient, sigma=sigma)
print('Generated grid')
# grid_resolution_parameter = 64      # example
# kernel_sigma = 0.025               # example
# sigma_levels = [kernel_sigma]      # start with single level


# grid = Grid.from_hopf_fzone(
#     material.point_group_matrices,
#     grid_resolution_parameter=grid_resolution_parameter,
#     sigma_levels=sigma_levels,
# )


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
    f.create_dataset("orientations", data = pruned_orient, compression=None)
    dset.attrs["units"] = "Arbitrary units"
    f.attrs["sigma"] = sigma
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
