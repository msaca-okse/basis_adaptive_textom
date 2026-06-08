"""
Extract one pixel's coefficients and the orientation grid from a reconstruction,
and save them to a compact HDF5 file for MATLAB analysis.

Rodrigues vector convention used here:
    r = tan(omega/2) * n_hat
where omega is the rotation angle and n_hat is the unit rotation axis.
This is the classical Rodrigues parametrisation (NOT the scipy rotation-vector
convention, which is omega * n_hat).  The conversion is done via scipy's
Rotation.as_rotvec() (gives omega * n_hat) and then rescaling by
tan(omega/2) / omega.

The output HDF5 file is straightforward to read in MATLAB:
    data = h5read('pixel_data.h5', '/coefficients');       % [K x 1]
    rods = h5read('pixel_data.h5', '/rodrigues_vectors');  % [3 x K]
    (MATLAB reverses dimension order relative to NumPy/h5py)
"""

import h5py
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
DATA_DIR = Path('/work3/msaca/textomo_run_022')
RECONSTRUCTION_NAME = 'reconstruction.h5'
OUTPUT_NAME = 'pixel_data.h5'

# Pixel to extract (row, col) — change as needed
PIXEL_ROW = 70
PIXEL_COL = 70

# ---------------------------------------------------------------------------
# LOAD RECONSTRUCTION
# ---------------------------------------------------------------------------
recon_path = DATA_DIR / RECONSTRUCTION_NAME
print(f"Loading reconstruction from {recon_path} ...")

with h5py.File(recon_path, 'r') as f:
    # x has shape (Nx, Ny, K)
    x = f['x'][()]                          # full reconstruction array
    orientations_matrices = f['orientations'][()]  # shape (K, 3, 3)

    # Copy all file-level metadata attributes
    meta = dict(f.attrs)
    # Copy dataset-level attributes for 'x'
    x_attrs = dict(f['x'].attrs)

print(f"Reconstruction shape (Nx, Ny, K): {x.shape}")
print(f"Orientations shape  (K, 3, 3):    {orientations_matrices.shape}")
print(f"File metadata: {meta}")

# ---------------------------------------------------------------------------
# EXTRACT ONE PIXEL
# ---------------------------------------------------------------------------
pixel_coeffs = x[PIXEL_ROW, PIXEL_COL, :]   # shape (K,)
K = pixel_coeffs.shape[0]
print(f"Extracted pixel ({PIXEL_ROW}, {PIXEL_COL}): {K} coefficients")

# ---------------------------------------------------------------------------
# CONVERT ORIENTATIONS TO RODRIGUES VECTORS  r = tan(omega/2) * n_hat
#
# scipy stores rotation matrices as R such that v' = R @ v (active convention).
# The stored matrices were saved as pruned_orient of shape (K, 3, 3).
# When passed to OrientationTree they are transposed, meaning the *stored*
# matrices are already in the convention used by the forward model.
# We convert each stored matrix directly with scipy and then rescale to
# Rodrigues form.
# ---------------------------------------------------------------------------
print("Converting rotation matrices to Rodrigues vectors ...")
rot = Rotation.from_matrix(orientations_matrices)   # infer from (K, 3, 3)

rotvec = rot.as_rotvec()          # shape (K, 3), each row = omega * n_hat
omega = np.linalg.norm(rotvec, axis=1, keepdims=True)   # shape (K, 1)

# Avoid division by zero for near-identity rotations
safe_omega = np.where(omega < 1e-10, 1.0, omega)
n_hat = rotvec / safe_omega                               # unit axis

tan_half = np.where(omega < 1e-10, 0.0, np.tan(omega / 2.0))  # (K, 1)

rodrigues = tan_half * n_hat      # shape (K, 3), r = tan(omega/2) * n_hat
print(f"Rodrigues vectors shape: {rodrigues.shape}")

# ---------------------------------------------------------------------------
# SAVE OUTPUT HDF5
# ---------------------------------------------------------------------------
out_path = DATA_DIR / OUTPUT_NAME
print(f"Saving output to {out_path} ...")

with h5py.File(out_path, 'w') as f:
    # Main datasets
    f.create_dataset('coefficients',      data=pixel_coeffs,   compression=None)
    f.create_dataset('rodrigues_vectors', data=rodrigues,       compression=None)
    f.create_dataset('orientations_rot_matrices', data=orientations_matrices, compression=None)

    # Pixel index
    f.attrs['pixel_row'] = PIXEL_ROW
    f.attrs['pixel_col'] = PIXEL_COL
    f.attrs['pixel_index_description'] = (
        'Row and column index into the (Nx, Ny, K) reconstruction array'
    )

    # Rodrigues convention note
    f.attrs['rodrigues_convention'] = (
        'r = tan(omega/2) * n_hat  '
        'where omega is the rotation angle (radians) and n_hat is the unit '
        'rotation axis.  This is the classical Rodrigues parametrisation. '
        'Shape: (K, 3).'
    )

    # Forward all file-level metadata from the reconstruction
    for key, val in meta.items():
        f.attrs[key] = val

    # Forward x-dataset attributes
    for key, val in x_attrs.items():
        f['coefficients'].attrs[key] = val

    f.attrs['reconstruction_file'] = str(recon_path)
    f.attrs['n_orientations'] = K

print("Done.")
print()
print("MATLAB loading example:")
print("  coeffs = h5read('pixel_data.h5', '/coefficients');       % [K x 1]")
print("  rods   = h5read('pixel_data.h5', '/rodrigues_vectors');  % [3 x K]  (MATLAB reverses dim order)")
print("  meta   = h5info('pixel_data.h5');  % inspect attributes")
