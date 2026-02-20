import h5py
import hdf5plugin
import numpy as np
from multiprocessing import Pool, Manager
import os
import sys
import gc
import argparse
import yaml

# -------------------------
# Parse command line args
# -------------------------
parser = argparse.ArgumentParser(description="Integrate diffraction projections using configuration file.")
parser.add_argument('--config', required=True, help='Path to YAML configuration file')
args = parser.parse_args()

# -------------------------
# Load YAML configuration
# -------------------------
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

pad = cfg['pad']
center = cfg['center']
pixel_size = cfg['pixel_size']
detector_distance = cfg['detector_distance']
N_theta = cfg['N_theta']
min_two_theta = cfg['min_two_theta']
N_workers = cfg['N_workers_integration']
factor = cfg['factor_integration']
N_chi = cfg['N_chi']
chunk_size = cfg['chunk_size_integration']
N_rot = cfg['N_rot']
datapath = cfg["datapath"]
diffraction_file = cfg["diffraction_file"]
output_file = cfg["integrated_file"]

# -------------------------
# Set up environment
# -------------------------
sys.path.append('/zhome/71/c/146676/material_tensor_tomo')
from texture_tomography.utils import polar


max_rad = 1475 // 2 + center[1] + pad
r_max = pixel_size * max_rad
max_two_theta = np.sin(r_max / detector_distance)
two_theta = np.linspace(min_two_theta, max_two_theta, N_theta)

# -------------------------
# Initialize output file
# -------------------------
with h5py.File(output_file, 'w') as f:
    f.attrs.update({
        'pad': pad,
        'center': center,
        'max_rad': max_rad,
        'pixel_size': pixel_size,
        'r_max': r_max,
        'detector_distance': detector_distance,
        'N_theta': N_theta,
        'max_two_theta': max_two_theta,
        'min_two_theta': min_two_theta
    })
    f.create_dataset('two_theta', data=two_theta)

# -------------------------
# Shared manager and lock
# -------------------------
manager = Manager()
lock = manager.Lock()

# -------------------------
# Worker function
# -------------------------
def projection_loader(projection_idx):
    print(f"Processing projection {projection_idx}")

    with h5py.File(datapath + diffraction_file, 'r') as f_in:
        dataset = f_in['entry/instrument/pilatus/data']
        data = dataset[362 * projection_idx : 362 * projection_idx + 362]
        data = np.pad(
            data[None, :, :, :],
            ((0, 0), (0, 0), (pad, 2 * center[0] + pad), (pad, 2 * center[1] + pad)),
            mode='constant'
        )

        out = polar.process_diffraction_cpu_to_gpu(
            data,
            num_phi=N_chi,
            factor=factor,
            two_theta_new=two_theta,
            detector_distance=detector_distance,
            r_max=r_max,
            chunk_size=chunk_size,
            return_to_cpu=True
        ).astype(np.float32).squeeze(axis=0)

    with lock:
        with h5py.File(output_file, 'a') as f_out:
            key = str(int(projection_idx))
            f_out.create_dataset(key, data=out, compression='gzip', compression_opts=4)

    del data, out, dataset
    gc.collect()

# -------------------------
# Parallel execution
# -------------------------
indices = np.arange(N_rot)
with Pool(processes=N_workers) as pool:
    pool.map(projection_loader, indices.astype(np.int32))

print(f"\nAll projections saved to {output_file}")
