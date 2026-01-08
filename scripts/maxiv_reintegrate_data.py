import os
import re
import h5py
import hdf5plugin
import multiprocessing as mp
import numpy as np
from multiprocessing import Pool
folder = "/dtu/3d-imaging-center/projects/2025_QIM_BlackBeauty/raw_data_extern/raw_data_aluminum_rod"

files = []
for f in os.listdir(folder):
    match = re.match(r"scan-(\d{4})_pilatus_integrated\.h5$", f)
    if match:
        num = int(match.group(1))
        if 48 <= num <= 58:
            files.append(os.path.join(folder, f))

files.sort()
print(files)



def process_translation(i_trans, input_path, N_rot, N_images_per_slice,
                        chi_af, theta_af, N_tot, N_trans):
    """Process one translation index and return its result."""
    out_partial = np.empty((N_rot, chi_af, theta_af), dtype=np.float32)

    with h5py.File(input_path, 'r') as file:
        indices = np.arange(N_tot)
        index_list_translations = np.array_split(indices, N_trans)
        sub_indices = np.array_split(index_list_translations[i_trans], N_rot // N_images_per_slice)
        sub_indices_input = np.array_split(np.arange(N_rot), N_rot // N_images_per_slice)

        for i_rot in range(N_rot // N_images_per_slice):
            current_indices = sub_indices[i_rot]
            current_indices_input = sub_indices_input[i_rot]

            dtc = file['entry/azint2d/data/I'][current_indices]
            N_images, chi_be, theta_be = dtc.shape

            dtc = dtc.reshape((N_images, chi_af, chi_be // chi_af, theta_af, theta_be // theta_af))
            dtc = dtc.sum(axis=(2, 4))

            out_partial[current_indices_input, :, :] = dtc

    return i_trans, out_partial


def process_h5(input_path, output_path,
               N_images_per_slice=100, N_theta=3000, N_rot=3003, N_proc=32,
               chi_af=90, theta_af=100):
    """Parallel HDF5 processing and output saving."""
    with h5py.File(input_path, 'r') as file:
        N_tot = len(file['entry/azint2d/data/I'])
        N_trans = N_tot // N_rot

    out_array = np.empty((N_rot, N_trans, chi_af, theta_af), dtype=np.float32)

    args = [(i_trans, input_path, N_rot, N_images_per_slice,
             chi_af, theta_af, N_tot, N_trans) for i_trans in range(N_trans)]

    with Pool(processes=N_proc) as pool:
        for i_trans, result in pool.starmap(process_translation, args):
            out_array[:, i_trans, :, :] = result

    # Save result to HDF5
    # Save result to HDF5: one dataset per rotation index
    with h5py.File(output_path, 'w') as f_out:
        for rot in range(N_rot):
            f_out.create_dataset(
                str(rot),
                data=out_array[rot],          # shape = (N_trans, chi_af, theta_af)
                compression='gzip'
            )
    return output_path




theta_af = 40
chi_af = 180


for i in range(len(files)):
    print('file: ', i)
    path_out = files[i][:-3] + '_re_x3.h5'
    process_h5(files[i],path_out, theta_af=theta_af, chi_af=chi_af)
    with h5py.File(files[i],'r') as file:
        two_theta = file['entry/azint2d/data/radial_axis'][:]
        theta_be = len(two_theta)
        two_theta = two_theta.reshape((theta_af, theta_be // theta_af))/180*np.pi
        two_theta = np.mean(two_theta, axis=1)

    with h5py.File(path_out, 'a') as f_out:
        f_out.create_dataset(
                'two_theta',
                data=two_theta,          # shape = (N_trans, chi_af, theta_af)
                compression='gzip'
            )



datapath = '/dtu/3d-imaging-center/projects/2025_QIM_BlackBeauty/raw_data_extern/raw_data_aluminum_rod/'

files = []
tmp = []
for f in os.listdir(datapath):
    m = re.match(r"scan-(\d{4})_pilatus_integrated_re_x3\.h5$", f)
    if m:
        num = int(m.group(1))
        if 48 <= num <= 58:
            tmp.append((num, os.path.join(datapath, f)))

# sort by num
tmp.sort(key=lambda x: x[0])
# keep only paths
files = [p for _, p in tmp]

all_data_list = []

for filename in files:
    print('Loading file ', filename)
    N_workers = 16  # adjust to match available CPUs

    # --- Get list of dataset keys (exclude metadata) ---
    with h5py.File(filename, 'r') as f:
        keys = [k for k in f.keys() if k != 'two_theta']

        keys = sorted(keys, key=lambda x: int(x))

    # --- Worker function ---
    def load_single(key):
        with h5py.File(filename, 'r') as f:
            arr = f[key][...]
            print(arr.shape)
        return arr[:-3]


    # --- Parallel load ---
    with Pool(processes=N_workers) as pool:
        data_list = pool.map(load_single, keys)

    # --- Combine into single array ---
    data = np.stack(data_list)
    all_data_list.append(data)
all_data = np.concatenate(all_data_list, axis=1)
all_data = all_data[:3000] # Remove the small overlap
mean_factor = 6 #3000 must be divisible by this factor)
N_rot,Nx,N_chi,N_theta = np.shape(all_data)
all_data_meaned = all_data.reshape(3000//mean_factor, mean_factor, Nx, N_chi, N_theta).mean(axis=1)
N_rot_meaned = 3000//mean_factor

with h5py.File(files[-1], 'r') as f:
    two_theta = f['two_theta'][:]

path_out = "/dtu/3d-imaging-center/projects/2025_QIM_BlackBeauty/raw_data_extern/raw_data_aluminum_rod/pilatus_integrated_re_anglemean_x3_3.h5"

with h5py.File(path_out, 'w') as f:
    for rot in range(N_rot_meaned):
        f.create_dataset(
            str(rot),
            data=all_data_meaned[rot],          # shape = (N_trans, chi_af, theta_af)
            compression='gzip'
        )
with h5py.File(path_out, 'a') as f_out:
    f_out.create_dataset(
            'two_theta',
            data=two_theta,          # shape = (N_trans, chi_af, theta_af)
            compression='gzip'
        )
    
