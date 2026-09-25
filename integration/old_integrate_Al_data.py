import pyFAI
import h5py, hdf5plugin
import numpy as np
import yaml
import argparse 


parser = argparse.ArgumentParser(description="Integrate diffraction projections using configuration file.")
parser.add_argument('--config', required=True, help='Path to YAML configuration file')
args = parser.parse_args()

# -------------------------
# Load YAML configuration
# -------------------------
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

poni_path = cfg['poni_path']
ai = pyFAI.load(poni_path)

datapath = cfg['datapath']
diffraction_files = cfg['diffraction_files']
N_trans_total = cfg['My']
N_theta = cfg['N_theta']
N_eta = cfg['N_eta']

output_file = cfg["integrated_file"]
# Batching: Choose the number of detecotr images to integrate at a time. 3000 should be divisible by it
batch_size = 300


def two_theta_to_q(two_theta, wavelength):
    # PyFAI uses 1/nm for q. Poni wavelength is in m
    return (4*np.pi/wavelength) * np.sin(0.5 * two_theta)

width_two_theta = 0.0075
peaks_two_theta = np.array([0.15253486, 0.17618918, 0.24949356 , 0.29284422, 0.30596612, 0.3537646, 0.38588877, 0.39604488, 0.43442364])
q_lower = two_theta_to_q(peaks_two_theta-width_two_theta, wavelength=ai.wavelength*1e9)
q_upper = two_theta_to_q(peaks_two_theta+width_two_theta, wavelength=ai.wavelength*1e9)
q_min = q_lower.min()
q_max = q_upper.max()
N_theta_total = 200
N_rings = 9

I_rings = np.zeros((batch_size, N_eta, N_rings), dtype=np.float32)

mask_list = []
delta_q = (q_max-q_min)/(N_theta_total)
q = np.linspace(q_min+delta_q/2, q_max-delta_q/2, N_theta_total)

for i_ring in range(N_rings):
    mask = (q >= q_lower[i_ring]) & (q < q_upper[i_ring])
    mask_list.append(mask)

N_Omega_raw = 3003
N_Omega_use = N_Omega_raw - 3

with h5py.File(datapath + output_file, "w") as fout:
    dset_out = fout.create_dataset(
        "I",
        shape=(N_Omega_use, N_trans_total, N_eta, N_theta),
        dtype=np.float32,
        chunks=(batch_size, 1, N_eta, N_theta),  # critical
        compression=None
        )


    trans_offset = 0

    for i_h5 in range(len(diffraction_files)):
        
        with h5py.File(datapath + diffraction_files[i_h5], 'r') as f:
            dataset = f['entry/instrument/pilatus/data']
            N_det, My, Mx = dataset.shape
            N_trans_file = N_det // N_Omega_raw
            N_trans_i = N_trans_file - 3 


            assert N_Omega_use % batch_size == 0

            batches_per_trans = N_Omega_use // batch_size
            N_batches = N_trans_i * batches_per_trans

            for i_batch in range(N_batches):
                i_trans = i_batch // batches_per_trans
                i_rot_start = (i_batch % batches_per_trans) * batch_size


                # IMPORTANT: stride by N_Omega_raw, not N_Omega_use
                i_det_start = i_trans * N_Omega_raw + i_rot_start
                data = dataset[i_det_start : i_det_start + batch_size]


                for i in range(batch_size):
                    res = ai.integrate2d(
                        data[i],
                        npt_rad=N_theta_total,
                        npt_azim=N_eta,
                        radial_range=(q_min, q_max),
                        method=("no", "csr", "opencl"),
                        )
                    I = res.intensity
                    for i_ring in range(N_rings):
                        I_rings[i,:,i_ring] = I[:,mask_list[i_ring]].sum(axis=1)



                dset_out[
                    i_rot_start : i_rot_start + batch_size,
                    trans_offset + i_trans,
                    :, :
                ] = I_rings

        trans_offset += N_trans_i