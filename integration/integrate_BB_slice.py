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
datapath = cfg['datapath']
diffraction_file = cfg['diffraction_file']
Nx = cfg['Nx']
N_rot = cfg['N_rot']
N_chi = cfg['N_chi']
N_theta = cfg['N_theta']
output_file = cfg["integrated_file"]


ai = pyFAI.load(poni_path)


with h5py.File(output_file, "w"):
    pass

with h5py.File(datapath + diffraction_file, 'r') as f:

    for i_N_rot in range(N_rot):

        dataset = f['entry/instrument/pilatus/data']
        data = dataset[362 * i_N_rot: 362 * i_N_rot + 362]

        data_rotation = []
        for i_Nx in range(Nx):

            res = ai.integrate2d(
                data[i_Nx],
                npt_rad=N_theta,
                npt_azim=N_chi,
                radial_range=(8.0, 40.0),
                method=("no", "csr", "opencl"),
            )
            intensity = res.intensity     # shape (npt_azim, npt_rad)
            q_A = res.radial*0.1        # 1D array
            chi = res.azimuthal           # 1D array
            two_theta_rad = 2.0 * np.arcsin(
                q_A * ai.wavelength*1e10 / (4.0 * np.pi)
            )

            data_rotation.append(intensity)
        data_rotation = np.stack(data_rotation).astype(np.float32)


        with h5py.File(output_file, 'a') as f_out:
            key = str(int(i_N_rot))
            f_out.create_dataset(key, data=data_rotation)

        

        print('Finished rotation ', i_N_rot)


with h5py.File(output_file, 'a') as f_out:
    f_out.create_dataset("two_theta", data=two_theta_rad)