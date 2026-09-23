import os
import pickle

import h5py
import ImageD11.columnfile
import ImageD11.parameters
import numpy as np

from . import paths


def get_omega(h5path):
    with h5py.File(h5path, "r") as f:
        omega = f[paths.OMEGAMOTOR][:]
    return omega


def get_dty(h5path):
    with h5py.File(h5path, "r") as f:
        dty = f[paths.DTYMOTOR][:]
    return dty


def get_frames(
    h5path,
    frame_number=None,
    start_index=None,
    end_index=None,
):
    with h5py.File(h5path, "r") as f:
        if start_index is not None and end_index is not None:
            frames = f[paths.FRAMES][start_index:end_index]
        elif frame_number is not None:
            frames = f[paths.FRAMES][frame_number]
        elif frame_number is None and start_index is None and end_index is None:
            frames = f[paths.FRAMES]
        else:
            raise ValueError(
                "Either start_index and end_index or frame_number or no arguments must be provided"
            )
    return frames


def get_number_of_dty_scans_in_file(h5path):
    omega = get_omega(h5path)
    dty = get_dty(h5path)
    omsteps_per_scan = np.unique(omega.round(4)).size
    assert len(dty) % omsteps_per_scan == 0, (
        f"len(dty) % omsteps_per_scan == {len(dty) % omsteps_per_scan}"
    )
    return len(dty) // omsteps_per_scan


def check_metadata(h5path, verbose=True):
    omega = get_omega(h5path)
    frames = get_frames(h5path)
    dty = get_dty(h5path)

    omsteps_per_scan = np.unique(omega.round(4)).size
    assert len(dty) % omsteps_per_scan == 0, (
        f"len(dty) % omsteps_per_scan == {len(dty) % omsteps_per_scan}"
    )
    number_of_frames_in_file = frames.shape[0]

    number_of_dty_scans_in_file = len(dty) // omsteps_per_scan
    assert number_of_frames_in_file % number_of_dty_scans_in_file == 0, (
        f"number_of_frames_in_file % number_of_dty_scans_in_file == {number_of_frames_in_file % number_of_dty_scans_in_file}"
    )
    assert (
        number_of_frames_in_file // number_of_dty_scans_in_file == omsteps_per_scan
    ), (
        f"number_of_frames_in_file // number_of_dty_scans_in_file == {number_of_frames_in_file // number_of_dty_scans_in_file}"
    )

    for i in range(number_of_dty_scans_in_file):
        assert np.allclose(
            dty[i * omsteps_per_scan : (i + 1) * omsteps_per_scan],
            dty[i * omsteps_per_scan],
        ), (
            f"dty[i * omsteps_per_scan : (i + 1) * omsteps_per_scan] != dty[i * omsteps_per_scan]: {dty[i * omsteps_per_scan : (i + 1) * omsteps_per_scan] != dty[i * omsteps_per_scan]}"
        )

    if verbose:
        print(f"number of dty scans in file: {number_of_dty_scans_in_file}")
        print(f"number of omega steps per dty scan: {omsteps_per_scan}")
        print(f"number of frames in file: {number_of_frames_in_file}")
        print(f"Detector shape: {frames.shape[1:]}")


def write_peaks(pks, h5path):
    """Write peak arrays to an HDF5 file.

    This function stores peak information needed by downstream
    ImageD11 processing. The input dictionary must contain arrays for
    the spatial centroids, detector coordinates, omega values, and
    intensity metrics. Each array is written as a separate dataset
    inside the HDF5 file.

    Args:
        pks (:obj:`dict`): Dictionary containing the arrays
            ``'s_raw'``, ``'f_raw'``, ``'dty'``, ``'o_raw'``,
            ``'s_1'``, ``'s_I'``.
        h5path (:obj:`str`): Path where the HDF5 file will be written.

    Raises:
        TypeError: If ``pks`` is not a :obj:`dict`.
        KeyError: If one or more required keys are missing.
    """
    if not isinstance(pks, dict):
        raise ValueError("pks must be a dictionary")

    req = {"s_raw", "f_raw", "dty", "o_raw", "s_1", "s_I"}
    if not req.issubset(pks):
        missing = req - set(pks)
        raise ValueError(f"pks missing required keys {missing}")

    dirname = os.path.dirname(h5path)
    if not os.path.isdir(dirname):
        raise ValueError(f"No such directory {dirname}")
    if os.path.exists(h5path):
        raise ValueError(f"File already exists : {h5path}")

    with h5py.File(h5path, "w") as f:
        for key in req:
            f.create_dataset(key, data=pks[key])


def read_peaks(h5path, par_path):
    """Read peak arrays from an HDF5 file and construct a columnfile.

    The function loads each dataset stored in the file and maps it
    back to the keys expected by :obj:`ImageD11.columnfile.colfile_from_dict`.
    It verifies that all required datasets exist before creating
    the output columnfile.

    Args:
        h5path (:obj:`str`): Path to the HDF5 file.
        par_path (:obj:`str`): Path to the parameters file.

    Returns:
        :obj:`ImageD11.columnfile.colfile`: Columnfile containing
        the peak information reconstructed from the stored datasets.


    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If one or more required datasets are not present.
    """
    if not os.path.exists(h5path):
        raise ValueError(h5path)

    parameters = ImageD11.parameters.read_par_file(par_path)

    keys_map = {
        "s_raw": "sc",
        "f_raw": "fc",
        "dty": "dty",
        "o_raw": "omega",
        "s_1": "Number_of_pixels",
        "s_I": "sum_intensity",
    }

    pks = {v: None for v in keys_map.values()}

    with h5py.File(h5path, "r") as f:
        for ds, outkey in keys_map.items():
            if ds not in f:
                raise KeyError(f"Dataset {ds} missing in file")
            pks[outkey] = f[ds][()].astype(np.float64)  # decompress

    colf = ImageD11.columnfile.colfile_from_dict(pks)
    colf.setparameters(parameters)

    return colf


def save_analysis_bundle(data, savepath, name):
    folder = os.path.join(savepath, name)
    os.makedirs(folder, exist_ok=True)

    pkl_path = os.path.join(folder, f"{name}.pkl")
    npz_path = os.path.join(folder, f"{name}.npz")

    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in data.items() if k not in arrays}

    meta["_npz_keys"] = tuple(arrays.keys())

    np.savez(npz_path, **arrays)

    with open(pkl_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    return folder


def load_analysis_bundle(savepath, name):
    folder = os.path.join(savepath, name)

    pkl_path = os.path.join(folder, f"{name}.pkl")
    npz_path = os.path.join(folder, f"{name}.npz")

    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)

    npz_keys = meta.pop("_npz_keys", ())

    arrays = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as z:
            arrays = (
                {k: z[k] for k in npz_keys} if npz_keys else {k: z[k] for k in z.files}
            )

    out = dict(meta)
    out.update(arrays)
    return out


def load_strain_txt():
    path = os.path.join(
        paths.PROCESS, "al1050_15pct_center_slice", "load_cell", "DIC-Data.txt"
    )
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return data[:, 5]


def load_force_txt():
    path = os.path.join(
        paths.PROCESS, "al1050_15pct_center_slice", "load_cell", "metadata.txt"
    )
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return data[:, 1]


def load_strain_and_force():
    strain = load_strain_txt()
    force = load_force_txt()
    return strain, force


if __name__ == "__main__":
    pass
