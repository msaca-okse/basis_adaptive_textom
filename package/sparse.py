from collections.abc import Iterable

import h5py
import numpy as np
from scipy.sparse import coo_matrix


# Written by Axel Henningsson and Mads Carlsen

def get_sparse_frames(framestack, omega_step_dgrs, ypos):
    """Get sparse frames from a framestack in scipy.sparse.coo_matrix format.

    Args:
        framestack (list): list of sparse frames
        omega_step_dgrs (float): omega step in degrees
        ypos (float): y-coordinate of the scan

    Returns:
        tuple: data, row, col, nnz, omega, y

    """
    data, row, col, nnz, omega, y = [], [], [], [], [], []
    for i, frame in enumerate(framestack):
        data.extend(frame.data)
        row.extend(frame.row)
        col.extend(frame.col)
        nnz.append(frame.nnz)
        omega.append(i * omega_step_dgrs)
        y.append(ypos)
    data = np.array(data)
    row = np.array(row)
    col = np.array(col)
    nnz = np.array(nnz)
    omega = np.array(omega)
    y = np.array(y)
    return data, row, col, nnz, omega, y


def _densify(h5_file_handle, ykey, frame_number):
    h = h5_file_handle[ykey]
    bins = np.concatenate(([0], np.cumsum(h["eiger/frames/nnz"])))
    a, b = bins[frame_number], bins[frame_number + 1]

    data = h["eiger/frames/intensity"][a:b]
    row = h["eiger/frames/row"][a:b]
    col = h["eiger/frames/col"][a:b]
    m = h5_file_handle["metadata/eiger/nrows"][()]
    n = h5_file_handle["metadata/eiger/ncols"][()]
    sparse_frame = coo_matrix((data, (row, col)), shape=(m, n))

    return sparse_frame.toarray()


def densify(h5_file_handle, ykey, frame_number):
    """Densify a sparse frame from a h5 file handle.

    See also scipy sparse coo_matrix for more information on the
    sparse format. We store rows columns and intensity values along
    with the number of nonzero elements in each frame (nnz). This
    allows for rapid reassambly of the sparse frame. coo_matrix is
    optimized for fast creation and not for arithmetic operations.

    Args:
        h5_file_handle (h5py.File): h5 file handle
        ykey (str): ykey to densify, i.e the h5 group name
            for the scan at a specific y-coordinate
        frame_number (int or iterable): frame number(s) to densify
            i.e the index of the omega step

    Returns:
        np.ndarray: densified frame
    """
    if isinstance(frame_number, int):
        return _densify(h5_file_handle, ykey, frame_number)
    elif isinstance(frame_number, Iterable):
        return np.array([_densify(h5_file_handle, ykey, i) for i in frame_number])


def write(
    filepath,
    ykey,
    framestack,
    omega_step_dgrs,
    ypos,
    sample,
    eiger,
    xrays,
    dty,
):
    """Write a sparse frame stack to a h5 file.

    Args:
        filepath (str): path to the h5 file
        ykey (str): ykey to densify, i.e the h5 group name
            for the scan at a specific y-coordinate
        framestack (list): list of sparse frames
        omega_step_dgrs (float): omega step in degrees
        ypos (float): y-coordinate of the scan
        sample (Sample): sample object
        eiger (Eiger): eiger object
        xrays (Xrays): xrays object
        dty (float): ystep size

    """
    data, row, col, nnz, omega, y = get_sparse_frames(framestack, omega_step_dgrs, ypos)

    with open("../data/samples/AL.cif") as f:
        cif = f.read()

    with h5py.File(filepath, "w") as hf:
        hf.create_dataset(ykey + "/eiger/frames/intensity", data=data)
        hf.create_dataset(ykey + "/eiger/frames/row", data=row)
        hf.create_dataset(ykey + "/eiger/frames/col", data=col)
        hf.create_dataset(ykey + "/eiger/frames/nnz", data=nnz)
        hf.create_dataset(ykey + "/eiger/frames/omega", data=omega)
        hf.create_dataset(ykey + "/eiger/frames/y", data=y)

        # Metadata from the simulation
        hf.create_dataset("metadata/motors/omegastepsize", data=omega_step_dgrs)
        hf.create_dataset(
            "metadata/wavelength",
            data=1 / (np.linalg.norm(xrays.wave_vector) / (2 * np.pi)),
        )
        hf.create_dataset("metadata/motors/ystep", data=dty)

        hf.create_dataset("metadata/eiger/pixelsize_y", data=eiger.pixel_size_y)
        hf.create_dataset("metadata/eiger/pixelsize_z", data=eiger.pixel_size_z)
        hf.create_dataset("metadata/eiger/distance", data=eiger.det_corner_0[0])
        hf.create_dataset("metadata/eiger/nrows", data=eiger.pixel_coordinates.shape[0])
        hf.create_dataset("metadata/eiger/ncols", data=eiger.pixel_coordinates.shape[1])

        # Input sample information
        hf.create_dataset(
            "metadata/sample/lattice/reference_cell", data=sample.phases[0].unit_cell
        )
        hf.create_dataset(
            "metadata/sample/lattice/sgname", data=sample.phases[0].sgname
        )
        hf.create_dataset("metadata/sample/material", data="Al")
        hf.create_dataset("metadata/sample/lattice/cif", data=cif)
        hf.create_dataset(
            "metadata/sample/mesh/vertices", data=sample.mesh_sample.coord
        )
        hf.create_dataset("metadata/sample/mesh/cells", data=sample.mesh_sample.enod)
        hf.create_dataset(
            "metadata/sample/lattice/orientation", data=sample.orientation_sample
        )
        hf.create_dataset("metadata/sample/lattice/strain", data=sample.strain_sample)
        hf.create_dataset("metadata/sample/lattice/strain", data=sample.strain_sample)