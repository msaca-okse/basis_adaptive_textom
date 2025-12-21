import numpy as np
import cupy as cp
from tqdm import tqdm



def cartesian_to_polar(matrix, num_phi=360, num_rad=None, max_radius = None,factor = 3):
    """ Convert a 2D CuPy matrix from Cartesian to Polar coordinates. """
    rows, cols = matrix.shape
    if max_radius is None:
        max_radius = (min(cols, rows) // 2)
    if num_rad is None:
        num_rad = max_radius


    num_phi_old = num_phi
    num_rad_old = num_rad
    num_phi = factor*num_phi
    num_rad = factor*num_rad
    rows, cols = matrix.shape
    bias_col, bias_row = 0, 0
    center_x, center_y = cols // 2 + bias_col, rows // 2 + bias_row
    

    # Create polar coordinate grid
    theta = np.linspace(0, 2 * np.pi, num_phi)  # Angles
    r = np.linspace(0, max_radius, num_rad)  # Radii
    R, Theta = np.meshgrid(r, theta)  # Create grid

    # Convert polar to Cartesian coordinates
    X = center_x + R * np.cos(Theta)
    Y = center_y + R * np.sin(Theta)

    # Bilinear interpolation
    X = np.clip(X, 0, cols - 1)
    Y = np.clip(Y, 0, rows - 1)
    x0, y0 = X.astype(np.int32), Y.astype(np.int32)  # Floor values
    x1, y1 = np.clip(x0 + 1, 0, cols - 1), np.clip(y0 + 1, 0, rows - 1)  # Ceiling values

    # Get pixel values from the original image
    Ia = matrix[y0, x0]
    Ib = matrix[y0, x1]
    Ic = matrix[y1, x0]
    Id = matrix[y1, x1]

    # Compute bilinear interpolation weights
    wa = (x1 - X) * (y1 - Y)
    wb = (X - x0) * (y1 - Y)
    wc = (x1 - X) * (Y - y0)
    wd = (X - x0) * (Y - y0)

    # Compute interpolated values
    polar_matrix = wa * Ia + wb * Ib + wc * Ic + wd * Id
    polar_matrix = polar_matrix.reshape(num_phi_old, factor, num_rad_old, factor)

    # Take the mean over the (4,4) blocks
    polar_matrix = polar_matrix.mean(axis=(1, 3))

    return polar_matrix, max_radius





def cartesian_to_polar_cupy(matrix, num_phi=360, num_rad=None, max_radius = None,factor = 3):
    """ Convert a 2D CuPy matrix from Cartesian to Polar coordinates. """
    rows, cols = matrix.shape
    if max_radius is None:
        max_radius = (min(cols, rows) // 2)
    if num_rad is None:
        num_rad = max_radius


    num_phi_old = num_phi
    num_rad_old = num_rad
    num_phi = factor*num_phi
    num_rad = factor*num_rad
    rows, cols = matrix.shape
    bias_col, bias_row = 0, 0
    center_x, center_y = cols // 2 + bias_col, rows // 2 + bias_row
    

    # Create polar coordinate grid
    theta = cp.linspace(0, 2 * cp.pi, num_phi)  # Angles
    r = cp.linspace(0, max_radius, num_rad)  # Radii
    R, Theta = cp.meshgrid(r, theta)  # Create grid

    # Convert polar to Cartesian coordinates
    X = center_x + R * cp.cos(Theta)
    Y = center_y + R * cp.sin(Theta)

    # Bilinear interpolation
    X = cp.clip(X, 0, cols - 1)
    Y = cp.clip(Y, 0, rows - 1)
    x0, y0 = X.astype(cp.int32), Y.astype(cp.int32)  # Floor values
    x1, y1 = cp.clip(x0 + 1, 0, cols - 1), cp.clip(y0 + 1, 0, rows - 1)  # Ceiling values

    # Get pixel values from the original image
    Ia = matrix[y0, x0]
    Ib = matrix[y0, x1]
    Ic = matrix[y1, x0]
    Id = matrix[y1, x1]

    # Compute bilinear interpolation weights
    wa = (x1 - X) * (y1 - Y)
    wb = (X - x0) * (y1 - Y)
    wc = (x1 - X) * (Y - y0)
    wd = (X - x0) * (Y - y0)

    # Compute interpolated values
    polar_matrix = wa * Ia + wb * Ib + wc * Ic + wd * Id
    polar_matrix = polar_matrix.reshape(num_phi_old, factor, num_rad_old, factor)

    # Take the mean over the (4,4) blocks
    polar_matrix = polar_matrix.mean(axis=(1, 3))

    return polar_matrix, max_radius




def detector_radius_to_twotheta(
    det_array,
    two_theta_new,
    detector_distance,
    r_max
):
    """
    Interpolate detector data from radius -> 2θ space.

    Parameters
    ----------
    det_array : ndarray, shape (n_azimuth, n_radius)
        Input array, axis=0 azimuth, axis=1 radius.
    two_theta_new : 1D ndarray
        Target 2θ values in radianss. Must lie within detector range.
    detector_distance : float
        Sample-to-detector distance (same units as r_max).
    r_max : float
        Maximum detector radius (same units as detector_distance).

    Returns
    -------
    out_array : ndarray, shape (n_azimuth, len(two_theta_new))
        Rebinned array in azimuth × 2θ space.
    """
    n_azimuth, n_radius = det_array.shape

    # radius axis (0..r_max)
    r = np.linspace(0, r_max, n_radius)

    # convert radius -> 2θ (radianss)
    two_theta = (np.arctan(r / detector_distance))

    # range check
    if (two_theta_new.min() < two_theta.min()) or (two_theta_new.max() > two_theta.max()):
        raise ValueError(
            f"Requested 2θ range {two_theta_new.min()}–{two_theta_new.max()} deg "
            f"outside detector range {two_theta.min()}–{two_theta.max()} deg."
        )

    # interpolate along axis=1 (radius → 2θ)
    out = np.empty((n_azimuth, len(two_theta_new)), dtype=det_array.dtype)
    for i in range(n_azimuth):
        out[i, :] = np.interp(two_theta_new, two_theta, det_array[i, :])

    return out




# def detector_radius_to_twotheta_cupy(
#     det_array: cp.ndarray,
#     two_theta_new,
#     detector_distance: float,
#     r_max: float
# ):
#     """
#     Interpolate detector data from radius -> 2θ space (CuPy version).
#     """
#     n_azimuth, n_radius = det_array.shape

#     # make sure inputs are cupy
#     two_theta_new = cp.asarray(two_theta_new)

#     # radius axis
#     r = cp.linspace(0, r_max, n_radius)

#     # convert radius -> 2θ (radianss)
#     two_theta = (cp.arctan(r / detector_distance))

#     # range check (cast to float to avoid cp.bool_ -> error)
#     if (float(two_theta_new.min()) < float(two_theta.min())) or \
#        (float(two_theta_new.max()) > float(two_theta.max())):
#         raise ValueError("Requested 2θ range outside detector range.")

#     # interpolate along axis=1
#     out = cp.empty((n_azimuth, len(two_theta_new)), dtype=det_array.dtype)
#     for i in range(n_azimuth):
#         out[i, :] = cp.interp(two_theta_new, two_theta, det_array[i, :])

#     return out


def detector_radius_to_twotheta_cupy(det_array, two_theta_new, detector_distance, r_max):
    n_azimuth, n_radius = det_array.shape
    two_theta_new = cp.asarray(two_theta_new)

    r = cp.linspace(0, r_max, n_radius)
    two_theta = (cp.arctan(r / detector_distance))

    # indices of bins
    idx = cp.searchsorted(two_theta, two_theta_new, side="left")
    idx = cp.clip(idx, 1, n_radius-1)  # valid range

    # x0, x1
    x0 = two_theta[idx-1]
    x1 = two_theta[idx]

    # values y0, y1 for all azimuths
    y0 = det_array[:, idx-1]   # shape (n_azimuth, n_new)
    y1 = det_array[:, idx]

    # linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    out = y0 + slope * (two_theta_new - x0)

    return out



def cartesian_to_polar_cupy_batch(matrix, num_phi=360, num_rad=None, max_radius=None, factor=3):
    """
    Batched Cartesian → Polar for CuPy arrays.

    Parameters
    ----------
    matrix : cp.ndarray, shape (B, H, W)
        Batch of 2D images.
    num_phi : int
        Number of angular bins.
    num_rad : int or None
        Number of radial bins (default = min(H,W)//2).
    max_radius : int or None
        Maximum radius (default = min(H,W)//2).
    factor : int
        Oversampling factor (for antialiasing).

    Returns
    -------
    polar_matrix : cp.ndarray, shape (B, num_phi, num_rad)
    max_radius : int
    """
    B, rows, cols = matrix.shape
    if max_radius is None:
        max_radius = min(cols, rows) // 2
    if num_rad is None:
        num_rad = max_radius

    num_phi_old = num_phi
    num_rad_old = num_rad
    num_phi = factor * num_phi
    num_rad = factor * num_rad

    center_x, center_y = cols // 2, rows // 2

    # polar grid
    theta = cp.linspace(0, 2 * cp.pi, num_phi)
    r = cp.linspace(0, max_radius, num_rad)
    R, Theta = cp.meshgrid(r, theta)   # (num_phi, num_rad)

    X = center_x + R * cp.cos(Theta)
    Y = center_y + R * cp.sin(Theta)

    # clip
    X = cp.clip(X, 0, cols - 1)
    Y = cp.clip(Y, 0, rows - 1)
    x0, y0 = X.astype(cp.int32), Y.astype(cp.int32)
    x1 = cp.clip(x0 + 1, 0, cols - 1)
    y1 = cp.clip(y0 + 1, 0, rows - 1)

    # expand to batch
    x0 = cp.broadcast_to(x0[None, :, :], (B, num_phi, num_rad))
    x1 = cp.broadcast_to(x1[None, :, :], (B, num_phi, num_rad))
    y0 = cp.broadcast_to(y0[None, :, :], (B, num_phi, num_rad))
    y1 = cp.broadcast_to(y1[None, :, :], (B, num_phi, num_rad))

    # bilinear interpolation
    Ia = matrix[cp.arange(B)[:, None, None], y0, x0]
    Ib = matrix[cp.arange(B)[:, None, None], y0, x1]
    Ic = matrix[cp.arange(B)[:, None, None], y1, x0]
    Id = matrix[cp.arange(B)[:, None, None], y1, x1]

    wa = (x1 - X) * (y1 - Y)
    wb = (X - x0) * (y1 - Y)
    wc = (x1 - X) * (Y - y0)
    wd = (X - x0) * (Y - y0)

    polar_matrix = wa*Ia + wb*Ib + wc*Ic + wd*Id
    polar_matrix = polar_matrix.reshape(B, num_phi_old, factor, num_rad_old, factor)
    polar_matrix = polar_matrix.mean(axis=(2,4))

    return polar_matrix, max_radius



def detector_radius_to_twotheta_cupy_batch(det_array, two_theta_new, detector_distance, r_max):
    """
    det_array: (B, n_azimuth, n_radius)
    two_theta_new: (n_new,)
    """
    B, n_azimuth, n_radius = det_array.shape
    two_theta_new = cp.asarray(two_theta_new)

    r = cp.linspace(0, r_max, n_radius)
    two_theta = (cp.arcsin(r / detector_distance))

    idx = cp.searchsorted(two_theta, two_theta_new, side="left")
    idx = cp.clip(idx, 1, n_radius-1)

    x0, x1 = two_theta[idx-1], two_theta[idx]
    y0 = det_array[:, :, idx-1]   # (B, n_azimuth, n_new)
    y1 = det_array[:, :, idx]

    slope = (y1 - y0) / (x1 - x0)
    out = y0 + slope * (two_theta_new - x0)

    return out   # shape (B, n_azimuth, n_new)


def process_diffraction_cupy(
    diffraction_4d_gpu, 
    num_phi, factor, 
    two_theta_new, detector_distance, r_max, 
    chunk_size=64
):
    """
    Process diffraction data (Cartesian->Polar->2θ) in chunks on GPU.

    Parameters
    ----------
    diffraction_4d_gpu : cp.ndarray, shape (Nx, Ny, H, W)
        Input diffraction data on GPU.
    num_phi : int
        Number of azimuth bins.
    factor : int
        Oversampling factor for antialiasing in polar conversion.
    two_theta_new : 1D array-like
        Target 2θ values in radianss.
    detector_distance : float
        Sample-to-detector distance.
    r_max : float
        Maximum detector radius.
    chunk_size : int
        Number of slices per chunk.

    Returns
    -------
    out : cp.ndarray, shape (Nx, Ny, num_phi, len(two_theta_new))
    """
    Nx, Ny, H, W = diffraction_4d_gpu.shape
    B = Nx * Ny
    two_theta_new_gpu = cp.asarray(two_theta_new)

    # allocate output array on GPU
    out = None

    # flatten (Nx,Ny) into batch dimension
    batch = diffraction_4d_gpu.reshape(B, H, W)
    print(B)
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk = batch[start:end]  # shape (b, H, W)
        print(chunk.shape)

        # Cartesian -> Polar
        pol_chunk, max_rad = cartesian_to_polar_cupy_batch(
            chunk, num_phi=num_phi, factor=factor
        )  # (b, num_phi, num_rad)

        # radius -> 2θ
        pol_reb_chunk = detector_radius_to_twotheta_cupy_batch(
            pol_chunk, two_theta_new_gpu, detector_distance, r_max
        )  # (b, num_phi, n_new)

        # allocate output once
        if out is None:
            out = cp.empty(
                (B, pol_reb_chunk.shape[1], pol_reb_chunk.shape[2]),
                dtype=pol_reb_chunk.dtype,
            )

        out[start:end] = pol_reb_chunk

    # reshape back to (Nx, Ny, num_phi, n_new)
    out = out.reshape(Nx, Ny, out.shape[1], out.shape[2])
    return out



def process_diffraction_cpu_to_gpu(
    diffraction_4d,        # numpy array, shape (Nx, Ny, H, W)
    num_phi, factor,
    two_theta_new, detector_distance, r_max,
    chunk_size=32,         # number of (i,j) slices per chunk
    return_to_cpu=True     # store results on CPU or GPU
):
    Nx, Ny, H, W = diffraction_4d.shape
    B = Nx * Ny
    two_theta_new_gpu = cp.asarray(two_theta_new)

    # flatten (Nx,Ny) into batch dimension for easy chunking
    flat = diffraction_4d.reshape(B, H, W)

    # figure out output shape using one test slice
    test_slice = diffraction_4d[0,0][None, :, :]   # (1, H, W)
    test_pol, max_rad = cartesian_to_polar_cupy_batch(
        cp.asarray(test_slice), num_phi=num_phi, factor=factor
    )
    test_reb = detector_radius_to_twotheta_cupy_batch(
        test_pol, two_theta_new_gpu, detector_distance, r_max
    )

    # output shape in flattened form: (B, num_phi, n_new)
    out_flat_shape = (B, test_reb.shape[1], test_reb.shape[2])

    # allocate container
    if return_to_cpu:
        out = np.empty(out_flat_shape, dtype=test_reb.dtype)
    else:
        out = cp.empty(out_flat_shape, dtype=test_reb.dtype)

    # loop over chunks
    for start in tqdm(range(0, B, chunk_size), desc="Processing"):
        end = min(start + chunk_size, B)

        # move chunk to GPU
        chunk_gpu = cp.asarray(flat[start:end])   # (b, H, W)

        # cartesian -> polar
        pol_chunk, _ = cartesian_to_polar_cupy_batch(
            chunk_gpu, num_phi=num_phi, factor=factor
        )

        # radius -> 2θ
        pol_reb_chunk = detector_radius_to_twotheta_cupy_batch(
            pol_chunk, two_theta_new_gpu, detector_distance, r_max
        )  # (b, num_phi, n_new)

        # store results
        if return_to_cpu:
            out[start:end] = cp.asnumpy(pol_reb_chunk)
        else:
            out[start:end] = pol_reb_chunk

        # free GPU memory
        del chunk_gpu, pol_chunk, pol_reb_chunk
        cp._default_memory_pool.free_all_blocks()

    # reshape back to (Nx, Ny, num_phi, n_new)
    out = out.reshape(Nx, Ny, out.shape[1], out.shape[2])
    return out




def spectra_radius_to_twotheta(
    spec_list,
    two_theta_new,
    detector_distance,
    r_max
):
    """
    Interpolate multiple 1D spectra from radius -> 2θ space.

    Parameters
    ----------
    spec_list : list of 1D ndarrays
        Each element is shape (n_radius,), an azimuthally integrated spectrum.
    two_theta_new : 1D ndarray
        Target 2θ values in radianss. Must lie within detector range.
    detector_distance : float
        Sample-to-detector distance (same units as r_max).
    r_max : float
        Maximum detector radius (same units as detector_distance).

    Returns
    -------
    out_array : ndarray, shape (len(spec_list), len(two_theta_new))
        Rebinned spectra in (spectrum index × 2θ) space.
    """
    n_spectra = len(spec_list)
    n_radius = len(spec_list[0])

    # radius axis (0..r_max)
    r = np.linspace(0, r_max, n_radius)

    # convert radius -> 2θ (radianss)
    two_theta = (np.arctan(r / detector_distance))

    # range check
    if (two_theta_new.min() < two_theta.min()) or (two_theta_new.max() > two_theta.max()):
        raise ValueError(
            f"Requested 2θ range {two_theta_new.min()}–{two_theta_new.max()} deg "
            f"outside detector range {two_theta.min()}–{two_theta.max()} deg."
        )

    # interpolate each spectrum
    out = np.empty((n_spectra, len(two_theta_new)), dtype=spec_list[0].dtype)
    for i, s in enumerate(spec_list):
        out[i, :] = np.interp(two_theta_new, two_theta, s)

    return out




def polar_to_cartesian_cupy_batch(polar_matrix, out_shape, max_radius, factor=3):
    """
    Batched Polar → Cartesian for CuPy arrays.

    Parameters
    ----------
    polar_matrix : cp.ndarray, shape (B, num_phi, num_rad)
        Batch of 2D polar images (azimuth, radius).
    out_shape : tuple (H, W)
        Output Cartesian image shape.
    max_radius : int
        Maximum radius used in the polar transform.
    factor : int
        Oversampling factor (for antialiasing).

    Returns
    -------
    cart_matrix : cp.ndarray, shape (B, H, W)
    """
    B, num_phi, num_rad = polar_matrix.shape
    H, W = out_shape
    center_x, center_y = W // 2, H // 2

    # Upsampled polar grid
    num_phi_up = factor * num_phi
    num_rad_up = factor * num_rad

    # Cartesian target grid
    y = cp.arange(H) - center_y
    x = cp.arange(W) - center_x
    X, Y = cp.meshgrid(x, y)   # (H, W)

    R = cp.sqrt(X**2 + Y**2)
    Theta = (cp.arctan2(Y, X) + 2*cp.pi) % (2*cp.pi)

    # Scale R, Theta to polar indices
    r_idx = R / max_radius * (num_rad_up - 1)
    theta_idx = Theta / (2*cp.pi) * (num_phi_up - 1)

    # Clip
    r_idx = cp.clip(r_idx, 0, num_rad_up - 1)
    theta_idx = cp.clip(theta_idx, 0, num_phi_up - 1)

    # Integer coordinates for bilinear interpolation
    r0 = r_idx.astype(cp.int32)
    r1 = cp.clip(r0 + 1, 0, num_rad_up - 1)
    t0 = theta_idx.astype(cp.int32)
    t1 = cp.clip(t0 + 1, 0, num_phi_up - 1)

    # Expand for batch
    r0 = cp.broadcast_to(r0[None, :, :], (B, H, W))
    r1 = cp.broadcast_to(r1[None, :, :], (B, H, W))
    t0 = cp.broadcast_to(t0[None, :, :], (B, H, W))
    t1 = cp.broadcast_to(t1[None, :, :], (B, H, W))

    # Bilinear interpolation in polar space
    Ia = polar_matrix[cp.arange(B)[:, None, None], t0, r0]
    Ib = polar_matrix[cp.arange(B)[:, None, None], t0, r1]
    Ic = polar_matrix[cp.arange(B)[:, None, None], t1, r0]
    Id = polar_matrix[cp.arange(B)[:, None, None], t1, r1]

    wa = (r1 - r_idx) * (t1 - theta_idx)
    wb = (r_idx - r0) * (t1 - theta_idx)
    wc = (r1 - r_idx) * (theta_idx - t0)
    wd = (r_idx - r0) * (theta_idx - t0)

    cart_matrix = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return cart_matrix




def twotheta_to_detector_radius_cupy_batch(tt_array, n_radius, detector_distance, r_max):
    """
    Inverse of detector_radius_to_twotheta_cupy_batch.
    Interpolates from two-theta space back to detector radius space.

    Parameters
    ----------
    tt_array : (B, n_azimuth, n_tt)
        Input array sampled on a uniform two-theta grid.
    n_radius : int
        Number of detector radius bins in the output.
    detector_distance : float
        Detector distance.
    r_max : float
        Maximum detector radius.

    Returns
    -------
    det_array : (B, n_azimuth, n_radius)
        Interpolated array on detector radius grid.
    """
    B, n_azimuth, n_tt = tt_array.shape

    # Define radius grid
    r = cp.linspace(0, r_max, n_radius)

    # Corresponding two-theta grid from detector geometry
    two_theta = cp.arctan(r / detector_distance)  # (n_radius,)

    # Original two-theta grid in input array
    two_theta_old = cp.linspace(two_theta.min(), two_theta.max(), n_tt)

    # Find where each new theta lies in the old grid
    idx = cp.searchsorted(two_theta_old, two_theta, side="left")
    idx = cp.clip(idx, 1, n_tt-1)

    x0, x1 = two_theta_old[idx-1], two_theta_old[idx]
    y0 = tt_array[:, :, idx-1]   # (B, n_azimuth, n_radius)
    y1 = tt_array[:, :, idx]

    slope = (y1 - y0) / (x1 - x0)
    det_array = y0 + slope * (two_theta - x0)

    return det_array  # shape (B, n_azimuth, n_radius)