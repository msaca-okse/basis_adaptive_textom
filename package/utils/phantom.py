import numpy as np
from scipy.ndimage import gaussian_filter1d


def make_sparse_phantom(K=100, nx=100, ny=100, S=10, n_regions=5,normalized_intensity=False, R=None, seed=None):
    """
    Create a phantom array A of shape (K, nx, ny).
    
    Parameters
    ----------
    K : int
        Length of the spectral / channel dimension.
    nx, ny : int
        Spatial dimensions.
    S : int
        Number of nonzero entries along K (sparse support).
    n_regions : int
        Number of distinct spatial regions inside the circle.
    R : float or None
        Radius of the valid region (centered at image center). If None, use min(nx, ny)/2.
    seed : int or None
        Random seed for reproducibility.
        
    Returns
    -------
    A : np.ndarray
        Phantom array of shape (K, nx, ny).
    regions : np.ndarray
        Region labels of shape (nx, ny). Pixels outside circle are -1.
    """
    rng = np.random.default_rng(seed)
    phantom = np.zeros((K, nx, ny), dtype=float)

    if R is None:
        R = min(nx, ny) / 2

    # --- make circular mask ---
    x = np.arange(nx) - nx/2 + 0.5
    y = np.arange(ny) - ny/2 + 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    circle_mask = (X**2 + Y**2) <= R**2

    # --- assign Voronoi-like regions inside the circle ---
    labels = -np.ones((nx, ny), dtype=int)  # -1 = outside
    centers = rng.integers([0, 0], [nx, ny], size=(n_regions, 2))
    for x in range(nx):
        for y in range(ny):
            if circle_mask[x, y]:
                dists = np.sum((centers - np.array([x, y]))**2, axis=1)
                labels[x, y] = np.argmin(dists)

    # --- assign a common sparse spectrum to each region ---
    for region_id in range(n_regions):
        # pick S random nonzero indices in K
        idx = rng.choice(K, size=S, replace=True)
        coef = np.zeros(K, dtype=float)
        coef[idx] = rng.random(S)

        mask = (labels == region_id)
        if normalized_intensity:
            coef = coef/coef.sum()

        phantom[:, mask] = coef[:, None]

    flat_labels = labels.ravel()
    unique_labels, first_idx = np.unique(flat_labels, return_index=True)
    order = np.argsort(first_idx)  # preserve order of appearance
    order = range(len(unique_labels))
    unique_labels = unique_labels[order]

    true_coeffs = []
    for lab in unique_labels:
        if lab == -1:
            # background spectrum = all zeros
            coef = np.zeros(phantom.shape[0])
        else:
            x, y = np.argwhere(labels == lab)[0]
            coef = phantom[:, x, y]
        true_coeffs.append(coef)

    true_coeffs = np.array(true_coeffs)




    return phantom.astype(np.float32), labels, true_coeffs, circle_mask



def classify_reconstruction(recovered, true_spectra, metric="l2"):
    """
    Classify each voxel in a reconstructed volume by nearest ground truth spectrum.

    Parameters
    ----------
    recovered : np.ndarray
        Reconstructed volume, shape (K, nx, ny).
    true_spectra : list or np.ndarray
        Array of shape (n_classes, K). Each row is a ground truth spectrum.
    true_labels : np.ndarray
        Label map from phantom, shape (nx, ny).
    mask : np.ndarray or None
        Boolean mask (nx, ny) of valid voxels. If None, all are valid.
    metric : str
        Distance metric: "l2" or "cosine".

    Returns
    -------
    predicted_labels : np.ndarray
        Predicted label map, shape (nx, ny).
    """
    K, nx, ny = recovered.shape
    n_classes = len(true_spectra)

    recovered_flat = recovered.reshape(K, nx*ny).T   # (nx*ny, K)
    print(recovered_flat.shape)

    if metric == "l2":
        dists = (np.abs(recovered_flat[:,None,:] - true_spectra[None, :, :])**2).sum(axis=2)
        #return dists
    elif metric == "l1":
        dists = (np.abs(recovered_flat[:,None,:] - true_spectra[None, :, :])).sum(axis=2)
        #return dists
    else:
        raise ValueError("Unsupported metric")

    # pick nearest spectrum
    pred_flat = np.argmin(dists, axis=1)

    predicted_labels = pred_flat.reshape(nx, ny)


    return predicted_labels-1, dists




def simulate_bragg_spectra(N_mat, two_theta, N_peaks, sigma=0.0, random_state=None):
    """
    Simulate N Bragg spectra.

    Parameters
    ----------
    N : int
        Number of spectra.
    two_theta : array_like
        1D array of 2θ values (x-axis).
    N_peaks : int or list of ints
        Number of peaks per spectrum.
        - If int, every spectrum has N_peaks peaks.
        - If list (length N), specifies number of peaks for each spectrum.
    sigma : float or list of floats, default=0.0
        Standard deviation of Gaussian convolution (in same units as two_theta spacing).
        - If 0, no convolution.
        - If list (length N), specifies sigma for each spectrum.
    random_state : int or np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    spectra : ndarray, shape (N_mat, len(two_theta))
        Simulated Bragg spectra.
    """

    rng = np.random.default_rng(random_state)
    two_theta = np.asarray(two_theta)
    n_points = len(two_theta)

    # Handle N_peaks (sparsity / number of peaks)
    if isinstance(N_peaks, int):
        S_list = [N_peaks] * N_mat
    elif isinstance(N_peaks, (list, np.ndarray)) and len(N_peaks) == N_mat:
        S_list = list(N_peaks)
    else:
        raise ValueError("N_peaks must be int or list/array of length N.")

    # Handle sigma
    if np.isscalar(sigma):
        sigma_list = [sigma] * N_mat
    elif isinstance(sigma, (list, np.ndarray)) and len(sigma) == N_mat:
        sigma_list = list(sigma)
    else:
        raise ValueError("sigma must be float or list/array of length N_mat.")

    spectra = np.zeros((N_mat, n_points), dtype=float)

    for i in range(N_mat):
        n_peaks = S_list[i]
        peak_positions = rng.choice(two_theta, size=n_peaks, replace=False)
        peak_intensities = rng.exponential(scale=1.0, size=n_peaks)  # positive distribution

        for pos, intensity in zip(peak_positions, peak_intensities):
            # Find nearest index in two_theta
            idx = np.argmin(np.abs(two_theta - pos))
            spectra[i, idx] += intensity

        # Apply Gaussian convolution if sigma > 0
        if sigma_list[i] > 0:
            # Convert sigma in units of two_theta to indices
            step = np.mean(np.diff(two_theta))
            sigma_in_bins = sigma_list[i] / step
            spectra[i] = gaussian_filter1d(spectra[i], sigma_in_bins)

    return spectra.astype(np.float32)


# def make_material_phantom(N_mat, phantom, labels, seed=None):
#     rng = np.random.default_rng(seed)

#     K, Nx, Ny = phantom.shape
#     material_phantom = np.zeros((N_mat, K, Nx, Ny), dtype=phantom.dtype)

#     unique_regions = np.unique(labels)
#     # Exclude background (-1)
#     regions = unique_regions[unique_regions >= 0]

#     # Assign a random material index to each region
#     region_to_material = {
#         region: rng.integers(low=0, high=N_mat) for region in regions
#     }

#     for region, L in region_to_material.items():
#         mask = (labels == region)
#         material_phantom[L, :, mask] = phantom[:, mask].T

#     return material_phantom, region_to_material



def make_material_phantom(K_list, nx=100, ny=100, S=10, n_regions=20, R = None, seed=None):
    N_mat = len(K_list)
    rng = np.random.default_rng(seed)

    _, labels, _, _ = make_sparse_phantom(K=1, nx=nx, ny=ny, S=1, n_regions=n_regions, R=R, seed=seed)

    unique_regions = np.unique(labels)
    # Exclude background (-1)
    regions = unique_regions[unique_regions >= 0]

    # Assign a random material index to each region
    region_to_material = {
        region: rng.integers(low=0, high=N_mat) for region in regions
    }

    material_labels = -1*np.ones((N_mat, nx, ny))

    for region, L in region_to_material.items():
        mask = (labels == region)
        material_labels[L][mask] = region


    material_coeffs = []
    for idx_K in range(N_mat):
        K = K_list[idx_K]
        material = np.zeros((K, nx, ny))
        unique_regions_material = np.unique(material_labels[idx_K])
        regions_material = unique_regions_material[unique_regions_material >= 0]
        for idx_region in range(len(regions_material)):
            region_id = regions_material[idx_region]
            idx = rng.choice(K, size=S, replace=True)
            coef = np.zeros(K, dtype=float)
            coef[idx] = rng.random(S)
            coef = coef/coef.sum()

            mask = (material_labels[idx_K] == region_id)

            material[:, mask] = coef[:, None]
        material_coeffs.append(material)


    return material_coeffs, material_labels


    # material_phantom = []
    # for idx_K in range(len(K_list)):
    #     material = np.zeros((K_list[idx_K], nx, ny))

    #     material_phantom.append(material)
    #     make_sparse_phantom(K=K_list[idx_K], nx=nx, ny=ny, S=S, n_regions=n_regions, R=R, seed=seed)

    # K, Nx, Ny = phantom.shape
    # material_phantom = np.zeros((N_mat, K, Nx, Ny), dtype=phantom.dtype)


    #     material_phantom[L, :, mask] = phantom[:, mask].T

    # return material_phantom, region_to_material