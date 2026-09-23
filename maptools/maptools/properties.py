from collections import deque

import numpy as np
import xfab.symmetry
import xfab.tools
from scipy.ndimage import binary_fill_holes
from scipy.spatial.transform import Rotation
from tqdm.notebook import tqdm
from xfab.symmetry import Umis


def FWHM(p):
    halfmax = np.max(p) / 2

    left = None
    right = None

    for i in range(len(p)):
        if p[i] > halfmax:
            left = i
            break
    for i in range(len(p) - 1, 0, -1):
        if p[i] > halfmax:
            right = i
            break

    if left is None or right is None:
        return 0

    if left - 1 < 0 or right + 1 >= len(p):
        return 0

    x1 = left - 1
    x2 = left
    xl = x1 + (halfmax - p[x1]) / (p[x2] - p[x1])

    x1 = right
    x2 = right + 1
    xr = x1 + (halfmax - p[x1]) / (p[x2] - p[x1])

    return xr - xl


def get_u_map(ubi_map, sample_mask):
    u_map = np.zeros_like(ubi_map)
    for i in range(ubi_map.shape[0]):
        for j in range(ubi_map.shape[1]):
            if sample_mask[i, j]:
                ubi = ubi_map[i, j]
                u = xfab.tools.ubi_to_u(ubi)
                u_map[i, j] = u
    return u_map


def _align(us, crystal_system):
    """Align a set of orientation matrices as closely as possible given a lattice symmetry."""
    rot = xfab.symmetry.rotations(crystal_system)
    ref = us[0]
    newu = np.zeros_like(us)
    for i, u in enumerate(us):
        mis = xfab.symmetry.Umis(u, ref, crystal_system)
        rotindex = np.argmin(mis[:, 1])
        newu[i, :, :] = u @ rot[rotindex]
    return newu


def kernel_average_misorientation(
    orientation_map,
    footprint,
    crystal_system,
    misorientation_threshold=(0, np.inf),
    mask=None,
    fill_value=np.nan,
):
    """Apply a kernel average misorientation (KAM) filter to the input orientation map.

    The KAM filter is designed as described here: https://mtex-toolbox.github.io/EBSDKAM.html

    Args:
        orientation_map (:obj:`numpy array`): the pixelated orientation matrix field, shape=(M, N, 3, 3).
        footprint (:obj:`numpy array`): boolean array defining the kenrel neighbourhood, shape=(m, n).
        crystal_system (:obj:int): crystal_system number must be one of 1: Triclinic, 2: Monoclinic,
            3: Orthorhombic, 4: Tetragonal, 5: Trigonal, 6: Hexagonal, 7: Cubic
        misorientation_threshold (tuple of :obj:`float`): Reject misorientations outside specified range.
            Defaults to (0, np.inf).
        mask (:obj:`numpy array`): Boolean array, where to skipp pixels, shape=(M, N). Defaults to None.
        fill_value (:obj:`numpy array`): To put where mask is false. Defaults to np.nan.

    Returns:
        :obj:`numpy array`: the scalar kernel average misorientation map, shape=(M, N).
    """
    assert footprint.shape[0] % 2 != 0
    assert footprint.shape[1] % 2 != 0

    m = footprint.shape[0] // 2
    n = footprint.shape[1] // 2
    kam_map = np.zeros((orientation_map.shape[0], orientation_map.shape[1]))
    lower_bound, upper_bound = misorientation_threshold

    for i in tqdm(range(m, orientation_map.shape[0] - m)):
        for j in range(n, orientation_map.shape[0] - n):
            if mask is not None and mask[i, j]:
                U = orientation_map[i, j]
                local_average_misorientation = 0
                number_of_pixels = 0
                for k in range(footprint.shape[0]):
                    for l in range(footprint.shape[1]):
                        if footprint[k, l] > 0:
                            row = i - m + k
                            col = j - n + l
                            if mask is not None and mask[row, col]:
                                u = orientation_map[row, col]
                                misorientation = Umis(U, u, crystal_system)[:, 1].min()
                                if (
                                    misorientation < upper_bound
                                    and misorientation > lower_bound
                                ):
                                    local_average_misorientation += misorientation
                                    number_of_pixels += 1
                if number_of_pixels > 0:
                    kam_map[i, j] += local_average_misorientation / number_of_pixels
            else:
                kam_map[i, j] = fill_value
    if mask is not None:
        kam_map[~mask] = fill_value
    return kam_map


def flood_fill(
    orientation_map,
    seed_point,
    footprint,
    crystal_system,
    local_disorientation_tolerance,
    global_disorientation_tolerance,
    mask=None,
    background_value=np.nan,
    fill_holes=False,
    max_grains=99,
    min_grain_size=0,
    verbose=False,
):
    assert footprint.shape[0] % 2 != 0
    assert footprint.shape[1] % 2 != 0

    M, N = orientation_map.shape[0], orientation_map.shape[1]
    segmentation = np.zeros((M, N))
    skipps = np.ones((M, N), dtype=bool)
    label = 1
    segmentation[seed_point[0], seed_point[1] : seed_point[1] + 3] = label
    done = False
    iteration = 0
    while not done and iteration < max_grains:
        rows, cols = np.where(mask * (segmentation == 0) * skipps)
        if len(rows) > 0:
            n = np.random.randint(0, len(rows))
            seed_point = (rows[n], cols[n])
            grain_mask = _flood(
                orientation_map,
                seed_point,
                footprint,
                crystal_system,
                local_disorientation_tolerance,
                global_disorientation_tolerance,
                mask,
            )
            if fill_holes:
                grain_mask = binary_fill_holes(grain_mask)
            if np.sum(grain_mask) > min_grain_size:
                segmentation[grain_mask] = label
                label += 1
            else:
                skipps[grain_mask] = False
            iteration += 1
            if verbose:
                print(
                    "Iteration ",
                    iteration,
                    ", found grain with : ",
                    np.sum(grain_mask),
                    " voxels @, seed_point, ",
                    np.sum(grain_mask),
                )
        else:
            done = True

    segmentation[segmentation == 0] = background_value
    return segmentation


def _flood(
    orientation_map,
    seed_point,
    footprint,
    crystal_system,
    local_disorientation_tolerance,
    global_disorientation_tolerance,
    mask,
):
    m = footprint.shape[0] // 2
    n = footprint.shape[1] // 2
    flood_mask = np.zeros(
        (orientation_map.shape[0], orientation_map.shape[1]), dtype=bool
    )

    i, j = seed_point
    flood_mask[i, j] = True
    if mask is not None and mask[i, j] == 0:
        raise ValueError("Seed point not in mask")

    unchartered_indices = deque([seed_point])
    while len(unchartered_indices) > 0:
        i, j = unchartered_indices.pop()
        U = orientation_map[i, j]

        if np.sum(flood_mask) < 20:
            us = _align(orientation_map[flood_mask, :, :], crystal_system)
            global_U = Rotation.from_matrix(us).mean().as_matrix()

        for k in range(footprint.shape[0]):
            for l in range(footprint.shape[1]):
                if footprint[k, l] > 0:
                    row = i - m + k
                    col = j - n + l
                    if not flood_mask[row, col] and mask is not None and mask[row, col]:
                        u = orientation_map[row, col]
                        misorientation = Umis(U, u, crystal_system)[:, 1].min()

                        if misorientation < local_disorientation_tolerance:
                            glob_misorientation = Umis(global_U, u, crystal_system)[
                                :, 1
                            ].min()
                            if glob_misorientation < global_disorientation_tolerance:
                                flood_mask[row, col] = True
                                unchartered_indices.appendleft((row, col))
    return flood_mask
