import numpy as np

def get_probed_coordinates(geom, integration_samples=1, full_circle_covered=True):
    """ Calculates and returns the probed polar and azimuthal coordinates on the unit sphere at
    each angle of projection and for each detector segment in the system's geometry.
    """
    n_proj = len(geom)
    n_seg = len(geom.detector_angles)
    probed_directions_zero_rot = np.zeros((n_seg, integration_samples, 3))
    # Impose symmetry if needed.
    if not full_circle_covered:
        shift = np.pi
    else:
        shift = 0
    det_bin_middles_extended = np.copy(geom.detector_angles)
    det_bin_middles_extended = np.insert(det_bin_middles_extended, 0, det_bin_middles_extended[-1] + shift)
    det_bin_middles_extended = np.append(det_bin_middles_extended, det_bin_middles_extended[1] + shift)

    for ii in range(n_seg):

        # Check if the interval from the previous to the next bin goes over the -pi +pi discontinuity
        before = det_bin_middles_extended[ii]
        now = det_bin_middles_extended[ii + 1]
        after = det_bin_middles_extended[ii + 2]

        if abs(before - now + 2 * np.pi) < abs(before - now):
            before = before + 2 * np.pi
        elif abs(before - now - 2 * np.pi) < abs(before - now):
            before = before - 2 * np.pi

        if abs(now - after + 2 * np.pi) < abs(now - after):
            after = after - 2 * np.pi
        elif abs(now - after - 2 * np.pi) < abs(now - after):
            after = after + 2 * np.pi

        # Generate a linearly spaced set of angles covering the detector segment
        start = 0.5 * (before + now)
        end = 0.5 * (now + after)
        inc = (end - start) / integration_samples
        angles = np.linspace(start + inc / 2, end - inc / 2, integration_samples)

        # Make the zero-rotation-frame vectors corresponding to the given angles
        probed_directions_zero_rot[ii, :, :] = np.cos(angles[:, np.newaxis]) * \
            geom.detector_direction_origin[np.newaxis, :]

        probed_directions_zero_rot[ii, :, :] += np.sin(angles[:, np.newaxis]) * \
            geom.detector_direction_positive_90[np.newaxis, :]

    twothetahalf = geom.two_theta / 2
    probed_directions_zero_rot = +probed_directions_zero_rot * np.cos(twothetahalf)\
        - np.sin(twothetahalf) * geom.p_direction_0

    # Initialize array for vectors
    probed_direction_vectors = np.zeros((n_proj, n_seg, integration_samples, 3), dtype=np.float64)
    # Calculate all the rotations
    probed_direction_vectors[...] = \
        np.einsum('kij,mli->kmlj', geom.rotations_as_array, probed_directions_zero_rot)

    return probed_direction_vectors

