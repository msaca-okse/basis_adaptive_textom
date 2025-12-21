from typing import Any

from math import floor
from numba import config, cuda, int32, float32, float64, void
import numpy as np


def _is_cuda_array(item: Any):
    """ Internal method which assists in debugging. """
    if config.ENABLE_CUDASIM:
        return False
    else:
        return cuda.is_cuda_array(item)


def john_transform_cuda(field: np.ndarray[float], projections: np.ndarray[float],
                        unit_vector_p: np.ndarray[float], unit_vector_j: np.ndarray[float],
                        unit_vector_k: np.ndarray[float], offsets_j: np.ndarray[float],
                        offsets_k: np.ndarray[float]):
    """ Frontend for performing the John transform with parallel
    GPU computing.

    Parameters
    ----------
    field
        The field to be projected, with 4 dimensions. The last index should
        have the same size as the last index of ``projections``. Can be either
        a `numpy.ndarray`, or a device array that implements the CUDA array
        interface. If a device array is given, no copying to device is needed.
    projections
        A 4-dimensional numpy array where the projections are stored.
        The first index runs over the different projection directions. Can be either
        a `numpy.ndarray`, or a device array that implements the CUDA array
        interface. If a device array is given, no copying or synchronization is needed.
    unit_vector_p
        The direction of projection in Cartesian coordinates.
    unit_vector_j
        One of the directions for the pixels of ``projection``.
    unit_vector_k
        The other direction for the pixels of ``projection``.
    offsets_j
        Offsets which align projections in the direction of `j`
    offsets_k
        Offsets which align projections in the direction of `k`.

    Notes
    -----
    For mathematical details, see :func:`~.john_transform.john_transform`.
    """
    # Always false for now as we are sticking to ``float32``.
    if False:
        cuda_float = float64
        numpy_float = np.float64
        cuda_x4_type = cuda.float64x4
    else:
        cuda_float = float32
        numpy_float = np.float32
        cuda_x4_type = cuda.float32x4

    # Find zeroth, first and second directions for indexing. Zeroth is the maximal projection direction.

    # (x, y, z), (y, x, z), (z, x, y)
    direction_0_index = np.argmax(abs(unit_vector_p), axis=1).reshape(-1, 1).astype(np.int32)
    direction_1_index = 1 * (direction_0_index == 0).astype(np.int32).reshape(-1, 1).astype(np.int32)
    direction_2_index = (2 - (direction_0_index == 2)).astype(np.int32).reshape(-1, 1).astype(np.int32)

    # Step size for direction 1 and 2.
    step_sizes_1 = (np.take_along_axis(unit_vector_p, direction_1_index, 1) /
                    np.take_along_axis(unit_vector_p, direction_0_index, 1)).astype(numpy_float).ravel()
    step_sizes_2 = (np.take_along_axis(unit_vector_p, direction_2_index, 1) /
                    np.take_along_axis(unit_vector_p, direction_0_index, 1)).astype(numpy_float).ravel()

    # Shape in each of the three directions.
    dimensions_0 = np.array(field.shape, dtype=numpy_float)[direction_0_index.ravel()]
    dimensions_1 = np.array(field.shape, dtype=numpy_float)[direction_1_index.ravel()]
    dimensions_2 = np.array(field.shape, dtype=numpy_float)[direction_2_index.ravel()]

    # Correction factor for length of line when taking a one-slice step.
    distance_multipliers = np.sqrt(1.0 + step_sizes_1 ** 2 + step_sizes_2 ** 2).astype(numpy_float)

    max_index = projections.shape[0]
    max_j = projections.shape[1]
    max_k = projections.shape[2]

    # CUDA chunking and memory size constants.
    channels = int(field.shape[-1])
    threads_j = int(8)
    threads_k = int(8)
    threads_angle = int(4)
    # Indices to navigate each projection. s is the surface positioning.
    k_vectors = unit_vector_k.astype(numpy_float)
    j_vectors = unit_vector_j.astype(numpy_float)
    s_vectors = (unit_vector_k * (-0.5 * max_k + offsets_k.reshape(-1, 1)) +
                 unit_vector_j * (-0.5 * max_j + offsets_j.reshape(-1, 1))).astype(numpy_float)

    # (0, 1, 2) if x main, (1, 0, 2) if y main, (2, 0, 1) if z main.
    direction_indices = np.stack((direction_0_index.ravel(),
                                  direction_1_index.ravel(),
                                  direction_2_index.ravel()), axis=1)

    # Bilinear interpolation over each slice.

    @cuda.jit(void(cuda_float[:, :, :, ::1], int32, cuda_float,
              cuda_float, cuda_float, int32[::1], cuda_float[::1]), device=True,
              lineinfo=True, fastmath=True, cache=True)
    def bilinear_interpolation(field: np.ndarray[float],
                               direction_0,
                               r0: float,
                               r1: float,
                               r2: float,
                               dimensions,
                               accumulator: np.ndarray[float]):
        """ Kernel for bilinear interpolation. Replaces texture interpolation."""
        if not (-1 <= r1 < dimensions[1] and -1 <= r2 < dimensions[2]):
            return
        # At edges, interpolate between value and 0.
        if (0 <= r1 < dimensions[1] - 1):
            r1_weight = int32(1)
        else:
            r1_weight = int32(0)

        if (0 <= r2 < dimensions[2] - 1):
            r2_weight = int32(1)
        else:
            r2_weight = int32(0)

        if int32(r1) == -1:
            r1_edge_weight = int32(1)
        else:
            r1_edge_weight = int32(0)

        if int32(r2) == -1:
            r2_edge_weight = int32(1)
        else:
            r2_edge_weight = int32(0)

        weight_1 = cuda_float((r1 - floor(r1)) * r1_weight) * (1 - r1_edge_weight)
        weight_2 = cuda_float((r2 - floor(r2)) * r2_weight) * (1 - r2_edge_weight)
        t = cuda_x4_type((1 - weight_1) * (1 - weight_2),
                         (weight_1 * weight_2),
                         ((1 - weight_1) * weight_2),
                         ((1 - weight_2) * weight_1))
        # Branch should be abstracted away by compiler, but could be done with pointer arithmetic.
        if (direction_0 == 0):
            x = int32(floor(r0))
            y = int32(floor(r1) + r1_edge_weight)
            y2 = y + r1_weight
            z = int32(floor(r2) + r2_edge_weight)
            z2 = z + r2_weight
            for i in range(accumulator.size):
                accumulator[i] += field[x, y, z, i] * t.x
                accumulator[i] += field[x, y2, z, i] * t.w
                accumulator[i] += field[x, y, z2, i] * t.z
                accumulator[i] += field[x, y2, z2, i] * t.y

        elif (direction_0 == 1):
            x = int32(floor(r1) + r1_edge_weight)
            x2 = x + r1_weight
            y = int32(floor(r0))
            z = int32(floor(r2) + r2_edge_weight)
            z2 = z + r2_weight
            for i in range(accumulator.size):
                accumulator[i] += field[x, y, z, i] * t.x
                accumulator[i] += field[x2, y, z, i] * t.w
                accumulator[i] += field[x, y, z2, i] * t.z
                accumulator[i] += field[x2, y, z2, i] * t.y

        elif (direction_0 == 2):
            x = int32(floor(r1) + r1_edge_weight)
            x2 = x + r1_weight
            y = int32(floor(r2) + r2_edge_weight)
            y2 = y + r2_weight
            z = int32(floor(r0))
            for i in range(accumulator.size):
                accumulator[i] += field[x, y, z, i] * t.x
                accumulator[i] += field[x2, y, z, i] * t.w
                accumulator[i] += field[x, y2, z, i] * t.z
                accumulator[i] += field[x2, y2, z, i] * t.y

    @cuda.jit(void(cuda_float[:, :, :, ::1], cuda_float[:, :, :, ::1]),
              lineinfo=True, fastmath=True, cache=True)
    def john_transform_inner(field: np.ndarray[float], projection: np.ndarray[float]):
        """ Performs the John transform of a field. Relies on a large number
        of pre-defined constants outside the kernel body. """
        index = cuda.blockIdx.y * threads_angle + cuda.threadIdx.y
        j = cuda.threadIdx.x + threads_j * (
                cuda.blockIdx.x % (
                    (max_j + threads_j - 1) // threads_j))

        if (j >= max_j) or (index >= max_index):
            return

        start_k = threads_k * (cuda.blockIdx.x // (
            (max_j + threads_j - 1) // threads_j))
        end_k = start_k + threads_k
        if end_k > max_k:
            end_k = max_k

        if start_k >= end_k:
            return

        # Define compile-time constants.
        step_size_1 = step_sizes_1[index]
        step_size_2 = step_sizes_2[index]
        k_vectors_c = k_vectors[index]
        j_vectors_c = j_vectors[index]
        s_vectors_c = s_vectors[index]
        dimensions_0_c = dimensions_0[index]
        dimensions_1_c = dimensions_1[index]
        dimensions_2_c = dimensions_2[index]
        dimensions = cuda.local.array(3, int32)
        dimensions[0] = dimensions_0_c
        dimensions[1] = dimensions_1_c
        dimensions[2] = dimensions_2_c
        direction_indices_c = direction_indices[index]
        distance_multiplier = distance_multipliers[index]

        # Could be chunked for very asymmetric samples.
        start_slice = 0
        end_slice = start_slice + dimensions[0]
        accumulator = cuda.local.array(channels, cuda_float)

        for k in range(start_k, end_k):
            for i in range(channels):
                accumulator[i] = 0.

            fj = cuda_float(j) + 0.5
            fk = cuda_float(k) + 0.5

            # Initial coordinates of projection.
            start_position_0 = (s_vectors_c[direction_indices_c[0]] +
                                fj * j_vectors_c[direction_indices_c[0]] +
                                fk * k_vectors_c[direction_indices_c[0]])
            start_position_1 = (s_vectors_c[direction_indices_c[1]] +
                                fj * j_vectors_c[direction_indices_c[1]] +
                                fk * k_vectors_c[direction_indices_c[1]])
            start_position_2 = (s_vectors_c[direction_indices_c[2]] +
                                fj * j_vectors_c[direction_indices_c[2]] +
                                fk * k_vectors_c[direction_indices_c[2]])

            # Centering w.r.t volume.

            position_0 = cuda_float(start_slice) + 0.5
            position_1 = cuda_float(step_size_1 * (cuda_float(start_slice) -
                                    0.5 * dimensions[0] - start_position_0 + 0.5) +
                                    start_position_1 + 0.5 * dimensions[1] - 0.5)
            position_2 = cuda_float(step_size_2 * (cuda_float(start_slice) -
                                    0.5 * dimensions[0] -
                                    start_position_0 + 0.5) +
                                    start_position_2 + 0.5 * dimensions[2] - 0.5)

            for i in range(start_slice, end_slice):
                bilinear_interpolation(field, direction_indices_c[0],
                                       position_0, position_1, position_2, dimensions, accumulator)
                position_0 += cuda_float(1.0)
                position_1 += step_size_1
                position_2 += step_size_2

            for i in range(channels):
                projection[index, j, k, i] = accumulator[i] * distance_multiplier

    # Launching of kernel.
    bpg = (((max_j + threads_j - 1) // threads_j) *
           ((max_k + threads_k - 1) // threads_k),
           ((projections.shape[0] + threads_angle - 1) // threads_angle))
    tpb = (threads_j, threads_angle)
    john_transform_grid = john_transform_inner[bpg, tpb]

    def transform_with_transfer(field: np.ndarray[float],
                                projections: np.ndarray[float]) -> np.ndarray[float]:
        if _is_cuda_array(field):
            assert field.dtype == 'float32'
            device_field = cuda.as_cuda_array(field)
            transfer_projections = False
        else:
            assert field.dtype == np.float32
            device_field = cuda.to_device(field)
            transfer_projections = True

        if _is_cuda_array(projections):
            assert projections.dtype == 'float32'
            device_projections = cuda.as_cuda_array(projections)
        else:
            assert projections.dtype == np.float32
            device_projections = cuda.to_device(projections)

        john_transform_grid(device_field, device_projections)
        if transfer_projections:
            return device_projections.copy_to_host()
        return device_projections
    return transform_with_transfer


def john_transform_adjoint_cuda(field: np.ndarray[float], projections: np.ndarray[float],
                                unit_vector_p: np.ndarray[float], unit_vector_j: np.ndarray[float],
                                unit_vector_k: np.ndarray[float], offsets_j: np.ndarray[float],
                                offsets_k: np.ndarray[float]):
    """ Frontend for performing the adjoint of the John transform with parallel
    GPU computing.

    Parameters
    ----------
    field
        The field into which the adjoint is projected, with 4 dimensions. The last index should
        have the same size as the last index of ``projections``. Can be either
        a `numpy.ndarray`, or a device array that implements the CUDA array
        interface. If a device array is given, no copying to device is needed.
    projections
        The projections from which the adjoint is calculated.
        The first index runs over the different projection directions. Can be either
        a `numpy.ndarray`, or a device array that implements the CUDA array
        interface. If a device array is given, no copying or synchronization is needed.
    unit_vector_p
        The direction of projection in Cartesian coordinates.
    unit_vector_j
        One of the directions for the pixels of ``projection``.
    unit_vector_k
        The other direction for the pixels of ``projection``.
    offsets_j
        Offsets which align projections in the direction of `j`
    offsets_k
        Offsets which align projections in the direction of `k`.

    Notes
    -----
    For mathematical details, see :func:`~.john_transform.john_transform_adjoint`.
    """
    # Always false as we are sticking to float32
    if False:
        cuda_float = float64
        numpy_float = np.float64
        cuda_x4_type = cuda.float64x4
    else:
        cuda_float = float32
        numpy_float = np.float32
        cuda_x4_type = cuda.float32x4

    max_p = projections.shape[0]
    max_j = projections.shape[1]
    max_k = projections.shape[2]
    max_x = field.shape[0]
    max_y = field.shape[1]
    max_z = field.shape[2]
    # CUDA chunking and memory size constants.
    channels = field.shape[-1]
    threads_z = int(8)
    # threads_angle = int(512)  # Not used for now.
    threads_x = int(8)
    threads_y = int(4)

    # Projection vectors. s for positioning the projection.
    p_vectors = unit_vector_p.astype(numpy_float)
    k_vectors = unit_vector_k.astype(numpy_float)
    j_vectors = unit_vector_j.astype(numpy_float)
    s_vectors = (unit_vector_k * (-0.5 * max_k + offsets_k.reshape(-1, 1)) +
                 unit_vector_j * (-0.5 * max_j + offsets_j.reshape(-1, 1))).astype(numpy_float)

    # Translate volume steps to normalized projection steps. Can add support for non-square voxels.
    vector_norm = np.einsum('...i, ...i', p_vectors, np.cross(j_vectors, k_vectors))
    norm_j = -np.cross(p_vectors, k_vectors) / vector_norm[..., None]
    norm_k = np.cross(p_vectors, j_vectors) / vector_norm[..., None]
    norm_offset_j = -np.einsum('...i, ...i', p_vectors, np.cross(s_vectors, k_vectors)) / vector_norm
    norm_offset_k = np.einsum('...i, ...i', p_vectors, np.cross(s_vectors, j_vectors)) / vector_norm

    @cuda.jit(void(cuda_float[:, :, :, ::1], cuda_float,
                   cuda_float, cuda_float, cuda_float[::1]), device=True,
              lineinfo=True, fastmath=True, cache=True)
    def bilinear_interpolation_projection(projection: np.ndarray[float],
                                          r0: float,
                                          r1: float,
                                          r2: float,
                                          accumulator: np.ndarray[float]):
        if not (-1 <= r1 < max_j and -1 <= r2 < max_k):
            return
        # At edges, use nearest-neighbor interpolation.
        if (0 <= r1 + 1 < max_j):
            r1_weight = int32(1)
        else:
            r1_weight = int32(0)

        if (0 <= r2 + 1 < max_k):
            r2_weight = int32(1)
        else:
            r2_weight = int32(0)

        if int32(r1) == -1:
            r1_edge_weight = 1
        else:
            r1_edge_weight = 0

        if int32(r2) == -1:
            r2_edge_weight = 1
        else:
            r2_edge_weight = 0

        y_weight = cuda_float((r1 - floor(r1)) * r1_weight) * (1 - r1_edge_weight)
        z_weight = cuda_float((r2 - floor(r2)) * r2_weight) * (1 - r2_edge_weight)
        x = int32(floor(r0))
        y = int32(floor(r1) + r1_edge_weight)
        y2 = y + r1_weight
        z = int32(floor(r2) + r2_edge_weight)
        z2 = z + r2_weight
        t = cuda_x4_type((1 - z_weight) * (1 - y_weight),
                         (z_weight * y_weight),
                         ((1 - z_weight) * y_weight),
                         ((1 - y_weight) * z_weight))
        for i in range(channels):
            accumulator[i] += projection[x, y, z, i] * t.x
            accumulator[i] += projection[x, y2, z, i] * t.z
            accumulator[i] += projection[x, y, z2, i] * t.w
            accumulator[i] += projection[x, y2, z2, i] * t.y

    @cuda.jit(void(cuda_float[:, :, :, ::1], cuda_float[:, :, :, ::1]),
              lineinfo=True, fastmath=True, cache=True)
    def john_transform_adjoint_inner(field: np.ndarray[float], projection: np.ndarray[float]):
        """ Performs the John transform of a field. Relies on a large number
        of pre-defined constants outside the kernel body. """
        # Indexing of volume coordinates.
        x = cuda.threadIdx.x + threads_x * (cuda.blockIdx.x % (
                       (max_x + threads_x - 1) // threads_x))
        y = cuda.threadIdx.y + threads_y * (cuda.blockIdx.x // (
                       (max_x + threads_x - 1) // threads_x))
        if (x >= max_x) or (y >= max_y):
            return

        # Stride in z.
        z = cuda.blockIdx.y * threads_z
        z_acc = cuda.local.array(channels, cuda_float)
        # Center of voxel and coordinate system.
        fx = x - 0.5 * max_x + 0.5
        fy = y - 0.5 * max_y + 0.5
        steps = min(max(max_z - z, 0), threads_z)

        # Compile time constants
        start_index = 0
        stop_index = max_p
        norm_j_c = norm_j[start_index:stop_index]
        norm_k_c = norm_k[start_index:stop_index]
        norm_offset_j_c = norm_offset_j[start_index:stop_index]
        norm_offset_k_c = norm_offset_k[start_index:stop_index]
        for ii in range(steps):
            fz = z - 0.5 * max_z + 0.5
            for j in range(channels):
                z_acc[j] = 0.0

            for a in range(start_index, stop_index):
                # Center with respect to projection.
                fj = (norm_offset_j_c[a] + fx * norm_j_c[a][0] +
                      fy * norm_j_c[a][1] + fz * norm_j_c[a][2] - 0.5)
                fk = (norm_offset_k_c[a] + fx * norm_k_c[a][0] +
                      fy * norm_k_c[a][1] + fz * norm_k_c[a][2] - 0.5)
                bilinear_interpolation_projection(projection,
                                                  a, fj, fk, z_acc)
            for i in range(channels):
                field[x, y, z, i] = z_acc[i]
            z += 1

    @cuda.jit(void(cuda_float[:, :, :, ::1], cuda_float,
              cuda_float, cuda_float, cuda_float[::1], int32), device=True,
              lineinfo=True, fastmath=True, cache=True)
    def bilinear_interpolation_projection_one_channel(projection: np.ndarray[float],
                                                      r0: float,
                                                      r1: float,
                                                      r2: float,
                                                      accumulator: np.ndarray[float],
                                                      index: int):
        if not (-1 <= r1 < max_j and -1 <= r2 < max_k):
            return
        # At edges, use nearest-neighbor interpolation.
        if (0 <= r1 + 1 < max_j):
            r1_weight = int32(1)
        else:
            r1_weight = int32(0)

        if (0 <= r2 + 1 < max_k):
            r2_weight = int32(1)
        else:
            r2_weight = int32(0)

        if int32(r1) == -1:
            r1_edge_weight = 1
        else:
            r1_edge_weight = 0

        if int32(r2) == -1:
            r2_edge_weight = 1
        else:
            r2_edge_weight = 0

        y_weight = cuda_float((r1 - floor(r1)) * r1_weight) * (1 - r1_edge_weight)
        z_weight = cuda_float((r2 - floor(r2)) * r2_weight) * (1 - r2_edge_weight)
        x = int32(floor(r0))
        y = int32(floor(r1) + r1_edge_weight)
        y2 = y + r1_weight
        z = int32(floor(r2) + r2_edge_weight)
        z2 = z + r2_weight
        t = cuda_x4_type((1 - z_weight) * (1 - y_weight),
                         (z_weight * y_weight),
                         ((1 - z_weight) * y_weight),
                         ((1 - y_weight) * z_weight))
        accumulator[index] += projection[x, y, z, 0] * t.x
        accumulator[index] += projection[x, y2, z, 0] * t.z
        accumulator[index] += projection[x, y, z2, 0] * t.w
        accumulator[index] += projection[x, y2, z2, 0] * t.y

    @cuda.jit(void(cuda_float[:, :, :, ::1], cuda_float[:, :, :, ::1]),
              lineinfo=True, fastmath=True, cache=True)
    def john_transform_adjoint_one_channel(field: np.ndarray[float],
                                           projection: np.ndarray[float]):
        """ Performs the John transform of a single-channel field."""
        # Indexing of volume coordinates.
        x = cuda.threadIdx.x + threads_x * (cuda.blockIdx.x % (
                       (max_x + threads_x - 1) // threads_x))
        y = cuda.threadIdx.y + threads_y * (cuda.blockIdx.x // (
                       (max_x + threads_x - 1) // threads_x))
        if (x >= max_x) or (y >= max_y):
            return

        # Stride in z.
        z = cuda.blockIdx.y * threads_z
        z_acc = cuda.local.array(threads_z, cuda_float)
        # Center of voxel and coordinate system.
        fx = x - 0.5 * max_x + 0.5
        fy = y - 0.5 * max_y + 0.5
        steps = min(max(max_z - z, 0), threads_z)

        # Compile time constants
        start_index = 0
        stop_index = max_p
        norm_j_c = norm_j[start_index:stop_index]
        norm_k_c = norm_k[start_index:stop_index]
        norm_offset_j_c = norm_offset_j[start_index:stop_index]
        norm_offset_k_c = norm_offset_k[start_index:stop_index]
        fz = z - 0.5 * max_z + 0.5
        for i in range(steps):
            z_acc[i] = 0.0
        """
        Chunking over projections is possible but of dubious value since
        it means reduction is necessary and thus pre-initialization
        of output array.
        """
        for a in range(start_index, stop_index):
            # Center with respect to projection.
            fj = norm_offset_j_c[a] + fx * norm_j_c[a][0] + fy * norm_j_c[a][1] + fz * norm_j_c[a][2] - 0.5
            fk = norm_offset_k_c[a] + fx * norm_k_c[a][0] + fy * norm_k_c[a][1] + fz * norm_k_c[a][2] - 0.5
            for i in range(steps):
                bilinear_interpolation_projection_one_channel(
                        projection,
                        a, fj, fk,
                        z_acc, i)
                fj += norm_j_c[a][2]
                fk += norm_k_c[a][2]
        for i in range(steps):
            field[x, y, z + i, 0] = z_acc[i]

    bpg = (((max_x + threads_x - 1) // threads_x) *
           ((max_y + threads_y - 1) // threads_y),
           ((max_z + threads_z - 1) // threads_z))
    tpb = (threads_x, threads_y)
    if channels > 1:
        john_transform_adjoint_grid = john_transform_adjoint_inner[bpg, tpb]
    else:
        john_transform_adjoint_grid = john_transform_adjoint_one_channel[bpg, tpb]

    def transform_with_transfer(field: np.ndarray[float],
                                projections: np.ndarray[float]) -> np.ndarray[float]:
        if _is_cuda_array(field):
            assert field.dtype == 'float32'
            device_field = cuda.as_cuda_array(field)
        else:
            assert field.dtype == np.float32
            device_field = cuda.to_device(field)

        if _is_cuda_array(projections):
            assert projections.dtype == 'float32'
            device_projections = cuda.as_cuda_array(projections)
            transfer_field = False
        else:
            assert projections.dtype == np.float32
            device_projections = cuda.to_device(projections)
            transfer_field = True
        john_transform_adjoint_grid(device_field,
                                    device_projections)
        if transfer_field:
            return device_field.copy_to_host()
        return device_field

    return transform_with_transfer





from math import floor
from numba import njit, int32, float32, float64, prange, void
from numpy.typing import NDArray
import numpy as np


def john_transform(
        field: NDArray[float],
        projections: NDArray[float],
        unit_vector_p: NDArray[float],
        unit_vector_j: NDArray[float],
        unit_vector_k: NDArray[float],
        offsets_j: NDArray[float],
        offsets_k: NDArray[float],
        float_type: str = 'float32') -> callable:
    r""" Frontend for performing the John transform with parallel
    CPU computing, using an algorithm akin to :func:`mumott.core.john_transform_cuda`.

    Parameters
    ----------
    field
        The field to be projected, with 4 dimensions. The last index should
        have the same size as the last index of :attr:`projections`.
    projections
        A 4-dimensional numpy array where the projections are stored.
        The first index runs over the different projection directions.
    unit_vector_p
        The direction of projection in Cartesian coordinates.
    unit_vector_j
        One of the directions for the pixels of :attr:`projections``.
    unit_vector_k
        The other direction for the pixels of :attr:`projections`.
    offsets_j
        Offsets which align projections in the direction of `j`
    offsets_k
        Offsets which align projections in the direction of `k`.
    float_type
        Whether to use 'float64' (default) or 'float32'. The argument should be supplied
        as a string. The types of :attr:`field` and :attr:`projections` must match this type.


    Notes
    -----
    The computation performed by this function may be written as

    .. math::

        p(I, J, K)_i = \sum_{s=0}^{N-1} d \cdot \sum_{t=0}^3 t_w V_i(\lfloor \mathbf{r}_j + d s \cdot \mathbf{v} \rfloor + \mathbf{t})

    where :math:`p(I, J, K)_i` is ``projection[I, J, K, i]``,  :math:`V_i` is ``volume[..., i]``,
    and :math:`\mathbf{v}` is ``unit_vector_p[I]``. :math:`N` is the number of voxels in the maximal direction
    of ``unit_vector_p[I]``. :math:`d` is the step length, and is given by the norm of
    :math:`\Vert \mathbf{v}_I \Vert / \vert \max(\mathbf{v})) \vert`.
    :math:`t_w` and :math:`\mathbf{t}` are weights and index shifts, respectively.
    The latter are necessary to perform bilinear interpolation between the four
    closest voxels in the two directions orthogonal to the
    maximal direction of :math:`\mathbf{v}`.
    :math:`\mathbf{r}_j` is the starting position for the ray,
    and is given by

    .. math::

        \mathbf{r}_j(J, K) = (J + 0.5 + o_J - 0.5 \cdot J_\text{max}) \cdot \mathbf{u} + \\
                (K + 0.5 + o_K - 0.5 \cdot K_\text{max}) \cdot \mathbf{w} + \\
                (\Delta_p) \cdot \mathbf{v} + 0.5 \cdot \mathbf{r}_\text{max} \\

    where :math:`o_J` is ``offsets_j[I]``,
    :math:`o_K` is ``offsets_k[I]``,
    :math:`\mathbf{u}` is ``unit_vector_j[I]``, :math:`\mathbf{w}` is ``unit_vector_k[I]``,
    :math:`J_\text{max}` and :math:`K_\text{max}` are ``projections.shape[1]`` and
    ``projections.shape[2]`` respectively, and :math:`\mathbf{r}_\text{max}` is
    ``volume.shape[:3]``. :math:`\Delta_p` is an additional offset that places the
    starting position at the edge of the volume.
    """ # noqa

    if float_type == 'float64':
        numba_float = float64
    elif float_type == 'float32':
        numba_float = float32
    else:
        raise ValueError('float_type must be either "float64" or "float32", '
                         f'but a value of {float_type} was specified!')

    if str(field.dtype) != float_type:
        raise TypeError('The dtype of the argument "field" must be the same '
                        f'as the dtype specified by "float_type" ({float_type}), but '
                        f'field.dtype is {field.dtype}!')
    if str(projections.dtype) != float_type:
        raise TypeError('The dtype of the argument "projections" must be the same '
                        f'as the dtype specified by "float_type" ({float_type}), but '
                        f'projections.dtype is {projections.dtype}!')

    # Find zeroth, first and second directions for indexing. Zeroth is the maximal projection direction.

    # (x, y, z), (y, x, z), (z, x, y)
    direction_0_index = np.argmax(abs(unit_vector_p), axis=1).reshape(-1, 1).astype(np.int32)
    direction_1_index = 1 * (direction_0_index == 0).astype(np.int32).reshape(-1, 1).astype(np.int32)
    direction_2_index = (2 - (direction_0_index == 2)).astype(np.int32).reshape(-1, 1).astype(np.int32)

    # Step size for direction 1 and 2.
    step_sizes_1 = (np.take_along_axis(unit_vector_p, direction_1_index, 1) /
                    np.take_along_axis(unit_vector_p, direction_0_index, 1)).astype(str(numba_float)).ravel()
    step_sizes_2 = (np.take_along_axis(unit_vector_p, direction_2_index, 1) /
                    np.take_along_axis(unit_vector_p, direction_0_index, 1)).astype(str(numba_float)).ravel()

    # Shape in each of the three directions.
    dimensions_0 = np.array(field.shape, dtype=str(numba_float))[direction_0_index.ravel()]
    dimensions_1 = np.array(field.shape, dtype=str(numba_float))[direction_1_index.ravel()]
    dimensions_2 = np.array(field.shape, dtype=str(numba_float))[direction_2_index.ravel()]

    # Correction factor for length of line when taking a one-slice step.
    distance_multipliers = np.sqrt(1.0 + step_sizes_1 ** 2 + step_sizes_2 ** 2).astype(str(numba_float))

    max_index = projections.shape[0]
    max_j = projections.shape[1]
    max_k = projections.shape[2]

    # CUDA chunking and memory size constants.
    channels = int(field.shape[-1])

    # Indices to navigate each projection. s is the surface positioning.
    k_vectors = unit_vector_k.astype(str(numba_float))
    j_vectors = unit_vector_j.astype(str(numba_float))
    s_vectors = (unit_vector_k * (-0.5 * max_k + offsets_k.reshape(-1, 1)) +
                 unit_vector_j * (-0.5 * max_j + offsets_j.reshape(-1, 1))).astype(str(numba_float))

    # (0, 1, 2) if x main, (1, 0, 2) if y main, (2, 0, 1) if z main.
    direction_indices = np.stack((direction_0_index.ravel(),
                                  direction_1_index.ravel(),
                                  direction_2_index.ravel()), axis=1)

    # Bilinear interpolation over each slice.

    @njit(void(numba_float[:, :, :, ::1], int32, numba_float,
               numba_float, numba_float, int32[::1], numba_float[::1]),
          fastmath=True, nogil=True, cache=True)
    def bilinear_interpolation(field: NDArray[float],
                               direction_0: int,
                               r0: float,
                               r1: float,
                               r2: float,
                               dimensions,
                               accumulator: NDArray[float]):
        """ Kernel for bilinear interpolation. Replaces texture interpolation."""
        if not ((0 <= r0 < dimensions[0]) and
                (-1 <= r1 < dimensions[1]) and
                (-1 <= r2 < dimensions[2])):
            return
        # At edges, use nearest-neighbor interpolation.
        r1_weight = int32(1 <= (int32(r1) + 1) < dimensions[1])
        r2_weight = int32(1 <= (int32(r2) + 1) < dimensions[2])
        r1_edge_weight = int32(int32(r1) == -1)
        r2_edge_weight = int32(int32(r2) == -1)
        weight_1 = numba_float((r1 - floor(r1)) * r1_weight)
        weight_2 = numba_float((r2 - floor(r2)) * r2_weight)
        t0 = (1 - weight_1) * (1 - weight_2)
        t1 = weight_1 * weight_2
        t2 = (1 - weight_1) * weight_2
        t3 = (1 - weight_2) * weight_1

        if direction_0 == 0:
            x = int32(r0)
            y = int32(r1) + r1_edge_weight
            z = int32(r2) + r2_edge_weight
            for i in range(accumulator.size):
                accumulator[i] += field[x, y, z, i] * t0
                accumulator[i] += field[x, y + r1_weight, z, i] * t3
                accumulator[i] += field[x, y, z + r2_weight, i] * t2
                accumulator[i] += field[x, y + r1_weight, z + r2_weight, i] * t1

        elif direction_0 == 1:
            x = int32(r1) + r1_edge_weight
            y = int32(r0)
            z = int32(r2) + r2_edge_weight
            for i in range(accumulator.size):
                accumulator[i] += field[x, y, z, i] * t0
                accumulator[i] += field[x + r1_weight, y, z, i] * t3
                accumulator[i] += field[x, y, z + r2_weight, i] * t2
                accumulator[i] += field[x + r1_weight, y, z + r2_weight, i] * t1

        elif direction_0 == 2:
            x = int32(r1) + r1_edge_weight
            y = int32(r2) + r2_edge_weight
            z = int32(r0)
            for i in range(accumulator.size):
                accumulator[i] += field[x, y, z, i] * t0
                accumulator[i] += field[x + r1_weight, y, z, i] * t3
                accumulator[i] += field[x, y + r2_weight, z, i] * t2
                accumulator[i] += field[x + r1_weight, y + r2_weight, z, i] * t1

    @njit(void(numba_float[:, :, :, ::1], numba_float[:, :, :, ::1]),
          fastmath=True, nogil=True, parallel=True, cache=True)
    def john_transform_inner(field: NDArray[float], projection: NDArray[float]):
        """ Performs the John transform of a field. Relies on a large number
        of pre-defined constants outside the kernel body. """
        for index in range(max_index):
            # Define compile-time constants.
            step_size_1 = step_sizes_1[index]
            step_size_2 = step_sizes_2[index]
            k_vectors_c = k_vectors[index]
            j_vectors_c = j_vectors[index]
            s_vectors_c = s_vectors[index]
            dimensions_0_c = dimensions_0[index]
            dimensions_1_c = dimensions_1[index]
            dimensions_2_c = dimensions_2[index]
            dimensions = np.empty(3, int32)
            dimensions[0] = dimensions_0_c
            dimensions[1] = dimensions_1_c
            dimensions[2] = dimensions_2_c
            direction_indices_c = direction_indices[index]
            distance_multiplier = distance_multipliers[index]

            for j in prange(max_j):
                accumulator = np.empty(channels, numba_float)
                # Could be chunked for very asymmetric samples.
                fj = numba_float(j) + 0.5
                for k in range(max_k):
                    for i in range(channels):
                        accumulator[i] = 0.

                    fk = numba_float(k) + 0.5

                    # Initial coordinates of projection.
                    start_position_0 = (s_vectors_c[direction_indices_c[0]] +
                                        fj * j_vectors_c[direction_indices_c[0]] +
                                        fk * k_vectors_c[direction_indices_c[0]])
                    start_position_1 = (s_vectors_c[direction_indices_c[1]] +
                                        fj * j_vectors_c[direction_indices_c[1]] +
                                        fk * k_vectors_c[direction_indices_c[1]])
                    start_position_2 = (s_vectors_c[direction_indices_c[2]] +
                                        fj * j_vectors_c[direction_indices_c[2]] +
                                        fk * k_vectors_c[direction_indices_c[2]])

                    # Centering w.r.t volume.
                    centering_step_1 = start_position_1 - step_size_1 * start_position_0
                    centering_step_2 = start_position_2 - step_size_2 * start_position_0

                    position_0 = numba_float(0) + 0.5
                    position_1 = step_size_1 * (numba_float(0) - 0.5 * dimensions[0] + 0.5) + \
                        centering_step_1 + 0.5 * dimensions[1] - 0.5
                    position_2 = step_size_2 * (numba_float(0) - 0.5 * dimensions[0] + 0.5) + \
                        centering_step_2 + 0.5 * dimensions[2] - 0.5

                    for i in range(dimensions[0]):
                        bilinear_interpolation(field, direction_indices_c[0],
                                               position_0, position_1, position_2, dimensions, accumulator)
                        position_0 += 1.0
                        position_1 += step_size_1
                        position_2 += step_size_2

                    for i in range(channels):
                        projection[index, j, k, i] = accumulator[i] * distance_multiplier

    def john_transform_wrapper(field, projection):
        john_transform_inner(field, projection)
        return projection

    return john_transform_wrapper


def john_transform_adjoint(field: NDArray[float], projections: NDArray[float],
                           unit_vector_p: NDArray[float], unit_vector_j: NDArray[float],
                           unit_vector_k: NDArray[float], offsets_j: NDArray[float],
                           offsets_k: NDArray[float], float_type: str = 'float64') -> callable:
    r""" Frontend for performing the adjoint of the John transform with parallel
    CPU computing, using an algorithm akin to :func:`mumott.core.john_transform_cuda`.

    Parameters
    ----------
    field
        The field into which the adjoint is projected, with 4 dimensions. The last index should
        have the same size as the last index of :attr:`projections`.
    projections
        The projections from which the adjoint is calculated.
        The first index runs over the different projection directions.
    unit_vector_p
        The direction of projection in Cartesian coordinates.
    unit_vector_j
        One of the directions for the pixels of :attr:`projections`.
    unit_vector_k
        The other direction for the pixels of :attr:`projections`.
    offsets_j
        Offsets which align projections in the direction of `j`
    offsets_k
        Offsets which align projections in the direction of `k`.
    float_type
        Whether to use 'float64' (default) or 'float32'. The argument should be supplied
        as a string. The types of :attr:`field` and :attr:`projections` must match this type.

    Notes
    -----
    The computation performed by this function may be written as

    .. math::

        V_i(\mathbf{x}) = \sum_{s=0}^{N-1} \cdot \sum_{t = 0}^{4} t_w p_i(s, \mathbf{x} \cdot \mathbf{u} + \Delta_J + t_J, \mathbf{x} \cdot \mathbf{w} + \Delta_K + t_K)

    :math:`N` is the total number of projections.
    :math:`V_i` is ``volume[:, i]``,
    and :math:`\mathbf{v}` is ``projection_vector[s]``. :math:`\mathbf{u}` is ``unit_vector_j[s]``,
    :math:`\mathbf{w}` is ``unit_vector_k[s]``. :math:`\Delta_J` and :math:`\Delta_K` are additional
    offsets based on the unit vectors, shapes, and offsets, which align the centers of the projection and volume,
    so that the intersection of each ray is correctly computed.
    :math:`t_W` are weights for bilinear interpolation between the four pixels nearest to each ray;
    :math:`t_J` and :math:`t_K` are the index offsets necessary to perform this interpolation.
    """ # noqa
    if float_type == 'float64':
        numba_float = float64
    elif float_type == 'float32':
        numba_float = float32
    else:
        raise ValueError('float_type must be either "numba_float" or "float32", '
                         f'but a value of {float_type} was specified!')

    if str(field.dtype) != float_type:
        raise TypeError('The dtype of the argument "field" must be the same '
                        f'as the dtype specified by "float_type" ({float_type}), but '
                        f'field.dtype is {field.dtype}!')

    if str(projections.dtype) != float_type:
        raise TypeError('The dtype of the argument "projections" must be the same '
                        f'as the dtype specified by "float_type" ({float_type}), but '
                        f'projections.dtype is {projections.dtype}!')

    max_j = projections.shape[1]
    max_k = projections.shape[2]

    max_x, max_y, max_z = field.shape[:3]
    max_index = projections.shape[0]

    # Projection vectors. s for positioning the projection.
    p_vectors = unit_vector_p.astype(str(numba_float))
    k_vectors = unit_vector_k.astype(str(numba_float))
    j_vectors = unit_vector_j.astype(str(numba_float))
    s_vectors = (unit_vector_k * (-0.5 * max_k + offsets_k.reshape(-1, 1)) +
                 unit_vector_j * (-0.5 * max_j + offsets_j.reshape(-1, 1))).astype(str(numba_float))

    # Translate volume steps to normalized projection steps. Can add support for non-square voxels.
    vector_norm = np.einsum('...i, ...i', p_vectors, np.cross(j_vectors, k_vectors))
    norm_j = -np.cross(p_vectors, k_vectors) / vector_norm[..., None]
    norm_k = np.cross(p_vectors, j_vectors) / vector_norm[..., None]
    norm_offset_j = -np.einsum('...i, ...i', p_vectors, np.cross(s_vectors, k_vectors)) / vector_norm
    norm_offset_k = np.einsum('...i, ...i', p_vectors, np.cross(s_vectors, j_vectors)) / vector_norm

    channels = field.shape[-1]

    @njit(void(numba_float[:, :, :, ::1], numba_float,
               numba_float, numba_float, numba_float[::1]),
          fastmath=True, cache=True, nogil=True)
    def bilinear_interpolation_projection(projection: NDArray[float],
                                          r0: float,
                                          r1: float,
                                          r2: float,
                                          accumulator: NDArray[float]):
        if not (-1 <= r1 < max_j and -1 <= r2 < max_k):
            return
        # At edges, use nearest-neighbor interpolation.
        r1_weight = int32(0 <= int32(r1) + 1 < max_j)
        r2_weight = int32(0 <= int32(r2) + 1 < max_k)
        r1_edge_weight = int32(int32(r1) == -1)
        r2_edge_weight = int32(int32(r2) == -1)
        y_weight = numba_float((r1 - floor(r1)) * r1_weight) * (1 - r1_edge_weight)
        z_weight = numba_float((r2 - floor(r2)) * r2_weight) * (1 - r2_edge_weight)
        x = int32(r0)
        y = int32(r1) + r1_edge_weight
        z = int32(r2) + r2_edge_weight
        t0 = (1 - z_weight) * (1 - y_weight)
        t1 = z_weight * y_weight
        t2 = (1 - z_weight) * y_weight
        t3 = (1 - y_weight) * z_weight
        for i in range(channels):
            accumulator[i] += projection[x, y, z, i] * t0
            accumulator[i] += projection[x, y + r1_weight, z, i] * t2
            accumulator[i] += projection[x, y, z + r2_weight, i] * t3
            accumulator[i] += projection[x, y + r1_weight, z + r2_weight, i] * t1

    @njit(void(numba_float[:, :, :, ::1], numba_float[:, :, :, ::1]),
          fastmath=True, cache=True, nogil=True, parallel=True)
    def john_transform_adjoint_inner(field: NDArray[float], projection: NDArray[float]):
        """ Performs the John transform of a field. Relies on a large number
        of pre-defined constants outside the kernel body. """
        # Indexing of volume coordinates.
        for x in range(max_x):
            fx = x - 0.5 * field.shape[0] + 0.5
            for y in prange(max_y):
                z_acc = np.empty(channels, numba_float)
                fy = y - 0.5 * field.shape[1] + 0.5
                for z in range(max_z):
                    # Center of voxel and coordinate system.
                    fz = z - 0.5 * field.shape[2] + 0.5

                    for j in range(channels):
                        z_acc[j] = 0.0

                    for a in range(max_index):
                        # Center with respect to projection.
                        fj = (norm_offset_j[a] + fx * norm_j[a][0] +
                              fy * norm_j[a][1] + fz * norm_j[a][2] - 0.5)
                        fk = (norm_offset_k[a] + fx * norm_k[a][0] +
                              fy * norm_k[a][1] + fz * norm_k[a][2] - 0.5)
                        bilinear_interpolation_projection(projection,
                                                          a, fj, fk, z_acc)
                    for i in range(channels):
                        field[x, y, z, i] = z_acc[i]

    def john_transform_adjoint_inner_wrapper(field, projection):
        john_transform_adjoint_inner(field, projection)
        return field
    return john_transform_adjoint_inner_wrapper
