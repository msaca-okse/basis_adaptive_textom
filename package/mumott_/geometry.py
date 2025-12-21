from __future__ import annotations
import logging
import tarfile
import tempfile
import json
import os
import codecs
import numpy as np
from typing import NamedTuple, Union
from scipy.spatial.transform import Rotation
from mumott_.hashing import list_to_hash
from mumott_.probed_coordinates import ProbedCoordinates
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class GeometryTuple(NamedTuple):
    """Tuple for passing and returning projection-wise geometry information.
    This is a helper class used by :class:`Geometry`.

    Attributes
    ----------
    rotation
        Rotation matrix. If the :attr:`angle` and :attr:`axis` arguments
        are given, this matrix should correspond to the
        matrix given by R_outer @ R_inner, where R_inner
        is defined by a rotation by :attr:`inner_angle` about
        :attr:`inner_axis`, and similarly for R_outer.
    j_offset
        Offset to align projection in the direction j.
    k_offset
        Offset to align projection in the direction k.
    inner_angle
        Angle of rotation about :attr:`inner_axis` in radians.
    outer_angle
        Angle of rotation about :attr:`outer_axis` in radians.
    inner_axis
        Inner rotation axis.
    outer_axis
        Outer rotation axis.
    """
    rotation: np.ndarray[float] = np.eye(3, dtype=float)
    j_offset: float = float(0)
    k_offset: float = float(0)
    inner_angle: float = None
    outer_angle: float = None
    inner_axis: np.ndarray[float] = None
    outer_axis: np.ndarray[float] = None

    def __hash__(self) -> int:
        to_hash = [self.rotation.ravel(), self.j_offset, self.k_offset,
                   self.inner_angle, self.outer_angle, self.inner_axis,
                   self.outer_axis]
        return int(list_to_hash(to_hash), 16)

    def __str__(self) -> str:
        wdt = 74
        s = []
        s += ['-' * wdt]
        s += ['GeometryTuple'.center(wdt)]
        s += ['-' * wdt]
        with np.printoptions(threshold=4, precision=5, linewidth=60, edgeitems=1):
            ss = ', '.join([f'{r}' for r in self.rotation])
            s += ['{:18} : {}'.format('rotation', ss)]
            s += ['{:18} : {}'.format('j_offset', self.j_offset)]
            s += ['{:18} : {}'.format('k_offset', self.k_offset)]
            s += ['{:18} : {}'.format('inner_angle', self.inner_angle)]
            s += ['{:18} : {}'.format('outer_angle', self.outer_angle)]
            ss = ', '.join([f'{r}' for r in self.inner_axis])
            s += ['{:18} : {}'.format('inner_axis', ss)]
            ss = ', '.join([f'{r}' for r in self.outer_axis])
            s += ['{:18} : {}'.format('outer_axis', ss)]
            s += ['{:18} : {}'.format('Hash', hex(hash(self))[2:8])]
        s += ['-' * wdt]
        return '\n'.join(s)

    def _repr_html_(self) -> str:
        s = []
        s += ['<h3>GeometryTuple</h3>']
        s += ['<table border="1" class="dataframe">']
        s += ['<thead><tr><th style="text-align: left;">Field</th><th>Size</th><th>Data</th></tr></thead>']
        s += ['<tbody>']
        with np.printoptions(threshold=4, edgeitems=2, precision=2, linewidth=40):
            s += ['<tr><td style="text-align: left;">Rotation</td>']
            s += [f'<td>{self.rotation.shape}</td><td>{self.rotation}</td></tr>']
            s += ['<tr><td style="text-align: left;">j_offset</td>']
            s += [f'<td>{1}</td><td>{self.j_offset:.4f}</td></tr>']
            s += ['<tr><td style="text-align: left;">k_offset</td>']
            s += [f'<td>{1}</td><td>{self.k_offset:.4f}</td></tr>']
            s += ['<tr><td style="text-align: left;">inner_angle</td>']
            s += [f'<td>{1}</td>'
                  f'<td>{self.inner_angle}']
            s += ['<tr><td style="text-align: left;">outer_angle</td>']
            s += [f'<td>{1}</td>'
                  f'<td>{self.outer_angle}</td>']
            s += ['<tr><td style="text-align: left;">inner_axis</td>']
            s += [f'<td>{self.inner_axis.shape}</td><td>{self.inner_axis}</td></tr>']
            s += ['<tr><td style="text-align: left;">outer_axis</td>']
            s += [f'<td>{self.outer_axis.shape}</td><td>{self.outer_axis}</td></tr>']
            s += [f'<td>{6}</td>'
                  f'<td>{hex(hash(self))[2:8]} (hash)</td></tr>']
        s += ['</tbody>']
        s += ['</table>']
        return '\n'.join(s)


class Geometry:
    """ Stores information about the system geometry.
    Instances of this class are used by
    :class:`DataContainer <mumott.data_handling.DataContainer>`
    and :class:`ProjectionStack <mumott.core.projection_stack.ProjectionStack>` to
    maintain geometry information.
    They can be stored as a file using the :meth:`write` method.
    This allows one to (re)create a :class:`Geometry` instance from an earlier
    and overwrite the geometry information read by a
    :class:`DataContainer <mumott.data_handling.DataContainer>` instance.
    This is useful, for example, in the context of alignment.

    Parameters
    ----------
    filename
        Name of file from which to read geometry information.
        Defaults to ``None``, in which case the instance is created with
        default parameters.
    """
    def __init__(self, filename: str = None):
        self._rotations = []
        self._j_offsets = []
        self._k_offsets = []
        self._p_direction_0 = np.array([0, 1, 0]).astype(float)
        self._j_direction_0 = np.array([1, 0, 0]).astype(float)
        self._k_direction_0 = np.array([0, 0, 1]).astype(float)
        self._detector_direction_origin = np.array([1, 0, 0]).astype(float)
        self._detector_direction_positive_90 = np.array([0, 0, 1]).astype(float)
        self._projection_shape = np.array([0, 0]).astype(np.int32)
        self._volume_shape = np.array([0, 0, 0]).astype(np.int32)
        self._detector_angles = np.array([]).astype(float)
        self._two_theta = np.array([0.0])
        self._reconstruction_rotations = []
        self._system_rotations = []
        self._inner_angles = []
        self._outer_angles = []
        self._inner_axes = []
        self._outer_axes = []
        self._full_circle_covered = False
        if filename is not None:
            self.read(filename)

    def write(self, filename: str) -> None:
        """Method for writing the current state of a :class:`Geometry` instance to file.

        Notes
        -----
        Any rotations in :attr:`reconstruction_rotations` and :attr:`system_rotations`
        will be applied to the :attr:`rotations` and system vectors respectively prior to writing.

        Parameters
        ----------
        filename
            Name of output file.
        """
        to_write = dict(_rotations=self.rotations_as_array.tolist(),
                        _j_offsets=self._j_offsets,
                        _k_offsets=self._k_offsets,
                        p_direction_0=self.p_direction_0.tolist(),
                        j_direction_0=self.j_direction_0.tolist(),
                        k_direction_0=self.k_direction_0.tolist(),
                        detector_direction_origin=self.detector_direction_origin.tolist(),
                        detector_direction_positive_90=self.detector_direction_positive_90.tolist(),
                        two_theta=self.two_theta.tolist(),
                        projection_shape=self.projection_shape.tolist(),
                        volume_shape=self.volume_shape.tolist(),
                        detector_angles=self.detector_angles.tolist(),
                        full_circle_covered=[bool(self.full_circle_covered)],
                        _inner_angles=self._inner_angles,
                        _outer_angles=self._outer_angles,
                        _inner_axes=[j.tolist() if j is not None else None for j in self._inner_axes],
                        _outer_axes=[j.tolist() if j is not None else None for j in self._outer_axes],
                        checksum=[hash(self)])
        with tarfile.open(name=filename, mode='w') as tar_file:
            for key, item in to_write.items():
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.close()
                with codecs.open(temp_file.name, 'w', encoding='utf-8') as tf:
                    json.dump(item, tf)
                    tf.flush()
                with open(temp_file.name, 'rb') as tt:
                    tar_info = tar_file.gettarinfo(arcname=key, fileobj=tt)
                    tar_file.addfile(tar_info, tt)
                os.remove(temp_file.name)

    def read(self, filename: str) -> None:
        """Method for reading the current state of a :class:`Geometry` instance from file.

        Parameters
        ----------
        filename
            Name of input file.
        """
        to_read = ['_rotations',
                   '_j_offsets',
                   '_k_offsets',
                   'p_direction_0',
                   'j_direction_0',
                   'k_direction_0',
                   'detector_direction_origin',
                   'detector_direction_positive_90',
                   'two_theta',
                   'projection_shape',
                   'volume_shape',
                   'detector_angles',
                   'full_circle_covered',
                   '_inner_angles',
                   '_outer_angles',
                   '_inner_axes',
                   '_outer_axes',
                   'checksum']
        with tarfile.open(name=filename, mode='r') as tar_file:
            for key in to_read:
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                try:
                    temp_file.write(tar_file.extractfile(key).read())
                except KeyError:
                    logger.warning(f'Key {key} not found!')
                    temp_file.close()
                    continue
                temp_file.close()
                with codecs.open(temp_file.name, 'r', encoding='utf-8') as f:
                    text = f.read()
                data_as_list = json.loads(text)
                for i, entry in enumerate(data_as_list):
                    if entry == 'null':
                        data_as_list[i] = None
                if key == 'checksum':
                    checksum = data_as_list[0]
                elif key in ('_rotations', '_inner_axes', '_outer_axes'):
                    setattr(self, key, [])
                    for entry in data_as_list:
                        # necessary check to avoid [array(None), ...] structure
                        if entry is not None:
                            entry = np.array(entry)
                        getattr(self, key).append(entry)
                elif key in ('_j_offsets', '_k_offsets', '_inner_angles', '_outer_angles'):
                    setattr(self, key, data_as_list)
                elif key in ('full_circle_covered'):
                    setattr(self, key, data_as_list[0])
                else:
                    setattr(self, key, np.array(data_as_list))
        if checksum != hash(self):
            logger.warning(f'Checksum does not match! Checksum is {checksum},'
                           f' but hash(self) is {hash(self)}. This may be due to'
                           ' version differences, but please proceed with caution!')

    def rotate_reconstruction(self,
                              A: np.ndarray[float] = None,
                              axis: np.ndarray[float] = None,
                              angle: np.ndarray[float] = None):
        r""" Rotates the reconstruction geometry. The given rotation matrix will modify the rotation
        matrix of each projection by multiplication from the right, such that

        .. math ::
            R_i' = R_i A

        where :math:`R_i` is the rotation matrix of projection :math:`i` and :math:`A` is the rotation matrix.
        For each projection, the system vectors are then rotated by

        .. math ::
            v_i = (R_i')^T v = A^T R_i^T v

        where :math:`v` corresponds to e.g., :attr:`p_direction_0`.

        Notes
        -----
        It is not possible to directly modify :attr:`rotations` after adding a reconstruction rotation.

        Parameters
        ----------
        A
            A 3-by-3 rotation matrix. If not given, then :attr:`axis` and :attr:`angle` must be provided.
        axis
            An axis, given as a unit length 3-vector, about which the rotation is defined. Not used
            if :attr:`A` is provided.
        angle
            The angle in radians of the rotation about :attr:`axis`. Not used if :attr:`A` is provided.
        """
        if A is None:
            A = Rotation.from_rotvec(axis * angle / np.linalg.norm(axis)).as_matrix()
        elif axis is not None or angle is not None:
            logger.warning('A provided along with axis and/or angle; axis/angle will be ignored!')

        self._reconstruction_rotations.append(A)

    def rotate_system_vectors(self,
                              A: np.ndarray[float] = None,
                              axis: np.ndarray[float] = None,
                              angle: np.ndarray[float] = None):
        r""" Rotates the system vectors. The given rotation matrix will modify the system vectors by

        .. math ::
            v' = A v

        where :math:`v` is a system vector, e.g., :attr:`p_direction_0`, and :math:`A` is the rotation matrix.
        For each projection, the system vectors are then rotated by

        .. math ::
            v_i = R_i^T A v

        where :math:`R_i` corresponds to :attr:`rotations` for projection :math:`i`.

        Notes
        -----
        It is not possible to directly modify the system vectors after adding a system rotation.

        Parameters
        ----------
        A
            A 3-by-3 rotation matrix. If not given, then :attr:`axis` and :attr:`angle` must be provided.
        axis
            An axis, given as a 3-vector, about which a rotation can be defined. Not used
            if :attr:`A` is provided.
        angle
            The angle in radians of the rotation about :attr:`axis`. Not used if :attr:`A` is provided.
        """
        if A is None:
            A = Rotation.from_rotvec(axis * angle / np.linalg.norm(axis)).as_matrix()
        elif axis is not None or angle is not None:
            logger.warning('A provided along with axis and/or angle; axis/angle will be ignored!')

        self._system_rotations.append(A)

    def append(self, value: GeometryTuple) -> None:
        """ Appends projection-wise geometry data provided as a
        :class:`GeometryTuple <mumott.core.geometry.GeometryTuple>`. """
        self._rotations.append(value.rotation)
        self._j_offsets.append(value.j_offset)
        self._k_offsets.append(value.k_offset)
        self._inner_angles.append(value.inner_angle)
        self._outer_angles.append(value.outer_angle)
        self._inner_axes.append(value.inner_axis)
        self._outer_axes.append(value.outer_axis)

    def insert(self, key: int, value: GeometryTuple) -> None:
        """ Inserts projection-wise data handed via a
        :class:`GeometryTuple <mumott.core.geometry.GeometryTuple>`. """
        self._rotations.insert(key, value.rotation)
        self._j_offsets.insert(key, value.j_offset)
        self._k_offsets.insert(key, value.k_offset)
        self._inner_angles.insert(key, value.inner_angle)
        self._outer_angles.insert(key, value.outer_angle)
        self._inner_axes.insert(key, value.inner_axis)
        self._outer_axes.insert(key, value.outer_axis)

    def __setitem__(self, key: int, value: GeometryTuple) -> None:
        """ Sets projection-wise data handed via a :class:`GeometryTuple`."""
        self._rotations[key] = value.rotation
        self._j_offsets[key] = value.j_offset
        self._k_offsets[key] = value.k_offset
        self._inner_angles[key] = value.inner_angle
        self._outer_angles[key] = value.outer_angle
        self._inner_axes[key] = value.inner_axis
        self._outer_axes[key] = value.outer_axis

    def __getitem__(self, key: int) -> GeometryTuple:
        """ Returns projection-wise data as a :class:`GeometryTuple`."""
        return GeometryTuple(rotation=self.rotations[key],
                             j_offset=self._j_offsets[key],
                             k_offset=self._k_offsets[key],
                             inner_angle=self._inner_angles[key],
                             outer_angle=self._outer_angles[key],
                             inner_axis=self._inner_axes[key],
                             outer_axis=self._outer_axes[key])

    def __delitem__(self, key: int) -> None:
        del self._rotations[key]
        del self._j_offsets[key]
        del self._k_offsets[key]
        del self._inner_angles[key]
        del self._outer_angles[key]
        del self._inner_axes[key]
        del self._outer_axes[key]

    def _get_probed_coordinates(self) -> NDArray[np.float_]:
        """
        Calculates and returns the probed polar and azimuthal coordinates on the unit sphere at
        each angle of projection and for each detector segment in the geometry of the system.
        """
        n_proj = len(self)
        n_seg = len(self.detector_angles)
        probed_directions_zero_rot = np.zeros((n_seg, 3, 3))
        # Impose symmetry if needed.
        if not self.full_circle_covered:
            shift = np.pi
        else:
            shift = 0
        det_bin_middles_extended = np.copy(self.detector_angles)
        det_bin_middles_extended = np.insert(det_bin_middles_extended, 0,
                                             det_bin_middles_extended[-1] + shift)
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
            angles = np.linspace(start, end, 3)

            # Make the zero-rotation-projection vectors corresponding to the given angles
            probed_directions_zero_rot[ii, :, :] = np.cos(angles[:, np.newaxis]) * \
                self.detector_direction_origin[np.newaxis, :]
            probed_directions_zero_rot[ii, :, :] += np.sin(angles[:, np.newaxis]) * \
                self.detector_direction_positive_90[np.newaxis, :]

        # Do wide-angle rotation
        n_twotheta = len(self.two_theta)
        twotheta = np.repeat(self.two_theta, n_seg)[:, np.newaxis, np.newaxis]
        probed_directions_zero_rot = np.tile(probed_directions_zero_rot, (n_twotheta, 1, 1))

        probed_directions_zero_rot = probed_directions_zero_rot * np.cos(twotheta/2)\
            - np.sin(twotheta/2) * self.p_direction_0

        # Initialize array for vectors
        probed_direction_vectors = np.zeros((n_proj,
                                             n_seg * n_twotheta,
                                             3,
                                             3), dtype=np.float64)
        # Calculate all the rotations
        probed_direction_vectors[...] = \
            np.einsum('kij,mli->kmlj',
                      self.rotations,
                      probed_directions_zero_rot)
        great_circle_offsets = np.einsum('kij,mli->kmlj',
                                         self.rotations,
                                         -np.sin(twotheta / 2) * self.p_direction_0)
        return ProbedCoordinates(probed_direction_vectors, great_circle_offsets)



        
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

        probed_directions_zero_rot = probed_directions_zero_rot * np.cos(twothetahalf)\
            - np.sin(twothetahalf) * geom.p_direction_0

        # Initialize array for vectors
        probed_direction_vectors = np.zeros((n_proj, n_seg, integration_samples, 3), dtype=np.float64)
        # Calculate all the rotations
        probed_direction_vectors[...] = \
            np.einsum('kij,mli->kmlj', geom.rotations_as_array, probed_directions_zero_rot)

        return probed_direction_vectors

    def delete_projections(self) -> None:
        """ Delete all projections."""
        self._rotations = []
        self._j_offsets = []
        self._k_offsets = []
        self._inner_angles = []
        self._outer_angles = []
        self._inner_axes = []
        self._outer_axes = []

    def _get_reconstruction_rotation(self) -> np.ndarray[float]:
        """ Internal method for composing reconstruction rotations. """
        reconstruction_rotation = np.eye(3, dtype=float)
        for r in self.reconstruction_rotations:
            reconstruction_rotation = reconstruction_rotation @ r
        return reconstruction_rotation

    def _get_system_rotation(self) -> np.ndarray[float]:
        """ Internal method for composing system rotations. """
        system_rotation = np.eye(3, dtype=float)
        for r in self.system_rotations:
            system_rotation = r @ system_rotation
        return system_rotation

    @property
    def system_rotations(self) -> list[np.ndarray[float]]:
        """ list of rotation matrices sequentially applied to the basis vectors of the system. """
        return self._system_rotations

    @system_rotations.setter
    def system_rotations(self, value: list[np.ndarray[float]]) -> list[np.ndarray[float]]:
        self._system_rotations = list(value)

    @property
    def reconstruction_rotations(self) -> list[np.ndarray[float]]:
        """ list of rotation matrices sequentially applied to the reconstruction geometry of the system. """
        return self._reconstruction_rotations

    @reconstruction_rotations.setter
    def reconstruction_rotations(self, value: list[np.ndarray[float]]) -> list[np.ndarray[float]]:
        self._reconstruction_rotations = list(value)

    @property
    def rotations(self) -> list[np.ndarray[float]]:
        """ Rotation matrices for the experimental rotation corresponding to each projection of data."""
        if len(self.reconstruction_rotations) > 0:
            reconstruction_rotation = self._get_reconstruction_rotation()
            return [r @ reconstruction_rotation for r in self._rotations]

        return self._rotations

    @property
    def rotations_as_array(self) -> np.ndarray[float]:
        """ Rotation matrices corresponding to each projection of data as an array."""
        if len(self) == 0:
            return np.array([])
        return np.stack(list(self.rotations), axis=0)

    @rotations.setter
    def rotations(self, value: Union[list, np.ndarray[float]]) -> None:
        if len(self._reconstruction_rotations) > 0:
            raise ValueError('Cannot modify rotations when reconstruction '
                             'rotations are in use.')
        self._rotations = list(value)

    @property
    def p_direction_0(self) -> np.ndarray[float]:
        """ The projection direction when no experimental rotation is applied."""
        if len(self._system_rotations) > 0:
            system_rotation = self._get_system_rotation()
            return system_rotation @ self._p_direction_0

        return self._p_direction_0

    @p_direction_0.setter
    def p_direction_0(self, value: np.ndarray[float]) -> None:
        if len(self.system_rotations) > 0:
            raise ValueError('Cannot modify system vectors when system '
                             'rotations are in use.')
        if np.size(value) != 3:
            raise ValueError('The size of the new value must be 3, but '
                             f'the provided value has size {np.size(value)}')
        self._p_direction_0[...] = value

    @property
    def j_direction_0(self) -> np.ndarray[float]:
        """ The direction corresponding to the first index in each projection
        when no experimental rotation is applied."""
        if len(self._system_rotations) > 0:
            system_rotation = np.eye(3)
            for r in self.system_rotations:
                system_rotation = system_rotation @ r
            return system_rotation @ self._j_direction_0

        return self._j_direction_0

    @j_direction_0.setter
    def j_direction_0(self, value: np.ndarray[float]) -> None:
        if len(self.system_rotations) > 0:
            raise ValueError('Cannot modify system vectors when system '
                             'rotations are in use.')
        if np.size(value) != 3:
            raise ValueError('The size of the new value must be 3, but '
                             f'the provided value has size {np.size(value)}')
        self._j_direction_0[...] = value

    @property
    def k_direction_0(self) -> np.ndarray[float]:
        """ The direction corresponding to the second index in each projection
        when no experimental rotation is applied."""
        if len(self._system_rotations) > 0:
            system_rotation = self._get_system_rotation()
            return system_rotation @ self._k_direction_0

        return self._k_direction_0

    @k_direction_0.setter
    def k_direction_0(self, value: np.ndarray[float]) -> None:
        if len(self.system_rotations) > 0:
            raise ValueError('Cannot modify system vectors when system '
                             'rotations are in use.')
        if np.size(value) != 3:
            raise ValueError('The size of the new value must be 3, but '
                             f'the provided value has size {np.size(value)}')
        self._k_direction_0[...] = value

    @property
    def detector_direction_origin(self) -> np.ndarray[float]:
        """ The direction at which the angle on the detector is zero,
        when no experimental rotation is applied."""
        if len(self._system_rotations) > 0:
            system_rotation = self._get_system_rotation()
            return system_rotation @ self._detector_direction_origin

        return self._detector_direction_origin

    @detector_direction_origin.setter
    def detector_direction_origin(self, value: np.ndarray[float]) -> None:
        if len(self.system_rotations) > 0:
            raise ValueError('Cannot modify system vectors when system '
                             'rotations are in use.')
        if np.size(value) != 3:
            raise ValueError('The size of the new value must be 3, but '
                             f'the provided value has size {np.size(value)}')
        self._detector_direction_origin[...] = value

    @property
    def detector_direction_positive_90(self) -> np.ndarray[float]:
        """ Rotation matrices corresponding to each projection of data."""
        if len(self._system_rotations) > 0:
            system_rotation = self._get_system_rotation()
            return system_rotation @ self._detector_direction_positive_90

        return self._detector_direction_positive_90

    @detector_direction_positive_90.setter
    def detector_direction_positive_90(self, value: np.ndarray[float]) -> None:
        if len(self.system_rotations) > 0:
            raise ValueError('Cannot modify system vectors when system '
                             'rotations are in use.')
        if np.size(value) != 3:
            raise ValueError('The size of the new value must be 3, but '
                             f'the provided value has size {np.size(value)}')
        self._detector_direction_positive_90[...] = value

    @property
    def j_offsets(self) -> list[float]:
        """Offsets to align projection in the direction j."""
        return self._j_offsets

    @property
    def j_offsets_as_array(self) -> np.ndarray[float]:
        """Offsets to align projection in the direction j as an array."""
        if len(self._j_offsets) == 0:
            return np.array([])
        return np.stack(self.j_offsets, axis=0)

    @j_offsets.setter
    def j_offsets(self, value: Union[list[float], np.ndarray[float]]) -> None:
        self._j_offsets = list(value)

    @property
    def outer_angles(self) -> list[float]:
        """Rotation angles for inner rotation, in radians."""
        return list(self._outer_angles)

    @property
    def outer_angles_as_array(self) -> np.ndarray[float]:
        """Rotation angles for inner rotations, in radians, as an array."""
        if len(self._outer_angles) == 0:
            return np.array([])
        return np.stack(self.outer_angles, axis=0)

    @outer_angles.setter
    def outer_angles(self, value: Union[list[float], np.ndarray[float]]) -> None:
        self._outer_angles = list(value)
        self._update_rotations()

    @property
    def inner_angles(self) -> list[float]:
        """Rotation angles for inner rotation, in radians."""
        return self._inner_angles

    @property
    def probed_coordinates(self) -> ProbedCoordinates:
        """ An array of 3-vectors with the (x, y, z)-coordinates
        on the reciprocal space map probed by the method.
        Structured as ``(N, K, 3, 3)``, where ``N``
        is the number of projections, ``K`` is the number of
        detector segments, the second-to-last axis contains
        start-, mid-, and endpoints, and the last axis contains the
        (x, y, z)-coordinates.

        Notes
        -----
        The number of detector segments is
        `len(geometry.detecor_angles)*len(geometry.two_theta)`
        i.e. the product of the number of two_theta bins times the number of
        azimuthal bins. As a default, only on two theta bin is used.
        When several two_theta bins are used, the second index corresponds
        to a raveled array, where the azimuthal is the fast index and
        two theta is the slow index.
        """
        return self._get_probed_coordinates()

    def _get_hashable_axes_and_angles(self) -> tuple[np.ndarray[float], ...]:
        """ Internal method for getting hashable ste of axes and angles,
        as well as checking if set is valid or if it contains any ``None``
        entries."""
        attributes = [self._inner_angles,
                      self._outer_angles,
                      self._inner_axes,
                      self._outer_axes]
        array_props = [self.inner_angles_as_array,
                       self.outer_angles_as_array,
                       self.inner_axes_as_array,
                       self.outer_axes_as_array]
        return_values = []
        for elements, arrays in zip(attributes, array_props):
            for a in elements:
                if a is None:
                    return_values.append(None)
                    break
            else:
                return_values.append(arrays)
        return tuple(return_values)

    @property
    def inner_angles_as_array(self) -> np.ndarray[float]:
        """Rotation angles for inner rotations, in radians, as an array."""
        if len(self._inner_angles) == 0:
            return np.array([])
        return np.stack(self.inner_angles, axis=0)

    @inner_angles.setter
    def inner_angles(self, value: Union[list[float], np.ndarray]) -> None:
        self._inner_angles = [j for j in value]
        self._update_rotations()

    @property
    def inner_axes(self) -> list[np.ndarray[float]]:
        """Inner rotation axes. All axes can be set
         at once using a single array with three entries."""
        return self._inner_axes

    @property
    def inner_axes_as_array(self) -> np.ndarray[float]:
        """Inner rotation axes as an array."""
        if len(self._inner_axes) == 0:
            return np.array([])
        return np.stack([j if j is not None else [None, None, None] for j in self._inner_axes], axis=0)

    @inner_axes.setter
    def inner_axes(self, value: Union[list[float], np.ndarray]) -> None:
        value = np.array(value)
        if value.ndim == 1:
            if value.shape != (3,):
                raise ValueError('inner_axes may be set using either '
                                 'a list/array of size-3 arrays or an array '
                                 'of shape (3,), but the provided array has shape '
                                 f'{value.shape}.')
            self._inner_axes = [value for _ in self._inner_axes]
        elif value.ndim == 2:
            if len(value) != len(self):
                raise ValueError('If inner_axes is set using a list/array of '
                                 'size-3 arrays, then it must be of the same length '
                                 f'as the Geometry instance ({len(self)}), '
                                 f'but it is of length {len(value)}.')
            if value[0].size != 3:
                raise ValueError('inner_axes may be set using either '
                                 'a list/array of size-3 arrays or an array '
                                 'of shape 3, but the provided array has shape '
                                 f'{value.shape}.')
            self._inner_axes = [j for j in value]
        else:
            raise ValueError('inner_axes must be set either with a list/array of '
                             f'shape ({len(self)}, 3) or with an array of shape '
                             f'(3,), but the provided array has shape {value.shape}!')
        self._update_rotations()

    @property
    def outer_axes(self) -> list[np.ndarray[float]]:
        """Inner rotation axes. All axes can be set
         at once using a single array with three entries."""
        return self._outer_axes

    @property
    def outer_axes_as_array(self) -> np.ndarray[float]:
        """Outer rotation axes as an array."""
        if len(self._outer_axes) == 0:
            return np.array([])
        return np.stack([j if j is not None else [None, None, None] for j in self._outer_axes], axis=0)

    @outer_axes.setter
    def outer_axes(self, value: Union[list[np.ndarray[float]], np.ndarray[float]]) -> None:
        value = np.array(value)
        if value.ndim == 1:
            if value.shape != (3,):
                raise ValueError('outer_axes may be set using either '
                                 'a list/array of size-3 arrays or an array '
                                 'of shape (3,), but the provided array has shape '
                                 f'{value.shape}.')
            self._outer_axes = [value for j in self._outer_axes]
        elif value.ndim == 2:
            if len(value) != len(self):
                raise ValueError('If outer_axes is set using a list/array of '
                                 'size-3 arrays, then it must be of the same length '
                                 f'as the Geometry instance ({len(self)}), but it is of length {len(value)}.')
            if value[0].size != 3:
                raise ValueError('outer_axes may be set using either '
                                 'a list/array of size-3 arrays or an array '
                                 'of shape 3, but the provided array has shape '
                                 f'{value.shape}.')
            self._outer_axes = [j for j in value]
        else:
            raise ValueError('outer_axes must be set either with a list/array of '
                             f'shape ({len(self)}, 3) or with an array of shape '
                             f'(3,), but the provided array has shape {value.shape}!')
        self._update_rotations()

    def _update_rotations(self) -> None:
        """Internal method for updating rotations based on changes
        to inner and outer axes. """
        can_update_rotations = np.all([h is not None for h in self._get_hashable_axes_and_angles()])
        if not can_update_rotations:
            logger.info('None values found in some axis or angle entries,'
                        ' rotations not updated.')
            return

        for i in range(len(self)):
            R_inner = Rotation.from_rotvec(self.inner_angles[i] * self.inner_axes[i]).as_matrix()
            R_outer = Rotation.from_rotvec(self.outer_angles[i] * self.outer_axes[i]).as_matrix()
            self.rotations[i] = R_outer @ R_inner

    @property
    def k_offsets(self) -> list[float]:
        """Offsets to align projection in the direction k."""
        return self._k_offsets

    @property
    def k_offsets_as_array(self) -> np.ndarray[float]:
        """Offsets to align projection in the direction k as an array."""
        if len(self._k_offsets) == 0:
            return np.array([])
        return np.stack(self.k_offsets, axis=0)

    @property
    def hash_rotations(self) -> str:
        """ A blake2b hash of :attr:`rotations_as_array`. """
        return list_to_hash([self.rotations_as_array])

    @property
    def hash_j_offsets(self) -> str:
        """ A blake2b hash of :attr:`j_offsets_as_array`. """
        return list_to_hash([self.j_offsets_as_array])

    @property
    def hash_k_offsets(self) -> str:
        """ A blake2b hash of :attr:`k_offsets_as_array`. """
        return list_to_hash([self.k_offsets_as_array])

    @property
    def hash_inner_angles(self) -> str:
        """ A blake2b hash of :attr:`inner_angle`. """
        return list_to_hash(self.inner_angles)

    @property
    def hash_outer_angles(self) -> str:
        """ A blake2b hash of :attr:`outer_anglesy`. """
        return list_to_hash(self.outer_angles)

    @property
    def hash_inner_axes(self) -> str:
        """ A blake2b hash of :attr:`inner_axes`. """
        return list_to_hash(self.inner_axes)

    @property
    def hash_outer_axes(self) -> str:
        """ A blake2b hash of :attr:`outer_axes`. """
        return list_to_hash(self.outer_axes)

    @property
    def projection_shape(self) -> NDArray[int]:
        """ 2D shape of the raster-scan. 1st element is the number of steps in the
        j-direction and the second is the number of steps in the k-direction."""
        return self._projection_shape

    @projection_shape.setter
    def projection_shape(self, value: NDArray[int]) -> None:
        if type(value) is not tuple:
            value = tuple(value)

        if len(value) != 2:
            raise ValueError('Length of projection_shape must be exactly 2.')

        if not all(isinstance(item, (int, np.integer)) for item in value):
            first_wrong = next((item for item in value if type(item) is not int))
            raise TypeError(f'{type(first_wrong)} cannot be interpreted as an integer.')

        self._projection_shape = np.array(value).astype(np.int32)

    @property
    def volume_shape(self) -> NDArray[int]:
        """ 3D shape of the reconstruction voxel array. 1st element is the number of points
        along the x-direction. 2nd is y and 3rd is z."""
        return self._volume_shape

    @volume_shape.setter
    def volume_shape(self, value: NDArray[int]) -> None:
        if type(value) is not tuple:
            value = tuple(value)

        if len(value) != 3:
            raise ValueError('Length of volume_shape must be exactly 2.')

        if not all(isinstance(item, (int, np.integer)) for item in value):
            first_wrong = next((item for item in value if type(item) is not int))
            raise TypeError(f'{type(first_wrong)} cannot be interpreted as an integer.')

        self._volume_shape = np.array(value).astype(np.int32)

    @property
    def detector_angles(self) -> NDArray(float):
        """ Azimuthal angles of detector segments in radians.
        One-dimensional sequence of center positions"""
        return self._detector_angles

    @detector_angles.setter
    def detector_angles(self, value: NDArray(float)) -> None:
        value = np.array(value).astype(float)
        if np.ndim(value) != 1:
            raise ValueError('Detector angles must be a one-dimensional sequence.')
        self._detector_angles = value

    @property
    def two_theta(self) -> NDArray(float):
        """ Scattering angle in radians. Can be list of angles, if multiple
        radial bins are used."""
        return self._two_theta

    @two_theta.setter
    def two_theta(self, value: NDArray(float)) -> None:

        value = np.array(value)

        if np.ndim(value) == 0:
            value = value.flatten()
        elif np.ndim(value) > 1:
            raise ValueError('Only scalars or one-dimensional sequences are valid values of two_theta.')

        self._two_theta = value.astype(float)

    @property
    def full_circle_covered(self) -> bool:
        """ Whether the azimuthal bins cover a half-circle of the detector (False)
        or the full circle (True). """
        return self._full_circle_covered

    @full_circle_covered.setter
    def full_circle_covered(self, value: bool) -> None:
        self._full_circle_covered = bool(value)

    @k_offsets.setter
    def k_offsets(self, value: Union[list[float], np.ndarray[float]]) -> None:
        self._k_offsets = list(value)

    def __hash__(self) -> int:
        to_hash = [self.rotations_as_array,
                   self.j_offsets_as_array,
                   self.k_offsets_as_array,
                   self.p_direction_0, self.j_direction_0, self.k_direction_0,
                   self.detector_direction_origin, self.detector_direction_positive_90,
                   self.two_theta,
                   self.projection_shape, self.volume_shape,
                   self.detector_angles, self.full_circle_covered,
                   *self._get_hashable_axes_and_angles()]
        return int(list_to_hash(to_hash), 16)

    def __len__(self) -> int:
        return len(self._rotations)

    def _get_str_representation(self, max_lines: int = 25) -> str:
        """ Retrieves a string representation of the object with specified
        maximum number of lines.

        Parameters
        ----------
        max_lines
            The maximum number of lines to return.
        """
        wdt = 74
        s = []
        s += ['-' * wdt]
        s += ['Geometry'.center(wdt)]
        s += ['-' * wdt]
        with np.printoptions(threshold=3, edgeitems=1, precision=3, linewidth=60):
            s += ['{:18} : {}'.format('hash_rotations',
                  self.hash_rotations[:6])]
            s += ['{:18} : {}'.format('hash_j_offsets',
                  self.hash_j_offsets[:6])]
            s += ['{:18} : {}'.format('hash_k_offsets',
                  self.hash_k_offsets[:6])]
            s += ['{:18} : {}'.format('p_direction_0', self.p_direction_0)]
            s += ['{:18} : {}'.format('j_direction_0', self.j_direction_0)]
            s += ['{:18} : {}'.format('k_direction_0', self.k_direction_0)]
            s += ['{:18} : {}'.format('hash_inner_angles', self.hash_inner_angles[:6])]
            s += ['{:18} : {}'.format('hash_outer_angles', self.hash_outer_angles[:6])]
            s += ['{:18} : {}'.format('hash_inner_axes', self.hash_inner_axes[:6])]
            s += ['{:18} : {}'.format('hash_outer_axes', self.hash_outer_axes[:6])]
            s += ['{:18} : {}'.format('detector_direction_origin', self.detector_direction_origin)]
            s += ['{:18} : {}'.format('detector_direction_positive_90', self.detector_direction_positive_90)]
            s += ['{:18} : {}°'.format('two_theta', np.rad2deg(self.two_theta))]
            s += ['{:18} : {}'.format('projection_shape', self.projection_shape)]
            s += ['{:18} : {}'.format('volume_shape', self.volume_shape)]
            s += ['{:18} : {}'.format('detector_angles', self.detector_angles)]
            s += ['{:18} : {}'.format('full_circle_covered', self.full_circle_covered)]
        s += ['-' * wdt]
        truncated_s = []
        leave_loop = False
        while not leave_loop:
            line = s.pop(0).split('\n')
            for split_line in line:
                if split_line != '':
                    truncated_s += [split_line]
                if len(truncated_s) > max_lines - 2:
                    if split_line != '...':
                        truncated_s += ['...']
                    if split_line != ('=' * wdt):
                        truncated_s += ['=' * wdt]
                    leave_loop = True
                    break
            if len(s) == 0:
                leave_loop = True
        return '\n'.join(truncated_s)

    def __str__(self) -> str:
        return self._get_str_representation()

    def _get_html_representation(self, max_lines: int = 25) -> str:
        """ Retrieves an html representation of the object with specified
        maximum number of lines.

        Parameters
        ----------
        max_lines
            The maximum number of lines to return.
        """
        s = []
        s += ['<h3>Geometry</h3>']
        s += ['<table border="1" class="dataframe">']
        s += ['<thead><tr><th style="text-align: left;">Field</th><th>Size</th><th>Data</th></tr></thead>']
        s += ['<tbody>']
        with np.printoptions(threshold=3, edgeitems=1, precision=2, linewidth=40):
            s += ['<tr><td style="text-align: left;">rotations</td>']
            s += [f'<td>{len(self.rotations)}</td>'
                  f'<td>{self.hash_rotations[:6]} (hash)</td></tr>']
            s += ['<tr><td style="text-align: left;">j_offsets</td>']
            s += [f'<td>{len(self.j_offsets)}</td>'
                  f'<td>{self.hash_j_offsets[:6]} (hash)</td></tr>']
            s += ['<tr><td style="text-align: left;">k_offsets</td>']
            s += [f'<td>{len(self.k_offsets)}</td>'
                  f'<td>{self.hash_k_offsets[:6]} (hash)</td></tr>']
            s += ['<tr><td style="text-align: left;">p_direction_0</td>']
            s += [f'<td>{len(self.p_direction_0)}</td><td>{self.p_direction_0}</td></tr>']
            s += ['<tr><td style="text-align: left;">j_direction_0</td>']
            s += [f'<td>{len(self.j_direction_0)}</td><td>{self.j_direction_0}</td></tr>']
            s += ['<tr><td style="text-align: left;">k_direction_0</td>']
            s += [f'<td>{len(self.k_direction_0)}</td><td>{self.k_direction_0}</td></tr>']
            s += ['<tr><td style="text-align: left;">inner_angles</td>']
            s += [f'<td>{len(self.inner_angles)}</td>'
                  f'<td>{self.hash_inner_angles[:6]} (hash)</td></tr>']
            s += ['<tr><td style="text-align: left;">outer_angles</td>']
            s += [f'<td>{len(self.outer_angles)}</td>'
                  f'<td>{self.hash_outer_angles[:6]} (hash)</td></tr>']
            s += ['<tr><td style="text-align: left;">inner_axes</td>']
            s += [f'<td>{len(self.inner_axes)}</td>'
                  f'<td>{self.hash_inner_axes[:6]} (hash)</td></tr>']
            s += ['<tr><td style="text-align: left;">outer_axes</td>']
            s += [f'<td>{len(self.outer_axes)}</td>'
                  f'<td>{self.hash_outer_axes[:6]} (hash)</td></tr>']
            s += ['<tr><td style="text-align: left;">detector_direction_origin</td>']
            s += [f'<td>{len(self.detector_direction_origin)}</td>'
                  f'<td>{self.detector_direction_origin}</td></tr>']
            s += ['<tr><td style="text-align: left;">detector_direction_positive_90</td>']
            s += [f'<td>{len(self.detector_direction_positive_90)}</td>'
                  f'<td>{self.detector_direction_positive_90}</td></tr>']
            s += ['<tr><td style="text-align: left;">two_theta</td>']
            s += [f'<td>{len(self.two_theta)}</td>'
                  '<td>' + f'{self.two_theta * 180 / np.pi}' + r'${}^{\circ}$</td>']
            s += ['<tr><td style="text-align: left;">projection_shape</td>']
            s += [f'<td>{len(self.projection_shape)}</td><td>{self.projection_shape}</td></tr>']
            s += ['<tr><td style="text-align: left;">volume_shape</td>']
            s += [f'<td>{len(self.volume_shape)}</td><td>{self.volume_shape}</td></tr>']
            s += ['<tr><td style="text-align: left;">detector_angles</td>']
            s += [f'<td>{len(self.detector_angles)}</td><td>{self.detector_angles}</td></tr>']
            s += ['<tr><td style="text-align: left;">full_circle_covered</td>']
            s += [f'<td>{1}</td><td>{self.full_circle_covered}</td></tr>']
        s += ['</tbody>']
        s += ['</table>']
        truncated_s = []
        line_count = 0
        leave_loop = False
        while not leave_loop:
            line = s.pop(0).split('\n')
            for split_line in line:
                truncated_s += [split_line]
                if '</tr>' in split_line:
                    line_count += 1
                    # Catch if last line had ellipses
                    last_tr = split_line
                if line_count > max_lines - 1:
                    if last_tr != '<tr><td style="text-align: left;">...</td></tr>':
                        truncated_s += ['<tr><td style="text-align: left;">...</td></tr>']
                    truncated_s += ['</tbody>']
                    truncated_s += ['</table>']
                    leave_loop = True
                    break
            if len(s) == 0:
                leave_loop = True
        return '\n'.join(truncated_s)

    def _repr_html_(self) -> str:
        return self._get_html_representation()
