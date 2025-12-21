import logging
import numpy as np

from abc import ABC, abstractmethod
from mumott_.probed_coordinates import ProbedCoordinates
from scipy.integrate import simpson, romb, trapezoid
from scipy.sparse import csr_array
from mumott_.hashing import list_to_hash


from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class BasisSet(ABC):

    """This is the base class from which specific basis sets are being derived.
    """

    def __init__(self,
                 probed_coordinates: ProbedCoordinates = None,
                 **kwargs):
        if probed_coordinates is None:
            probed_coordinates = ProbedCoordinates()
        self.probed_coordinates = probed_coordinates
        self._integration_mode = kwargs.get('integration_mode', 'simpson')
        if self._integration_mode not in ('simpson', 'romberg', 'trapezoid', 'midpoint'):
            raise ValueError('integration_mode must be "simpson" (for integration with Simpson\'s rule), '
                             ' "midpoint" (for center-of-segment approximation), "romberg", or "trapezoid"!')
        self._integration_tolerance = kwargs.get('integration_tolerance', 1e-5)
        self._integration_maxiter = kwargs.get('integration_maxiter', 10)
        self._n_integration_starting_points = kwargs.get('n_integration_starting_points', 3)
        self._enforce_sparsity = kwargs.get('enforce_sparsity', False)
        self._sparsity_count = kwargs.get('sparsity_count', 3)

    @property
    def probed_coordinates(self) -> ProbedCoordinates:
        return self._probed_coordinates

    @probed_coordinates.setter
    def probed_coordinates(self, pc: ProbedCoordinates) -> None:
        self._probed_coordinates = pc

    @abstractmethod
    def forward(self,
                coefficients: NDArray,
                indices: NDArray = None) -> NDArray:
        pass

    @abstractmethod
    def gradient(self,
                 coefficients: NDArray,
                 indices: NDArray = None) -> NDArray:
        pass

    @abstractmethod
    def get_spherical_harmonic_coefficients(self,
                                            coefficients: NDArray,
                                            ell_max: int = None,) -> NDArray:
        pass

    @abstractmethod
    def _get_projection_matrix(self, probed_coordinates: ProbedCoordinates):
        pass

    def generate_map(self,
                     coefficients: NDArray[float],
                     resolution_in_degrees: int = 5,
                     map_half_sphere: bool = True) -> tuple[NDArray]:
        """ Generate a (theta, phi) map of the function modeled by the input coefficients.
        If :attr:`map_half_sphere=True` (default) a map of only the z>0 half sphere is returned.

        Parameters
        ----------
        coefficients
            One dimensional numpy array with length ``len(self)`` containing the coefficients
            of the function to be plotted.
        resolution_in_degrees
            The resoution of the map in degrees. The map uses eqidistant lines in longitude
            and latitude.
        map_half_sphere
            If `True` returns a map of the z>0 half sphere.

        Returns
        -------
        map_intensity
            Intensity values of the map.
        map_theta
            Polar cooridnates of the map.
        map_phi
            Azimuthal coordinates of the map.
        """
        # Generate coordinates.
        if map_half_sphere:
            steps = int(np.ceil(90/resolution_in_degrees))
            map_theta = np.linspace(0, np.pi/2, steps + 1)
        else:
            steps = int(np.ceil(180/resolution_in_degrees))
            map_theta = np.linspace(0, np.pi, steps + 1)
        steps = int(np.ceil(360/resolution_in_degrees))
        map_phi = np.linspace(0, 2*np.pi, steps + 1)
        map_theta, map_phi = np.meshgrid(map_theta, map_phi, indexing='ij')

        # Create a ProbedCoordinates to pass into `_get_projection_matrix`
        x = np.cos(map_phi)*np.sin(map_theta)
        y = np.sin(map_phi)*np.sin(map_theta)
        z = np.cos(map_theta)
        vector = np.stack((x, y, z), axis=-1)
        probed_coordinates = ProbedCoordinates()
        probed_coordinates.vector = vector[:, :, np.newaxis, :]

        # Evaluate map intensity
        basis_funciton_values = self._get_projection_matrix(probed_coordinates)[:, :, 0, :]
        map_intensity = np.einsum('m,tpm->tp', coefficients, basis_funciton_values)
        return map_intensity, map_theta, map_phi

    def _make_projection_matrix_sparse(self, matrix: np.ndarray[float]) -> np.ndarray[float]:
        if self._enforce_sparsity:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    sorted_args = np.argsort(matrix[i, j, :])
                    for k in sorted_args[:-self._sparsity_count]:
                        matrix[i, j, k] = 0.

    def _get_integrated_projection_matrix(self, probed_coordinates: ProbedCoordinates = None):
        """ Uses Simpson's rule to integrate the basis set representation over
        each detector segment on the unit sphere."""
        # use 0 and -1 for backwards compatibility
        if probed_coordinates is None:
            probed_coordinates = self.probed_coordinates
        start = probed_coordinates.vector[..., 0, :]
        # don't normalize in-place to avoid modifying probed_coordinates
        start = start / np.linalg.norm(start, axis=-1)[..., None]
        start = start[..., None, :]
        end = probed_coordinates.vector[..., -1, :]
        end = end / np.linalg.norm(end, axis=-1)[..., None]
        end = end[..., None, :]
        # if segments lie on small circle, correct using waxs offset vector before slerp
        offset = probed_coordinates.great_circle_offset[..., 0, :]
        offset = offset[..., None, :]
        # when run initially with probed_coordinates = None or similar
        if np.allclose(start, end):
            return self._get_projection_matrix(probed_coordinates)[:, :, 0]
        if self._integration_mode == 'midpoint':
            # Just use central point to get the projection matrix.
            pc = ProbedCoordinates(probed_coordinates.vector[..., 1:2, :])
            return self._get_projection_matrix(pc)[..., 0, :]
        # segment length is subtended angle between start and end
        corr_start = start - offset
        corr_end = end - offset
        segment_length = np.arccos(np.einsum('...i, ...i', corr_start, corr_end))

        def slerp(t):
            omega = segment_length.reshape(segment_length.shape + (1,))
            t = t.reshape(1, 1, -1, 1)
            # avoid division by 0, use 1st order approximation
            where = np.isclose(abs(omega[..., 0, 0]), 0.)
            sin_omega = np.sin(omega)
            sin_tomega = np.sin(t * omega)
            sin_1mtomega = np.sin((1 - t) * omega)
            a = np.zeros(start.shape[:2] + (t.shape[2], 1), dtype=float)
            b = np.zeros(start.shape[:2] + (t.shape[2], 1), dtype=float)
            a[~where] = sin_1mtomega[~where] / sin_omega[~where]
            b[~where] = sin_tomega[~where] / sin_omega[~where]
            # sin(ax) / sin(x) ~ a for x ~ 0
            a[where] = (1 - t) * np.ones_like(omega[where])
            b[where] = t * np.ones_like(omega[where])
            return a * corr_start + b * corr_end + offset

        def quadrature(matrix, t):
            if self._integration_mode == 'simpson':
                return simpson(matrix, x=t, axis=-2)
            elif self._integration_mode == 'romberg':
                return romb(matrix, dx=1 / (t.size - 1), axis=-2)
            elif self._integration_mode == 'trapezoid':
                return trapezoid(matrix, x=t, axis=-2)
        number_of_points = self._n_integration_starting_points
        t = np.linspace(0, 1, number_of_points)
        pc = ProbedCoordinates(slerp(t))
        old_matrix = self._get_projection_matrix(pc)
        # get an initial matrix for comparison
        old_matrix = quadrature(old_matrix, t)
        for i in range(self._integration_maxiter):
            # double the number of intervals in each iteration
            number_of_points += max(number_of_points - 1, 1)
            t = np.linspace(0, 1, number_of_points)
            vector = slerp(t)
            pc = ProbedCoordinates(vector)
            # integrate all matrices using simpson's rule
            new_matrix = quadrature(self._get_projection_matrix(pc), t)
            norm = np.max(abs(new_matrix - old_matrix)) / np.max(abs(new_matrix))
            if norm < self._integration_tolerance:
                break
            old_matrix = new_matrix
        else:
            logger.warning('Projection matrix did not converge! '
                           'Try increasing integration_maxiter or reducing integration_tolerance.')
        self._make_projection_matrix_sparse(new_matrix)
        return new_matrix

    @property
    def integration_mode(self) -> str:
        """
        Mode of integration for calculating projection matrix.
        Accepted values are ``'simpson'``, ``'romberg'``, ``'trapezoid'``,
        and ``'midpoint'``.
        """
        return self._integration_mode

    @integration_mode.setter
    def integration_mode(self, val) -> None:
        if val not in ('simpson', 'midpoint', 'romberg'):
            raise ValueError('integration_mode must have value "midpoint", '
                             '"romberg", "trapezoid" or "simpson", '
                             f'but a value of {val} was given!')
        self._integration_mode = val
        self._projection_matrix = self._get_integrated_projection_matrix()

    @abstractmethod
    def get_output(self,
                   coefficients: NDArray) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    # @abstractmethod
    # def __dict__(self) -> dict:
    #     pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def _repr_html_(self) -> str:
        pass

    @property
    def csr_representation(self) -> tuple:
        """ The projection matrix as a stack of sparse matrices in
        CSR representation as a tuple. The information in the tuple consists of
        the 3 dense matrices making up the representation,
        in the order ``(pointers, indices, data)``."""
        nnz = np.max((self._projection_matrix > 0).sum((-1, -2)))
        sparse_data = np.zeros((self._projection_matrix.shape[0], nnz), dtype=np.float32)
        sparse_indices = np.zeros((self._projection_matrix.shape[0], nnz), dtype=np.int32)
        sparse_pointers = np.zeros((self._projection_matrix.shape[0],
                                    self._projection_matrix.shape[1] + 1), dtype=np.int32)
        for i in range(self._projection_matrix.shape[0]):
            sparse_matrix = csr_array(self._projection_matrix[i])
            sparse_matrix.eliminate_zeros()
            sparse_data[i, :sparse_matrix.nnz] = sparse_matrix.data
            sparse_indices[i, :sparse_matrix.nnz] = sparse_matrix.indices
            sparse_pointers[i, :len(sparse_matrix.indptr)] = sparse_matrix.indptr
        return sparse_pointers, sparse_indices, sparse_data
