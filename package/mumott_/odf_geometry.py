# --- odf_geometry.py ---

from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray
from mumott_.geometry import Geometry
from scipy.spatial.transform import Rotation as R

# import your own classes
from mumott_.probed_coordinates import ProbedCoordinates



@dataclass
class ODFGeometry:
    """
    Skeleton geometry describing the ODF sampling for a given experiment.

    You should fill in: how to construct ProbedCoordinates from
    (rotations, two-theta ring(s), azimuthal bins, etc).

    Attributes
    ----------
    detector_angles : array-like (M,)
        Azimuthal bin centers in radians.
    two_theta : array-like (R,)
        Scattering angles (radians). If you use multiple rings, M_total = M * R.
    ell_max : int
        Spherical-harmonics band-limit.
    enforce_friedel_symmetry : bool
        If True, restricts ℓ to even values.
    """

    azimuthal_angles: Sequence[float] = (0.0,)
    two_theta: Sequence[float] = (0.0,)
    ell_max: int = 0
    enforce_friedel_symmetry: bool = False
    rotation: Sequence[float] = (0.0,)   # assume list of rotation angles about z
    kernel_scale_parameter: float = 1.0
    grid_scale: int = 4

    # class-level constants
    _p_direction_0: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=float))
    _j_direction_0: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0], dtype=float))
    _k_direction_0: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1], dtype=float))

    # this is filled after init
    geom: object = field(init=False)

    def __post_init__(self):
        # set up Geometry
        self.geom = Geometry()
        self.geom._p_direction_0 = self._p_direction_0
        self.geom._j_direction_0 = self._j_direction_0
        self.geom._k_direction_0 = self._k_direction_0

        self.geom.detector_angles = self.azimuthal_angles
        self.geom.two_theta = self.two_theta
        self.geom.full_circle_covered = True

        # build rotation matrices about z
        Rz = [
            R.from_rotvec(np.array([0., 0., 1.]) * angle).as_matrix()
            for angle in self.rotation
        ]
        self.geom.rotations = Rz
    


    def _build_probed_coordinates(self) -> ProbedCoordinates:
        """
        Return ProbedCoordinates with vectors shaped (N, M_total, I, 3)
        and optional great_circle_offset of same broadcastable shape.

        TODO: Replace this stub with your actual construction (you can
        port the logic from your mumott Geometry._get_probed_coordinates()).
        """
        az = np.asarray(self.azimuthal_angles, dtype=float)   # (M,)
        tt = np.asarray(self.two_theta, dtype=float)         # (R,)
        M = az.size * tt.size
        

        self.probed_coordinates = self.geom._get_probed_coordinates()
        return self.probed_coordinates


    def build_probed_coordinates(self) -> ProbedCoordinates:
        """
        Return ProbedCoordinates with vectors shaped (N, M_total, I, 3)
        and optional great_circle_offset of same broadcastable shape.

        TODO: Replace this stub with your actual construction (you can
        port the logic from your mumott Geometry._get_probed_coordinates()).
        """
        az = np.asarray(self.azimuthal_angles, dtype=float)   # (M,)
        tt = np.asarray(self.two_theta, dtype=float)         # (R,)
        M = az.size * tt.size
        

        self.probed_coordinates = self.geom.get_probed_coordinates()
        return self.probed_coordinates
    

    def _calculate_basis_vectors(self) -> None:
        """ Calculates the basis vectors for the John transform, one projection vector
        and two coordinate vectors. """
        self._basis_vector_projection = np.einsum(
            'kij,i->kj', self.geom.rotations_as_array, self._p_direction_0)
        self._basis_vector_j = np.einsum(
            'kij,i->kj', self.geom.rotations_as_array, self._j_direction_0)
        self._basis_vector_k = np.einsum(
            'kij,i->kj', self.geom.rotations_as_array, self._k_direction_0)
        

    def _get_john_transform_parameters(self,indices: NDArray[int] = None) -> Tuple:
        self._calculate_basis_vectors()
        if indices is None:
            indices = np.s_[:]
        vector_p = self._basis_vector_projection[indices]
        vector_j = self._basis_vector_j[indices]
        vector_k = self._basis_vector_k[indices]
        #j_offsets = self._geometry.j_offsets_as_array[indices]
        #k_offsets = self._geometry.k_offsets_as_array[indices]
        return vector_p, vector_j, vector_k,# j_offsets, k_offsets)


    # Helpers
    @property
    def n_segments(self) -> int:
        return int(np.size(self.azimuthal_angles) * np.size(self.two_theta))
