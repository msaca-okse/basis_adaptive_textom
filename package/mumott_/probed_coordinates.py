from dataclasses import dataclass, field
from mumott_.hashing import list_to_hash
import numpy as np



def _default_vector():
    """ Factory function needed by dataclass """
    return np.array([1., 0., 0.]).reshape(1, 1, 1, 3)


def _default_offset_vector():
    """ Factory function needed by dataclass """
    return np.array([0., 0., 0.]).reshape(1, 1, 1, 3)


@dataclass
class ProbedCoordinates:
    """ A small container class for probed coordinates for the
    :class:` BasisSet <mumott.methods.BasisSet>`.

    Parameters
    ----------
    vector : NDArray[float]
        The coordinates on the sphere probed at each detector segment by the
        experimental method. Should be structured ``(N, M, I, 3)`` where ``N``
        is the number of projections, ``M`` is the number of detector segments,
        ``I`` is the number of points on the detector to be integrated over,
        and the last index gives the ``(x, y, z)`` components of the coordinates.
        ``I`` can be as small as ``1``, but the array still needs to be structured
        in this way explicitly.
        By default, the value will be ``np.array([1., 0., 0.]).reshape(1, 1, 1, 3)``.

    great_circle_offset : np.ndarray[float]
        The vector which offsets the probed coordinate vectors from lying on a
        great circle. Must have the same number of dimensions as :attr:`vector`, and be
        broadcastable (dimensions where the value would be repeated may be ``1``).
        This vector is used when interpolating coordinates that lie on a small circle.

    """
    vector: np.ndarray[float] = field(default_factory=_default_vector)
    great_circle_offset: np.ndarray[float] = field(default_factory=_default_offset_vector)

    def __hash__(self) -> int:
        return int(list_to_hash([self.vector, self.great_circle_offset]), 16)

    @property
    def to_spherical(self) -> tuple:
        """ Returns spherical coordinates of :attr:`vector`, in the order
        ``(radius, polar angle, azimuthal angle)``. """
        r = np.linalg.norm(self.vector, axis=-1)
        theta = np.arccos(self.vector[..., 2] / r)
        phi = np.arctan2(self.vector[..., 1], self.vector[..., 0])
        return (r, theta, phi)
    
