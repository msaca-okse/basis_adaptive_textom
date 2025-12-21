import logging
from typing import Any, Dict, Iterator, Tuple

import numpy as np
from numpy.typing import NDArray
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing.pool import ThreadPool
#from multiprocessing.Pool import ThreadPool


from mumott_.probed_coordinates import ProbedCoordinates#, SphericalHarmonicMapper
from mumott_.hashing import list_to_hash
from mumott_.tensor_operations import (framewise_contraction_transpose_pool, framewise_contraction_pool)

from mumott_.base_basis_set import BasisSet
from mumott_.spherical_harmonics import SphericalHarmonics


logger = logging.getLogger(__name__)


class GaussianKernels(BasisSet):
    r""" Basis set class for gaussian kernels, a simple local representation on the sphere.
    The kernels follow a pseudo-even distribution similar to that described by
    Y. Kurihara `in 1965 <https://doi.org/10.1175/1520-0493%281965%29093%3C0399%3ANIOTPE%3E2.3.CO%3B2>`_,
    except with offsets added at the poles.

    Notes
    -----
    The Gaussian kernel at location :math:`\rho_i` is given by

    .. math ::
        N_i \exp\left[ -\frac{1}{2} \left(\frac{d(\rho_i, r)}{\sigma}\right)^2 \right]

    .. math ::
        \sigma = \frac{\nu \pi}{2 (g + 1)}

    where :math:`\nu` is the kernel scale parameter and :math:`g` is the grid scale, and

    .. math ::
        d(\rho, r) = \arctan_2(\Vert \rho \times r \Vert, \rho \cdot r),

    that is, the great circle distance from the kernel location :math:`\rho` to the
    probed location :math:`r`. If Friedel symmetry is assumed, the expression is instead

    .. math ::
        d(\rho, r) = \arctan_2(\Vert \rho \times r \Vert, \vert \rho \cdot r \vert)

    The normalization factor :math:`\rho_i` is given by

    .. math ::
        N_i = \sum_j \exp\left[ -\frac{1}{2} \left( \frac{d(\rho_i, \rho_j)}{\sigma} \right)^2 \right]

    where the sum goes over the coordinates of all grid points. This leads to an
    approximately even spherical function, such that a set of coefficients which are all equal
    is approximately isotropic, to the extent possible with respect to restrictions
    imposed by grid resolution and scale parameter.

    Parameters
    ----------
    probed_coordinates : ProbedCoordinates
        Optional. A container with the coordinates on the sphere probed at each detector segment by the
        experimental method. Its construction from the system geometry is method-dependent.
        By default, an empty instance of :class:`mumott.ProbedCoordinates` is created.
    grid_scale : int
        The size of the coordinate grid on the sphere. Denotes the number of azimuthal rings between the
        pole and the equator, where each ring has between ``2`` and ``2 * grid_scale`` points
        along the azimuth.
    kernel_scale_parameter : float
        The scale parameter of the kernel in units of :math:`\frac{\pi}{2 (g + 1)}`, where
        :math:`g` is ``grid_scale``.
    enforce_friedel_symmetry : bool
        If set to ``True``, Friedel symmetry will be enforced, using the assumption that points
        on opposite sides of the sphere are equivalent.
    kwargs
        Miscellaneous arguments which relate to segment integrations can be
        passed as keyword arguments:

            integration_mode
                 Mode to integrate line segments on the reciprocal space sphere. Possible options are
                 ``'simpson'``, ``'midpoint'``, ``'romberg'``, ``'trapezoid'``.
                 ``'simpson'``, ``'trapezoid'``, and ``'romberg'`` use adaptive
                 integration with the respective quadrature rule from ``scipy.integrate``.
                 ``'midpoint'`` uses a single mid-point approximation of the integral.
                 Default value is ``'simpson'``.
            n_integration_starting_points
                 Number of points used in the first iteration of the adaptive integration.
                 The number increases by the rule ``N`` &larr; ``2 * N - 1`` for each iteration.
                 Default value is 3.
            integration_tolerance
                 Tolerance for the maximum relative error between iterations before the integral
                 is considered converged. Default is ``1e-5``.
            integration_maxiter
                 Maximum number of iterations. Default is ``10``.
            enforce_sparsity
                 If ``True``, makes matrix sparse by limiting the number of basis set elements
                 that can map to each segment. Default is ``False``.
            sparsity_count
                 Number of basis set elements that can map to each segment,
                 if ``enforce_sparsity`` is set to ``True``. Default is ``3``.
        """
    def __init__(self,
                 probed_coordinates: ProbedCoordinates = None,
                 grid_scale: int = 4,
                 kernel_scale_parameter: float = 1.,
                 enforce_friedel_symmetry: bool = False,
                 N_splits: int = 1,
                 projections_per_proc: int = 10,
                 **kwargs):
        super().__init__(probed_coordinates, **kwargs)
        self._probed_coordinates_hash = hash(self.probed_coordinates)
        self._grid_scale = grid_scale
        self._kernel_scale_parameter = kernel_scale_parameter
        self._enforce_friedel_symmetry = enforce_friedel_symmetry
        self.N_splits = N_splits
        self.projections_per_proc = projections_per_proc
        #self._projection_matrix = np.ascontiguousarray(self._get_integrated_projection_matrix().transpose([0,2,1]))
        self._projection_matrix = np.ascontiguousarray(self._get_integrated_projection_matrix_fast(N_splits = N_splits).transpose([0,2,1]))
        self.M_rot = self._projection_matrix.shape[0]


    # def __getstate__(self):
    #     # Copy state but drop the big attribute
    #     state = self.__dict__.copy()
    #     if "_projection_matrix" in state:
    #         del state["_projection_matrix"]
    #     return state

    # def __setstate__(self, state):
    #     # Restore everything else, leave projection_matrix unset
    #     self.__dict__.update(state)
    #     self._projection_matrix = None  # or reload lazily if needed


    def _get_integrated_projection_matrix_fast(self, probed_coordinates=None, N_splits=1):
        if probed_coordinates is None:
            probed_coordinates = self.probed_coordinates

        M_rot = probed_coordinates.vector.shape[0]

        # Split and build ProbedCoordinates objects
        probed_split = np.array_split(probed_coordinates.vector, M_rot//self.projections_per_proc, axis=0)
        probed_list = [ProbedCoordinates(vector=vec) for vec in probed_split]
        print('Instantiating basis change matrix with #processes = ', N_splits)
        # Use pathos Pool, but call the *top-level* wrapper
        with Pool(nodes=N_splits) as pool:
            results = pool.map(
                self._call_get_integrated_projection_matrix,
                [(coords) for coords in probed_list]
            )

        out = np.concatenate(results, axis=0)
        return out


    def _call_get_integrated_projection_matrix(self,args):
        coords = args
        proj_mat =  self._get_integrated_projection_matrix(probed_coordinates=coords)
        return proj_mat


    def _get_kurihara_mesh(self, N) -> Tuple[NDArray, NDArray]:
        phi = []
        theta = []
        for i in np.arange(N, -1, -1):
            for j in np.arange(0, (2 * (i + 0.5))):
                phi.append((j + 0.5) / (2 * (i + 0.5)) * np.pi)
                phi.append((j + 0.5) / (2 * (i + 0.5)) * -np.pi)
                theta.append((i + 0.5) / (N + 1) * np.pi / 2)
                theta.append((i + 0.5) / (N + 1) * np.pi / 2)
        theta = np.array(theta)
        phi = np.mod(phi, 2 * np.pi)

        if not self._enforce_friedel_symmetry:
            theta = np.concatenate((theta, np.arccos(-np.cos(theta))))
            phi = np.concatenate((phi, phi))
        return theta, phi

    def get_inner_product(self,
                          u: NDArray,
                          v: NDArray) -> NDArray:
        r""" Retrieves the inner product of two coefficient arrays, that is to say,
        the sum-product over the last axis.

        Parameters
        ----------
        u
            The first coefficient array, of arbitrary shape and dimension, so long as
            the number of coefficients equals the length of this :class:`GaussianKernels` instance.
        v
            The second coefficient array, of the same shape as :attr:`u`.
        """
        assert u.shape[-1] == len(self)
        assert u.shape == v.shape
        return np.einsum('...i, ...i -> ...', u, v, optimize='greedy')

    def _get_spherical_distances(self,
                                 theta_1: NDArray[float], theta_2: NDArray[float],
                                 phi_1: NDArray[float], phi_2: NDArray[float],
                                 radius: float = 1.,
                                 enforce_friedel_symmetry: bool = False) -> NDArray[float]:
        r""" Function for obtaining the distances between two point sets
        on a sphere, possibly with Friedel symmetry enforced.
        Arrays can have any shape, but they must all be broadcastable, and the polar angles of
        each set must have the same shape as the azimuthal angles.
        If the first and second set of points have the same shape, then the distances will be
        computed pointwise. Otherwise, the distances will be computed by broadcasting.

        Parameters
        ----------
        theta_1
            The polar angle of the first set of points, defined as :math:`\arccos(z_1)`.
        theta_2
            The polar angle fo the second set of points, defined as :math:`\arccos(z_2)`.
        phi_2
            The azimuthal angle of the first set of points, defined as :math:`\arctan_2(y_1, x_1)`.
        phi_2
            The azimuthal angle of the second set of points, defined as :math:`\arctan_2(y_2, x_2)`.
        radius
            The radius of the sphere. Default is `1`, i.e., describing a unit sphere.
        enforce_friedel_symmetry
            If ``True`` (default), the point :math:`(x, y, z)` will be considered as
            equivalent to the point :math:`(-x, -y, -z)` and the maximum possible distance on the sphere will
            be :math:`\frac{\sqrt{x^2 + y^2 + z^2} \pi}{2}`, i.e., at a 90-degree angle.
            Otherwise, the two points will be considered distinct and the maximum
            distance will be :math:`\sqrt{x^2 + y^2 + z^2} \pi`.
        """
        phi_diff = abs(phi_1 - phi_2)
        sine_factor = np.sin(theta_1) * np.sin(theta_2) * np.cos(phi_diff)
        cosine_factor = np.cos(theta_1) * np.cos(theta_2)
        if enforce_friedel_symmetry:
            return radius * np.arccos(abs(sine_factor + cosine_factor).clip(0., 1.))
        else:
            return radius * np.arccos((sine_factor + cosine_factor).clip(-1., 1.))

    def _get_projection_matrix(self, probed_coordinates: ProbedCoordinates = None) -> NDArray[float]:
        """ Computes the matrix necessary for forward and gradient calculations.
        Called when the coordinate system has been updated, or one of
        ``kernel_scale_parameter`` or ``grid_scale`` has been changed."""
        if probed_coordinates is None:
            probed_coordinates = self.probed_coordinates
        theta, phi = self._get_kurihara_mesh(self._grid_scale)
        phi = phi.reshape(1, 1, 1, -1)
        theta = theta.reshape(1, 1, 1, -1)
        _, probed_polar_angles, probed_azim_angles = probed_coordinates.to_spherical
        probed_polar_angles = probed_polar_angles[..., np.newaxis]
        probed_azim_angles = probed_azim_angles[..., np.newaxis]
        # Find distances to all probed detector points on sphere
        distances = self._get_spherical_distances(theta, probed_polar_angles,
                                                  phi, probed_azim_angles,
                                                  enforce_friedel_symmetry=self._enforce_friedel_symmetry)

        # Probe at location of each kernel function to normalize over sphere
        mesh_distances = self._get_spherical_distances(
            theta.reshape(-1, 1), theta.reshape(1, -1),
            phi.reshape(-1, 1), phi.reshape(1, -1),
            enforce_friedel_symmetry=self._enforce_friedel_symmetry)
        std = (self._kernel_scale_parameter * np.sqrt(2 * np.pi)) / (2 * (self._grid_scale + 1))
        matrix = np.exp(-(1 / 2) * (distances / std) ** 2)
        norm_matrix = np.exp(-(1 / 2) * (mesh_distances / std) ** 2)
        # The normalization factor is the inverse of the unnormalized function value at each grid point
        norm_factors = np.reciprocal(norm_matrix.sum(-1).reshape(1, 1, 1, -1))
        return matrix * norm_factors
    

    def worker_forward(self,args):  ### Is is really best to pass down the indices,
        # or should i rather pass down the matrix
        submatrices, coeffs = args
        out = np.zeros(coeffs.shape[:-1] + (submatrices.shape[2],), coeffs.dtype)
        framewise_contraction_transpose_pool(submatrices, coeffs, out)
        return out


    def worker_adjoint(self,args):
        submatrices, dat = args
        out = np.zeros(dat.shape[:-1] + (submatrices.T.shape[1],), dat.dtype)
        framewise_contraction_pool(submatrices, dat, out)
        return out


    def forward(self,
                coefficients: NDArray,
                indices: NDArray = None) -> NDArray:
        """ Carries out a forward computation of projections from Gaussian kernel space
        into detector space, for one or several tomographic projections.

        Parameters
        ----------
        coefficients
            An array of coefficients, of arbitrary shape so long as the last
            axis has the same size as :attr:`~.GaussianKernels.kernel_scale_parameter`, and if
            :attr:`indices` is ``None`` or greater than one, the first axis should have the
            same length as :attr:`indices`
        indices
            Optional. Indices of the tomographic projections for which the forward
            computation is to be performed. If ``None``, the forward computation will
            be performed for all projections.

        Returns
        -------
            An array of values on the detector corresponding to the :attr:`coefficients` given.
            If :attr:`indices` contains exactly one index, the shape is ``(coefficients.shape[:-1], J)``
            where ``J`` is the number of detector segments. If :attr:`indices` is ``None`` or contains
            several indices, the shape is ``(N, coefficients.shape[1:-1], J)`` where ``N``
            is the number of tomographic projections for which the computation is performed.

        Notes
        -----
        The assumption is made in this implementation that computations over several
        indices act on sets of images from different projections. For special usage
        where multiple projections of entire fields are desired, it may be better
        to use :attr:`projection_matrix` directly. This also applies to
        :meth:`gradient`.
        """
        self._update()
        output = np.zeros(coefficients.shape[:-1] + (self._projection_matrix.shape[2],),
                          coefficients.dtype)
        if indices is None:
            framewise_contraction_transpose(self._projection_matrix,
                                            coefficients,
                                            output)
        elif indices.size == 1:
            np.einsum('ijk, ...k -> ...j',
                      self._projection_matrix[indices],
                      coefficients,
                      out=output,
                      optimize='greedy',
                      casting='unsafe')
        else:

            
            indices_split = np.array_split(indices, self.M_rot//self.projections_per_proc)
            coeffs_split = np.array_split(coefficients, self.M_rot//self.projections_per_proc, axis=0)


            tasks = [(self.projection_matrix[idx], coeffs) for idx, coeffs in zip(indices_split, coeffs_split)]
            with Pool(nodes=self.N_splits) as pool:
                results = pool.map(self.worker_forward, tasks)

            output = np.concatenate(results, axis=0)
            # framewise_contraction_transpose_pool(self._projection_matrix[indices],
            #                                coefficients,
            #                                output)
        return output
    



    def adjoint(self,
                data: NDArray,
                indices: NDArray = None) -> NDArray:
        """ Carries out a forward computation of projections from spherical harmonic space
        into detector space, for one or several tomographic projections.

        Parameters
        ----------
        coefficients
            An array of coefficients, of arbitrary shape so long as the last
            axis has the same size as :attr:`~.SphericalHarmonics.ell_indices`, and if
            :attr:`indices` is `None` or greater than one, the first axis should have the
            same length as :attr:`indices`
        indices
            Optional. Indices of the tomographic projections for which the forward
            computation is to be performed. If ``None``, the forward computation will
            be performed for all projections.

        Returns
        -------
            An array of values on the detector corresponding to the :attr:`coefficients` given.
            If :attr:`indices` contains exactly one index, the shape is ``(coefficients.shape[:-1], J)``
            where ``J`` is the number of detector segments. If :attr:`indices` is ``None`` or contains
            several indices, the shape is ``(N, coefficients.shape[1:-1], J)`` where ``N``
            is the number of tomographic projections for which the computation is performed.

        Notes
        -----
        The assumption is made in this implementation that computations over several
        indices act on sets of images from different projections. For special usage
        where multiple projections of entire fields is desired, it may be better
        to use :attr:`projection_matrix` directly. This also applies to
        :meth:`gradient`.
        """
        self._update()
        output = np.zeros(data.shape[:-1] + (self._projection_matrix.T.shape[1],),
                        data.dtype)
        
        if indices is None:
            NotImplementedError

        else:
            indices_split = np.array_split(indices, self.M_rot//self.projections_per_proc)
            data_split = np.array_split(data, self.M_rot//self.projections_per_proc, axis=0)

            tasks = [(self._projection_matrix[idx], dat) for idx, dat in zip(indices_split, data_split)]

            with Pool(nodes=self.N_splits) as pool:
                results = pool.map(self.worker_adjoint, tasks)

            output = np.concatenate(results, axis=0)
            # framewise_contraction_pool(self._projection_matrix[indices],
            #                     data,
            #                     output)
            return output





    def gradient(self,
                 coefficients: NDArray,
                 indices: NDArray = None) -> NDArray:
        """ Carries out a gradient computation of projections from Gaussian kernel space
        into detector space for one or several tomographic projections.

        Parameters
        ----------
        coefficients
            An array of coefficients (or residuals) of arbitrary shape so long as the last
            axis has the same size as the number of detector segments.
        indices
            Optional. Indices of the tomographic projections for which the gradient
            computation is to be performed. If ``None``, the gradient computation will
            be performed for all projections.

        Returns
        -------
            An array of gradient values based on the :attr:`coefficients` given.
            If :attr:`indices` contains exactly one index, the shape is ``(coefficients.shape[:-1], J)``
            where ``J`` is the number of detector segments. If indices is ``None`` or contains
            several indices, the shape is ``(N, coefficients.shape[1:-1], J)`` where ``N``
            is the number of tomographic projections for which the computation is performed.

        Notes
        -----
        When solving an inverse problem, one should not attempt to optimize the
        coefficients directly using the gradient one obtains by applying this method to the data.
        Instead, one must either take the gradient of the residual between the
        :meth:`~.GaussianKernels.forward` computation of the coefficients and the data.
        Alternatively one can apply both the forward and the gradient computation to the
        coefficients to be optimized, and the gradient computation to the data, and treat
        the residual of the two as the gradient of the optimization coefficients. The approaches
        are algebraically equivalent, but one may be more efficient than the other in some
        circumstances. However, normally, the projection between detector and
        ``GaussianKernel`` space is only a small part of the overall computation,
        so there is typically not much to be gained from optimizing it.
        """
        self._update()
        output = np.zeros(coefficients.shape[:-1] + (self._projection_matrix.shape[2],),
                          coefficients.dtype)
        if indices is None:
            framewise_contraction(self._projection_matrix,
                                  coefficients,
                                  output)
        elif indices.size == 1:
            np.einsum('ikj, ...k -> ...j',
                      self._projection_matrix[indices],
                      coefficients,
                      out=output,
                      optimize='greedy',
                      casting='unsafe')
        else:
            framewise_contraction(self._projection_matrix[indices],
                                  coefficients,
                                  output)
        return output

    def get_amplitudes(self, coefficients: NDArray[float],
                       probed_coordinates: ProbedCoordinates = None) -> NDArray[float]:
        """ Computes the amplitudes of the spherical function represented by the provided
        :attr:`coefficients` at the :attr:`probed_coordinates`.

        Parameters
        ----------
        coefficients
            An array of coefficients of arbitrary shape, provided that the
            last dimension contains the coefficients for one spherical function.
        probed_coordinates
            An instance of :class:`mumott.core.ProbedCoordinates` with its :attr:`vector`
            attribute indicating the points of the sphere for which to evaluate the amplitudes.
        """
        if probed_coordinates is None:
            probed_coordinates = self._probed_coordinates
        matrix = self._get_projection_matrix(probed_coordinates)
        self._make_projection_matrix_sparse(matrix)
        return np.einsum('ij, ...j', matrix.squeeze(), coefficients, optimize='greedy')

    def get_spherical_harmonic_coefficients(self, coefficients: NDArray, ell_max: int = None):
        """ Computes the spherical harmonic coefficients of the spherical function
        represented by the provided :attr:`coefficients` using a Driscoll-Healy grid.

        For details on the Driscoll-Healy grid, see
        `the SHTools page <https://shtools.github.io/SHTOOLS/grid-formats.html>`_ for a
        comprehensive overview.

        Parameters
        ----------
        coefficients
            An array of coefficients of arbitrary shape, provided that the
            last dimension contains the coefficients for one function.
        ell_max
            The bandlimit of the spherical harmonic expansion. By default, it is ``2 * grid_scale``.

        """
        if ell_max is None:
            ell_max = 2 * self._grid_scale
        dh_grid_size = 4 * (self._grid_scale + 1)
        mapper = SphericalHarmonicMapper(ell_max=ell_max, polar_resolution=dh_grid_size,
                                         azimuthal_resolution=dh_grid_size,
                                         enforce_friedel_symmetry=self._enforce_friedel_symmetry)
        coordinates = ProbedCoordinates(mapper.unit_vectors.reshape((1, -1, 1, 3)))
        amplitudes = self.get_amplitudes(coefficients, coordinates)\
            .reshape(coefficients.shape[:-1] + (dh_grid_size, dh_grid_size))
        spherical_harmonics_coefficients = mapper.get_harmonic_coefficients(amplitudes)
        return spherical_harmonics_coefficients

    def get_output(self,
                   coefficients: NDArray) -> Dict[str, Any]:
        r""" Returns a dictionary of output data for a given array of basis set coefficients.

        Parameters
        ----------
        coefficients
            An array of coefficients of arbitrary shape and dimensions, except
            its last dimension must be the same length as the :attr:`len` of this instance.
            Computations only operate over the last axis of :attr:`coefficients`, so derived
            properties in the output will have the shape ``(*coefficients.shape[:-1], ...)``.

        Returns
        -------
            A dictionary containing two sub-dictionaries, ``basis_set`` and ``spherical_harmonic_analysis``.
            ``basis_set`` contains information particular to :class:`GaussianKernels`, whereas
            ``spherical_harmonic_analysis`` contains an analysis of the spherical function using
            a spherical harmonic transform.

        Notes
        -----
        In detail, the two sub-dictionaries ``basis_set`` and ``spherical_harmonic_analysis``
        have the following members:

        basis_set
            name
                The name of the basis set, i.e., ``'GaussianKernels'``
            coefficients
                A copy of :attr:`coefficients`.
            grid_scale
                A copy of :attr:`~.GaussianKernels.grid_scale`.
            kernel_Scale_paramter
                A copy of :attr:`~.GaussianKernels.kernel_scale_parameter`.
            enforce_friedel_symmetry
                A copy of :attr:`~.GaussianKernels.enforce_friedel_symmetry`.
            projection_matrix
                A copy of :attr:`~.GaussianKernels.projection_matrix`.
        spherical_harmonic_analysis
            An analysis of the spherical function in terms of
            spherical harmonics. See
            :meth:`SphericalHarmonics.get_output <SphericalHarmonics.get_output()>` for details.
        """
        assert coefficients.shape[-1] == len(self)
        # Update to ensure non-dirty output state.
        self._update()
        sph_coefficients = self.get_spherical_harmonic_coefficients(coefficients)
        spherical_harmonics_set = SphericalHarmonics(ell_max=2 * self._grid_scale,
                                                     enforce_friedel_symmetry=self._enforce_friedel_symmetry)
        sph_output = spherical_harmonics_set.get_output(sph_coefficients)
        output_dictionary = {}

        # basis set-specific information
        basis_set = {}
        output_dictionary['basis_set'] = basis_set
        basis_set['name'] = type(self).__name__
        basis_set['coefficients'] = coefficients.copy()
        basis_set['grid_scale'] = self._grid_scale
        basis_set['grid'] = self.grid
        basis_set['kernel_scale_parameter'] = self._kernel_scale_parameter
        basis_set['enforce_friedel_symmetry'] = self._enforce_friedel_symmetry
        basis_set['projection_matrix'] = self._projection_matrix.copy()
        basis_set['hash'] = hex(hash(self))

        # Analysis is more easily done in spherical harmonic space.
        output_dictionary['spherical_harmonic_analysis'] = sph_output

        return output_dictionary

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """ Allows class to be iterated over and in particular be cast as a dictionary.
        """
        yield 'name', type(self).__name__
        yield 'grid_scale', self._grid_scale
        yield 'kernel_scale_parameter', self._kernel_scale_parameter
        yield 'enforce_friedel_symmetry', self._enforce_friedel_symmetry
        yield 'projection_matrix', self._projection_matrix
        yield 'hash', hex(hash(self))[2:]

    def __len__(self) -> int:
        return self._projection_matrix.shape[-1]

    def __hash__(self) -> int:
        """Returns a hash reflecting the internal state of the instance.

        Returns
        -------
            A hash of the internal state of the instance,
            cast as an ``int``.
        """
        to_hash = [self._grid_scale,
                   self.grid,
                   self._kernel_scale_parameter,
                   self._enforce_friedel_symmetry,
                   self._projection_matrix,
                   self._probed_coordinates_hash]
        return int(list_to_hash(to_hash), 16)

    def _update(self) -> None:
        # We only run updates if the hashes do not match.
        if self.is_dirty:
            self._projection_matrix = self._get_integrated_projection_matrix()
            self._probed_coordinates_hash = hash(self._probed_coordinates)

    @property
    def is_dirty(self) -> bool:
        return hash(self._probed_coordinates) != self._probed_coordinates_hash

    @property
    def projection_matrix(self) -> NDArray:
        """ The matrix used to project spherical functions from the unit sphere onto the detector.
        If ``v`` is a vector of gaussian kernel coefficients, and ``M`` is the ``projection_matrix``,
        then ``M @ v`` gives the corresponding values on the detector segments associated with
        each projection. ``M[i] @ v`` gives the values on the detector segments associated with
        projection ``i``.

        If ``r`` is a residual between a projection from Gaussian kernel to detector space and data from
        projection ``i``, then ``M[i].T @ r`` gives the associated gradient in Gaussian kernel space.
        """
        self._update()
        return self._projection_matrix

    @property
    def grid_scale(self) -> int:
        """ The number of azimuthal rings from each pole to the equator in the
        spherical grid.
        """
        return self._grid_scale

    @grid_scale.setter
    def grid_scale(self, val: int) -> None:
        if val < 0 or val != round(val):
            raise ValueError('grid_scale must be a non-negative integer,'
                             f' but a value of {val} was given!')
        self._grid_scale = val
        self._projection_matrix = self._get_integrated_projection_matrix()

    @property
    def kernel_scale_parameter(self) -> float:
        """ The scale parameter for each kernel.
        """
        return self._kernel_scale_parameter

    @kernel_scale_parameter.setter
    def kernel_scale_parameter(self, val: float) -> float:
        self._kernel_scale_parameter = val
        self._projection_matrix = self._get_integrated_projection_matrix()

    @property
    def enforce_friedel_symmetry(self) -> bool:
        """ If ``True``, Friedel symmetry is enforced, i.e., the point
        :math:`-r` is treated as equivalent to :math:`r`. """
        return self._enforce_friedel_symmetry

    @property
    def grid(self) -> Tuple[NDArray['float'], NDArray['float']]:
        r""" Returns the polar and azimuthal angles of the grid used by the basis.

        Returns
        -------
            A ``Tuple`` with contents ``(polar_angle, azimuthal_angle)``, where the
            polar angle is defined as :math:`\arccos(z)`.
        """
        return self._get_kurihara_mesh(self._grid_scale)

    @property
    def grid_hash(self) -> str:
        """ Returns a hash of :attr:`grid`.
        """
        return list_to_hash([self.grid])

    @property
    def projection_matrix_hash(self) -> str:
        """ Returns a hash of :attr:`projection_matrix`.
        """
        return list_to_hash([self.projection_matrix])

    def __str__(self) -> str:
        wdt = 74
        s = []
        s += ['-' * wdt]
        s += ['GaussianKernels'.center(wdt)]
        s += ['-' * wdt]
        with np.printoptions(threshold=4, edgeitems=2, precision=5, linewidth=60):
            s += ['{:18} : {}'.format('grid_scale', self.grid_scale)]
            s += ['{:18} : {}'.format('grid_hash', self.grid_hash)]
            s += ['{:18} : {}'.format('enforce_friedel_symmetry', self.enforce_friedel_symmetry)]
            s += ['{:18} : {}'.format('kernel_scale_parameter', self.kernel_scale_parameter)]
            s += ['{:18} : {}'.format('projection_matrix_hash', self.projection_matrix_hash)]
            s += ['{:18} : {}'.format('hash', hex(hash(self))[2:8])]
        s += ['-' * wdt]
        return '\n'.join(s)

    def _repr_html_(self) -> str:
        s = []
        s += [f'<h3>{self.__class__.__name__}</h3>']
        s += ['<table border="1" class="dataframe">']
        s += ['<thead><tr><th style="text-align: left;">Field</th><th>Size</th><th>Data</th></tr></thead>']
        s += ['<tbody>']
        with np.printoptions(threshold=4, edgeitems=2, precision=2, linewidth=40):
            s += ['<tr><td style="text-align: left;">grid_scale</td>']
            s += [f'<td>1</td><td>{self.grid_scale}</td></tr>']
            s += ['<tr><td style="text-align: left;">grid_hash</td>']
            s += [f'<td>{len(self.grid_hash)}</td><td>{self.grid_hash[:6]}</td></tr>']
            s += ['<tr><td style="text-align: left;">kernel_scale_parameter</td>']
            s += [f'<td>1</td><td>{self.kernel_scale_parameter}</td></tr>']
            s += ['<tr><td style="text-align: left;">enforce_friedel_symmetry</td>']
            s += [f'<td>1</td>'
                  f'<td>{self.enforce_friedel_symmetry}</td></tr>']
            s += ['<tr><td style="text-align: left;">projection_matrix</td>']
            s += [f'<td>{len(self.projection_matrix_hash)}</td>'
                  f'<td>{self.projection_matrix_hash[:6]}</td></tr>']
            s += ['<tr><td style="text-align: left;">hash</td>']
            s += [f'<td>{len(hex(hash(self)))}</td><td>{hex(hash(self))[2:8]}</td></tr>']
        s += ['</tbody>']
        s += ['</table>']
        return '\n'.join(s)

