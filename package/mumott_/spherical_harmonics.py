import logging
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import sph_harm
from pathos.multiprocessing import ProcessingPool as Pool


from mumott_.base_basis_set import BasisSet
from mumott_.probed_coordinates import ProbedCoordinates
from mumott_.hashing import list_to_hash
from mumott_.tensor_operations import (framewise_contraction_transpose_pool, framewise_contraction_pool)
from multiprocessing import Pool


logger = logging.getLogger(__name__)


class SphericalHarmonics(BasisSet):
    """ Basis set class for spherical harmonics, the canonical representation
    of polynomials on the unit sphere and a simple way of representing
    band-limited spherical functions which allows for easy computations of statistics
    and is suitable for analyzing certain symmetries.

    Parameters
    ----------
    ell_max : int
        The bandlimit of the spherical functions that you want to be able to represent.
        A good rule of thumb is that :attr:`ell_max` should not exceed the
        number of detector segments minus 1.
    probed_coordinates : ProbedCoordinates
        Optional. A container with the coordinates on the sphere probed at each detector segment by the
        experimental method. Its construction from the system geometry is method-dependent.
        By default, an empty instance of
        :class:`ProbedCoordinates <mumott.core.probed_coordinates.ProbedCoordinates>` is created.
    enforce_friedel_symmetry : bool
        If set to ``True``, Friedel symmetry will be enforced, using the assumption that points
        on opposite sides of the sphere are equivalent. This results in only even ``ell`` being used.
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
    """
    def __init__(self,
                 probed_coordinates: ProbedCoordinates = None,
                 ell_max: int = 0,
                 enforce_friedel_symmetry: bool = False,
                 N_splits: int = 1,
                 **kwargs):
        super().__init__(probed_coordinates, **kwargs)
        self._probed_coordinates_hash = hash(self.probed_coordinates)
        self._ell_max = ell_max
        self._ell_indices = np.zeros(1)
        self._emm_indices = np.zeros(1)
        # Compute initial values for indices and matrix.
        self._enforce_friedel_symmetry = enforce_friedel_symmetry
        self._calculate_coefficient_indices()
        #self._projection_matrix = np.ascontiguousarray(self._get_integrated_projection_matrix().transpose([0,2,1]))
        self._projection_matrix = np.ascontiguousarray(self._get_integrated_projection_matrix_fast(N_splits = N_splits).transpose([0,2,1]))



    def _get_integrated_projection_matrix_fast(self, probed_coordinates=None, N_splits=1):
        if probed_coordinates is None:
            probed_coordinates = self.probed_coordinates

        M_rot = probed_coordinates.vector.shape[0]
        n_splits = min(N_splits, M_rot)

        # Split and build ProbedCoordinates objects
        probed_split = np.array_split(probed_coordinates.vector, n_splits, axis=0)
        probed_list = [ProbedCoordinates(vector=vec) for vec in probed_split]
        print('Instantiating basis change matrix with #processes = ', n_splits)
        # Use pathos Pool, but call the *top-level* wrapper
        with Pool(nodes=n_splits) as pool:
            results = pool.map(
                _call_get_integrated_projection_matrix,
                [(self, coords) for coords in probed_list]
            )

        out = np.concatenate(results, axis=0)
        return out



    def _calculate_coefficient_indices(self) -> None:
        """
        Computes the attributes :attr:`~.SphericalHarmonics.ell_indices` and
        :attr:`~.SphericalHarmonics.emm_indices`. Called when :attr:`~.SphericalHarmonics.ell_max`
        changes.
        """
        if self._enforce_friedel_symmetry:
            divisor = 2
        else:
            divisor = 1

        mm = np.zeros((self._ell_max + 1) * (self._ell_max // divisor + 1), dtype=int)
        ll = np.zeros((self._ell_max + 1) * (self._ell_max // divisor + 1), dtype=int)
        count = 0
        for h in range(0, self._ell_max + 1, divisor):
            for i in range(-h, h + 1):
                ll[count] = h
                mm[count] = i
                count += 1
        self._ell_indices = ll
        self._emm_indices = mm

    def _get_projection_matrix(self, probed_coordinates: ProbedCoordinates = None) -> None:
        """ Computes the matrix necessary for forward and gradient calculations.
        Called when the coordinate system has been updated or ``ell_max`` has changed."""
        if probed_coordinates is None:
            probed_coordinates = self._probed_coordinates
        _, probed_polar_angles, probed_azim_angles = probed_coordinates.to_spherical
        # retrieve complex spherical harmonics with emm >= 0, shape (N, M, len(ell_indices), K)
        complex_factors = sph_harm(abs(self._emm_indices)[np.newaxis, np.newaxis, np.newaxis, ...],
                                   self._ell_indices[np.newaxis, np.newaxis, np.newaxis, ...],
                                   probed_azim_angles[..., np.newaxis],
                                   probed_polar_angles[..., np.newaxis])
        # cancel Condon-Shortley phase factor in scipy.special.sph_harm
        condon_shortley_factor = (-1.) ** self._emm_indices
        # 4pi normalization factor and complex-to-real normalization factor for m != 0
        norm_factor = np.sqrt(4 * np.pi) * \
            np.sqrt(1 + (self._emm_indices != 0).astype(int)) * condon_shortley_factor
        matrix = norm_factor[np.newaxis, np.newaxis, np.newaxis, ...] * (
            (self._emm_indices >= 0)[np.newaxis, np.newaxis, np.newaxis, ...] * complex_factors.real +
            (self._emm_indices < 0)[np.newaxis, np.newaxis, np.newaxis, ...] * complex_factors.imag)
        return matrix
    


    def forward(self,
                coefficients: NDArray,
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
        assert coefficients.shape[-1] == self._ell_indices.size
        self._update()
        output = np.zeros(coefficients.shape[:-1] + (self._projection_matrix.shape[2],),
                          coefficients.dtype)

        if indices is None:
            NotImplementedError
            
        else:
            framewise_contraction_transpose_pool(self._projection_matrix[indices],
                                coefficients,
                                output)
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
            framewise_contraction_pool(self._projection_matrix[indices],
                                data,
                                output)
            return output

    def gradient(self,
                 coefficients: NDArray,
                 indices: NDArray = None) -> NDArray:
        """ Carries out a gradient computation of projections from spherical harmonic space
        into detector space, for one or several tomographic projections.

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
            An array of gradient values based on the `coefficients` given.
            If :attr:`indices` contains exactly one index, the shape is ``(coefficients.shape[:-1], J)``
            where ``J`` is the number of detector segments. If indices is ``None`` or contains
            several indices, the shape is ``(N, coefficients.shape[1:-1], J)`` where ``N``
            is the number of tomographic projections for which the computation is performed.

        Notes
        -----
        When solving an inverse problem, one should not to attempt to optimize the
        coefficients directly using the ``gradient`` one obtains by applying this method to the data.
        Instead, one must either take the gradient of the residual between the
        :meth:`~.SphericalHarmonics.forward` computation of the coefficients and the data.
        Alternatively one can apply both the forward and the gradient computation to the
        coefficients to be optimized, and the gradient computation to the data, and treat
        the residual of the two as the gradient of the optimization coefficients. The approaches
        are algebraically equivalent, but one may be more efficient than the other in some
        circumstances.
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

    def get_inner_product(self,
                          u: NDArray,
                          v: NDArray,
                          resolve_spectrum: bool = False,
                          spectral_moments: List[int] = None) -> NDArray:
        r""" Retrieves the inner product of two coefficient arrays.

        Notes
        -----
            The canonical inner product in a spherical harmonic representation
            is :math:`\sum_\ell N(\ell) \sum_m u_m^\ell v_m^\ell`, where :math:`N(\ell)` is
            a normalization constant (which is unity for the :math:`4\pi` normalization).
            This inner product is a rotational invariant. The rotational invariance also holds
            for any partial sums over :math:`\ell`.
            One can define a function of :math:`\ell` that returns such
            products, namely :math:`S(\ell, u, v) = N(\ell)\sum_m u_m^\ell v_m^\ell`,
            called the spectral power function.
            The sum :math:`\sum_{\ell = 1}S(\ell)` is equal to the covariance of the
            band-limited spherical functions represented by :math:`u` and :math:`v`, and each
            :math:`S(\ell, u, v)` is the contribution to the covariance of the band :math:`\ell`.
            See also
            `the SHTOOLS documentation <https://shtools.github.io/SHTOOLS/real-spherical-harmonics.html>`_
            for an excellent overview of this.

        Parameters
        ----------
        u
            The first coefficient array, of arbitrary shape and dimension,
            except the last dimension must be the same as the length of
            :attr:`~.SphericalHarmonics.ell_indices`.
        v
            The second coefficient array, of the same shape as ``u``.
        resolve_spectrum
            Optional. Whether to resolve the product according to each frequency band, given by the
            coefficients of each ``ell`` in :attr:`~SphericalHarmonics.ell_indices`.
            Defaults to ``False``, which means that the sum of every component of the spectrum is returned.
            If ``True``, components are returned in order of ascending ``ell``.
            The ``ell`` included in the spectrum depends on :attr:`spectral_moments`.
        spectral_moments
            Optional. List of particular values of ``ell`` to calculate the inner product for.
            Defaults to ``None``, which is identical to including all values of ``ell``
            in the calculation.
            If :attr:`spectral_moments` contains all nonzero values of ``ell`` and
            :attr:`resolve_spectrum` is ``False``, the covariance of
            :attr:`v` and :attr:`u` will be calculated (the sum of the inner product over all non-zero ``ell``
            If ``resolve_spectrum`` is ``True``, the covariance per `ell` in ``spectral_moments``, will
            be calculated, i.e., the inner products will not be summed over.

        Returns
        -------
            An array of the inner products of the spherical functions represented by ``u`` and ``v``.
            Has the shape ``(u.shape[:-1])`` if :attr:`resolve_spectrum` is ``False``,
            ``(u.shape[:-1] + (ell_max // 2 + 1,))`` if :attr:`resolve_spectrum` is ``True`` and
            ``spectral_moments`` is ``None``, and finally the shape
            ``(u.shape[:-1] + (np.unique(spectral_moments).size,))`` if :attr:`resolve_spectrum` is ``True``
            and :attr:`spectral_moments` is a list of integers found
            in :attr:`~.SphericalHarmonics.ell_indices`
         """
        assert u.shape == v.shape
        assert u.shape[-1] == self._ell_indices.size
        if not resolve_spectrum:
            if spectral_moments is None:
                return np.einsum('...i, ...i -> ...', u, v)
            # pick out only the subset where ell matches the provided spectral moments
            where = np.any([np.equal(self._ell_indices, ell) for ell in spectral_moments], axis=0)
            return np.einsum('...i, ...i -> ...', u[..., where], v[..., where])
        if spectral_moments is None:
            which_ell = np.unique(self._ell_indices)
        else:
            which_ell = np.unique(spectral_moments)
            which_ell = [ell for ell in which_ell if np.any(np.equals(self._ell_indices, ell))]
        power_spectrum = np.zeros((*u.shape[:-1], which_ell.size))
        # power spectrum for any one ell is given by inner product over each ell
        for i, ell in enumerate(which_ell):
            power_spectrum[..., i] = np.einsum('...i, ...i -> ...',
                                               u[..., self._ell_indices == ell],
                                               v[..., self._ell_indices == ell],
                                               optimize='greedy')
        return power_spectrum

    def get_covariances(self,
                        u: NDArray,
                        v: NDArray,
                        resolve_spectrum: bool = False) -> NDArray:
        """ Returns the covariances of the spherical functions represented by
        two coefficient arrays.

        Parameters
        ----------
        u
            The first coefficient array, of arbitrary shape
            except its last dimension must be the same length as
            the length of :attr:`~SphericalHarmonics.ell_indices`.
        v
            The second coefficient array, of the same shape as :attr:`u`.
        resolve_spectrum
            Optional. Whether to resolve the product according to each frequency band, given by the
            coefficients of each ``ell`` in :attr:`~.SphericalHarmonics.ell_indices`.
            Default value is ``False``.

        Returns
        -------
            An array of the covariances of the spherical functions represented by ``u`` and ``v``.
            Has the shape ``(u.shape[:-1])`` if `resolve_spectrum` is ``False``, and
            ``(u.shape[:-1] + (ell_max // 2 + 1,))`` if `resolve_spectrum` is ``True``, where
            ``ell_max`` is :attr:`.SphericalHarmonics.ell_max`.

        Notes
        -----
        Calling this function is equivalent to calling :func:`~.SphericalHarmonics.get_inner_product`
        with ``spectral_moments=np.unique(ell_indices[ell_indices > 0])`` where ``ell_indices`` is
        :attr:`.SphericalHarmonics.ell_indices`. See the note to
        :func:`~.SphericalHarmonics.get_inner_product` for mathematical details.
        """
        spectral_moments = np.unique(self._ell_indices[self._ell_indices > 0])
        return self.get_inner_products(u, v, resolve_spectrum, spectral_moments)

    def get_output(self,
                   coefficients: NDArray) -> Dict[str, Any]:
        r""" Returns a dictionary of output data for a given array of spherical harmonic coefficients.

        Parameters
        ----------
        coefficients
            An array of coefficients of arbitrary shape and dimensions, except
            its last dimension must be the same length as :attr:`~.SphericalHarmonics.ell_indices`.
            Computations only operate over the last axis of :attr:`coefficients`, so derived
            properties in the output will have the shape ``(*coefficients.shape[:-1], ...)``.

        Returns
        -------
            A dictionary containing two sub-dictionaries, ``basis_set`` and ``spherical_functions``.
            ``basis_set`` contains information particular to :class:`SphericalHarmonics`, whereas
            ``spherical_functions`` contains information about the spherical functions
            represented by the :attr:`coefficients` which are not specific to the chosen representation.

        Notes
        -----
        In detail, the two sub-dictionaries ``basis_set`` and ``spherical_functions`` have the following
        members:

        basis_set
            name
                The name of the basis set, i.e., ``'SphericalHarmonicParameters'``
            coefficients
                A copy of :attr:`coefficients`.
            ell_max
                A copy of :attr:`~.SphericalHarmonics.ell_max`.
            ell_indices
                A copy of :attr:`~.SphericalHarmonics.ell_indices`.
            emm_indices
                A copy of :attr:`~.SphericalHarmonics.emm_indices`.
            projection_matrix
                A copy of :attr:`~.SphericalHarmonics.projection_matrix`.
        spherical_functions
            means
                The spherical means of each function represented by :attr:`coefficients`.
            variances
                The spherical variances of each function represented by :attr:`coefficients`.
                If :attr:`~.ell_max` is ``0``, all variances will equal zero.
            r2_tensors
                The traceless symmetric rank-2 tensor component of each function represented by
                :attr:`coefficients`, in 6-element form, in the order ``[xx, yy, zz, yz, xz, xy]``,
                i.e., by the Voigt convention.
                The matrix form can be recovered as r2_tensors[..., tensor_to_matrix_indices],
                yielding matrix elements ``[[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]``.
                If :attr:`~.ell_max` is ``0``, all tensors have elements
                [1, 0, -1, 0, 0, 0].
            tensor_to_matrix_indices
                A list of indices to help recover the matrix from the 2-element form of the
                rank-2 tensors, equalling precisely ``[[0, 5, 4], [5, 1, 3], [4, 3, 2]]``
            eigenvalues
                The eigenvalues of the rank-2 tensors, sorted in ascending order in the last index.
                If :attr:`~.ell_max` is ``0``, the eigenvalues will always be (1, 0, -1)
            eigenvectors
                The eigenvectors of the rank-2 tensors, sorted with their corresponding eigenvectors
                in the last index. Thus, ``eigenvectors[..., 0]`` gives the eigenvector corresponding
                to the smallest eigenvalue, and ``eigenvectors[..., 2]`` gives the eigenvector
                corresponding to the largest eigenvalue. Generally, one of these two eigenvectors
                is used to define the orientation of a function, depending on whether it is
                characterized by a minimum (``0``) or a maximum (``2``). The middle eigenvector (``1``)
                is typically only used for visualizations.
                If :attr:`~.ell_max` is ``0``, the eigenvectors will be the Cartesian basis
                vectors.
            main_orientations
                The estimated main orientations from the largest absolute eigenvalues.
                If :attr:`~.ell_max` is ``0``, the main orientation will be the x-axis.
            main_orientation_symmetries
                The strength or definiteness of the main orientation, calculated from
                the quotient of the absolute middle and signed largest eigenvalues of the
                rank-2 tensor.
                If ``0``, the orientation is totally ambiguous. The orientation is completely
                transversal if the value is ``-1`` (orientation represents a minimum),
                and completely longitudinal if the value is ``1`` (orientation represents a maximum).
                If :attr:`~.ell_max` is ``0``, the main orientations are all totally ambiguous.
            normalized_standard_deviations
                A relative measure of the overall anisotropy of the spherical functions. Equals
                :math:`\sqrt{\sigma^2 / \mu}`, where :math:`\sigma^2` is the variance
                and :math:`\mu` is the mean. The places where :math:`\mu=0` have been
                set to ``0``. If :attr:`~.ell_max` is ``0``, the normalized standard
                deviations will always be zero.
            power_spectra
                The spectral powers of each ``ell`` in :attr:`~.SphericalHarmonics.ell_indices`,
                for each spherical function, sorted in ascending ``ell``.
                If :attr:`~.ell_max` is ``0``, each function will have only one element, equal
                to the mean squared.
            power_spectra_ell
                An array containing the corresponding ``ell`` to each of the last indices
                in :attr:`power_spectra`. Equal to ``np.unique(ell_indices)``.
        """
        assert coefficients.shape[-1] == self._ell_indices.size
        # Update to ensure non-dirty output state.
        self._update()
        output_dictionary = {}

        # basis set-specific information
        basis_set = {}
        output_dictionary['basis_set'] = basis_set
        basis_set['name'] = type(self).__name__
        basis_set['coefficients'] = coefficients.copy()
        basis_set['ell_max'] = self._ell_max
        basis_set['ell_indices'] = self._ell_indices.copy()
        basis_set['emm_indices'] = self._emm_indices.copy()
        basis_set['projection_matrix'] = self._projection_matrix.copy()
        basis_set['hash'] = hex(hash(self))

        # general properties of spherical function
        spherical_functions = {}
        output_dictionary['spherical_functions'] = spherical_functions
        spherical_functions['means'] = coefficients[..., 0]
        if coefficients.shape[-1] > 1:
            spherical_functions['variances'] = (coefficients[..., 1:] ** 2).sum(-1)
        else:
            spherical_functions['variances'] = np.zeros_like(coefficients[..., 0])
        r2_tensor, eigvect, eigval = self._get_rank2_tensor_analysis(coefficients)
        spherical_functions['eigenvectors'] = eigvect
        spherical_functions['r2_tensors'] = np.concatenate(
             (np.diagonal(r2_tensor, offset=0, axis1=-2, axis2=-1),
              r2_tensor[..., 1, [2]], r2_tensor[..., 0, [2]], r2_tensor[..., 0, [1]]), axis=-1)
        spherical_functions['tensor_to_matrix_indices'] = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
        spherical_functions['eigenvalues'] = eigval

        # estimate main orientation from absolute eigenvalues and select from eigenvectors
        spherical_functions['main_orientations'] = \
            np.take_along_axis(eigvect,
                               np.argmax(abs(eigval), axis=-1).reshape(eigval.shape[:-1] + (1, 1)), axis=-1)
        spherical_functions['main_orientations'] = spherical_functions['main_orientations'][..., 0]
        # estimate main orientation strength as quotient between middle and largest absolute eigenvalue.
        eigargs = np.argmax(abs(eigval), axis=-1).reshape(eigval.shape[:-1] + (1,))
        spherical_functions['main_orientation_symmetries'] = 2 * abs(eigval[..., 1][..., None]) / \
            np.take_along_axis(eigval, eigargs, axis=-1)

        # estimate overall relative anisotropy as quotient betewen standard deviation and mean
        valid_indices = spherical_functions['means'] > 0  # mask points where mean is zero or negative
        normalized_std = np.zeros_like(spherical_functions['means'])
        normalized_std[valid_indices] = np.sqrt(spherical_functions['variances'][valid_indices]) / \
            spherical_functions['means'][valid_indices]
        spherical_functions['normalized_standard_deviations'] = normalized_std
        spherical_functions['power_spectra'] = self.get_inner_product(coefficients,
                                                                      coefficients,
                                                                      resolve_spectrum=True)
        spherical_functions['power_spectra_ell'] = np.unique(self._ell_indices)
        return output_dictionary

    def get_spherical_harmonic_coefficients(
        self,
        coefficients: NDArray[float],
        ell_max: int = None
    ) -> NDArray[float]:
        """ Convert a set of spherical harmonics coefficients to a different :attr:`ell_max`
        by either zero-padding or truncation and return the result.

        Parameters
        ----------
        coefficients
            An array of coefficients of arbitrary shape, provided that the
            last dimension contains the coefficients for one function.
        ell_max
            The band limit of the spherical harmonic expansion.
        """

        if coefficients.shape[-1] != len(self):
            raise ValueError(f'The number of coefficients ({coefficients.shape[-1]}) does not '
                             f'match the expected value. ({len(self)})')

        if self._enforce_friedel_symmetry:
            num_coeff_output = (ell_max+1) * (ell_max+2) // 2
        elif not self._enforce_friedel_symmetry:
            num_coeff_output = (ell_max+1)**2

        output_array = np.zeros((*coefficients.shape[:-1], num_coeff_output))
        output_array[..., :min(len(self), num_coeff_output)] = \
            coefficients[..., :min(len(self), num_coeff_output)]
        return output_array

    def _get_rank2_tensor_analysis(self, coefficients: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """ Performs an analysis of the rank-2 tensor components of the functions represented by the
        given coefficients.

        Parameters
        ----------
        coefficients
            An array of coefficients of arbitrary shape, as long as the last axis has the same shape
            as :attr:`~.SphericalHarmonics.ell_indices`.

        Returns
        -------
            A tuple with three memberrs.
            First, the traceless rank-2 tensor components of all the
            functions represented by the :attr:`coefficients`, in 3-by-3 matrix form stored
            in the last two indices: ``[[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]``
            Second, eigenvalues of the rank-2 tensors, sorted in ascending order.
            Third, eigenvectors associated to each eigenvalue, sorted in the same order in the last index.
            Thus, ``eigenvectors[..., 0]`` returns the eigenvectors corresponding to the smallest eigenvalue.

        Notes
        -----
        This method handles the edge case where :attr:`~.SphericalHarmonics.ell_max` is ``0`` gracefully.
        In this case, all elements of :attr:`r2_tensors` will be ``[[1, 0, 0], [0, 0, 0], [0, 0, -1]]``,
        all elements of :attr:`eigenvalues` will be ``[-1, 0, 1]``, and all elements of
        :attr:`eigenvectors` will ``[[1, 0, 0], [0, 1, 0], [0, 0, 1]]``.

        """
        if self._ell_max < 2:
            logger.info('Note: ell_max < 2, so rank-2 tensors will all be diagonal matrices with diagonal'
                        ' [1, 0, -1], eigenvalues will be [1, 0, -1],'
                        ' and eigenvectors will be Cartesian basis vectors.')
            eigenvalues = np.zeros(coefficients.shape[:-1] + (3,), dtype=coefficients.dtype)
            eigenvectors = np.zeros(coefficients.shape[:-1] + (3, 3), dtype=coefficients.dtype)
            eigenvalues[...] = [-1, 0, 1]
            eigenvectors[..., 0, 0] = 1.
            eigenvectors[..., 1, 1] = 1.
            eigenvectors[..., 2, 2] = 1.
            r2_tensor = np.zeros(coefficients.shape[:-1] + (3, 3), dtype=coefficients.dtype)
            r2_tensor[..., 0, 0] = 1
            r2_tensor[..., 1, 1] = 0
            r2_tensor[..., 2, 2] = -1
            return r2_tensor, eigenvectors, eigenvalues
        A = coefficients[..., self._ell_indices == 2]

        # Normalizing coefficients
        c1 = np.sqrt(15)
        c2 = np.sqrt(5)
        c3 = np.sqrt(15)
        r2_tensor = np.zeros(coefficients.shape[:-1] + (3, 3), dtype=coefficients.dtype)
        r2_tensor[..., 0, 0] = c3 * A[..., 4] - c2 * A[..., 2]
        r2_tensor[..., 1, 1] = -c3 * A[..., 4] - c2 * A[..., 2]
        r2_tensor[..., 2, 2] = c2 * 2 * A[..., 2]
        r2_tensor[..., 0, 1] = c1 * A[..., 0]
        r2_tensor[..., 1, 0] = c1 * A[..., 0]
        r2_tensor[..., 2, 1] = c1 * A[..., 1]
        r2_tensor[..., 1, 2] = c1 * A[..., 1]
        r2_tensor[..., 2, 0] = c1 * A[..., 3]
        r2_tensor[..., 0, 2] = c1 * A[..., 3]
        w, v = np.linalg.eigh(r2_tensor.reshape(-1, 3, 3))

        # Some complicated sorting logic to sort eigenvectors per ascending eigenvalues.
        sorting = np.argsort(w, axis=1).reshape(-1, 3, 1)
        v = v.transpose(0, 2, 1)
        v = np.take_along_axis(v, sorting, axis=1)
        v = v.transpose(0, 2, 1)
        v = v / np.sqrt(np.sum(v ** 2, axis=1).reshape(-1, 1, 3))
        eigenvalues = w.reshape(coefficients.shape[:-1] + (3,))
        eigenvectors = v.reshape(coefficients.shape[:-1] + (3, 3,))
        return r2_tensor.reshape(*coefficients.shape[:-1], 3, 3), eigenvectors, eigenvalues

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """ Allows class to be iterated over and in particular be cast as a dictionary.
        """
        yield 'name', type(self).__name__
        yield 'ell_max', self._ell_max
        yield 'ell_indices', self._ell_indices
        yield 'emm_indices', self._emm_indices
        yield 'projection_matrix', self._projection_matrix
        yield 'hash', hex(hash(self))[2:]

    def __len__(self) -> int:
        return len(self._ell_indices)

    def __hash__(self) -> int:
        """Returns a hash reflecting the internal state of the instance.

        Returns
        -------
            A hash of the internal state of the instance,
            cast as an ``int``.
        """
        to_hash = [self._ell_max,
                   self._ell_indices,
                   self._emm_indices,
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
        If ``v`` is a vector of spherical harmonic coefficients, and ``M`` is the ``projection_matrix``,
        then ``M @ v`` gives the corresponding values on the detector segments associated with
        each projection. ``M[i] @ v`` gives the values on the detector segments associated with
        projection ``i``.

        If ``r`` is a residual between a projection from spherical to detector space and data from
        projection ``i``, then ``M[i].T @ r`` gives the associated gradient in spherical harmonic
        space.
        """
        self._update()
        return self._projection_matrix

    @property
    def ell_max(self) -> int:
        r""" The maximum ``ell`` used to represent spherical functions.

        Notes
        -----
        The word ``ell`` is used to represent the cursive small L, also written
        :math:`\ell`, often used as an index for the degree of the Legendre polynomial
        in the definition of the spherical harmonics.
        """
        return self._ell_max

    @ell_max.setter
    def ell_max(self, val: int) -> NDArray[int]:
        if (val % 2 != 0 and self._enforce_friedel_symmetry) or val < 0 or val != round(val):
            raise ValueError('ell_max must be an even (if Friedel symmetry is enforced),'
                             ' non-negative integer,'
                             f' but a value of {val} was given!')
        self._ell_max = val
        self._calculate_coefficient_indices()
        self._projection_matrix = self._get_integrated_projection_matrix()

    @property
    def ell_indices(self) -> NDArray[int]:
        r""" The ``ell`` associated with each coefficient and its corresponding
        spherical harmonic. Updated when :attr:`~.SphericalHarmonics.ell_max` changes.

        Notes
        -----
        The word ``ell`` is used to represent the cursive small L, also written
        :math:`\ell`, often used as an index for the degree of the Legendre polynomial
        in the definition of the spherical harmonics.
        """
        return self._ell_indices

    @property
    def emm_indices(self) -> NDArray[int]:
        r""" The ``emm`` associated with each coefficient and its corresponding
        spherical harmonic. Updated when :attr:`~.SphericalHarmonics.ell_max` changes.

        Notes
        -----
        For consistency with :attr:`~.SphericalHarmonics.ell_indices`, and to avoid
        visual confusion with other letters, ``emm`` is
        used to represent the index commonly written :math:`m` in mathematical notation,
        the frequency of the sine-cosine parts of the spherical harmonics,
        often called the spherical harmonic order.
        """
        return self._emm_indices

    def __str__(self) -> str:
        wdt = 74
        s = []
        s += ['-' * wdt]
        s += ['SphericalHarmonics'.center(wdt)]
        s += ['-' * wdt]
        with np.printoptions(threshold=4, edgeitems=2, precision=5, linewidth=60):
            s += ['{:18} : {}'.format('Maximum "ell"', self.ell_max)]
            s += ['{:18} : {}'.format('"ell" indices', self.ell_indices)]
            s += ['{:18} : {}'.format('"emm" indices', self.emm_indices)]
            s += ['{:18} : {}'.format('Projection matrix', self.projection_matrix)]
            s += ['{:18} : {}'.format('Hash', hex(hash(self))[2:8])]
        s += ['-' * wdt]
        return '\n'.join(s)

    def _repr_html_(self) -> str:
        s = []
        s += ['<h3>SphericalHarmonics</h3>']
        s += ['<table border="1" class="dataframe">']
        s += ['<thead><tr><th style="text-align: left;">Field</th><th>Size</th><th>Data</th></tr></thead>']
        s += ['<tbody>']
        with np.printoptions(threshold=4, edgeitems=2, precision=2, linewidth=40):
            s += ['<tr><td style="text-align: left;">Maximum "ell"</td>']
            s += [f'<td>{1}</td><td>{self.ell_max}</td></tr>']
            s += ['<tr><td style="text-align: left;">"ell" indices</td>']
            s += [f'<td>{len(self.ell_indices)}</td><td>{self.ell_indices}</td></tr>']
            s += ['<tr><td style="text-align: left;">"emm" indices</td>']
            s += [f'<td>{len(self.emm_indices)}</td><td>{self.emm_indices}</td></tr>']
            s += ['<tr><td style="text-align: left;">Coefficient projection matrix</td>']
            s += [f'<td>{self.projection_matrix.shape}</td><td>{self.projection_matrix}</td></tr>']
            s += ['<tr><td style="text-align: left;">Hash</td>']
            s += [f'<td>{len(hex(hash(self)))}</td><td>{hex(hash(self))[2:8]}</td></tr>']
        s += ['</tbody>']
        s += ['</table>']
        return '\n'.join(s)
    

def _call_get_integrated_projection_matrix(args):
    self_obj, coords = args
    return self_obj._get_integrated_projection_matrix(probed_coordinates=coords)
