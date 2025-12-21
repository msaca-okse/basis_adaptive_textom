import numpy as np
from numba import jit


"""
NB: Seems faster to simply let numba work out types and shapes than doing it upfront,
after the first call. There is no parallelization with `prange` because there was
little to no visible gains in the cases checked, probably since the
matrix multiplication should already be parallelized. This may need to be revisited
in the future.
"""


#@jit(nopython=True)
def _framewise_contraction(matrices: np.ndarray[float],
                           data: np.ndarray[float],
                           out: np.ndarray[float]) -> None:
    """This is the internal function that carries out calculations."""
    # NB: In principle could consider cache size but have not had much success with this.
    for i in range(len(matrices)):
        np.dot(data[i], matrices[i], out=out[i])


def framewise_contraction(matrices: np.ndarray[float],
                          data: np.ndarray[float],
                          out: np.ndarray[float]) -> None:
    """This function is a more efficient implementation of the
    series of matrix multiplications which is carried out the expression
    ``np.einsum('ijk, inmj -> inmk', matrices, data, out=out)``, using
    ``numba.jit``, for a four-dimensional :attr:`data` array.
    Used to linearly transform data from one representation to another.

    Parameters
    ----------
    matrices
        Stack of matrices with shape ``(i, j, k)`` where ``i`` is the
        stacking dimension.
    data
        Stack of data with shape ``(i, n, m, ..., j)`` where ``i`` is the
        stacking dimension and ``j`` is the representation dimension.
    out
        Output array with shape ``(i, n, m, ..., k)`` where ``i`` is the
        stacking dimension and ``k`` is the representation dimension.
        Will be modified in-place and must be contiguous.

    Notes
    -----
    All three arrays will be cast to ``out.dtype``.
    """
    # This is inefficient to do in the jitted function.
    dtype = out.dtype
    if matrices.dtype != dtype:
        matrices = matrices.astype(dtype)
    if data.dtype != dtype:
        data = data.astype(dtype)
    data = data.reshape(len(matrices), -1, data.shape[-1])
    out = out.reshape(len(matrices), -1, out.shape[-1])
    _framewise_contraction(matrices, data, out)


# @jit(nopython=True)
# def _framewise_contraction_transpose(matrices: np.ndarray[float],
#                                      data: np.ndarray[float],
#                                      out: np.ndarray[float]) -> None:
#     """Internal function for matrix contraction."""
#     for i in range(len(matrices)):
#         np.dot(data[i], matrices[i].T, out=out[i])


#@jit(nopython=True)
def _framewise_contraction_transpose_kronecker(matrix: np.ndarray[float],
                                     data: np.ndarray[float],
                                     out: np.ndarray[float]) -> None:
    """Internal function for matrix contraction."""
    """
    matrix: (M, K)
    data:   (N, K)
    out:    (N, M)
    Computes out[i] = data[i] @ matrix.T
    """
    N = data.shape[0]
    for i in range(N):
        for j in range(matrix.shape[0]):  # M
            s = 0.0
            for k in range(matrix.shape[1]):  # K
                s += data[i, k] * matrix[j, k]
            out[i, j] = s

#@jit(nopython=True)
def _framewise_contraction_np2(matrix: np.ndarray,
                                        data:   np.ndarray,
                                        out:    np.ndarray) -> None:
    """
    matrix: (M, K)
    data:   (K, N)
    out:    (M, N)  = matrix @ data
    """
    # Preconditions for zero-copy BLAS call:
    #  - dtypes identical (float32 or float64)
    #  - arrays contiguous (or at least not requiring implicit copies)
    np.dot(data, matrix, out=out)  # calls GEMM, no temporary result









@jit(nopython=True)
def _framewise_contraction(matrices: np.ndarray[float],
                                     data: np.ndarray[float],
                                     out: np.ndarray[float]) -> None:
    """Internal function for matrix contraction."""
    for i in range(len(matrices)):
        np.dot(data[i], matrices[i].T, out=out[i])


@jit(nopython=True)
def _framewise_contraction_transpose(matrices: np.ndarray[float],
                                     data: np.ndarray[float],
                                     out: np.ndarray[float]) -> None:
    """Internal function for matrix contraction."""
    for i in range(len(matrices)):
        np.dot(data[i], matrices[i], out=out[i])


def framewise_contraction_transpose_pool(matrices: np.ndarray[float],
                                    data: np.ndarray[float],
                                    out: np.ndarray[float]) -> np.ndarray[float]:
    """This function is a more efficient implementation of the
    series of matrix multiplications which is carried out the expression
    ``np.einsum('ijk, inmk -> inmj', matrices, data, out=out)``, using
    ``numba.jit``, for a four-dimensional :attr:`data` array.
    Used to linearly transform data from one representation to another.

    Parameters
    ----------
    matrices
        Stack of matrices with shape ``(i, j, k)`` where ``i`` is the
        stacking dimension.
    data
        Stack of data with shape ``(i, n, m, ..., k)`` where ``i`` is the
        stacking dimension and ``k`` is the representation dimension.
    out
        Output array with shape ``(i, n, m, ..., j)`` where ``i`` is the
        stacking dimension and ``j`` is the representation dimension.
        Will be modified in-place and must be contiguous.

    Notes
    -----
    All three arrays will be cast to ``out.dtype``.
    """
    dtype = out.dtype
    if matrices.dtype != dtype:
        matrices = matrices.astype(dtype)
    if data.dtype != dtype:
        data = data.astype(dtype)
    _framewise_contraction_transpose(matrices, data, out)



def framewise_contraction_pool(matrices: np.ndarray[float],
                                    data: np.ndarray[float],
                                    out: np.ndarray[float]) -> np.ndarray[float]:
    """This function is a more efficient implementation of the
    series of matrix multiplications which is carried out the expression
    ``np.einsum('ijk, inmk -> inmj', matrices, data, out=out)``, using
    ``numba.jit``, for a four-dimensional :attr:`data` array.
    Used to linearly transform data from one representation to another.

    Parameters
    ----------
    matrices
        Stack of matrices with shape ``(i, j, k)`` where ``i`` is the
        stacking dimension.
    data
        Stack of data with shape ``(i, n, m, ..., k)`` where ``i`` is the
        stacking dimension and ``k`` is the representation dimension.
    out
        Output array with shape ``(i, n, m, ..., j)`` where ``i`` is the
        stacking dimension and ``j`` is the representation dimension.
        Will be modified in-place and must be contiguous.

    Notes
    -----
    All three arrays will be cast to ``out.dtype``.
    """
    dtype = out.dtype
    if matrices.dtype != dtype:
        matrices = matrices.astype(dtype)
    if data.dtype != dtype:
        data = data.astype(dtype)
    _framewise_contraction(matrices, data, out)