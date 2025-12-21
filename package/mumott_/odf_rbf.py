# odf_sh_image_operator.py
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray
import numpy as np
from numba import jit

from cil.framework import ImageGeometry, ImageData, BlockDataContainer, AcquisitionGeometry, AcquisitionData
from cil.optimisation.operators import LinearOperator

from mumott_.spherical_harmonics import SphericalHarmonics
from mumott_.gaussian_kernels import GaussianKernels
from mumott_.odf_geometry import ODFGeometry
from pathos.multiprocessing import ProcessingPool as Pool
try:
    import cupy as cp
except ImportError:
    cp = None





class ODFGK(LinearOperator):
    """
    B: coefficients -> ODF-detector values

    Domain  (ig_in):  ImageGeometry with shape (Nx, Ny, K)
                      z=K (SH coeffs)
    Range   (ig_out): ImageGeometry with shape (Nx, Ny, M)
                      z=M (detector segments)

    Notes
    -----
    - Only `direct()` is implemented (and returns ImageData).
    - `adjoint()` is left as a skeleton (raises NotImplementedError).
    - The operator treats the z-dimension as if it were the channel axis.
    """

    def __init__(
        self,
        B_matrix: NDArray[np.float32],
        ag_in: AcquisitionGeometry,
        ag_out: AcquisitionGeometry,
        dtype=np.float32,
        device: str = 'cpu',
        projections_per_proc: int = 10,
        N_processes: int = 32,
        **sh_kwargs
    ):
        self.ag_in   = ag_in
        self.ag_out  = ag_out
        self.dtype   = dtype
        self.projections_per_proc = projections_per_proc
        self.device = device
        self.N_processes = N_processes


        # Shapes from SH projection matrix
        N_rot, K, N_chi, N_theta = map(int, B_matrix.shape)
        N_seg = N_chi*N_theta
        self.B_matrix = B_matrix.reshape(N_rot, K, N_seg)
        self.N_rot, self.N_seg, self.K = N_rot, N_seg, K


        # put on GPU if requested
        if self.device == 'gpu':
            if cp is None:
                raise ImportError("CuPy is required for GPU mode.")
            self.B_matrix_gpu = cp.asarray(self.B_matrix, dtype=self.dtype)
        else:
            self.B_matrix_gpu = None
        
        # Initialise LinearOperator with geometries
        super().__init__(domain_geometry=self.ag_in, range_geometry=self.ag_out)

    # ------------ LinearOperator API ------------

    def direct(self, x: ImageData) -> ImageData:
        """
        x: AcquisitionData on ag_in with array shape (K, N_rot, Mx)
        returns AcquisitionData on ag_out with array shape (N_rot, Mx, M)
        """
        if isinstance(x, AcquisitionData):
            xin = x.as_array()
        else:
            return f"Unknown type: {type(x)}"


        if xin.dtype != self.dtype:
            xin = xin.astype(self.dtype, copy=False)

        if tuple(xin.shape) != tuple(self.ag_in.shape):
            raise ValueError(f"Input shape {xin.shape} != ag_in.shape {self.ag_in.shape}")

        xin_t = np.ascontiguousarray(xin.transpose([1,2,0]))  # from (K, N_rot, Mx) to (N_rot, Mx, K)  

        #: (N_rot, Mx, K) -> (N_rot, Mx, N_seg)

        if self.device == 'cpu':
            yin_t = self.forward_cpu(xin_t)
        elif self.device == 'gpu':
            yin_t = self.forward_gpu(xin_t)
        else:
            raise ValueError(f"Unknown device: {self.device}")


        if tuple(yin_t.shape) != tuple(self.ag_out.shape):
            raise RuntimeError(
                f"Computed output shape {yin_t.shape} doesn't match ag_out.shape {self.ag_out.shape}."
            )


        return AcquisitionData(yin_t, geometry=self.ag_out)


        

    def adjoint(self, y: ImageData):
        """
        y: AcquisitionData on ag_out with array shape (M_rot, Mx, M)
        returns AcquisitionData on ag_in with array shape (K, M_rot, Mx)
        """
        if isinstance(y, AcquisitionData):
            yin = y.as_array()
        else:
            return f"Unknown type: {type(y)}"
        
        if yin.dtype != self.dtype:
            yin = yin.astype(self.dtype, copy=False)

        if tuple(yin.shape) != tuple(self.ag_out.shape):
            raise ValueError(f"Input shape {yin.shape} != ag_out.shape {self.ag_out.shape}")
        
        if self.device == 'cpu':
            xin = self.adjoint_cpu(yin)
        elif self.device == 'gpu':
            xin = self.adjoint_gpu(yin)
        else:
            raise ValueError(f"Unknown device: {self.device}")


        xin_t = np.ascontiguousarray(xin.transpose([2,0,1])).astype(np.float32)  # from (K, M_rot, Mx) to (M_rot, Mx, K)
        if tuple(xin_t.shape) != tuple(self.ag_in.shape):
            raise RuntimeError(
                f"Computed output shape {xin_t.shape} doesn't match ag_in.shape {self.ag_in.shape}."
            )
        return AcquisitionData(xin_t, geometry=self.ag_in)


    


    def _validate_geometries(self):
        # ig_in: (Nx, Ny, K)
        if self.ig_in.shape[0] != self.K:
            raise ValueError(f"ig_in.shape[-1]={self.ig_in.shape[-1]} must equal K={self.K}")

        # ig_out: (Nx, Ny, M)
        if self.ig_out.shape[0] != self.M:
            raise ValueError(f"ig_out.shape[-1]={self.ig_out.shape[-1]} must equal M={self.M}")

        if self.ig_in.shape[1:] != self.ig_out.shape[1:]:
            raise ValueError(f"Spatial (x,y) mismatch: ig_in {self.ig_in.shape[1:]} vs ig_out {self.ig_out.shape[1:]}")
        

        
    def forward_cpu(self, coefficients):
        indices = np.arange(self.N_rot)
        indices_split = np.array_split(indices, self.N_rot//self.projections_per_proc)
        coeffs_split = np.array_split(coefficients, self.N_rot//self.projections_per_proc, axis=0)


        tasks = [(self.B_matrix[idx], coeffs) for idx, coeffs in zip(indices_split, coeffs_split)]
        with Pool(nodes=self.N_processes) as pool:
            results = pool.map(worker_forward_cpu, tasks)

        output = np.concatenate(results, axis=0)
        return output




        
    def adjoint_cpu(self, data):
        indices = np.arange(self.N_rot)
        indices_split = np.array_split(indices, self.N_rot//self.projections_per_proc)
        data_split = np.array_split(data, self.N_rot//self.projections_per_proc, axis=0)


        tasks = [(self.B_matrix[idx], data_) for idx, data_ in zip(indices_split, data_split)]
        with Pool(nodes=self.N_processes) as pool:
            results = pool.map(worker_adjoint_cpu, tasks)

        output = np.concatenate(results, axis=0)
        return output
    

    def forward_gpu(self, coefficients: np.ndarray) -> np.ndarray:
        """
        GPU version of forward.
        coefficients: (N_rot, Mx, K)
        B_matrix_gpu: (N_rot, K, N_seg)
        returns:      (N_rot, Mx, N_seg)
        """
        coeffs_gpu = cp.asarray(coefficients, dtype=self.dtype)
        # Batched matrix multiplication
        out_gpu = cp.matmul(coeffs_gpu, self.B_matrix_gpu)
        out_cpu = cp.asnumpy(out_gpu)
        del coeffs_gpu, out_gpu
        cp.cuda.runtime.deviceSynchronize()
        return out_cpu
    
    def adjoint_gpu(self, data: np.ndarray) -> np.ndarray:
        """
        GPU version of adjoint.
        data: (N_rot, Mx, N_seg)
        B_matrix_gpu: (N_rot, K, N_seg)
        returns: (N_rot, Mx, K)
        """
        data_gpu = cp.asarray(data, dtype=self.dtype)
        out_gpu = cp.matmul(data_gpu, cp.transpose(self.B_matrix_gpu, (0, 2, 1)))
        out_cpu = cp.asnumpy(out_gpu)
        del data_gpu, out_gpu
        cp.cuda.runtime.deviceSynchronize()
        return out_cpu




@jit(nopython=True)
def worker_forward_cpu_jit(submatrices, coeffs, out):
    for i in range(len(submatrices)):
        np.dot(coeffs[i], submatrices[i], out=out[i])


def worker_forward_cpu(args):
    submatrices, coeffs = args
    out = np.zeros(coeffs.shape[:-1] + (submatrices.shape[2],), coeffs.dtype)
    worker_forward_cpu_jit(submatrices, coeffs, out)
    return out



@jit(nopython=True)
def worker_adjoint_cpu_jit(submatrices, data, out):
    for i in range(len(submatrices)):
        np.dot(data[i], submatrices[i].T, out=out[i])


def worker_adjoint_cpu(args):
    submatrices, data = args
    out = np.zeros(data.shape[:-1] + (submatrices.shape[1],),
                data.dtype)
    worker_adjoint_cpu_jit(submatrices, data, out)
    return out