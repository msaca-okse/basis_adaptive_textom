# odf_sh_image_operator.py
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray
import numpy as np
from numba import jit
import h5py
import time

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
        B_matrix_path: None,
        material_key: None,
        N_theta: None,
        N_chi: None,
        N_rot: None,
        K: None,
        two_thetas: None,
        peak_width: None,
        ag_in: AcquisitionGeometry,
        ag_out: AcquisitionGeometry,
        dtype=np.float32,
        device: str = 'cpu',
        projections_per_proc: int = 10,
        N_processes: int = 32,
        verbose: bool = False,
        convolve=True,
        sobolev = 0,
        keep_in_GPU = False,
        **sh_kwargs
    ):
        self.B_matrix_path = B_matrix_path
        self.material_key = material_key
        self.ag_in   = ag_in
        self.ag_out  = ag_out
        self.dtype   = dtype
        self.projections_per_proc = projections_per_proc
        self.device = device
        self.N_processes = N_processes
        self.two_thetas = np.array(two_thetas).astype(np.float32)
        self.peak_width = peak_width
        self.verbose = verbose
        self.convolve = convolve
        self.sobolev = sobolev
        self.keep_in_GPU = keep_in_GPU
        self.B_gpu = None


        # Shapes from SH projection matrix
        #N_rot, K, N_chi, N_theta = map(int, B_matrix.shape)
        N_seg = N_chi*N_theta
        self.N_chi = N_chi
        self.N_theta = N_theta
        self.N_rot, self.N_seg, self.K = N_rot, N_seg, K


        with h5py.File(self.B_matrix_path, "r") as f:
            g = f[self.material_key]
            basis_np = g["data"][...]
            intensity_np = g["intensity"][...]
            peak_positions_np = g["peak_positions"][...]

        self.N_peaks = basis_np.shape[2]//self.N_chi
        basis_function_arrays = basis_np.reshape((self.N_rot, self.K, self.N_chi, self.N_peaks))

        self.intensity = intensity_np
        self.peak_positions = peak_positions_np
        self.basis_function_arrays = (
            basis_function_arrays
            / (basis_function_arrays.sum(axis=(0,1,2), keepdims=True) + 1e-3)
            * self.intensity[None, None, None, :]* basis_function_arrays.sum(axis=(0,1,2,3))/1000

        ).astype(np.float32, copy=False)
        
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


        #if xin.dtype != self.dtype:
        #    xin = xin.astype(self.dtype, copy=False)

        if tuple(xin.shape) != tuple(self.ag_in.shape):
            raise ValueError(f"Input shape {xin.shape} != ag_in.shape {self.ag_in.shape}")

        xin_t = np.ascontiguousarray(xin.transpose([1,2,0]))  # from (K, N_rot, Mx) to (N_rot, Mx, K)  

        #: (N_rot, Mx, K) -> (N_rot, Mx, N_seg)

        if self.device == 'gpu':
            yin_t = self.forward_gpu(xin_t)
            
        else:
            raise ValueError(f"Device must be 'gpu': {self.device}")


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
        

        #if yin.dtype != self.dtype:
        #    yin = yin.astype(self.dtype, copy=False)

        if tuple(yin.shape) != tuple(self.ag_out.shape):
            raise ValueError(f"Input shape {yin.shape} != ag_out.shape {self.ag_out.shape}")
        
        if self.device == 'gpu':

            xin = self.adjoint_gpu(yin)
        else:
            raise ValueError(f"Device must be 'gpu': {self.device}")


        xin_t = np.ascontiguousarray(xin.transpose([2,0,1]))  # from (K, M_rot, Mx) to (M_rot, Mx, K)
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

        flag = False
        t0 = time.perf_counter()
        if self.convolve:
            if self.B_gpu is None:
                flag = True
                basis_gpu = cp.asarray(self.basis_function_arrays, dtype=cp.float32)
                two_thetas_gpu = cp.asarray(self.two_thetas, dtype=cp.float32)
                peaks_gpu = cp.asarray(self.peak_positions, dtype=cp.float32)  # (N_peaks,)
                diff = cp.abs(two_thetas_gpu[:, None] - peaks_gpu[None, :])   # (N_theta, N_peaks)
                min_dist = cp.min(diff, axis=1)                                # (N_theta,)
                theta_mask = min_dist < (2.5 * self.peak_width)
                full_mask = cp.tile(theta_mask, self.N_chi)
                self.theta_mask = cp.asnumpy(theta_mask)
                self.full_mask = cp.asnumpy(full_mask)
                self.full_idx = np.nonzero(cp.asnumpy(full_mask))[0]
                self.N_theta_mask = int(np.sum(theta_mask))

            
                dt = float(self.two_thetas[1] - self.two_thetas[0])
                inv_norm = 1.0 / cp.sqrt(2.0 * cp.pi * (self.peak_width ** 2))

                # diff: (N_peaks, N_t2)
                diff = peaks_gpu[:, None] - two_thetas_gpu[None, theta_mask]
                gaussian = (inv_norm * cp.exp(-0.5 * (diff / self.peak_width) ** 2) * dt).astype(cp.float32)
                print(print('full mask shape', full_mask.shape))
                print('gaussian shape', gaussian.shape)

                # ---- contract over peaks (axis=3 of basis, axis=0 of gaussian) ----
                # result: (Nrot, K, N_chi, N_t2)
                B_gpu = cp.tensordot(basis_gpu, gaussian, axes=([3], [0]))
                B_gpu = B_gpu.reshape(self.N_rot, self.K, self.N_chi * self.N_theta_mask)


                if self.keep_in_GPU:
                    self.B_gpu = B_gpu
            else:
                B_gpu = self.B_gpu


        if self.verbose:
            print(f"creating B_matrix took {time.perf_counter() - t0:.3f} s")

        coeffs_gpu = cp.asarray(coefficients, dtype=self.dtype)
        # Batched matrix multiplication

        t1 = time.perf_counter()
        out_gpu = cp.matmul(coeffs_gpu, B_gpu)
        Nx = out_gpu.shape[1]
        out_gpu_full = cp.zeros((self.N_rot, Nx, self.N_chi*self.N_theta), dtype=self.dtype)
        out_gpu_full[..., self.full_idx] = out_gpu # faster to preallocate on gpu, but more memory heavy though...
        out_cpu = cp.asnumpy(out_gpu_full).astype(np.float32)

        if self.sobolev<0:
            k = cp.fft.rfftfreq(self.N_chi)              # frequencies in cycles per sample
            omega2 = (2 * cp.pi * k) ** 2
            weights = (1.0 + omega2) ** (self.sobolev/2)
            out_gpu = out_gpu.reshape(self.N_rot, Nx, self.N_chi, self.N_theta)
            out_gpu = cp.fft.rfft(out_gpu, axis=2)*weights[None,None,:,None]
            
            z_real = out_gpu.real
            z_imag = out_gpu.imag
            out_gpu = cp.concatenate([z_real, z_imag], axis=2).astype(cp.float32)
            out_gpu = out_gpu.reshape(self.N_rot, Nx, 2*(self.N_chi//2+1)*self.N_theta)

        if self.verbose:
            print(f"Matrix matrix product took {time.perf_counter() - t1:.3f} s")

        


        if self.convolve and flag:
            del coeffs_gpu, out_gpu, gaussian, diff, inv_norm, dt, peaks_gpu, two_thetas_gpu, basis_gpu
        else:
            del coeffs_gpu, out_gpu

        cp.cuda.runtime.deviceSynchronize()
        if self.verbose:
            print(f"forward() took {time.perf_counter() - t0:.3f} s")
            print('Matrix array is size', np.prod(B_gpu.shape)*4/1e9)
        if not self.keep_in_GPU:
            del B_gpu
        return out_cpu
    
    def adjoint_gpu(self, data: np.ndarray) -> np.ndarray:
        """
        GPU version of adjoint.
        data: (N_rot, Mx, N_seg)
        B_matrix_gpu: (N_rot, K, N_seg)
        returns: (N_rot, Mx, K)
        """

        t0 = time.perf_counter()
        flag = False
        if self.convolve:
            if self.B_gpu is None:
                flag = True
                basis_gpu = cp.asarray(self.basis_function_arrays, dtype=cp.float32)
                two_thetas_gpu = cp.asarray(self.two_thetas, dtype=cp.float32)
                peaks_gpu = cp.asarray(self.peak_positions, dtype=cp.float32)  # (N_peaks,)
                diff = cp.abs(two_thetas_gpu[:, None] - peaks_gpu[None, :])   # (N_theta, N_peaks)
                min_dist = cp.min(diff, axis=1)                                # (N_theta,)
                theta_mask = min_dist < (2.5 * self.peak_width)
                full_mask = cp.tile(theta_mask, self.N_chi)
                self.theta_mask = cp.asnumpy(theta_mask)
                self.full_mask = np.nonzero(cp.asnumpy(full_mask))[0]
                self.N_theta_mask = int(np.sum(theta_mask))

            
                dt = float(self.two_thetas[1] - self.two_thetas[0])
                inv_norm = 1.0 / cp.sqrt(2.0 * cp.pi * (self.peak_width ** 2))

                # diff: (N_peaks, N_t2)
                diff = peaks_gpu[:, None] - two_thetas_gpu[None, theta_mask]
                gaussian = (inv_norm * cp.exp(-0.5 * (diff / self.peak_width) ** 2) * dt).astype(cp.float32)
                print(print('full mask shape', full_mask.shape))
                print('gaussian shape', gaussian.shape)

                # ---- contract over peaks (axis=3 of basis, axis=0 of gaussian) ----
                # result: (Nrot, K, N_chi, N_t2)
                B_gpu = cp.tensordot(basis_gpu, gaussian, axes=([3], [0]))
                B_gpu = B_gpu.reshape(self.N_rot, self.K, self.N_chi * self.N_theta_mask)


                if self.keep_in_GPU:
                    self.B_gpu = B_gpu
            else:
                B_gpu = self.B_gpu

        if self.verbose:
            print(f"creating B_matrix took {time.perf_counter() - t0:.3f} s")


        t1 = time.perf_counter()
        data_gpu = cp.asarray(data, dtype=cp.float32)
        data_gpu = data_gpu[..., self.full_idx]

        Nx = data_gpu.shape[1]
        if self.sobolev<0:
            k = cp.fft.rfftfreq(self.N_chi)              # frequencies in cycles per sample
            omega2 = (2 * cp.pi * k) ** 2
            weights = (1.0 + omega2) ** (self.sobolev/2)
            data_gpu = data_gpu.reshape((self.N_rot, Nx, 2*(self.N_chi//2+1), self.N_theta))

            n_half = data_gpu.shape[2] // 2
            r_real = data_gpu[:,:, :n_half,:]
            r_imag = data_gpu[:,:, n_half:,:]
            data_gpu = r_real + 1j * r_imag
            data_gpu = cp.fft.irfft(data_gpu*weights[None,None,:,None], axis=2)
            data_gpu = data_gpu.reshape(self.N_rot, Nx, self.N_chi*self.N_theta)
            

        out_gpu = cp.matmul(data_gpu, cp.transpose(B_gpu, (0, 2, 1)))
        if self.verbose:
            print(f"Matrix matrix product took {time.perf_counter() - t1:.3f} s")

        out_cpu = cp.asnumpy(out_gpu).astype(np.float32)

        if self.convolve and flag:
            del data_gpu, out_gpu, gaussian, diff, inv_norm, dt, peaks_gpu, two_thetas_gpu, basis_gpu
        else:
            del data_gpu, out_gpu
        cp.cuda.runtime.deviceSynchronize()
        if self.verbose:
            print(f"adjoint() took {time.perf_counter() - t0:.3f} s")
            print('Matrix array is size', np.prod(B_gpu.shape)*4/1e9)
        del B_gpu
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

