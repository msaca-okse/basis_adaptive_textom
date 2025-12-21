# odf_sh_image_operator.py
from typing import Optional, Sequence, Tuple
from numpy.typing import NDArray
import numpy as np
import h5py
import time
import gc

from cil.framework import ImageGeometry, ImageData, BlockDataContainer, AcquisitionGeometry, AcquisitionData
from cil.optimisation.operators import LinearOperator


import pyopencl as cl
import pyopencl.array as clarray
import gratopy
from pyclblast import gemmStridedBatched







# ---- OpenCL kernels (module-level) ----

EXPAND_GAUSSIAN_KERNEL_SRC = r"""
__kernel void expand_gaussian_peaks(
    __global const float *basis,
    __global const float *gaussian,
    __global float *out,
    int R, int K, int C, int P, int T
){
    int gid = get_global_id(0);

    int t   = gid % T;
    int tmp = gid / T;
    int c   = tmp % C;
    tmp    /= C;
    int k   = tmp % K;
    int r   = tmp / K;

    if (r >= R) return;

    float acc = 0.0f;
    int base_basis = (((r*K + k)*C + c)*P);
    int base_g     = t;

    for (int p = 0; p < P; ++p) {
        acc += basis[base_basis + p] * gaussian[p*T + t];
    }

    out[(((r*K + k)*C + c)*T + t)] = acc;
}
"""

TRANSPOSE_KERNEL_SRC = r"""
__kernel void transpose_k_nrot_mx(
    __global const float *inp,
    __global float *out,
    int K, int R, int Mx,
    int total   // = K*R*Mx
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int k = gid % K;
    int tmp = gid / K;
    int x = tmp % Mx;
    int r = tmp / Mx;

    out[(r*Mx + x)*K + k] = inp[(k*R + r)*Mx + x];
}
"""

TRANSPOSE_KERNEL_F_TO_C = r"""
__kernel void transpose_d_omega_k_f_to_c(
    __global const float *inp,   // (d, ω, K) Fortran
    __global float *out,         // (ω, d, K) C
    const int D,                 // number of detectors (d)
    const int O,                 // number of rotations (ω)
    const int K,                 // number of coefficients
    const int total              // = D * O * K
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    // Decompose linear index assuming output C-order (ω, d, K)
    int k = gid % K;
    int tmp = gid / K;
    int d = tmp % D;
    int o = tmp / D;

    // Fortran index: d + D*(o + O*k)
    int in_idx = d + D * (o + O * k);

    out[gid] = inp[in_idx];
}
"""


BTRANSPOSE_KERNEL_SRC = r"""
__kernel void transpose_B_r_k_nsub_to_r_nsub_k(
    __global const float *inp,   // (R, K, Nsub)
    __global float *out,         // (R, Nsub, K)
    int R, int K, int Nsub,
    int total                    // = R*K*Nsub
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int j = gid % Nsub;
    int tmp = gid / Nsub;
    int k = tmp % K;
    int r = tmp / K;

    // out[r,j,k] = inp[r,k,j]
    out[(r * Nsub + j) * K + k] = inp[(r * K + k) * Nsub + j];
}
"""

TRANSPOSE_KERNEL_ADJOINT = r"""
__kernel void transpose_r_mx_k_to_k_r_mx(
    __global const float *inp,   // (R, Mx, K)
    __global float *out,         // (K, R, Mx)
    int R, int Mx, int K,
    int total                    // = R*Mx*K
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int k = gid % K;
    int tmp = gid / K;
    int x = tmp % Mx;
    int r = tmp / Mx;

    out[(k*R + r)*Mx + x] = inp[(r*Mx + x)*K + k];
}
"""

TRANSPOSE_KERNEL_ADJOINT_C_TO_F = r"""
__kernel void transpose_omega_d_k_c_to_d_omega_k_f(
    __global const float *inp,   // (ω, d, K) C-order
    __global float *out,         // (d, ω, K) Fortran-order
    const int O,                 // number of omega
    const int D,                 // number of detectors
    const int K,                 // number of coefficients
    const int total              // = O * D * K
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    // Decompose gid assuming INPUT C-order (ω, d, K)
    int k = gid % K;
    int tmp = gid / K;
    int d = tmp % D;
    int o = tmp / D;

    // Input index (C-order)
    int in_idx = (o * D + d) * K + k;

    // Output index (Fortran-order)
    int out_idx = d + D * (o + O * k);

    out[out_idx] = inp[in_idx];
}
"""




ACCUMULATE_KERNEL_SRC = r"""
__kernel void accumulate_segments(
    __global float *out_full,
    __global const float *out_sub,
    __global const int *idx,
    int R, int Mx, int Nsub, int Nfull,
    int total // = R*Mx*Nsub
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int s = gid % Nsub;
    int tmp = gid / Nsub;
    int x = tmp % Mx;
    int r = tmp / Mx;
    if (r >= R) return;

    int full_s = idx[s];
    // optional extra safety while debugging:
    if ((unsigned)full_s >= (unsigned)Nfull) return;

    out_full[(r*Mx + x)*Nfull + full_s] += out_sub[(r*Mx + x)*Nsub + s];
}

"""


BATCHED_GEMM_KERNEL_SRC = r"""
// Simple tiled batched GEMM: C[r,m,n] = sum_k A[r,m,k] * B[r,k,n]
//
// Layout assumptions (row-major / C-order contiguous):
//   A: (R, M, K) contiguous with fastest axis K
//   B: (R, K, N) contiguous with fastest axis N
//   C: (R, M, N) contiguous with fastest axis N
//
// Global NDRange: (N, M, R)  i.e. x=n, y=m, z=r
//
// Tune TS for your GPU (16 or 32 typical).
#ifndef TS
#define TS 16
#endif

__kernel void batched_gemm_rmn(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int R,
    const int M,
    const int K,
    const int N
){
    const int n = (int)get_global_id(0); // column in N
    const int m = (int)get_global_id(1); // row in M
    const int r = (int)get_global_id(2); // batch index

    if (r >= R || m >= M || n >= N) return;

    // Local indices within the tile
    const int ln = (int)get_local_id(0);
    const int lm = (int)get_local_id(1);

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float acc = 0.0f;

    // Base pointers for batch r
    const int A0 = (r * M) * K; // start of A[r,:,:]
    const int B0 = (r * K) * N; // start of B[r,:,:]
    const int C0 = (r * M) * N; // start of C[r,:,:]

    // Iterate over K in tiles of TS
    for (int k0 = 0; k0 < K; k0 += TS) {

        // Load A tile: Asub[lm][ln] = A[r, m, k0+ln]
        int ak = k0 + ln;
        if (ak < K) Asub[lm][ln] = A[A0 + m*K + ak];
        else        Asub[lm][ln] = 0.0f;

        // Load B tile: Bsub[lm][ln] = B[r, k0+lm, n]
        int bk = k0 + lm;
        if (bk < K) Bsub[lm][ln] = B[B0 + bk*N + n];
        else        Bsub[lm][ln] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial dot
        #pragma unroll
        for (int t = 0; t < TS; ++t) {
            acc += Asub[lm][t] * Bsub[t][ln];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[C0 + m*N + n] = acc;
}
"""
GATHER_LAST_AXIS_KERNEL = r"""
__kernel void gather_last_axis(
    __global const float *inp,   // (R, Mx, Nfull)
    __global float *out,          // (R, Mx, Nsub)
    __global const int *idx,      // (Nsub)
    int R,
    int Mx,
    int Nsub,
    int Nfull
) {
    int gid = get_global_id(0);
    int total = R * Mx * Nsub;
    if (gid >= total) return;

    int j = gid % Nsub;
    int tmp = gid / Nsub;
    int x = tmp % Mx;
    int r = tmp / Mx;

    int src = idx[j];

    out[(r * Mx + x) * Nsub + j] =
        inp[(r * Mx + x) * Nfull + src];
}
"""


X_SLICING_KERNEL = r"""
__kernel void gather_coeffs_k_slice(
    __global const float *inp,   // (Ktot, R, Mx)
    __global float *out,         // (Ki,   R, Mx)
    int i0,
    int Ki,
    int R,
    int Mx,
    int Ktot
){
    int gid = get_global_id(0);
    int total = Ki * R * Mx;
    if (gid >= total) return;

    int x = gid % Mx;
    int tmp = gid / Mx;
    int r = tmp % R;
    int k = tmp / R;

    out[(k*R + r)*Mx + x] =
        inp[((k + i0)*R + r)*Mx + x];
}
"""

K_SLICING_KERNEL = r"""
__kernel void slice_k_lastaxis_f(
    __global const float *inp,   // (d, ω, Ktot) Fortran
    __global float *out,         // (d, ω, Ki)   Fortran
    const int D,                 // number of detectors
    const int O,                 // number of rotations
    const int Ktot,              // total K
    const int i0,                // starting K index
    const int Ki,                // number of K to extract
    const int total              // = D * O * Ki
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    // Decompose gid assuming Fortran layout of output
    int d = gid % D;
    int tmp = gid / D;
    int o = tmp % O;
    int k = tmp / O;   // k in [0, Ki)

    // Input K index
    int kin = k + i0;

    // Fortran indexing
    int out_idx = d + D * (o + O * k);
    int in_idx  = d + D * (o + O * kin);

    out[out_idx] = inp[in_idx];
}
"""


SCATTER_LASTAXIS_KERNEL = r"""
__kernel void scatter_k_lastaxis_f(
    __global float *dst,        // (d, ω, Ktot) Fortran
    __global const float *src,  // (d, ω, Ki)   Fortran
    const int D,
    const int O,
    const int Ktot,
    const int i0,
    const int Ki,
    const int total             // = D * O * Ki
){
    int gid = get_global_id(0);
    if (gid >= total) return;

    int d = gid % D;
    int tmp = gid / D;
    int o = tmp % O;
    int k = tmp / O;

    int dst_k = k + i0;

    int src_idx = d + D * (o + O * k);
    int dst_idx = d + D * (o + O * dst_k);

    dst[dst_idx] = src[src_idx];
}
"""




class PFO(LinearOperator):
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
        material_keys: None,
        N_theta: None,
        N_chi: None,
        N_rot: None,
        Nx: None,
        K_list: None,
        two_thetas: None,
        peak_width: None,
        ig_in: ImageGeometry,
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
        self.material_keys = material_keys
        self.ig_in   = ig_in
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
        self.K_list = K_list
        self.N_mat = len(K_list)
        self.B_gpu = None

        # Shapes from SH projection matrix
        #N_rot, K, N_chi, N_theta = map(int, B_matrix.shape)
        N_seg = N_chi*N_theta
        self.N_chi = N_chi
        self.N_theta = N_theta
        self.N_rot, self.N_seg = N_rot, N_seg
        self.Nx = Nx
        self.K_sum = int(sum(self.K_list))

        # --- 1) Create a context/queue once ---
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)

        # --- 2) Create ProjectionSettings using THIS queue ---
        self.PS = gratopy.ProjectionSettings(
            self.queue,
            gratopy.PARALLEL,
            (Nx, Nx, self.K_sum),
            N_rot,
            Nx,
            image_width=Nx,
            detector_width=Nx,
            detector_shift=2
        )
        assert self.queue.context.int_ptr == self.ctx.int_ptr

        # --- 4) NOW build all kernels/programs in the FINAL context ---
        self.prg = cl.Program(
            self.ctx,
            EXPAND_GAUSSIAN_KERNEL_SRC
            + TRANSPOSE_KERNEL_SRC
            + ACCUMULATE_KERNEL_SRC
            + BATCHED_GEMM_KERNEL_SRC
            + GATHER_LAST_AXIS_KERNEL
            + BTRANSPOSE_KERNEL_SRC
            + TRANSPOSE_KERNEL_ADJOINT
            + X_SLICING_KERNEL
            + TRANSPOSE_KERNEL_F_TO_C
            + K_SLICING_KERNEL
            + TRANSPOSE_KERNEL_ADJOINT_C_TO_F
            + SCATTER_LASTAXIS_KERNEL
        ).build(options=["-D", "TS=16"])

        # --- 5) Grab kernels from the program ---
        self.batched_gemm_kernel = self.prg.batched_gemm_rmn
        self.expand_kernel = self.prg.expand_gaussian_peaks
        self.transpose_kernel = self.prg.transpose_k_nrot_mx
        self.accumulate_kernel = self.prg.accumulate_segments
        self.gather_kernel = self.prg.gather_last_axis
        self.btranspose_kernel = self.prg.transpose_B_r_k_nsub_to_r_nsub_k
        self.transpose_r_mx_k_to_k_r_mx_kernel = self.prg.transpose_r_mx_k_to_k_r_mx
        self.gather_coeffs_kernel = self.prg.gather_coeffs_k_slice
        self.transpose_d_omega_k_f_to_c = self.prg.transpose_d_omega_k_f_to_c
        self.slice_k_lastaxis_f = self.prg.slice_k_lastaxis_f
        self.transpose_omega_d_k_c_to_d_omega_k_f = self.prg.transpose_omega_d_k_c_to_d_omega_k_f
        self.scatter_k_lastaxis_f = self.prg.scatter_k_lastaxis_f



        self.N_peaks_list = []
        self.basis_np_list = []
        self.intensity_np_list = []
        self.peak_positions_np_list = []
        self.basis_function_arrays_list = []
        self.full_idx_list = []
        self.theta_mask_list = []
        self.full_mask_list = []
        self.N_theta_mask_list = []
        self.basis_gpu_nonexpanded = []

        for i_mat in range(self.N_mat):
            with h5py.File(self.B_matrix_path, "r") as f:
                g = f[self.material_keys[i_mat]]
                basis_np = g["data"][...]
                intensity_np = g["intensity"][...]
                peak_positions_np = g["peak_positions"][...]

            self.basis_np_list.append(basis_np)
            self.intensity_np_list.append(intensity_np)
            self.peak_positions_np_list.append(peak_positions_np)
            

            self.N_peaks = basis_np.shape[2]//self.N_chi
            self.N_peaks_list.append(self.N_peaks)

            basis_function_arrays = basis_np.reshape((self.N_rot, self.K_list[i_mat], self.N_chi, self.N_peaks))
            basis_function_arrays = (
                basis_function_arrays
                / (basis_function_arrays.sum(axis=(0,1,2), keepdims=True) + 1e-3)
                * intensity_np[None, None, None, :]* basis_function_arrays.sum(axis=(0,1,2,3))/1000
                ).astype(np.float32, copy=False)
            
            self.basis_function_arrays_list.append(basis_function_arrays)
            self.offsets = np.zeros(len(self.K_list)+1, dtype=int)
            self.offsets[1:] = np.cumsum(self.K_list)


            diff = np.abs(self.two_thetas[:, None] - peak_positions_np[None, :])   # (N_theta, N_peaks)
            min_dist = np.min(diff, axis=1)                                # (N_theta,)
            theta_mask = min_dist < (2.5 * self.peak_width)
            full_mask = np.tile(theta_mask, self.N_chi)
            self.theta_mask_list.append((theta_mask))
            self.full_mask_list.append((full_mask))
            self.full_idx_list.append(np.nonzero(full_mask)[0])
            self.N_theta_mask_list.append(int(np.sum(theta_mask)))

            self.basis_gpu_nonexpanded.append(clarray.to_device(
                self.queue,
                np.asarray(self.basis_function_arrays_list[i_mat], dtype=np.float32)
            ))

        self.full_idx_gpu_list = []

        for idx in self.full_idx_list:
            idx_np = np.asarray(idx, dtype=np.int32)
            idx_gpu = clarray.to_device(self.queue, idx_np)
            self.full_idx_gpu_list.append(idx_gpu)



        if self.keep_in_GPU:
            self.setup_matrices_to_device()
        
        # Initialise LinearOperator with geometries
        super().__init__(domain_geometry=self.ig_in, range_geometry=self.ag_out)


    def get_c(self, coeffs_gpu, i_mat):
        """
        coeffs_gpu: clarray (K_total, R, Mx)
        returns:    clarray (K_i,     R, Mx)
        """
        queue = self.queue

        i0 = self.offsets[i_mat]
        i1 = self.offsets[i_mat + 1]
        Ki = i1 - i0

        R = self.N_rot
        Mx = self.Nx
        Ktot = coeffs_gpu.shape[0]

        out = clarray.empty(queue, (Ki, R, Mx), dtype=np.float32)

        total = Ki * R * Mx
        self.gather_coeffs_kernel(
            queue,
            (total,),
            None,
            coeffs_gpu.data,
            out.data,
            np.int32(i0),
            np.int32(Ki),
            np.int32(R),
            np.int32(Mx),
            np.int32(Ktot),
        )

        return out
    

    def get_c_opencl_fortran(self, coeffs_gpu, i_mat):
        """
        Slice coefficients along last axis (K) on GPU.

        coeffs_gpu: clarray, shape (d, ω, K_total), order='F'
        returns:    clarray, shape (d, ω, K_i),     order='F'
        """

        # --- K slicing info ---
        i0 = self.offsets[i_mat]
        i1 = self.offsets[i_mat + 1]
        Ki = i1 - i0

        # --- dimensions ---
        D, O, Ktot = coeffs_gpu.shape

        # --- allocate output (Fortran order!) ---
        out = clarray.empty(
            self.queue,
            (D, O, Ki),
            dtype=np.float32,
            order="F",
        )

        total = D * O * Ki

        # --- launch slicing kernel ---
        self.slice_k_lastaxis_f(
            self.queue,
            (total,),
            None,
            coeffs_gpu.data,
            out.data,
            np.int32(D),
            np.int32(O),
            np.int32(Ktot),
            np.int32(i0),
            np.int32(Ki),
            np.int32(total),
        )

        return out



    def setup_matrix(self, i_mat):

        basis_gpu = self.basis_gpu_nonexpanded[i_mat]  # clarray.Array
        peaks_gpu = np.array(self.peak_positions_np_list[i_mat], dtype=np.float32)  # (N_peaks,)

    
        dt = float(self.two_thetas[1] - self.two_thetas[0])
        inv_norm = 1.0 / np.sqrt(2.0 * np.pi * (self.peak_width ** 2))

        # diff: (N_peaks, N_t2)
        diff = peaks_gpu[:, None] - self.two_thetas[None, self.theta_mask_list[i_mat]]
        gaussian_np = (inv_norm * np.exp(-0.5 * (diff / self.peak_width) ** 2) * dt).astype(np.float32)

        gaussian_gpu = clarray.to_device(self.queue, gaussian_np)

        R = self.N_rot
        K = self.K_list[i_mat]
        C = self.N_chi
        P = basis_gpu.shape[3]
        T = gaussian_np.shape[1]

        out_gpu = clarray.empty(
            self.queue, (R, K, C, T), dtype=np.float32
        )

        global_size = (R * K * C * T,)

        self.expand_kernel(
            self.queue,
            global_size,
            None,
            basis_gpu.data,
            gaussian_gpu.data,
            out_gpu.data,
            np.int32(R),
            np.int32(K),
            np.int32(C),
            np.int32(P),
            np.int32(T),
        )

        return out_gpu.reshape(R, K, C * T)
    
    def setup_matrices_to_device(self):

        self.B_gpu_list = []
        for i_mat in range(self.N_mat):
            B_gpu = self.setup_matrix(i_mat)
            self.B_gpu_list.append(B_gpu)





    def direct(self, x: ImageData, out: Optional[ImageData] = None) -> ImageData:
        """
        OpenCL forward operator
        """
        t_total_start = time.perf_counter()

        t_upload_total    = 0.0
        t_get_c_total     = 0.0
        t_transpose_total = 0.0
        t_forward_total   = 0.0
        t_download_total  = 0.0

        if not isinstance(x, ImageData):
            raise TypeError(type(x))

        xin = x.as_array()
        if tuple(xin.shape) != tuple(self.ig_in.shape):
            raise ValueError("Shape mismatch")

        # ---------------- upload FULL coeff array ----------------
        t0 = time.perf_counter()
        xin_f = np.asfortranarray(xin, dtype=np.float32)

        coeffs_gpu_full = clarray.to_device(
            self.queue,
            xin_f
        )
        assert coeffs_gpu_full.flags.f_contiguous
        self.queue.finish()  # for correct timing + safe start
        t_upload_total += time.perf_counter() - t0

        # ---------------- gratopy projection ----------------
        coeffs_gpu_full_sino = clarray.zeros(
            self.queue,
            (self.PS.n_detectors, self.PS.n_angles, self.K_sum),
            dtype=np.float32,
            order="F",
        )
        assert coeffs_gpu_full_sino.queue is self.queue

        gratopy.forwardprojection(coeffs_gpu_full, self.PS, sino=coeffs_gpu_full_sino)
        assert coeffs_gpu_full_sino.flags.f_contiguous
        # Ensure projection completed before we start slicing/transpose
        self.queue.finish()

        # ---------------- allocate output ----------------
        t0 = time.perf_counter()
        yin_gpu = clarray.zeros(
            self.queue,
            (self.N_rot, self.Nx, self.N_chi * self.N_theta),
            dtype=np.float32,
            order="C",
        )
        self.queue.finish()
        t_upload_total += time.perf_counter() - t0

        # ---------------- main loop ----------------
        for i_mat in range(self.N_mat):

            K_i = self.K_list[i_mat]
            total = self.N_rot * self.Nx * K_i

            # ---- slice coefficients (GPU) ----
            t0 = time.perf_counter()
            coeffs_sub_gpu = self.get_c_opencl_fortran(coeffs_gpu_full_sino, i_mat)
            self.coeffs_sub_gpu = coeffs_sub_gpu
            assert self.queue.context.int_ptr == self.transpose_d_omega_k_f_to_c.context.int_ptr

            self.queue.finish()  # needed because we'll free / reuse soon
            t_get_c_total += time.perf_counter() - t0

            # ---- transpose (d,ω,K)[F] -> (ω,d,K)[C] ----
            coeffs_t_gpu = clarray.empty(
                self.queue,
                (self.N_rot, self.Nx, K_i),
                dtype=np.float32,
                order="C",
            )

            t0 = time.perf_counter()
            self.transpose_d_omega_k_f_to_c(
                self.queue,
                (total,),
                None,
                coeffs_sub_gpu.data,
                coeffs_t_gpu.data,
                np.int32(self.Nx),        # D
                np.int32(self.N_rot),     # O
                np.int32(K_i),            # K
                np.int32(total),
            )
            self.queue.finish()  # must finish before deleting coeffs_sub_gpu
            t_transpose_total += time.perf_counter() - t0

            del coeffs_sub_gpu

            # ---- forward operator ----
            t0 = time.perf_counter()
            self.forward_gpu_opencl(yin_gpu, coeffs_t_gpu, i_mat)
            self.queue.finish()  # must finish before deleting coeffs_t_gpu
            t_forward_total += time.perf_counter() - t0

            del coeffs_t_gpu

        # ---------------- download result ----------------
        t0 = time.perf_counter()
        yin_cpu = yin_gpu.get()  # implicit sync
        t_download_total += time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        if self.verbose:
            print("\n=== DIRECT() OpenCL TIMING ===")
            print(f"GPU upload total        : {t_upload_total:.4f} s")
            print(f"get_c (GPU slice) total : {t_get_c_total:.4f} s")
            print(f"transpose total        : {t_transpose_total:.4f} s")
            print(f"forward_gpu total      : {t_forward_total:.4f} s")
            print(f"GPU download total     : {t_download_total:.4f} s")
            print("--------------------------------")
            print(f"TOTAL direct() time    : {t_total:.4f} s")
            print("================================\n")

        # ---- write into out if provided ----
        if out is None:
            return AcquisitionData(yin_cpu, geometry=self.ag_out)
        else:
            out.fill(yin_cpu)
            return out





    def adjoint(self, y: ImageData,  out: Optional[AcquisitionData] = None):

        t_total_start = time.perf_counter()

        t_upload_total   = 0.0
        t_adjoint_total  = 0.0
        t_transpose_total= 0.0
        t_download_total = 0.0

        # ---------------- input ----------------
        if not isinstance(y, AcquisitionData):
            raise TypeError(type(y))

        yin = y.as_array()
        if tuple(yin.shape) != tuple(self.ag_out.shape):
            raise ValueError("Shape mismatch")

        queue = self.queue

        # ---------------- upload once ----------------
        t0 = time.perf_counter()
        yin_gpu = clarray.to_device(
            queue,
            np.require(yin, dtype=np.float32, requirements=["C"])
        )
        t_upload_total += time.perf_counter() - t0

        # ---------------- loop over materials ----------------
        O  = self.N_rot
        D  = self.Nx
        x_full_gpu = clarray.zeros(
            queue,
            (D, O, self.K_sum),
            dtype=np.float32,
            order="F",
        )
        for i_mat in range(self.N_mat):

            # ---- adjoint GPU ----
            t0 = time.perf_counter()
            xin_gpu = self.adjoint_gpu_opencl(yin_gpu, i_mat)
            queue.finish()
            t_adjoint_total += time.perf_counter() - t0

            # ---- transpose to (K, R, Mx) ----
            t0 = time.perf_counter()
            # ---- transpose (ω, d, K)[C] -> (d, ω, K)[F] ----
            K  = self.K_list[i_mat]
            total = O * D * K

            xin_gpu_t = clarray.empty(
                queue,
                (D, O, K),          # (d, ω, K)
                dtype=np.float32,
                order="F",
            )

            t0 = time.perf_counter()
            self.transpose_omega_d_k_c_to_d_omega_k_f(
                queue,
                (total,),
                None,
                xin_gpu.data,
                xin_gpu_t.data,
                np.int32(O),
                np.int32(D),
                np.int32(K),
                np.int32(total),
            )
            queue.finish()
            t_transpose_total += time.perf_counter() - t0

            i0 = self.offsets[i_mat]
            Ki = self.K_list[i_mat]
            total = D * O * Ki

            self.scatter_k_lastaxis_f(
                queue,
                (total,),
                None,
                x_full_gpu.data,
                xin_gpu_t.data,
                np.int32(D),
                np.int32(O),
                np.int32(self.K_sum),
                np.int32(i0),
                np.int32(Ki),
                np.int32(total),
            )
            queue.finish()

        x_img_gpu = clarray.zeros(
            queue,
            (self.Nx, self.Nx, self.K_sum),
            dtype=np.float32,
            order="F",
        )

        gratopy.backprojection(
            x_full_gpu,
            self.PS,
            img=x_img_gpu
        )

        # ---------------- download / write output ----------------
        t0 = time.perf_counter()

        if out is None:
            # create new ImageData
            x_cpu_f = x_img_gpu.get()  # implicit sync
            x_cpu = np.ascontiguousarray(x_cpu_f)
            result = ImageData(x_cpu, geometry=self.ig_in)
        else:
            # Write into provided container
            if not isinstance(out, ImageData):
                raise TypeError(type(out))

            if tuple(out.shape) != tuple(self.ig_in.shape):
                raise ValueError("Shape mismatch for out")

            # Download into a temporary array first
            x_cpu_f = x_img_gpu.get()

            # Enforce C-order
            out_arr = out.as_array()
            out_arr[...] = np.ascontiguousarray(x_cpu_f)

            result = out

        t_download_total += time.perf_counter() - t0
        queue.finish()

        t_total = time.perf_counter() - t_total_start

        if self.verbose:
            print("\n=== ADJOINT() OpenCL TIMING ===")
            print(f"GPU upload total      : {t_upload_total:.4f} s")
            print(f"adjoint_gpu total     : {t_adjoint_total:.4f} s")
            print(f"transpose total       : {t_transpose_total:.4f} s")
            print(f"GPU download total    : {t_download_total:.4f} s")
            print("--------------------------------")
            print(f"TOTAL adjoint() time  : {t_total:.4f} s")
            print("================================\n")

        return result
            


    def _validate_geometries(self):
        # ig_in: (Nx, Ny, K)
        if self.ig_in.shape[0] != self.K_sum:
            raise ValueError(f"ig_in.shape[-1]={self.ig_in.shape[-1]} must equal K={self.K_sum}")

        # ig_out: (Nx, Ny, M)
        if self.ig_out.shape[0] != self.M:
            raise ValueError(f"ig_out.shape[-1]={self.ig_out.shape[-1]} must equal M={self.M}")

        if self.ig_in.shape[1:] != self.ig_out.shape[1:]:
            raise ValueError(f"Spatial (x,y) mismatch: ig_in {self.ig_in.shape[1:]} vs ig_out {self.ig_out.shape[1:]}")
        




    def adjoint_gpu_opencl(self, data_gpu, i_mat):
        """
        data_gpu: clarray (R, Mx, Nseg_full)
        returns:  clarray (R, Mx, K_i)
        """
        queue = self.queue
        R = self.N_rot
        Mx = self.Nx
        K  = self.K_list[i_mat]
        Nsub = len(self.full_idx_list[i_mat])
        Nfull = self.N_chi * self.N_theta

        # ---- gather Y_sub = data[..., idx] into (R, Mx, Nsub) ----
        data_gpu_sub = clarray.empty(queue, (R, Mx, Nsub), dtype=np.float32)
        global_size = (R * Mx * Nsub,)

        self.gather_kernel(
            queue, global_size, None,
            data_gpu.data,
            data_gpu_sub.data,
            self.full_idx_gpu_list[i_mat].data,
            np.int32(R),
            np.int32(Mx),
            np.int32(Nsub),
            np.int32(Nfull),
        )

        # ---- build B_gpu: (R, K, Nsub) ----
        if self.keep_in_GPU:
            B_gpu = self.B_gpu_list[i_mat]
        else:
            B_gpu = self.setup_matrix(i_mat)

        # ---- explicit transpose: BT_gpu = (R, Nsub, K) ----
        BT_gpu = clarray.empty(queue, (R, Nsub, K), dtype=np.float32)
        total = R * K * Nsub
        self.btranspose_kernel(
            queue, (total,), None,
            B_gpu.data,
            BT_gpu.data,
            np.int32(R),
            np.int32(K),
            np.int32(Nsub),
            np.int32(total),
        )


        # ---- output (R, Mx, K) ----
        out_gpu = clarray.empty(queue, (R, Mx, K), dtype=np.float32)

        # ---- batched adjoint GEMM (no transpose flags now) ----
        batched_gemm_adj_clblast(queue, data_gpu_sub, BT_gpu, out_gpu, R, Mx, Nsub, K)

        return out_gpu



    def forward_gpu_opencl(self, out_gpu_full, coeffs_gpu, i_mat):
        """
        OpenCL forward operator for one material.

        Parameters
        ----------
        out_gpu_full : clarray.Array
            Shape (R, Mx, Nseg_full), accumulated in-place
        coeffs_gpu : clarray.Array
            Shape (R, Mx, K_i)
        i_mat : int
            Material index
        """

        queue = self.queue

        # ---------------- build / fetch B ----------------
        # B_gpu: (R, K_i, Nsub)
        if self.keep_in_GPU:
            B_gpu = self.B_gpu_list[i_mat]
        else:
            B_gpu = self.setup_matrix(i_mat)

        R = self.N_rot
        Mx = self.Nx
        K  = self.K_list[i_mat]
        Nsub = B_gpu.shape[2]

        # ---------------- allocate sub-output (reuse-friendly) ----------------
        out_sub = clarray.empty(
            queue,
            (R, Mx, Nsub),
            dtype=np.float32,
            order="C",
        )

        # ---------------- batched GEMM ----------------
        # out_sub[r,m,n] = sum_k coeffs_gpu[r,m,k] * B_gpu[r,k,n]
        self.coeffs_gpu = coeffs_gpu
        self.B_gpu = B_gpu
        batched_gemm_clblast(
            self.queue,
            coeffs_gpu,
            B_gpu,
            out_sub,
            R=self.N_rot,
            M=self.Nx,
            K=self.K_list[i_mat],
            N=Nsub,
        )

        # ---------------- accumulate into full sinogram ----------------
        idx_gpu = self.full_idx_gpu_list[i_mat]

        global_size = (R * Mx * Nsub,)


        total = np.int32(R) * np.int32(Mx) * np.int32(Nsub)
        self.out_sub = out_sub
        self.accumulate_kernel(
            queue,
            global_size,
            None,
            out_gpu_full.data,
            out_sub.data,
            idx_gpu.data,
            np.int32(R),
            np.int32(Mx),
            np.int32(Nsub),
            np.int32(out_gpu_full.shape[2]),
            total,
        )

        # Explicit deletes are optional; buffers become reusable when out of scope
        del out_sub, B_gpu



def batched_gemm_clblast(queue, A3, B3, C3, R, M, K, N):
    """
    A3: (R, M, K) clarray
    B3: (R, K, N) clarray
    C3: (R, M, N) clarray
    """

    # reshape views (NO COPY)
    A = A3.reshape((R*M, K))
    B = B3.reshape((R*K, N))
    C = C3.reshape((R*M, N))

    # leading dimensions (row-major)
    a_ld = K
    b_ld = N
    c_ld = N

    # strides between batches (in elements)
    a_stride = M * K
    b_stride = K * N
    c_stride = M * N

    gemmStridedBatched(
        queue,
        M, N, K,
        R,              # batch_count
        A, B, C,        # MUST be 2D
        a_ld, b_ld, c_ld,
        a_stride, b_stride, c_stride,
        alpha=1.0,
        beta=0.0,
    )




def batched_gemm_adj_clblast(queue, Y3, BT3, X3, R, Mx, Nsub, K):
    """
    Y3:  (R, Mx,   Nsub)  clarray
    BT3: (R, Nsub, K)     clarray  (explicitly transposed B)
    X3:  (R, Mx,   K)     clarray (output)
    """

    # 2D views (NO COPY)
    A = Y3.reshape((R * Mx, Nsub))   # (R*Mx, Nsub)
    B = BT3.reshape((R * Nsub, K))   # (R*Nsub, K)
    C = X3.reshape((R * Mx, K))      # (R*Mx, K)

    # leading dimensions (row-major)
    a_ld = Nsub
    b_ld = K
    c_ld = K

    # batch strides (in elements)
    a_stride = Mx * Nsub
    b_stride = Nsub * K
    c_stride = Mx * K

    gemmStridedBatched(
        queue,
        Mx, K, Nsub,     # m, n, k
        R,               # batch_count
        A, B, C,
        a_ld, b_ld, c_ld,
        a_stride, b_stride, c_stride,
        alpha=1.0,
        beta=0.0,
        a_transp=False,
        b_transp=False,  # <<< now no ambiguity
    )