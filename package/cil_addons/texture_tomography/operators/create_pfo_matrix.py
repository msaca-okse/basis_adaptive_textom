PF_KERNEL_SRC = r"""
__kernel void pfmatrix_eval(
    __global const float *coords,     // (R, C, P, 3) flattened
    __global const float *grid_inv,   // (K, 9) flattened row-major (3x3) = inverse rotation matrices
    __global const float *sym_ops,    // (G, 9) flattened row-major (3x3)
    __global const float *hvecs,      // (P, 3) normalized h-vectors
    __global float *out,              // (R, K, C, P) flattened
    const int R,
    const int K,
    const int C,
    const int P,
    const int G,
    const float inv_sigma2,           // 1/sigma^2
    const float norm_factor           // 1/(8*pi*sigma^2)  (or whatever you want)
){
    int gid = get_global_id(0);
    int total = R * K * C * P;
    if (gid >= total) return;

    // decode gid for layout out[r,k,c,p]
    int p = gid % P;
    int tmp = gid / P;
    int c = tmp % C;
    tmp /= C;
    int k = tmp % K;
    int r = tmp / K;

    // --- load coordinate v = coords[r,c,p,:] ---
    int coord_base = ((r * C + c) * P + p) * 3;
    float vx = coords[coord_base + 0];
    float vy = coords[coord_base + 1];
    float vz = coords[coord_base + 2];

    // --- apply inverse grid rotation q = grid_inv[k] * v ---
    int Rb = k * 9;
    float qx = grid_inv[Rb + 0]*vx + grid_inv[Rb + 1]*vy + grid_inv[Rb + 2]*vz;
    float qy = grid_inv[Rb + 3]*vx + grid_inv[Rb + 4]*vy + grid_inv[Rb + 5]*vz;
    float qz = grid_inv[Rb + 6]*vx + grid_inv[Rb + 7]*vy + grid_inv[Rb + 8]*vz;

    // --- load normalized h-vector for this reflection p ---
    int hb = p * 3;
    float hx0 = hvecs[hb + 0];
    float hy0 = hvecs[hb + 1];
    float hz0 = hvecs[hb + 2];

    float w = 0.0f;

    // loop over symmetry operations
    for (int g = 0; g < G; ++g) {
        int Sb = g * 9;

        // h_rot = sym_ops[g] * h0
        float hx = sym_ops[Sb + 0]*hx0 + sym_ops[Sb + 1]*hy0 + sym_ops[Sb + 2]*hz0;
        float hy = sym_ops[Sb + 3]*hx0 + sym_ops[Sb + 4]*hy0 + sym_ops[Sb + 5]*hz0;
        float hz = sym_ops[Sb + 6]*hx0 + sym_ops[Sb + 7]*hy0 + sym_ops[Sb + 8]*hz0;

        // dot = h_rot · q
        float dot = hx*qx + hy*qy + hz*qz;

        // gaussian_arg = -(1 - dot)/sigma^2
        float arg1 = -(1.0f - dot) * inv_sigma2;
        if (arg1 > -6.0f) w += exp(arg1);

        // Friedel partner: -(1 + dot)/sigma^2
        float arg2 = -(1.0f + dot) * inv_sigma2;
        if (arg2 > -6.0f) w += exp(arg2);
    }

    out[gid] = w * norm_factor;
}
"""




import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

def build_pf_program(ctx: cl.Context) -> cl.Program:
    return cl.Program(ctx, PF_KERNEL_SRC).build()

def pfmatrix_eval_gpu(
    queue: cl.CommandQueue,
    pfo_kernel: cl.Kernel,
    coords_gpu: clarray.Array,
    grid_inv_gpu: clarray.Array,
    sym_ops_gpu: clarray.Array,
    hvecs_gpu: clarray.Array,
    R: int, K: int, C: int, P: int, G: int,
    sigma: float,
    out_gpu: clarray.Array
):

    assert coords_gpu.dtype == np.float32
    assert grid_inv_gpu.dtype == np.float32
    assert sym_ops_gpu.dtype == np.float32
    assert hvecs_gpu.dtype == np.float32

    # output
    if out_gpu is None:
        out_gpu = clarray.empty(queue, (R, K, C, P), dtype=np.float32, order="C")

    inv_sigma2 = np.float32(1.0 / (sigma * sigma))

    # matches: weights_array /2 * (1/2*1/sqrt(2*pi)**2/sigma**2)
    # sqrt(2*pi)**2 = 2*pi => factor = 1/(8*pi*sigma^2)
    norm_factor = np.float32(1.0 / (8.0 * np.pi * sigma * sigma))

    total = R * K * C * P
    pfo_kernel(
        queue,
        (total,),
        None,
        coords_gpu.data,
        grid_inv_gpu.data,
        sym_ops_gpu.data,
        hvecs_gpu.data,
        out_gpu.data,
        np.int32(R), np.int32(K), np.int32(C), np.int32(P), np.int32(G),
        inv_sigma2,
        norm_factor
    )
    return out_gpu
