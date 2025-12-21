# prox_operators.py
import pyopencl as cl
import pyopencl.array as clarray
import numpy as np

PROX_KERNELS = r"""
__kernel void prox_nonneg(__global float *x, int total) {
    int gid = get_global_id(0);
    if (gid < total && x[gid] < 0.0f) x[gid] = 0.0f;
}

__kernel void prox_l1(__global float *x, float lambda, int total) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    float v = x[gid];
    float a = fabs(v) - lambda;
    x[gid] = (a > 0.0f) ? copysign(a, v) : 0.0f;
}

__kernel void prox_nonneg_l1(__global float *x, float lambda, int total) {
    int gid = get_global_id(0);
    if (gid >= total) return;
    float v = x[gid] - lambda;
    x[gid] = (v > 0.0f) ? v : 0.0f;
}
"""

def build_prox_program(ctx):
    return cl.Program(ctx, PROX_KERNELS).build()


def prox_nonneg(queue, prg, x_gpu):
    total = x_gpu.size
    prg.prox_nonneg(
        queue, (total,), None,
        x_gpu.data, np.int32(total)
    )
    return x_gpu


def prox_l1(queue, prg, x_gpu, lam, tau):
    total = x_gpu.size
    prg.prox_l1(
        queue, (total,), None,
        x_gpu.data,
        np.float32(lam * tau),
        np.int32(total)
    )
    return x_gpu


def prox_nonneg_l1(queue, prg, x_gpu, lam, tau):
    total = x_gpu.size
    prg.prox_nonneg_l1(
        queue, (total,), None,
        x_gpu.data,
        np.float32(lam * tau),
        np.int32(total)
    )
    return x_gpu
