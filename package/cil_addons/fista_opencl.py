# fista_opencl.py
import time
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.clmath as clmath

# -------------------- FISTA helper kernels --------------------

FISTA_KERNELS = r"""
__kernel void residual_axpb(
    __global const float *Ax,
    __global const float *b,
    __global float *r,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    r[gid] = Ax[gid] - b[gid];
}

// v = y - tau * grad
__kernel void grad_step(
    __global const float *y,
    __global const float *grad,
    __global float *v,
    const float tau,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    v[gid] = y[gid] - tau * grad[gid];
}

// y = x + beta * (x - x_old)
__kernel void extrapolate(
    __global const float *x,
    __global const float *x_old,
    __global float *y,
    const float beta,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    float xv = x[gid];
    y[gid] = xv + beta * (xv - x_old[gid]);
}

// x_old <- x (copy kernel to avoid enqueue_copy corner-cases with strides)
__kernel void copy_buf(
    __global const float *src,
    __global float *dst,
    const int total
){
    int gid = get_global_id(0);
    if (gid >= total) return;
    dst[gid] = src[gid];
}
"""

def build_fista_program(ctx: cl.Context) -> cl.Program:
    return cl.Program(ctx, FISTA_KERNELS).build()


# -------------------- FISTA implementation --------------------

class FISTAOpenCL:
    """
    Solve: min_x 0.5||A x - b||^2 + g(x)
    with FISTA on GPU.

    x layout: (Nx, Ny, K) Fortran (but we treat it as flat for kernels)
    Ax layout: (O, D, Nseg) C
    """

    def __init__(self, operator, prox_kind="nonneg", lam=0.0, L=None, tau=None):
        """
        operator: object with fields .ctx, .queue and methods:
                  direct(x_gpu) -> y_gpu
                  adjoint(y_gpu) -> x_gpu
        prox_kind: "nonneg" | "l1" | "nonneg_l1"
        lam: lambda for l1 / nonneg_l1
        L: Lipschitz constant of grad f; tau defaults to 1/L
        tau: override step-size directly (if provided, L can be None)
        """
        self.op = operator
        self.ctx = operator.ctx
        self.queue = operator.queue

        self.fista_prg = build_fista_program(self.ctx)

        # prox program must be built on SAME context
        from package.cil_addons.prox import build_prox_program, prox_nonneg, prox_l1, prox_nonneg_l1
        self.prox_prg = build_prox_program(self.ctx)
        self._prox_nonneg = prox_nonneg
        self._prox_l1 = prox_l1
        self._prox_nonneg_l1 = prox_nonneg_l1

        self.prox_kind = prox_kind
        self.lam = float(lam)

        if tau is None:
            if L is None:
                raise ValueError("Provide either L or tau")
            self.tau = 1.0 / float(L)
        else:
            self.tau = float(tau)

    def _apply_prox(self, x_gpu):
        if self.prox_kind == "nonneg":
            return self._prox_nonneg(self.queue, self.prox_prg, x_gpu)
        elif self.prox_kind == "l1":
            return self._prox_l1(self.queue, self.prox_prg, x_gpu, self.lam, self.tau)
        elif self.prox_kind == "nonneg_l1":
            return self._prox_nonneg_l1(self.queue, self.prox_prg, x_gpu, self.lam, self.tau)
        else:
            raise ValueError(f"Unknown prox_kind: {self.prox_kind}")

    def run(
        self,
        x0_gpu: clarray.Array,
        out_gpu: clarray.Array,
        niter: int,
        verbose: int = 0,
        diagnostics_interval: int = 1,
    ):
        """
        x0_gpu: clarray (Nx, Ny, K) float32, order='F'
        out_gpu:  clarray (O, D, Nseg) float32, order='C'
        returns x_gpu solution (same layout as x0_gpu)

        verbose=0 disables printing
        verbose>=1 prints per diagnostics_interval
        """
        q = self.queue

        # ---- basic checks (cheap, but catches 90% of “garbage” bugs) ----
        if not isinstance(x0_gpu, clarray.Array) or not isinstance(out_gpu, clarray.Array):
            raise TypeError("x0_gpu and out_gpu must be pyopencl.array.Array")

        if x0_gpu.queue is None or out_gpu.queue is None:
            raise ValueError("Arrays must have a queue attached (created via clarray on a queue)")

        if x0_gpu.dtype != np.float32 or out_gpu.dtype != np.float32:
            raise TypeError("This FISTA assumes float32 arrays")

        if x0_gpu.queue.context.int_ptr != self.ctx.int_ptr:
            raise ValueError("x0_gpu context != operator context")
        if out_gpu.queue.context.int_ptr != self.ctx.int_ptr:
            raise ValueError("out_gpu context != operator context")

        # ---- persistent buffers ----
        # x, y, x_old, v, grad all same shape as x
        x = clarray.empty(q, x0_gpu.shape, dtype=np.float32, order="F")
        y = clarray.empty(q, x0_gpu.shape, dtype=np.float32, order="F")
        x_old = clarray.empty(q, x0_gpu.shape, dtype=np.float32, order="F")
        v = clarray.empty(q, x0_gpu.shape, dtype=np.float32, order="F")
        grad = None  # created after first adjoint (depends on op output layout)

        # copy x0 -> x,y,x_old
        total_x = x0_gpu.size
        self.fista_prg.copy_buf(q, (total_x,), None, x0_gpu.data, x.data, np.int32(total_x))
        self.fista_prg.copy_buf(q, (total_x,), None, x0_gpu.data, y.data, np.int32(total_x))
        self.fista_prg.copy_buf(q, (total_x,), None, x0_gpu.data, x_old.data, np.int32(total_x))
        q.finish()

        # We'll allocate Ax and residual after first forward (operator defines exact shape/order)
        Ax = None
        r = None

        t = 1.0

        # timers (optional)
        t_forward = 0.0
        t_residual = 0.0
        t_adjoint = 0.0
        t_gradstep = 0.0
        t_prox = 0.0
        t_extrap = 0.0
        t_obj = 0.0

        t_total_start = time.perf_counter()

        for k in range(niter):
            # ---- Ax = A(y) ----
            t0 = time.perf_counter()
            Ax_new = self.op.direct(y)  # returns clarray
            q.finish()
            t_forward += time.perf_counter() - t0

            if Ax is None:
                Ax = Ax_new
                # residual buffer (same shape as Ax)
                r = clarray.empty(q, Ax.shape, dtype=np.float32, order="C")
            else:
                # reuse Ax buffer: copy Ax_new -> Ax then drop Ax_new
                total_Ax = Ax.size
                self.fista_prg.copy_buf(q, (total_Ax,), None, Ax_new.data, Ax.data, np.int32(total_Ax))
                q.finish()
                del Ax_new

            # ---- r = Ax - b ----
            t0 = time.perf_counter()
            total_Ax = Ax.size
            self.fista_prg.residual_axpb(q, (total_Ax,), None, Ax.data, out_gpu.data, r.data, np.int32(total_Ax))
            q.finish()
            t_residual += time.perf_counter() - t0

            # ---- grad = A*(r) ----
            t0 = time.perf_counter()
            grad_new = self.op.adjoint(r)  # returns clarray shaped like x (Fortran)
            q.finish()
            t_adjoint += time.perf_counter() - t0

            if grad is None:
                grad = grad_new
            else:
                # reuse grad buffer
                total_x = grad.size
                self.fista_prg.copy_buf(q, (total_x,), None, grad_new.data, grad.data, np.int32(total_x))
                q.finish()
                del grad_new

            # ---- v = y - tau*grad (fused) ----
            t0 = time.perf_counter()
            total_x = y.size
            self.fista_prg.grad_step(
                q, (total_x,), None,
                y.data, grad.data, v.data,
                np.float32(self.tau),
                np.int32(total_x)
            )
            q.finish()
            t_gradstep += time.perf_counter() - t0

            # ---- prox: x <- prox(v)   (in-place on v, then copy to x) ----
            t0 = time.perf_counter()
            self._apply_prox(v)   # modifies v in-place
            q.finish()
            # copy v -> x
            self.fista_prg.copy_buf(q, (total_x,), None, v.data, x.data, np.int32(total_x))
            q.finish()
            t_prox += time.perf_counter() - t0

            # ---- momentum update (CPU scalar) ----
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_new

            # ---- y = x + beta*(x - x_old) (fused) ----
            t0 = time.perf_counter()
            self.fista_prg.extrapolate(
                q, (total_x,), None,
                x.data, x_old.data, y.data,
                np.float32(beta),
                np.int32(total_x)
            )
            q.finish()
            t_extrap += time.perf_counter() - t0

            # ---- x_old <- x ----
            self.fista_prg.copy_buf(q, (total_x,), None, x.data, x_old.data, np.int32(total_x))
            # (no finish; next iteration will sync anyway)

            t = t_new

            # ---- diagnostics ----
            if verbose and ((k + 1) % diagnostics_interval == 0 or k == 0 or k == niter - 1):
                t0 = time.perf_counter()

                # f = 0.5 * ||Ax - b||^2 = 0.5 * ||r||^2
                # reduction returns 0-dim clarray, .get() pulls scalar only
                r2 = clarray.vdot(r, r).get()  # float
                fval = 0.5 * float(r2)

                gval = 0.0
                if self.prox_kind in ("l1", "nonneg_l1") and self.lam != 0.0:
                    # g = lam * ||x||_1
                    gval = self.lam * float(clarray.sum(clmath.fabs(x)).get())

                obj = fval + gval

                # relative change ||x - x_old|| / ||x||
                # here x_old already equals x (we copied), so measure via y-x maybe not useful.
                # Instead, track norm(x) and norm(grad) as diagnostics:
                xnorm = float(np.sqrt(clarray.vdot(x, x).get()))
                gnorm = float(np.sqrt(clarray.vdot(grad, grad).get()))


                t_obj += time.perf_counter() - t0

                print(
                    f"[iter {k+1:4d}/{niter}] "
                    f"obj={obj:.6e}  f={fval:.6e}  g={gval:.6e}  "
                    f"||x||={xnorm:.6e}  ||grad||={gnorm:.6e}  "
                    f"tau={self.tau:.3e}  beta={beta:.3e}"
                )

        total_time = time.perf_counter() - t_total_start

        if verbose:
            print("\n=== FISTA(OpenCL) TIMING SUMMARY ===")
            print(f"forward (A) total     : {t_forward:.4f} s")
            print(f"residual total        : {t_residual:.4f} s")
            print(f"adjoint (A*) total    : {t_adjoint:.4f} s")
            print(f"grad step total       : {t_gradstep:.4f} s")
            print(f"prox total            : {t_prox:.4f} s")
            print(f"extrap total          : {t_extrap:.4f} s")
            print(f"diagnostics total     : {t_obj:.4f} s")
            print("-----------------------------------")
            print(f"TOTAL                 : {total_time:.4f} s")
            print("===================================\n")

        return x
