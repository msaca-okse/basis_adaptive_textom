"""Fits for the eta-omega figures of every ring (appendix of the article): the reconstructions of
texture_tomography/textomo_{adaptive,uniform}.ipynb fitted to all rings, and again with one ring left out
(its weights set to zero), to show how well the model predicts a ring it was not fitted to.

    python visualization/ring_holdout/fit_rings.py                 every task not done yet
    python visualization/ring_holdout/fit_rings.py --tasks uniform_wo02,adaptive_all
    python visualization/ring_holdout/fit_rings.py --list

Tasks: adaptive_all, adaptive_wo00 ... adaptive_wo13 (every ring left out in turn), uniform_all, and
uniform_wo02 / uniform_wo09 (220 and 333+511 left out). One adaptive fit takes about 17 min and one uniform fit
about an hour on an A100 or L40S GPU, so the tasks are claimed with a lock directory each: several processes
(e.g. one per GPU or cluster job) can work through the list at the same time. Re-running skips finished tasks.

The problem and the options are those of the two notebooks: the data reduced to 500 omega bins of 6 frames,
the ring sums equalised, zero weight at |eta| within 10 deg of +-90 deg and in empty eta bins; the adaptive
basis rotated by Rz(+50 deg), sigma 0.4 deg, thinned at 0.1 deg (K 48 017); the uniform grid 500 000 random
orientations (seed 4200), sigma 2 deg, thinned at 1.5 deg (K 82 198); coefficients streamed from host memory;
L = 1.1 x 6 streamed power iterations (seed 0, once per basis, the same L for every task of a basis);
FISTA-Huber, delta 5.3e4, non-negative, 200 iterations from zero.

Outputs in PROCESS/ring_holdout/: cache/ (the reduced data, 1.5 GB, rebuilt when missing), and per task
<task>.npz (predicted eta-omega images summed over translations and at the translation through the rotation
axis), <task>.json (residual and correlations of every ring) and <task>_coeffs.h5 (the coefficients,
(K, Ny, Nx)). The figures are made by make_figures.py.
"""
import argparse
import gc
import importlib.metadata
import json
import os
import socket
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from diffractom import FISTAHuber, Grid, Material, SinglePhaseForwardOperator, estimate_L_power_streamed  # noqa: E402
from diffractom.operators.single_phase_forward_operator import group_reflections_into_rings  # noqa: E402
from integration.frame_loader import OUTPUT_PATH, PROCESS  # noqa: E402

OUT = os.path.join(PROCESS, "ring_holdout")
CACHE = os.path.join(OUT, "cache")
LOCKS = os.path.join(OUT, "locks")

N_FRAMES = 3000  # one full turn
OMEGA_BIN = 6  # frames per omega bin
OMEGA_OFFSET_DEG = 50.0
NITER = 200
HUBER_DELTA = 5.3e4
N_RINGS = 14
UNIFORM_HOLDOUT = [2, 9]  # 220 and 333+511 (the first double ring)
GRIDS = {"adaptive": dict(sigma_deg=0.4, theta_deg=0.1), "uniform": dict(sigma_deg=2.0, theta_deg=1.5)}


def diffractom_version():
    """The installed version (set DIFFRACTOM_VERSION when diffractom is imported from a source tree instead)."""
    return os.environ.get("DIFFRACTOM_VERSION") or importlib.metadata.version("diffractom")


def all_tasks():
    return (["adaptive_all"] + [f"adaptive_wo{r:02d}" for r in range(N_RINGS)]
            + ["uniform_all"] + [f"uniform_wo{r:02d}" for r in UNIFORM_HOLDOUT])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- data, reduced as in the notebooks
def prepare_cache():
    done = os.path.join(CACHE, "done")
    if os.path.isfile(done):
        return
    try:
        os.mkdir(os.path.join(CACHE, "lock"))
    except FileExistsError:
        log("another process reduces the data; waiting")
        while not os.path.isfile(done):
            time.sleep(15)
        return
    from integration.integrated_data import IntegratedData

    t0 = time.perf_counter()
    data = IntegratedData(OUTPUT_PATH)  # polarization-corrected integration
    assert data.polarization_corrected is True
    positions = np.round(data.translations, 4)
    unique_positions = np.unique(positions)  # overlapping scan files: pairs of positions are averaged
    n_omega = N_FRAMES // OMEGA_BIN
    reduced = np.zeros((len(unique_positions), n_omega, data.n_eta, data.n_rings), dtype=np.float32)
    count = np.zeros(len(unique_positions))
    for t in range(data.n_translations):
        u = np.searchsorted(unique_positions, positions[t])
        reduced[u] += data.I[t, :N_FRAMES].reshape(n_omega, OMEGA_BIN, data.n_eta, data.n_rings).sum(axis=1)
        count[u] += 1
    reduced /= count[:, None, None, None].astype(np.float32)
    arr = np.ascontiguousarray(reduced.transpose(1, 0, 2, 3))  # (N_Omega, My, N_eta, N_rings)
    del reduced
    ring_sums = arr.sum(axis=(0, 1, 2), dtype=np.float64)
    arr *= (ring_sums.sum() / ring_sums).astype(np.float32)  # every ring the same total intensity
    np.save(os.path.join(CACHE, "arr.npy"), arr)
    rotation_axis_mm = float(np.load(os.path.join(PROCESS, "al1050_15pct_center_slice", "rotation_axis_com.npy")))
    meta = {
        "data": OUTPUT_PATH,
        "omega_step": float(np.median(np.diff(data.omega_deg))),
        "omega_bin_deg": data.omega_deg[:N_FRAMES].reshape(n_omega, OMEGA_BIN).mean(axis=1).tolist(),
        "eta_deg": np.asarray(data.eta_deg).tolist(),
        "unique_positions": unique_positions.tolist(),
        "rotation_axis_mm": rotation_axis_mm,
        "translation_step_mm": float(np.median(np.diff(unique_positions))),
        "wavelength_A": float(data.wavelength_A),
        "lattice": data.material["lattice_params"],
        "hkl_list": [list(map(int, hkl)) for ring in data.rings for hkl in ring["hkl"]],
        "two_theta_rad": np.asarray(data.two_theta_rad).tolist(),
        "ring_labels": ["+".join("".join(str(abs(int(v))) for v in hkl) for hkl in ring) for ring in data.hkl],
    }
    with open(os.path.join(CACHE, "meta.json"), "w") as f:
        json.dump(meta, f)
    open(done, "w").close()
    log(f"data reduced and cached in {time.perf_counter() - t0:.0f} s")


def fitted_bins(arr, eta):
    """(N_eta, N_rings) weights of the notebooks: zero near the rotation axis and in empty eta bins."""
    weights = np.ones(arr.shape[2:], dtype=np.float32)
    weights[np.abs(np.abs(eta) - 90) < 10, :] = 0
    profile = np.asarray(arr.mean(axis=(0, 1)))
    weights[profile < 0.05 * np.median(profile, axis=0, keepdims=True)] = 0
    return weights


def load_problem(mmap=False):
    arr = np.load(os.path.join(CACHE, "arr.npy"), mmap_mode="r" if mmap else None)
    with open(os.path.join(CACHE, "meta.json")) as f:
        meta = json.load(f)
    N_Omega, My, N_eta, N_theta = arr.shape
    weights = fitted_bins(arr, np.array(meta["eta_deg"]))
    cfg = {
        "energy": 12.398 / meta["wavelength_A"],
        "Nx": 115,
        "Ny": 115,
        "My": My,
        "N_Omega": N_Omega,
        "N_eta": N_eta,
        "angle_range": [-OMEGA_OFFSET_DEG, N_FRAMES * meta["omega_step"] - OMEGA_OFFSET_DEG],
        "cor_offset": -meta["rotation_axis_mm"] / meta["translation_step_mm"],
        "eta_angle_range": [0, 360],
        "j_direction_0": [0, -1, 0],
        "k_direction_0": [0, 0, 1],
        "p_direction_0": [1, 0, 0],
        "detector_direction_origin": [0, 1, 0],
        "detector_direction_positive_90": [0, 0, 1],
    }
    lat = meta["lattice"]
    mat = Material.from_lattice_parameters(
        a=lat["a"], b=lat["b"], c=lat["c"], alpha=lat["alpha"], beta=lat["beta"], gamma=lat["gamma"],
        symmetry_group="cubic", wavelength_A=meta["wavelength_A"], min_two_theta=0.0, max_two_theta=0.0,
        hkl_list=np.array(meta["hkl_list"]))
    rings = group_reflections_into_rings(mat)
    ring_two_theta = np.array([mat.reflections["two_theta"][r[0]] for r in rings])
    assert len(rings) == N_theta and np.allclose(ring_two_theta, meta["two_theta_rad"], rtol=1e-4)
    i_center = int(np.argmin(np.abs(np.array(meta["unique_positions"]) - meta["rotation_axis_mm"])))
    return arr, weights, cfg, mat, meta, i_center


def make_grid(name):
    p = GRIDS[name]
    if name == "adaptive":
        basis = np.load(os.path.join(REPO, "adaptive_basis", "basis.npy"))
        basis = R.from_euler("z", OMEGA_OFFSET_DEG, degrees=True).as_matrix() @ basis
        grid = Grid.from_rotation_matrices(basis, np.deg2rad(p["sigma_deg"]))
    else:
        np.random.seed(4200)
        grid = Grid.from_random_fundamental_zone(500000, "cubic", np.deg2rad(p["sigma_deg"]))
    grid.prune_close_orientations(theta_deg=p["theta_deg"])
    return grid


# ---------------------------------------------------------------- tasks
def claim(task):
    if os.path.isfile(os.path.join(OUT, task + ".done")):
        return False
    try:
        os.mkdir(os.path.join(LOCKS, task))
    except FileExistsError:  # another process works on it (remove the lock directory if that one died)
        return False
    with open(os.path.join(LOCKS, task, "owner"), "w") as f:
        f.write(f"{os.environ.get('LSB_JOBID', '?')} {socket.gethostname()} {time.ctime()}\n")
    return True


def scores(arr, pred, fitted, weights, i_center):
    """Per ring: relative residual and correlation over every fitted bin, and of the eta-omega images."""
    out = []
    for r in range(arr.shape[3]):
        m = fitted[:, r]
        d = arr[:, :, m, r].astype(np.float64)
        p = pred[:, :, m, r].astype(np.float64)
        ds, ps = d.sum(axis=1), p.sum(axis=1)
        dc, pc = d[:, i_center], p[:, i_center]
        out.append({
            "rel_residual": float(np.linalg.norm(d - p) / np.linalg.norm(d)),
            "corr_bins": float(np.corrcoef(d.ravel(), p.ravel())[0, 1]),
            "corr_summed": float(np.corrcoef(ds.ravel(), ps.ravel())[0, 1]),
            "corr_center": float(np.corrcoef(dc.ravel(), pc.ravel())[0, 1]),
            "rel_residual_summed": float(np.linalg.norm(ds - ps) / np.linalg.norm(ds)),
            "rel_residual_center": float(np.linalg.norm(dc - pc) / np.linalg.norm(dc)),
            "fitted": bool(weights[:, r].any()),
        })
    return out


def run_task(task, op, L, arr, weights_base, meta, i_center):
    grid_name, which = task.split("_")
    w = weights_base.copy()
    left_out = None
    if which.startswith("wo"):
        left_out = int(which[2:])
        w[:, left_out] = 0
    N_Omega, My, N_eta, N_theta = arr.shape
    b = arr.reshape(N_Omega, My, N_eta * N_theta)
    x = np.zeros(op.coeff_shape, dtype=np.float32)  # host memory, streamed through the GPU (as the notebooks)
    solver = FISTAHuber(op, prox_kind="nonneg", L=L, huber_delta=HUBER_DELTA)
    t0 = time.perf_counter()
    x = solver.run(x, b, niter=NITER, weights=w, verbose=1, diagnostics_interval=10)
    fista_time = time.perf_counter() - t0
    pred = op.direct(x).reshape(N_Omega, My, N_eta, N_theta)
    del solver
    sc = scores(arr, pred, weights_base > 0, w, i_center)
    wsel = (w > 0)[None, None]
    res = float(np.sqrt(np.sum(((arr - pred) * wsel) ** 2, dtype=np.float64)
                        / np.sum((arr * wsel) ** 2, dtype=np.float64)))
    np.savez(os.path.join(OUT, task + ".npz"),
             summed=pred.sum(axis=1, dtype=np.float64).astype(np.float32),  # (N_Omega, N_eta, N_rings)
             center=np.ascontiguousarray(pred[:, i_center]),                # (N_Omega, N_eta, N_rings)
             i_center=i_center)
    info = {"task": task, "grid": grid_name, "left_out": left_out, "K": int(op.coeff_shape[0]), "L": L,
            "niter": NITER, "huber_delta": HUBER_DELTA, "fista_time_s": fista_time,
            "rel_residual_fitted": res, "rings": sc, "diffractom": diffractom_version(),
            "host": socket.gethostname(), "job": os.environ.get("LSB_JOBID")}
    with open(os.path.join(OUT, task + ".json"), "w") as f:
        json.dump(info, f, indent=1)
    with h5py.File(os.path.join(OUT, task + "_coeffs.h5"), "w") as f:
        ds = f.create_dataset("coeffs", data=x, compression="gzip", compression_opts=1)
        ds.attrs["axes"] = "(K, Ny, Nx)"
        f.attrs["grid"] = grid_name
        f.attrs["left_out_ring"] = -1 if left_out is None else left_out
        f.attrs["sigma_deg"] = GRIDS[grid_name]["sigma_deg"]
        f.attrs["theta_deg"] = GRIDS[grid_name]["theta_deg"]
        f.attrs["niter"] = NITER
        f.attrs["huber_delta"] = HUBER_DELTA
    open(os.path.join(OUT, task + ".done"), "w").close()
    lo = "" if left_out is None else (f", left-out ring {meta['ring_labels'][left_out]}: rel residual "
                                      f"{sc[left_out]['rel_residual']:.3f}, corr summed {sc[left_out]['corr_summed']:.3f}")
    log(f"{task} done: FISTA {fista_time:.0f} s, fitted rel residual {res:.4f}{lo}")
    del x, pred
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default=None, help="comma-separated tasks (default: all)")
    parser.add_argument("--list", action="store_true", help="list the tasks and whether they are done")
    args = parser.parse_args()
    tasks = all_tasks() if args.tasks is None else args.tasks.split(",")
    unknown = set(tasks) - set(all_tasks())
    if unknown:
        raise SystemExit(f"unknown tasks: {sorted(unknown)}")
    if args.list:
        for t in tasks:
            print(t, "done" if os.path.isfile(os.path.join(OUT, t + ".done")) else "")
        return
    for d in (CACHE, LOCKS):
        os.makedirs(d, exist_ok=True)
    log(f"diffractom {diffractom_version()}, {socket.gethostname()}, tasks {tasks}")
    prepare_cache()
    arr, weights_base, cfg, mat, meta, i_center = load_problem()
    log(f"data {arr.shape}, centre translation {i_center}")
    op, op_grid, L = None, None, None
    for task in tasks:
        if not claim(task):
            continue
        grid_name = task.split("_")[0]
        if grid_name != op_grid:
            if op is not None:
                op.release_streamer()
                del op
                gc.collect()
            t0 = time.perf_counter()
            op = SinglePhaseForwardOperator(cfg=cfg, material=mat, grid=make_grid(grid_name), max_gb=1.0,
                                            verbose=True, normalized=True, reserve_coefficient_arrays=0)
            op_grid = grid_name
            log(f"{grid_name}: K = {op.coeff_shape[0]}, operator built in {time.perf_counter() - t0:.0f} s")
            lpath = os.path.join(OUT, f"L_{grid_name}.json")
            if os.path.isfile(lpath):
                with open(lpath) as f:
                    L = json.load(f)["L"]
            else:
                L = 1.1 * float(estimate_L_power_streamed(op, niter=6, seed=0, verbose=1))
                with open(lpath, "w") as f:
                    json.dump({"L": L, "diffractom": diffractom_version()}, f)
            log(f"L = {L:.4e}")
        log(f"start {task}")
        run_task(task, op, L, arr, weights_base, meta, i_center)
    log("finished")


if __name__ == "__main__":
    main()
