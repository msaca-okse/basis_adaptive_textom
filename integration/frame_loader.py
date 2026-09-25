import os
import numpy as np
import h5py, hdf5plugin  # noqa: F401  (hdf5plugin registers detector compression filters)
import matplotlib.pyplot as plt

ROOT = "/dtu/3d-imaging-center/projects/2025_QIM_BlackBeauty/raw_data_extern/2025_Danmax_Al1050"
PROCESS = os.path.join(ROOT, "process")
RAW = os.path.join(ROOT, "raw")

OMEGA_MOTOR = "/entry/measurement/tom_ry"
DTY_MOTOR = "/entry/measurement/im_x"
FRAMES = "/entry/measurement/pilatus"
MASKFILE = os.path.join(PROCESS, "mask.npy")
PARAMETERS_AL1050 = os.path.join(PROCESS, "al1050_15pct_center_slice", "al1050.par")
PONI_PATH = os.path.join(PROCESS, "LaB6_34p798keV_244p89mm.poni")

INTEGRATED_FILENAME = "scan-0048-0058_integrated"
INTEGRATED_PATH = os.path.join(RAW, INTEGRATED_FILENAME)

RAW_SCANS = {
    "scan-0048": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0048.h5"),
    "scan-0049": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0049.h5"),
    "scan-0050": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0050.h5"),
    "scan-0051": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0051.h5"),
    "scan-0052": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0052.h5"),
    "scan-0053": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0053.h5"),
    "scan-0054": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0054.h5"),
    "scan-0055": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0055.h5"),
    "scan-0056": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0056.h5"),
    "scan-0057": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0057.h5"),
    "scan-0058": os.path.join(RAW, "al1050_15pct_center_slice", "scan-0058.h5"),
}

# Threshold (in dty motor units) used to detect a translation step between two
# consecutive frames. Must be well above the in-block motor jitter and well
# below the actual step size between translations -- check this against your
# stage's readback noise before trusting the segmentation.
DTY_STEP_TOL = 1e-3


class DanMaxFile:
    """
    One HDF5 file: rotation is the fast (inner) axis, translation the slow
    (outer) axis, so the frame axis is a concatenation of N_trans_file
    contiguous rotation blocks. Block boundaries are found from the actual
    im_x values, not assumed -- the number of translations per file, and even
    the number of rotation steps per block, are read from the data.
    """

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self._file = None
        self._segments = None  # list of (start, stop) frame-index ranges

    def open(self):
        self._file = h5py.File(self.path, "r")
        omega_raw = self._file[OMEGA_MOTOR][()]
        dty_raw = self._file[DTY_MOTOR][()]
        n_det = self._file[FRAMES].shape[0]
        if not (len(omega_raw) == len(dty_raw) == n_det):
            raise ValueError(
                f"{self.name}: frame/motor length mismatch -- "
                f"{n_det} frames, {len(omega_raw)} omega values, "
                f"{len(dty_raw)} dty values."
            )
        self._omega_raw = omega_raw
        self._dty_raw = dty_raw
        self._segments = self._find_translation_segments(dty_raw)
        return self

    @staticmethod
    def _find_translation_segments(dty, tol=DTY_STEP_TOL):
        """Split the frame axis into contiguous blocks of constant dty."""
        jumps = np.where(np.abs(np.diff(dty)) > tol)[0] + 1
        bounds = np.concatenate(([0], jumps, [len(dty)]))
        return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def n_translations(self):
        return len(self._segments)

    def n_omega(self, i_trans_local):
        start, stop = self._segments[i_trans_local]
        return stop - start

    def omega(self, i_trans_local):
        start, stop = self._segments[i_trans_local]
        return self._omega_raw[start:stop]

    def dty(self, i_trans_local):
        start, stop = self._segments[i_trans_local]
        vals = self._dty_raw[start:stop]
        if vals.std() > DTY_STEP_TOL:
            raise ValueError(
                f"{self.name} block {i_trans_local}: im_x varies by "
                f"{vals.std():.4g} within a detected block (tol {DTY_STEP_TOL}). "
                "Segmentation likely picked up motor jitter as a real step -- "
                "check DTY_STEP_TOL against your stage's readback noise."
            )
        return float(vals.mean())

    def frame(self, i_trans_local, i_omega):
        start, _ = self._segments[i_trans_local]
        return self._file[FRAMES][start + i_omega]

    def frames(self, i_trans_local, omega_slice=slice(None)):
        """Lazy, contiguous read of a block (or a slice within it) -- nothing
        is loaded into memory until this is indexed/sliced further."""
        start, stop = self._segments[i_trans_local]
        idx = range(*omega_slice.indices(stop - start))
        return self._file[FRAMES][start + idx.start : start + idx.stop : idx.step]


class DanMaxDataset:
    """
    s3dxrd loader for this experiment. Translation blocks are discovered per
    file from the actual dty readback (see DanMaxFile), then merged across
    all files and sorted by actual translation position into one global
    translation index. Rotation-step count is expected to be constant across
    every block -- this is *checked* against the data on open, not assumed.

        with DanMaxDataset() as ds:
            ds.n_translations
            ds.translations()            # (N_y,) actual dty values, sorted
            ds.omega(i_trans)            # actual tom_ry values for that block
            ds.frames(i_trans)           # lazy (N_omega, My, Mx) view
    """

    def __init__(self, raw_scans=RAW_SCANS):
        self._files = [DanMaxFile(name, path) for name, path in raw_scans.items()]
        self._index = []  # list of (file_idx, local_i_trans), global order

    def __enter__(self):
        for f in self._files:
            f.open()

        entries = []
        for fi, f in enumerate(self._files):
            for li in range(f.n_translations):
                entries.append((fi, li, f.dty(li)))
        entries.sort(key=lambda e: e[2])
        self._index = [(fi, li) for fi, li, _ in entries]

        n_omega_values = {self.n_omega(i) for i in range(self.n_translations)}
        if len(n_omega_values) > 1:
            raise ValueError(
                "Expected the same number of rotation steps in every "
                f"translation block; found varying counts: {sorted(n_omega_values)}. "
                "Either the data genuinely has ragged blocks, or DTY_STEP_TOL "
                "is mis-tuned and segmentation is splitting/merging blocks "
                "incorrectly -- inspect specific files before proceeding."
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        for f in self._files:
            f.close()

    # ---- generic loader interface expected by the notebooks ----

    @property
    def n_translations(self):
        return len(self._index)

    def n_omega(self, i_trans):
        fi, li = self._index[i_trans]
        return self._files[fi].n_omega(li)

    def omega(self, i_trans):
        fi, li = self._index[i_trans]
        return self._files[fi].omega(li)

    def translations(self):
        return np.array([self._files[fi].dty(li) for fi, li in self._index])

    def frame(self, i_trans, i_omega):
        fi, li = self._index[i_trans]
        return self._files[fi].frame(li, i_omega)

    def frames(self, i_trans, omega_slice=slice(None)):
        fi, li = self._index[i_trans]
        return self._files[fi].frames(li, omega_slice)

    def mask(self):
        return np.load(MASKFILE)

    def poni_path(self):
        return PONI_PATH



def dark(fontsize=28):
    plt.style.use("dark_background")
    ticksize = fontsize
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.rcParams["font.family"] = "Nimbus Roman"
