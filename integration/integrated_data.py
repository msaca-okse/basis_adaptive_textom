import json
import os
import numpy as np


class IntegratedData:
    """
    Loader for the output of the ring integration: a .json file with all
    parameters and a .bin file holding the memory-mapped intensity array.

        data = IntegratedData("/path/to/scan-0048-0058_integrated")   # no suffix
        data.I            # np.memmap, shape (n_translations, n_omega, n_eta, n_rings)
        data.q_nm         # (n_rings,) ring centre positions
        data.translations # (n_translations,) dty motor positions
        data.omega_deg    # (n_omega,) shared rotation sequence
        data.eta_deg      # (n_eta,) azimuthal bin centres (see note in the class)
        data.corrections  # dict of applied intensity corrections, {} if not recorded
        data.polarization_corrected  # True / False / None (None = json predates the record)

    The array is opened read-only and lazily -- nothing is loaded into memory
    until you slice it.
    """

    def __init__(self, path_without_suffix, require_complete=True):
        base = str(path_without_suffix)
        # plain string concatenation on purpose: Path.with_suffix would
        # mangle names that already contain a dot
        self.json_path = base + ".json"
        self.bin_path = base + ".bin"

        with open(self.json_path) as f:
            self.params = json.load(f)
        p = self.params

        if require_complete and not p["integration_complete"]:
            raise RuntimeError(
                f"{self.json_path} says integration_complete=False -- the full "
                "integration hasn't finished (or was interrupted), so the .bin "
                "file may be partially filled with zeros. Pass "
                "require_complete=False to open it anyway."
            )

        out = p["output"]
        self.shape = tuple(out["shape"])
        self.dtype = np.dtype(out["dtype"])
        self.axes = out["shape_axes"]

        # check the .bin matches the json, using the *given* path rather than
        # the absolute bin_path stored in the json (files may have been moved)
        expected_bytes = int(np.prod(self.shape)) * self.dtype.itemsize
        actual_bytes = os.path.getsize(self.bin_path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{self.bin_path} is {actual_bytes} bytes but the json implies "
                f"{expected_bytes} bytes (shape {self.shape}, {self.dtype})."
            )

        self.I = np.memmap(self.bin_path, dtype=self.dtype, mode="r", shape=self.shape,
                           order=out.get("memmap_order", "C"))

        # scan info
        scan = p["scan"]
        self.n_translations = scan["n_translations"]
        self.n_omega = scan["n_omega"]
        self.translations = np.array(scan["translations_dty"])
        self.omega_deg = np.array(scan["omega_deg"])

        # ring info
        rings = p["rings"]
        self.n_eta = rings["n_eta"]
        self.q_width_nm = rings["q_width_nm"]
        self.rings = rings["selected"]  # list of dicts: q_nm, hkl, multiplicity, ...
        self.q_nm = np.array([r["q_nm"] for r in self.rings])
        self.hkl = [r["hkl"] for r in self.rings]  # list of lists: several families can share one ring
        self.multiplicity = [r["multiplicity"] for r in self.rings]
        self.d_spacing_A = np.array([r["d_spacing_A"] for r in self.rings])
        self.two_theta_rad = np.array([r["two_theta_rad"] for r in self.rings])

        # azimuthal axis: stored explicitly in the json at integration time
        if "eta_deg" not in rings or "eta_range_deg" not in rings:
            raise KeyError(
                f"{self.json_path} has no 'eta_deg' / 'eta_range_deg' in its "
                "'rings' section (written by an older version of the parameter "
                "cell). If that run used pyFAI's default azimuth range, you can "
                "patch the json rather than re-integrating -- see the notes on "
                "patching an existing json."
            )
        self.eta_range_deg = tuple(rings["eta_range_deg"])
        self.eta_deg = np.array(rings["eta_deg"])
        if len(self.eta_deg) != self.n_eta:
            raise ValueError(
                f"eta_deg has {len(self.eta_deg)} entries but n_eta={self.n_eta}"
            )

        # geometry / material metadata, passed through untouched
        self.geometry = p["geometry"]
        self.wavelength_A = p["geometry"]["wavelength_A"]
        self.material = p["material"]

        # intensity corrections and integration settings; absent in jsons written
        # before they were recorded, which means "unknown", not "not applied"
        self.corrections = p.get("corrections", {})
        self.integration = p.get("integration", {})
        pol = self.corrections.get("polarization")
        self.polarization_corrected = None if pol is None else bool(pol["applied"])

        # consistency checks between the parts of the json
        assert self.I.shape[0] == self.n_translations
        assert self.I.shape[1] == self.n_omega
        assert self.I.shape[2] == self.n_eta
        assert self.I.shape[3] == len(self.rings)

    @property
    def n_rings(self):
        return len(self.rings)

    def summary(self):
        gb = np.prod(self.shape) * self.dtype.itemsize / 1e9
        lines = [
            f"IntegratedData: {self.json_path}",
            f"  array   : {self.shape}  axes={self.axes}  {self.dtype}  ({gb:.2f} GB, memmap)",
            f"  dty     : {self.n_translations} steps, {self.translations.min():.4g} .. {self.translations.max():.4g}",
            f"  omega   : {self.n_omega} steps, {self.omega_deg.min():.4g} .. {self.omega_deg.max():.4g} deg",
            f"  eta     : {self.n_eta} bins, range {self.eta_range_deg[0]:g}..{self.eta_range_deg[1]:g} deg",
            f"  rings   : {self.n_rings}, q_width = {self.q_width_nm} nm^-1",
        ]
        for i, r in enumerate(self.rings):
            hkls = ", ".join(str(tuple(h)) for h in r["hkl"])
            lines.append(f"    [{i:2d}] q = {r['q_nm']:8.3f} nm^-1   hkl: {hkls}")

        if self.corrections:
            lines.append("  corrections:")
            for name, c in self.corrections.items():
                extra = {k: v for k, v in c.items() if k not in ("applied", "note")}
                lines.append(f"    {name:12s} applied={c.get('applied')}  {extra if extra else c.get('note', '')}")
        else:
            lines.append("  corrections: not recorded (json predates it) -- polarization_corrected=None (unknown)")
        if self.integration:
            lines.append(f"  integration: {self.integration}")
        else:
            lines.append("  integration: settings not recorded")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

    def close(self):
        # drop the reference so the OS can release the mapping
        self.I = None