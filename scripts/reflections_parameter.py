import os
import numpy as np
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import h5py
import argparse
import yaml


parser = argparse.ArgumentParser(description="Integrate diffraction projections using configuration file.")
parser.add_argument('--config', required=True, help='Path to YAML configuration file')
args = parser.parse_args()

# Resolve to absolute path if relative
config_path = args.config
if not os.path.isabs(config_path):
    # Resolve relative to the current script’s directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

wavelength_kev = cfg['wavelength']
cutoff_paramter = cfg['reflection_cufoff']
pad = cfg['pad']
center = cfg['center']
pixel_size = cfg['pixel_size']
detector_distance = cfg['detector_distance']
min_two_theta = cfg['min_two_theta']
cif_path = cfg['cif_path']
h5_filename = cfg['reflections_folder']

wavelength = 12.398/wavelength_kev

max_rad = 1475 // 2 + center[1] + pad
r_max = pixel_size * max_rad
max_two_theta = np.sin(r_max / detector_distance)

# Directory with CIF files

calc = XRDCalculator(wavelength=wavelength)
cif_files = sorted([f for f in os.listdir(cif_path) if f.endswith(".cif")])

materials = []

for file in cif_files:
    full_path = os.path.join(cif_path, file)
    print("=" * 100)
    print(f"Material: {file}")

    try:
        atoms = read(full_path)
        structure = AseAtomsAdaptor.get_structure(atoms)

        # Symmetry
        sga = SpacegroupAnalyzer(structure)
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
        crystal_system = sga.get_crystal_system()

        print(f"  Space group: {sg_symbol} (No. {sg_number})")
        print(f"  Crystal system: {crystal_system}\n")

        # Compute normalized pattern
        pattern = calc.get_pattern(structure, two_theta_range=(min_two_theta*180/np.pi, max_two_theta*180/np.pi))
        y_norm = np.array(pattern.y)

        # Undo pymatgen normalization by approximate scattering power
        Z2_sum = sum(site.specie.Z**2 for site in structure)
        y_abs = y_norm * Z2_sum

        materials.append({
            "file": file,
            "sg": sg_symbol,
            "num": sg_number,
            "system": crystal_system,
            "pattern": pattern,
            "structure": structure,
            "y_abs": y_abs,
        })

    except Exception as e:
        print(f"Error processing {file}: {e}\n")

# Global normalization
global_max = max(np.max(m["y_abs"]) for m in materials)

# Output folder
out_dir = os.path.join(cif_path, h5_filename)
os.makedirs(out_dir, exist_ok=True)

for m in materials:
    pattern = m["pattern"]
    structure = m["structure"]
    y_global = m["y_abs"] / global_max * 100

    # Intensity cutoff (1% of max intensity for this material)
    intensity_cutoff = cutoff_paramter * np.max(y_global)

    # Lattice parameters
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

    hkl_list = []
    intensity_list = []
    two_theta_list = []
    d_list = []
    multiplicity_list = []

    for i, peak in enumerate(pattern.hkls):
        d_val = float(pattern.d_hkls[i])
        twotheta = float(pattern.x[i])
        total_intensity = float(y_global[i])

        # Skip weak reflections
        if total_intensity < intensity_cutoff:
            continue

        for refl in peak:
            hkl = refl.get("hkl", (0, 0, 0))
            if not isinstance(hkl, (list, tuple)):
                hkl = (0, 0, 0)
            hkl = tuple(int(x) for x in hkl)
            mult = int(refl.get("multiplicity", 1))

            hkl_list.append(hkl)
            intensity_list.append(total_intensity)
            two_theta_list.append(twotheta)
            d_list.append(d_val)
            multiplicity_list.append(mult)

    # Skip materials where all reflections were too weak
    if not hkl_list:
        print(f"Skipped {m['file']} — all reflections below cutoff.")
        continue

    # Output filename
    base_name = os.path.splitext(m["file"])[0]
    out_file = os.path.join(out_dir, f"{base_name}.h5")

    # Save to HDF5
    with h5py.File(out_file, "w") as f:
        f.create_dataset("hkl_list", data=np.array(hkl_list, dtype=int))
        f.create_dataset("intensity_list", data=np.array(intensity_list, dtype=float))
        f.create_dataset("two_theta_list", data=np.array(two_theta_list, dtype=float))
        f.create_dataset("d_list", data=np.array(d_list, dtype=float))
        f.create_dataset("multiplicity_list", data=np.array(multiplicity_list, dtype=int))

        f.attrs["space_group"] = f"{m['sg']} (No. {m['num']}), {m['system']}"
        f.attrs["a"] = a
        f.attrs["b"] = b
        f.attrs["c"] = c
        f.attrs["alpha"] = alpha
        f.attrs["beta"] = beta
        f.attrs["gamma"] = gamma

    print(f"Saved: {out_file}")