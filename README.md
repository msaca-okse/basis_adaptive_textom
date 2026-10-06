# Basis-adaptive texture tomography of an Al1050 sample

Reconstruction of the orientation distribution in every voxel of a slice through an aluminium (Al1050)
sample, measured by scanning 3DXRD at the DanMAX beamline, MAX IV. The orientation distributions are
expanded in an adaptive basis, built from the orientations found by point-by-point indexing of the same
data, and fitted to the azimuthally integrated diffraction data with
[diffractom](https://doi.org/10.5281/zenodo.23185512) (version 0.2.0).

## Installation

This code was written for **diffractom version 0.2.0** (https://doi.org/10.5281/zenodo.23185512) and needs
exactly that release: later versions of diffractom may change its interface. Install the release from its tag
(not from the main branch), with the conda environment it ships, which pins the tested versions of its
dependencies:

```bash
git clone --branch v0.2.0 https://github.com/msaca-okse/diffractom.git
cd diffractom
conda env create -f environment.yml
conda activate diffractom
python -m pip install -e .
cd ..
```

The same release can also be downloaded as an archive from the Zenodo record above. Check the installed
version with

```bash
python -c "import importlib.metadata as m; print(m.version('diffractom'))"   # 0.2.0
```

Then, in the same (activated) environment, clone this repository and install the packages its notebooks need:

```bash
git clone https://github.com/msaca-okse/basis_adaptive_textom.git
cd basis_adaptive_textom
python -m pip install -r requirements.txt
python -m pip install -e maptools   # helpers for the peak segmentation and indexing
```

The integration and the reconstructions run on a GPU through OpenCL.

## Data

The dataset is published by MAX IV: *Scanning 3DXRD dataset of tensile aluminum specimen*, PID
`20.500.14080/ff73157c-3675-46da-9bd4-bf91679551b3`,
https://scicat.maxiv.lu.se/datasets/20.500.14080%2Fff73157c-3675-46da-9bd4-bf91679551b3.
Set its location as `ROOT` in `integration/frame_loader.py` and in `maptools/maptools/paths.py`.

## Pipeline

The steps are notebooks, run from the repository root in this order. Each notebook documents its
parameter choices.

1. **Peak segmentation**, `adaptive_basis/segment_peaks/01-04`: choose the segmentation parameters,
   segment the diffraction peaks in every frame, check the result and merge the peaks into one table.
2. **Indexing and adaptive basis**, `adaptive_basis/indexing/01-04`: find the centre of rotation, index
   the slice point by point, refine the map, and extract the adaptive basis, `adaptive_basis/basis.npy`.
   The basis is included in the repository, so steps 1 and 2 can be skipped.
3. **Integration**, `integration/01_inspect_integrate.ipynb`: choose the powder rings and integrate every
   frame azimuthally around them, with polarization correction. The result, `I[translation, omega, eta,
   ring]`, is read with `integration/integrated_data.py`. Dataset-specific reading is in
   `integration/frame_loader.py`.
4. **Texture tomography**, `texture_tomography/textomo_adaptive.ipynb`: reconstruction with the adaptive
   basis; `textomo_uniform.ipynb`: the same with a uniform orientation grid, for comparison.
5. **Figures**, `visualization/paper_figures.ipynb`: IPF maps, kernel average misorientation, and the
   orientation distribution in single voxels.

## Citation

If you use this code, please cite the article it accompanies:

> Martin Sæbye Carøe, Mads Allerup Carlsen, Felix Tristan Frankus, Adam André William Cretton,
> Michela La Bella, Innokentiy Kantor, Mads Ry Vogel Jørgensen, Henning Friis Poulsen,
> Jakob Sauer Jørgensen, Nils Axel Henningsson. "Bridging powder and multi-crystal diffraction with
> basis-adaptive texture tomography". In preparation.

The data: *Scanning 3DXRD dataset of tensile aluminum specimen*, MAX IV, PID
`20.500.14080/ff73157c-3675-46da-9bd4-bf91679551b3`. The texture tomography library: diffractom v0.2.0,
https://doi.org/10.5281/zenodo.23185512.

## License

The code in this repository is licensed under the Apache License 2.0 (see `LICENSE`).
