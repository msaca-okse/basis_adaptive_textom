# Basis-adaptive texture tomography of an Al1050 sample

Reconstruction of the orientation distribution in every voxel of a slice through an aluminium (Al1050)
sample, measured by scanning 3DXRD at the DanMAX beamline, MAX IV. The orientation distributions are
expanded in an adaptive basis, built from the orientations found by point-by-point indexing of the same
data, and fitted to the azimuthally integrated diffraction data with
[diffractom](https://doi.org/10.5281/zenodo.23920227) (version 0.3.0).

## Installation

This code was written for **diffractom version 0.3.0** (https://doi.org/10.5281/zenodo.23920227) and needs
exactly that release: later versions of diffractom may change its interface. Install the release from its tag
(not from the main branch), with the conda environment it ships, which pins the tested versions of its
dependencies:

```bash
git clone --branch v0.3.0 https://github.com/msaca-okse/diffractom.git
cd diffractom
conda env create -f environment.yml
conda activate diffractom
python -m pip install -e .
cd ..
```

The same release can also be downloaded as an archive from the Zenodo record above. Check the installed
version with

```bash
python -c "import importlib.metadata as m; print(m.version('diffractom'))"   # 0.3.0
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
   orientation distribution in single voxels (`GRID = "uniform"` in its first code cell makes the same
   figures for the uniform grid). `visualization/data_misfit.ipynb`: the data against the predictions of
   both reconstructions in measurement coordinates (eta-omega images of the rings, sinograms, agreement per
   ring).
6. **Rings left out**, `visualization/ring_holdout/`: `fit_rings.py` repeats both reconstructions fitted to
   all rings and with one ring left out at a time (every ring for the adaptive basis; 220 and 333+511 for the
   uniform grid), `make_figures.py` makes the eta-omega figures of every ring and the appendix figure of the
   article from them. One adaptive fit takes about 17 minutes and one uniform fit about an hour on a 40 GB
   GPU; the script can run in several processes at once (see its docstring).

The rotation angles of the peaks: the segmentation labels every frame with its motor position `tom_ry`,
which is the start of the frame's rotation interval, not its centre. The peak positions in omega, the indexed
orientations and therefore the adaptive basis are rotated by half a frame (0.06 deg) about the rotation axis.
The integration and the texture tomography use the frame centres, so the data are not affected; the basis is
kept as it was used in the article.

## Changes in version 2.1.0

* diffractom 0.3.0 (was 0.2.0). The pole figures of the basis functions are integrated over each
  (omega, eta) bin by Gauss-Legendre quadrature with as few nodes as keep an error bound below 5 % of the
  peak (`quadrature="auto"`; 2 x 2 nodes per bin for the adaptive basis, one for the uniform grid) instead of
  being evaluated at the bin centre; the Gaussians are cut at 4 sigma (was 3); the tomographic projection
  uses the pixel footprint (trapezoid) kernel.
* `textomo_adaptive`, `textomo_uniform`, `paper_figures` and `data_misfit` were run again with it. The
  misorientation histogram of `paper_figures` is taken at pixel (73, 53) (was (75, 55)).
* New: `visualization/ring_holdout/` (the appendix figure with rings left out of the fit). Its fits for the
  article were made with diffractom main (90ec767, after 0.3.0); fitted to all rings they agree with the 0.3.0
  reconstructions of the notebooks to 2e-6 (relative difference of the coefficients).

## Citation

If you use this code, please cite the article it accompanies:

> Martin Sæbye Carøe, Mads Allerup Carlsen, Felix Tristan Frankus, Adam André William Cretton,
> Michela La Bella, Innokentiy Kantor, Mads Ry Vogel Jørgensen, Henning Friis Poulsen,
> Jakob Sauer Jørgensen, Nils Axel Henningsson. "Bridging powder and multi-crystal diffraction with
> basis-adaptive texture tomography". In preparation.

The data: *Scanning 3DXRD dataset of tensile aluminum specimen*, MAX IV, PID
`20.500.14080/ff73157c-3675-46da-9bd4-bf91679551b3`. The texture tomography library: diffractom v0.3.0,
https://doi.org/10.5281/zenodo.23920227.

## License

The code in this repository is licensed under the Apache License 2.0 (see `LICENSE`).
