# Basis-adaptive texture tomography of an Al1050 sample

Reconstruction of the orientation distribution in every voxel of a slice through an aluminium (Al1050)
sample, measured by scanning 3DXRD at the DanMAX beamline, MAX IV. The orientation distributions are
expanded in an adaptive basis, built from the orientations found by point-by-point indexing of the same
data, and fitted to the azimuthally integrated diffraction data with
[diffractom](https://doi.org/10.5281/zenodo.20431767).

## Installation

Install diffractom following https://github.com/msaca-okse/diffractom, which creates a conda environment.
In that environment:

```bash
git clone https://github.com/msaca-okse/basis_adaptive_textom.git
cd basis_adaptive_textom
python -m pip install -r requirements.txt
python -m pip install -e maptools   # helpers for the peak segmentation and indexing
```

The integration and the reconstructions run on a GPU through OpenCL.

## Data

The dataset will be made available; a link will be added here. Set its location as `ROOT` in
`integration/frame_loader.py` and in `maptools/maptools/paths.py`.

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

## License

See `LICENSE`.
