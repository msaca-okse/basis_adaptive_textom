# Adaptive basis texture tomography of aluminum sample

This repository accompanies the manuscript:

"Bridging powder and multi-crystal diffraction with basis-adaptive texture tomography"
Martin Sæbye Carøe, Mads Allerup Carlsen, Felix Tristan Frankus, Adam André William Cretton, 
Michela La Bella, Innokentiy Kantor, Mads Ry Vogel Jørgensen, Henning Friis Poulsen, 
Jakob Sauer Jørgensen, Nils Axel Henningsson

## Overview

This code does texture tomography reconstructions using the library *diffractom* https://doi.org/10.5281/zenodo.20431767 of aluminum sample data
measured at DanMAX at MAX-IV. 

## Installation

Install the "diffractom" library, by following the instructions on "https://github.com/msaca-okse/diffractom". In the process, you will create a conda environment. Now clone this repository and install required dependencies inside the conda environment:

```bash
git clone https://github.com/msaca-okse/basis_adaptive_textom.git
cd basis_adaptive_textom
python -m pip install -r requirements.txt
```

If you want to run the full peak segmentation + indexing pipeline, you will need to install the maptools package found in the repo via pip:
```bash
cd maptools
python -m pip install .
```



## Running the code

### Step 1
To run the code, you will need to download the dataset. This can be found at: [Link to published dataset].

### Step 2
This step is for running the peak segmentation+indexing part of the pipeline. Running the peak segmentation+indexing code produces the file "adaptive_basis.h5", which can be found in the repo. You can skip this step and go to step 3.
Alternatively, in order to run the segmentation-indexing notebooks, you will need to configurate the paths "ROOT" and "CODE" in "maptools/maptools/paths" to set the absolute paths to the data folder and the code repository. Then run the notebooks found in the "adaptive_basis" folder in order, starting with "segment_peaks", then "indexing". These notebooks contain the parameters used for extracting the adaptive basis that was used for the reconstruction in the article.

### Step 3
In order to run the reconstructions, a config file must be created. This config file includes information about paths for cif and poni files, which can be found with the dataset. You must also set paths for where reconstructions and intermediate steps are saved. The config file contains the parameters used for integration, and parameters used in the reconstruction algorithm.
An example config file can be found in the configs folder.

Once the config has been set, you can run the azimuthal binning/integration script as follows:
```bash
python integration/integrate_Al_data.py --config configs/aluminum_config.yaml
```


Reconstructions can be made by running the script

```bash
python scripts/textomo_adaptive.py --config configs/aluminum_config.yaml
```

The reconstructions can be visualized by running scripts and notebooks found in the folder post_reconstruction_analysis.

## Data

The dataset required for running this code will be made available soon. A link will be added here.

## Citation

If you use this repository, please also cite the underlying software it builds on:

Carøe, Martin Sæbye (2026). *diffractom*. Zenodo. https://doi.org/10.5281/zenodo.20431767
