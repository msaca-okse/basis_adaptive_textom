# Adaptive basis texture tomography of aluminum sample

This repository accompanies the manuscript:

"Bridging powder and multi-crystal diffraction with basis-adaptive texture tomography"
Martin Sæbye Carøe, Mads Allerup Carlsen, Felix Tristan Frankus, Adam André William Cretton, 
Michela La Bella, Innokentiy Kantor, Mads Ry Vogel Jørgensen, Henning Friis Poulsen, 
Jakob Sauer Jørgensen, Nils Axel Henningsson

## Overview

This code does texture tomography reconstructions using the library *diffractom* https://doi.org/10.5281/zenodo.20431767 of aluminum sample data
measure at DanMAX at MAX-IV. 

## Installation

Clone the repository:

```bash
git clone https://github.com/msaca-okse/texture_tomography.git
cd texture_tomography
```

## Running the code
In order to run the reconstructions, a config file must be created. This config file includes information about paths for cif and poni files,
paths for where the scripts save reconstructions and intermediate steps, parameters used in integration, and parameters used i nthe reconstructoin algorithm.
Examples of config files can be found in the configs folder.

Once the config has been set, the user can run the aximuthal binning script as follows:
```bash
python integration/integrate_Al_data.py --config configs/aluminum_config.yaml
```
Reconstructions can be made by running the script

```bash
python scripts/textomo2.py --config configs/aluminum_config.yaml
```

The reconstructions can be visualized by running scripts and notebooks found in the folder post_reconstruction_analysis.

## Data

The dataset required for running this code will be made available soon. A link will be added here.

## Citation

If you use this repository, please also cite the underlying software it builds on:

Carøe, Martin Sæbye (2026). *diffractom*. Zenodo. https://doi.org/10.5281/zenodo.20431767
