#!/bin/bash
#BSUB -q hpc
#BSUB -J reintegration
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -oo log/%J.out
#BSUB -eo error/%J.err

# ensure log directories exist (relative to this .sh file)
mkdir -p log error

# initialize conda
export PATH="/zhome/71/c/146676/miniconda3/bin:$PATH"
source /zhome/71/c/146676/miniconda3/etc/profile.d/conda.sh
conda activate
conda activate cil


python ../scripts/maxiv_reintegrate_data.py
