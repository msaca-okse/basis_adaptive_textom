#!/bin/bash
#BSUB -J Integration
#BSUB -q gpua100
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000]"
#BSUB -M 4000
#BSUB -W 15:00
#BSUB -oo log/%J.out
#BSUB -eo error/%J.err

# -- end of LSF options -- 

# initialize conda
export PATH="/zhome/71/c/146676/miniconda3/bin:$PATH"
source /zhome/71/c/146676/miniconda3/etc/profile.d/conda.sh
conda activate
conda activate textom

python ../integration/integrate_BB_slice.py --config ../configs/bb_config_1.yaml