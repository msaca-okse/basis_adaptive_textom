#!/bin/bash
#BSUB -J Integration
#BSUB -q gpuqimalma94
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2000]"
#BSUB -M 2000
#BSUB -W 15:00
#BSUB -oo log/%J.out
#BSUB -eo error/%J.err

# -- end of LSF options -- 

# initialize conda
export PATH="/zhome/71/c/146676/miniconda3/bin:$PATH"
source /zhome/71/c/146676/miniconda3/etc/profile.d/conda.sh
conda activate
conda activate cil

python ../integration/integrate_Al_data.py --config ../configs/aluminum_config_single_large.yaml