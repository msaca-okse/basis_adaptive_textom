#!/bin/bash
#BSUB -J TextureTomography
#BSUB -q gpua100
# #BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000]"        # 40 GB host RAM
#BSUB -M 4000                      # hard memory limit
#BSUB -W 15:00
#BSUB -oo log/%J.out
#BSUB -eo error/%J.err

# initialize conda
export PATH="/zhome/71/c/146676/miniconda3/bin:$PATH"
source /zhome/71/c/146676/miniconda3/etc/profile.d/conda.sh

conda activate textom

python ../scripts/textomo2.py
