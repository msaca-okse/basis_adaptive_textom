#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- name ---
#BSUB -J Recon
# -- choose queue --
# For gpu write gpuv100, gpua100, gpua10, gpua40. Check availability by bqueues -l gpua100, eg
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -e logs/%J.err
#BSUB -o logs/%J.out
#BSUB -M 16000
#BSUB -R "rusage[mem=16000]"
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 15:00 
# -- Number of cores requested -- 
#BSUB -n 16
# -- Specify the distribution of the cores: on a separate nodes --
#BSUB -R "span[hosts=1]"
# Array job: N tasks, one per folder
#BSUB -J dfrctn_sim
# -- end of LSF options -- 

source /zhome/71/c/146676/miniconda3/bin/activate && conda activate cil

python integrate_segments.py
