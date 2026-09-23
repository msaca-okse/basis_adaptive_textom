import os
import socket

import psutil


def info():
    mem = psutil.virtual_memory()
    print("hostname:", socket.gethostname())
    print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID"))
    print("SLURM_JOB_NODELIST:", os.environ.get("SLURM_JOB_NODELIST"))
    print("NPROCS:", len(os.sched_getaffinity(0)))
    print("Memory available:", mem.available / 1e9, "GB")
