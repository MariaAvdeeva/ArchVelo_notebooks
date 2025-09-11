#!/bin/bash
#
#SBATCH -N1 --ntasks-per-node=128 --exclusive

module load modules/2.2-20230808
module load python/3.8.16
python3 ArchVelo_cl13_slurm.py