#!/bin/bash

# #SBATCH -N1 --ntasks-per-node=128 --exclusive

echo "Slurm job started"

# Run your Python script (add arguments if necessary)
python 3_ArchVelo_arm.py

echo "Slurm job finished"
