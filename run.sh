#!/bin/bash
#SBATCH -J Train
#SBATCH -o RetinalVesselSegmentationTesting.o%j
#SBATCH --mail-user=sporalas@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=8 -N 1
#SBATCH -t 12:0:0
#SBATCH --gpus=volta:1
#SBATCH --mem=16GB

python test.py
