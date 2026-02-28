#!/bin/bash
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --chdir=/remote/bigmem03a/spinner/SNforMuTau/regimes
#SBATCH --array=0-11

source ../venv/bin/activate
python mainTR.py $SLURM_ARRAY_TASK_ID
