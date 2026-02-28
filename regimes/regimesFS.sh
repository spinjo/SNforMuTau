#!/bin/bash
#SBATCH --partition=gshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/remote/bigmem03a/spinner/SNforMuTau/regimes
#SBATCH --array=0-7

source ../venv/bin/activate
python mainFS.py $SLURM_ARRAY_TASK_ID
