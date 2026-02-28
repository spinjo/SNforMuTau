#!/bin/bash
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --chdir=/remote/bigmem03a/spinner/SNforMuTau/mutau
#SBATCH --array=0-399

source ../venv/bin/activate
python mainTR.py $SLURM_ARRAY_TASK_ID
