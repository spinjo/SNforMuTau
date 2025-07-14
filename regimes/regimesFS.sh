#!/bin/sh
#SBATCH -D /users/ttp/jspinner/calc_v5/regimes
#SBATCH -p albatros,empire
#SBATCH -e errFS.txt
#SBATCH -o outFS.txt
#SBATCH --mem 1G

source ../venv/bin/activate
python mainFS.py $SLURM_ARRAY_TASK_ID
