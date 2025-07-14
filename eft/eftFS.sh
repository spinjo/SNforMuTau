#!/bin/sh
#SBATCH -D /users/ttp/jspinner/calc_v5/eft
#SBATCH -p albatros,empire
#SBATCH -e errFS.txt
#SBATCH -o outFS.txt

source ../venv/bin/activate
python mainFS.py $SLURM_ARRAY_TASK_ID
