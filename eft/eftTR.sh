#!/bin/sh
#SBATCH -D /users/ttp/jspinner/calc_v5/eft
#SBATCH -p albatros,empire
#SBATCH -o outTR.txt
#SBATCH -e errTR.txt

source ../venv/bin/activate
python mainTR.py $SLURM_ARRAY_TASK_ID
