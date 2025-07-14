#!/bin/sh
#SBATCH -D /users/ttp/jspinner/calc_v5/mutau
#SBATCH -p albatros,empire
#SBATCH -o outTR.txt
#SBATCH -e errTR.txt
#SBATCH --mem 1G

source ../venv/bin/activate
python mainTR.py $SLURM_ARRAY_TASK_ID
