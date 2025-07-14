#!/bin/sh
#SBATCH -D /users/ttp/jspinner/calc_v5/eft
#SBATCH -p albatros,moon,empire
#SBATCH -o outTR.txt
#SBATCH -e errTR.txt

source ../venv/bin/activate
python eftTable.py
