#!/bin/bash
#SBATCH --partition=gshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/remote/bigmem03a/spinner/SNforMuTau/eft

source ../venv/bin/activate
python eftTable.py
