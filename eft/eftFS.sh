#!/bin/bash
#PBS -q medium_bookworm
#PBS -l nodes=1:ppn=4:medium_bookworm
#PBS -l walltime=5:00:00
#PBS -l vmem=10gb
#PBS -d /remote/bigmem03a/spinner/SNforMuTau/eft
#PBS -t 0-7

source ../venv/bin/activate
python mainFS.py $PBS_ARRAYID
