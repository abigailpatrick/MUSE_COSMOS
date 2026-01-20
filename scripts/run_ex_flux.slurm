#!/bin/bash
#SBATCH --job-name=muse_extract
#SBATCH --output=logs/muse_flux_%j.out
#SBATCH --error=logs/muse_flux_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=100G

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

python run_ex_flux.py # got the loop to pick sources etc. 

