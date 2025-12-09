#!/bin/bash
#SBATCH --job-name=origin_pipeline
#SBATCH --output=logs/origin_%A_%a.out
#SBATCH --error=logs/origin_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --array=1-27

set -euo pipefail

# Ensure logs dir exists even if submitting from clean dir
mkdir -p logs

# Use a non-interactive backend for matplotlib
export MPLBACKEND=Agg

# Keep threaded libs from over-subscribing
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Load Conda 
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39


cd /cephfs/apatrick/musecosmos/scripts/

# If your script is saved elsewhere, update this path:
PY_SCRIPT=/cephfs/apatrick/musecosmos/scripts/originsub_all.py

# Pass the array index to the script as the source ID
srun python "$PY_SCRIPT" --source-id "${SLURM_ARRAY_TASK_ID}"
