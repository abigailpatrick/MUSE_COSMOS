#!/bin/bash
#SBATCH --job-name=fsf_creation
#SBATCH --partition=all
#SBATCH --output=fsf_creation_%j.log
#SBATCH --error=fsf_creation_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=480G
#SBATCH --cpus-per-task=4


# Activate conda environment
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

cd /cephfs/apatrick/musecosmos/scripts/

echo "======================================="
echo "FSF model creation job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "======================================="

# Reduce memory pressure by disabling OpenMP threading in libraries
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run your FSF creation Python script
/usr/bin/time -v python FSF_creation2.py 2>&1 | tee fsf_creation_${SLURM_JOB_ID}_runtime.log

echo "======================================="
echo "Job finished at: $(date)"
echo "======================================="
