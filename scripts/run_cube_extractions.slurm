#!/bin/bash
#SBATCH --job-name=muse_extract
#SBATCH --output=logs/muse_extract_%j.out
#SBATCH --error=logs/muse_extract_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G

# Load Python
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39
# Run using existing subcubes
srun python /cephfs/apatrick/musecosmos/scripts/cube_extractions_var.py \
    --cube /cephfs/apatrick/musecosmos/scripts/aligned/mosaics/big_cube/MEGA_CUBE_withFSF_VAR.fits \
    --output /cephfs/apatrick/musecosmos/dataproducts/extractions/ \
    --use_saved_subcubes
  