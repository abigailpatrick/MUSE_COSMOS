#!/bin/bash
#SBATCH --array=1-27
#SBATCH --job-name=optdxdy
#SBATCH --output=optdxdy_%j.log
#SBATCH --error=optdxdy_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=20G

python optimise_spatial_offset_lya.py \
  --csv jels_halpha_candidates_mosaic_all.csv \
  --row-index ${SLURM_ARRAY_TASK_ID} \
  --cube-dir /cephfs/apatrick/musecosmos/dataproducts/extractions \
  --outfile offsets_${SLURM_ARRAY_TASK_ID}.csv

