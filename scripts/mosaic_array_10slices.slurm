#!/bin/bash
#SBATCH --job-name=mosaic10
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mosaic10_%A_%a.out
#SBATCH --error=logs/mosaic10_%A_%a.err

# Activate conda environment
. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

# Paths
CUBES_DIR="/cephfs/apatrick/musecosmos/reduced_cubes/norm"
OFFSETS_TXT="/cephfs/apatrick/musecosmos/scripts/aligned/offsets.txt"
OUTPUT_DIR="/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/full"
TMP_BASE="/cephfs/apatrick/musecosmos/scripts/aligned/tmp_slices"

# Slice mapping: 97 + 350 * index
SLICE_NUM=$((97 + 350 * SLURM_ARRAY_TASK_ID))
echo "Task $SLURM_ARRAY_TASK_ID -> slice $SLICE_NUM"

# Unique temp dir
SLICE_TMP_DIR="${TMP_BASE}/slice_${SLICE_NUM}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SLICE_TMP_DIR"

# Run mosaic + VAP + line histogram
python /cephfs/apatrick/musecosmos/scripts/mosaic_chunk_var.py \
    --path "$CUBES_DIR" \
    --offsets_txt "$OFFSETS_TXT" \
    --slice $SLICE_NUM \
    --output_file "$OUTPUT_DIR/var_mosaic_slice_${SLICE_NUM}.fits" \
    --plotting \
    --tmp_dir "$SLICE_TMP_DIR"

# Clean up temp
rm -rf "$SLICE_TMP_DIR"
echo "Temporary directory $SLICE_TMP_DIR removed."
