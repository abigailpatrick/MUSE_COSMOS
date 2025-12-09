#!/bin/bash
#SBATCH --job-name=overlay10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/overlay10_%j.out
#SBATCH --error=logs/overlay10_%j.err

. /usr/local/anaconda/3.13/etc/profile.d/conda.sh
conda activate env39

OUTPUT_DIR="/cephfs/apatrick/musecosmos/scripts/aligned/mosaics/full"

# Create the overlay figure for the 10 slices
#python /cephfs/apatrick/musecosmos/scripts/plot_overlay_ratio_histograms.py \
 # --output_dir "$OUTPUT_DIR" \
  #--start 97 --step 350 --count 10 \
  #--bins 140  \
 # --clip_lo 1 --clip_hi 99 \
 # --outfile "overlay_ratio_hist_10slices_2.png"

# Create the overlay figure for 30 slices
python /cephfs/apatrick/musecosmos/scripts/plot_overlay_ratio_histograms.py \
--output_dir "$OUTPUT_DIR" \
--starts 100 97 103 \
--step 350 --count 10 \
--bins 140 --clip_lo 1 --clip_hi 99 \
--outfile "overlay_ratio_hist_30slices.png"