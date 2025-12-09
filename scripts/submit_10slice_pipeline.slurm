#!/bin/bash
# submit_10slice_pipeline.slurm

ARRAY_JOBID=$(sbatch --parsable mosaic_array_10slices.slurm)
echo "Submitted array job: $ARRAY_JOBID"

sbatch --dependency=afterok:${ARRAY_JOBID} overlay_after_array.slurm
echo "Submitted overlay job dependent on $ARRAY_JOBID"
