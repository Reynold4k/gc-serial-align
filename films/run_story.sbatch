#!/bin/bash
# Render both replicates' films as a SLURM array. Roughly 40 minutes and under
# 8 GB each on one core; the four threads are for the x264 encoder.
#
# Edit the account, partition and the GCALIGN_* paths for your installation,
# then:  sbatch run_story.sbatch
#SBATCH --job-name=gc_story
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --array=1-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=story_%A_%a.log
set -euo pipefail

# Rendering is single-threaded numpy plus PIL; letting BLAS spawn threads only
# adds contention on a one-core allocation.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gc_story_${SLURM_JOB_ID}"

export GCALIGN_NINE_SLICE_DIR=/path/to/_build/nine_slice
export GCALIGN_FILM_OUT=/path/to/film_output
export GCALIGN_SPATEO_REP1=/path/to/rep1_bin20_aligned
export GCALIGN_SPATEO_REP2=/path/to/rep2_bin20_aligned
export GCALIGN_GC_FAMILY_CSV=/path/to/new_region_merged_gc_id.csv
export GCALIGN_GC_REGIONS_DIR=/path/to/02_GC_selection
export GCALIGN_SLICES_REP1=CHIP_A,CHIP_B,CHIP_C,CHIP_D
export GCALIGN_SLICES_REP2=CHIP_E,CHIP_F,CHIP_G,CHIP_H

PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$PYTHON" -u "$HERE/film_paths.py"
"$PYTHON" -u "$HERE/render_story.py" --rep "rep${SLURM_ARRAY_TASK_ID}"
