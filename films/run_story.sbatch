#!/bin/bash
# Render the reconstruction films of finished gcalign runs: one array task per
# run, each writing the full film, the short edit, their checks and the README
# loops. Fill in every CHANGE_ME and /path/to, then:  sbatch run_story.sbatch
#
# Cost per run on 16 cores: the full film (86 s at 1080p60) renders in about
# 5 min and the short edit in about 2 min. SLURM reports a peak of about 23 GB
# for the full film; the frame workers share their textures, so the real
# footprint is lower, but the per-process accounting counts them repeatedly.
#SBATCH --job-name=gcalign_film
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=film_%A_%a.log
set -euo pipefail

# one entry per array task
CONFIGS=(/path/to/rep1.yaml /path/to/rep2.yaml)        # finished runs
LABELS=(rep1 rep2)                                       # tag shown top right
SPATEO=(/path/to/rep1_spateo_run /path/to/rep2_spateo_run)   # act 1; "" opens on act 2
OUT=/path/to/films
PYTHON=${PYTHON:-python3}
# export GCALIGN_FFMPEG=/path/to/ffmpeg                 # otherwise PATH, then imageio-ffmpeg

i=${SLURM_ARRAY_TASK_ID:-0}
export TMPDIR=/tmp/gcfilm_${SLURM_JOB_ID:-$$}_${i}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
# frames are rendered by forked worker processes; keep BLAS to one thread each
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=$TMPDIR

cfg=${CONFIGS[$i]}
name=${LABELS[$i]}
export GCALIGN_SPATEO_DIR=${SPATEO[$i]}
mkdir -p "$OUT"

"$PYTHON" -u -m gcalign.film render --config "$cfg" --label "$name" --out "$OUT/$name.mp4"
"$PYTHON" -u -m gcalign.film render --config "$cfg" --label "$name" --preset short \
          --out "$OUT/${name}_short.mp4"
"$PYTHON" -u -m gcalign.film check "$OUT/$name.mp4"
"$PYTHON" -u -m gcalign.film check "$OUT/${name}_short.mp4"
"$PYTHON" -u -m gcalign.film loops --short "$OUT/${name}_short.mp4" --names "${name}_film" \
          --closeup "$OUT/$name.mp4" --out "$OUT/loops_$name"
