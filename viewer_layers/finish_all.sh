#!/usr/bin/env bash
# Build every optional viewer layer for one finished gcalign run, then rewrite its viewer.
#
# Usage:
#   viewer_layers/finish_all.sh CONFIG.yaml RCTD_WEIGHTS_DIR [--output-dir RUN_DIR]
#
# Set in the environment:
#   FOLLICLE_CHANNEL  IF channel that marks the B-cell follicle (for example IgD_AF647)
#   ZONE_CHANNEL      IF channel that is high in the GC light zone (for example CD4_AF594)
#   WEIGHTS_PATTERN   file-name pattern of one ST section's RCTD table, {source} = its chip id
#                     (default "*{source}*.csv*")
#   PYTHON            interpreter with gcalign's dependencies (default: python3)
#
# It writes, inside the run directory:
#   05_viewer/follicle_points.npz   B-cell follicle points          (build_follicle_layer.py)
#   05_viewer/dzlz_points.npz       dark-zone / light-zone points   (build_dzlz_layer.py)
#   04_columns/zone_blocks.npz      dark-zone / light-zone solids   (build_zone_blocks.py)
#   05_viewer/if_images.json        IF images with GC outlines      (build_if_images.py)
#   05_viewer/<viewer>.html         the viewer, rebuilt with all of the above
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
CONFIG=$1
WEIGHTS=$2
shift 2
EXTRA=("$@")                      # passed to every step, e.g. --output-dir RUN_DIR

: "${FOLLICLE_CHANNEL:?set FOLLICLE_CHANNEL to the IF channel that marks the B-cell follicle}"
: "${ZONE_CHANNEL:?set ZONE_CHANNEL to the IF channel that is high in the light zone}"
PATTERN=${WEIGHTS_PATTERN:-"*{source}*.csv*"}
PY=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$HERE/..${PYTHONPATH:+:$PYTHONPATH}"   # the gcalign next to this folder

echo "== 1/5 B-cell follicle points"
"$PY" "$HERE/build_follicle_layer.py" --config "$CONFIG" --weights "$WEIGHTS" \
  --weights-pattern "$PATTERN" --follicle-channel "$FOLLICLE_CHANNEL" ${EXTRA[@]+"${EXTRA[@]}"}
echo "== 2/5 dark-zone / light-zone points"
"$PY" "$HERE/build_dzlz_layer.py" --config "$CONFIG" --weights "$WEIGHTS" \
  --weights-pattern "$PATTERN" --zone-channel "$ZONE_CHANNEL" ${EXTRA[@]+"${EXTRA[@]}"}
echo "== 3/5 dark-zone / light-zone solids"
"$PY" "$HERE/build_zone_blocks.py" --config "$CONFIG" --weights "$WEIGHTS" \
  --weights-pattern "$PATTERN" --zone-channel "$ZONE_CHANNEL" ${EXTRA[@]+"${EXTRA[@]}"}
echo "== 4/5 IF images"
"$PY" "$HERE/build_if_images.py" --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}
echo "== 5/5 viewer"
"$PY" -m gcalign.cli run --config "$CONFIG" --only viewer ${EXTRA[@]+"${EXTRA[@]}"}
