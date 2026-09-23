#!/bin/bash
# A slide-deck copy of each master film: same 1920x1080 at 60 fps, same frames,
# just a cheaper rate. The master stays the archive copy; this is the one that
# travels. Roughly 3x smaller.
#
#   export GCALIGN_FILM_OUT=/.../film_output
#   ./make_compact.sh rep1 rep2
set -euo pipefail

OUT="${GCALIGN_FILM_OUT:?set GCALIGN_FILM_OUT to the directory holding the films}"
CRF="${GCALIGN_COMPACT_CRF:-26}"
FF="${GCALIGN_FFMPEG:-$(command -v ffmpeg || true)}"
if [ -z "$FF" ]; then
  FF=$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())') \
    || { echo "no ffmpeg: set GCALIGN_FFMPEG, or pip install imageio-ffmpeg" >&2; exit 1; }
fi

for rep in "$@"; do
  src="$OUT/${rep}_nine_section_reconstruction.mp4"
  dst="$OUT/${rep}_nine_section_reconstruction_compact.mp4"
  [ -f "$src" ] || { echo "missing $src" >&2; continue; }
  # encode to a temp name and rename, so a second run cannot half-overwrite a
  # copy someone is already using
  "$FF" -y -v error -i "$src" -c:v libx264 -preset slow -crf "$CRF" -pix_fmt yuv420p \
        -threads 4 -movflags +faststart "$dst.tmp.mp4"
  mv -f "$dst.tmp.mp4" "$dst"
  printf '%s  %.1f MB -> %.1f MB\n' "$rep" \
    "$(stat -c%s "$src" | awk '{print $1/1e6}')" "$(stat -c%s "$dst" | awk '{print $1/1e6}')"
done
