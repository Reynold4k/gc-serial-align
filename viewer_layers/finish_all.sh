#!/bin/bash
set -euo pipefail
W=/path/to/rctd_work
S9=/path/to/nine_section_output
D=/path/to/viewer_bundle
G=$S9/gc-serial-align
PY=/path/to/spateo-env/bin/python
cd "$W"
echo "########## 1. four-slice RCTD HTMLs"
for r in rep1 rep2; do
  python3 build_rctd_four_slice.py --rep $r --weights weights --out ${r}_four_slice_rctd.html 2>&1 \
    | grep -E "spots \(|wrote|joined" | sed "s/^/  [$r] /"
done
echo "########## 2. nine-slice follicle point clouds"
for r in rep1 rep2; do
  python3 build_follicle_layer.py --rep $r --weights weights --out "$S9/$r/05_viewer/follicle_points.npz" 2>&1 \
    | tail -2 | sed "s/^/  [$r] /"
done
echo "########## 3. nine-slice DZ/LZ blocks"
for r in rep1 rep2; do
  $PY build_zone_blocks.py --rep $r --weights weights --out "$S9/$r/04_columns" 2>&1 \
    | tail -2 | sed "s/^/  [$r] /"
done
echo "########## 4. regenerate nine-slice viewers"
cd "$G"
sed 's|^output_dir: ./out/rep1|output_dir: '"$S9"'/rep1|' configs/rep1_mouse_spleen.yaml > /tmp/rep1_local.yaml
for r in rep1 rep2; do
  CFG=configs/${r}_mouse_spleen.yaml; [ "$r" = rep1 ] && CFG=/tmp/rep1_local.yaml
  $PY -c "
import sys; sys.path.insert(0,'.')
from gcalign import cli
sys.argv=['gcalign','run','--config','$CFG','--only','viewer']
cli.main()" 2>&1 | grep -E "follicle backdrop|wrote" | sed "s/^/  [$r] /"
done
echo "########## 5. deploy"
cp "$W/rep1_four_slice_rctd.html" "$W/rep2_four_slice_rctd.html" "$D/viewers/"
cp "$S9/rep1/05_viewer/nine_slice_gc_columns_rep1.html" "$D/viewers/rep1_nine_slice_columns.html"
cp "$S9/rep2/05_viewer/nine_slice_gc_columns_rep2.html" "$D/viewers/rep2_nine_slice_columns.html"
ls -la "$D/viewers/" | awk '{printf "  %8s  %s\n",$5,$9}' | grep html
echo "DONE"
