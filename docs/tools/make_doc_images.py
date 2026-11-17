#!/usr/bin/env python
"""Copy the figures of a run into docs/img at the size the documentation uses.

    python docs/tools/make_doc_images.py FIGURES_DIR [--gc 19]

FIGURES_DIR is the folder written by `gcalign figures --gc <id>` (06_figures/ of
a run). Photographic figures are saved as JPEG, charts as PNG, 1800 px wide at
most. The documentation images are made from our replicate 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

WIDTH = 1800
# (source inside FIGURES_DIR, file name in docs/img)
PLAN = [
    ("sections.png", "sections.jpg"),
    ("overlay.png", "overlay.jpg"),
    ("columns.png", "columns.jpg"),
    ("alignment.png", "alignment.png"),
    ("orientation.png", "orientation.png"),
    ("presence.png", "presence.png"),
    ("gc_closeups/GC{gc}.png", "gc_closeup.jpg"),
    ("columns/GC{gc}.png", "column_gc{gc}.jpg"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("figures_dir", type=Path)
    parser.add_argument("--gc", type=int, default=19, help="GC id used for the single-GC examples")
    parser.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "img")
    args = parser.parse_args()
    for source, target in PLAN:
        src = args.figures_dir / source.format(gc=args.gc)
        dst = args.out / target.format(gc=args.gc)
        if not src.exists():
            print(f"missing {src}")
            continue
        image = Image.open(src).convert("RGB")
        if image.width > WIDTH:
            image = image.resize((WIDTH, round(image.height * WIDTH / image.width)), Image.LANCZOS)
        if dst.suffix == ".jpg":
            image.save(dst, quality=90, optimize=True, subsampling=0)
        else:
            image.save(dst, optimize=True)
        print(f"{dst.name:22s} {image.width} x {image.height}  {dst.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
