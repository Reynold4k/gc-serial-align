#!/usr/bin/env python3
"""Every installation-specific path the film renderer needs, in one place.

This repository keeps cluster paths out of committed code, so the values below
are placeholders.  Point them at your own run by exporting the environment
variables - nothing here has to be edited:

    export GCALIGN_NINE_SLICE_DIR=/.../_build/nine_slice   # holds rep1/, rep2/
    export GCALIGN_FILM_OUT=/.../film_output               # where films land
    export GCALIGN_SPATEO_REP1=/.../rep1_bin20_aligned     # act 1 only
    export GCALIGN_SPATEO_REP2=/.../rep2_bin20_aligned
    export GCALIGN_GC_FAMILY_CSV=/.../new_region_merged_gc_id.csv
    export GCALIGN_GC_REGIONS_DIR=/.../02_GC_selection
    export GCALIGN_SLICES_REP1=CHIP_A,CHIP_B,CHIP_C,CHIP_D
    export GCALIGN_SLICES_REP2=CHIP_E,CHIP_F,CHIP_G,CHIP_H

Acts 2-5 need only the first two.  Act 1 replays the Stereo-seq registration's
own stored iterations, which live with that run rather than with gcalign's
output, which is why it needs the rest.  ``python film_paths.py`` prints what is
set and what is still missing.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

PLACEHOLDER = "/path/to"


def _path(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


def _names(env: str, default: str) -> list[str]:
    return [s for s in os.environ.get(env, default).split(",") if s]


# -- gcalign's own output ---------------------------------------------------
# The nine-section output directory: one subdirectory per replicate, each with
# 01_if_gc/, 02_global_align/, 03_gc_match/, 04_columns/, validation_report.json.
NINE_SLICE_DIR = _path("GCALIGN_NINE_SLICE_DIR", f"{PLACEHOLDER}/_build/nine_slice")

# Where films, preview stills and the validation record are written.
FILM_OUT = _path("GCALIGN_FILM_OUT", f"{PLACEHOLDER}/film_output")

# rep{n}_verified_icp_replay.json, written by scripts/replay_saved_icp.py.
# Defaults to FILM_OUT because that is where that script puts them.
ICP_REPLAY_DIR = _path("GCALIGN_ICP_REPLAY_DIR", str(FILM_OUT))

# -- the upstream Stereo-seq registration, for act 1 only -------------------
# The directory holding <slice>_gc_bin20_aligned_all_stages.h5ad, i.e. the
# Spateo run with uns/iter_rigid, uns/iter_nonrigid and uns/iter_deformable
# still in it.  Without the stored iterations there is no act 1 to replay.
SPATEO_ITER_DIR = {
    "rep1": _path("GCALIGN_SPATEO_REP1", f"{PLACEHOLDER}/rep1_bin20_aligned"),
    "rep2": _path("GCALIGN_SPATEO_REP2", f"{PLACEHOLDER}/rep2_bin20_aligned"),
}

# Curated cross-section GC grouping: columns rep, slice, new_gc_id, merged_gc_id.
GC_FAMILY_CSV = _path("GCALIGN_GC_FAMILY_CSV",
                      f"{PLACEHOLDER}/new_region_merged_gc_id.csv")

# One <slice>/<slice>_gc_regions.parquet per slice, columns x, y, gc_region_new.
GC_REGIONS_DIR = _path("GCALIGN_GC_REGIONS_DIR", f"{PLACEHOLDER}/02_GC_selection")

# Chip ids in cutting order, matching the short labels in validation_report.json.
SLICES = {
    "rep1": _names("GCALIGN_SLICES_REP1", "CHIP_A,CHIP_B,CHIP_C,CHIP_D"),
    "rep2": _names("GCALIGN_SLICES_REP2", "CHIP_E,CHIP_F,CHIP_G,CHIP_H"),
}

# -- tools ------------------------------------------------------------------
# Any ffmpeg with libx264.  imageio-ffmpeg ships a static one if the system has
# none; set GCALIGN_FFMPEG to use a specific binary.
FFMPEG = os.environ.get("GCALIGN_FFMPEG", "") or shutil.which("ffmpeg") or ""

FONT_DIR = _path("GCALIGN_FONT_DIR", "/usr/share/fonts/dejavu-sans-fonts")


def ffmpeg() -> str:
    """The encoder, falling back to the one imageio-ffmpeg vendors."""
    if FFMPEG:
        return FFMPEG
    try:
        import imageio_ffmpeg
    except ImportError as exc:                                   # pragma: no cover
        raise RuntimeError(
            "no ffmpeg found: set GCALIGN_FFMPEG, put ffmpeg on PATH, "
            "or pip install imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def report() -> int:
    """Print which paths resolve and which are still placeholders."""
    rows = [
        ("GCALIGN_NINE_SLICE_DIR", NINE_SLICE_DIR, True),
        ("GCALIGN_FILM_OUT", FILM_OUT, True),
        ("GCALIGN_ICP_REPLAY_DIR", ICP_REPLAY_DIR, True),
        ("GCALIGN_SPATEO_REP1", SPATEO_ITER_DIR["rep1"], False),
        ("GCALIGN_SPATEO_REP2", SPATEO_ITER_DIR["rep2"], False),
        ("GCALIGN_GC_FAMILY_CSV", GC_FAMILY_CSV, False),
        ("GCALIGN_GC_REGIONS_DIR", GC_REGIONS_DIR, False),
    ]
    bad = 0
    for env, value, needed_always in rows:
        if PLACEHOLDER in str(value):
            state = "PLACEHOLDER"
        elif value.exists():
            state = "ok"
        else:
            state = "MISSING"
        if state != "ok":
            bad += 1
        note = "" if needed_always else "   (act 1 only)"
        print(f"  {state:12s} {env:26s} {value}{note}")
    for rep, names in SLICES.items():
        print(f"  {'':12s} {'SLICES[' + rep + ']':26s} {', '.join(names)}")
    print(f"  {'ok' if FFMPEG or True else '':12s} {'ffmpeg':26s} {FFMPEG or '(imageio-ffmpeg)'}")
    fonts = "ok" if FONT_DIR.exists() else "MISSING"
    print(f"  {fonts:12s} {'GCALIGN_FONT_DIR':26s} {FONT_DIR}")
    return bad


if __name__ == "__main__":
    print("film_paths:")
    n = report()
    print(f"\n{n} path(s) still need setting." if n else "\nall paths resolve.")
