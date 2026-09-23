#!/usr/bin/env python3
"""Pull frames back out of the finished MP4 and tile them, so the cut can be
checked as a whole rather than one rendered still at a time.

Also reports what the container actually says - duration, frame count, rate -
which is the only way to be sure the encode kept every frame that was fed in.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import film_paths as P                                               # noqa: E402

OUT = P.FILM_OUT


def probe(exe, path):
    """ffprobe is not shipped with imageio-ffmpeg, so read the banner instead."""
    out = subprocess.run([exe, "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True).stderr
    return [ln.strip() for ln in out.splitlines()
            if "Duration" in ln or "Stream #" in ln]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--tile", type=int, default=384)
    args = ap.parse_args()

    exe = P.ffmpeg()
    mp4 = OUT / f"{args.rep}_nine_section_reconstruction.mp4"
    audit = json.loads((OUT / f"{args.rep}_story_validation.json").read_text())
    dur = audit["duration_seconds"]
    for line in probe(exe, mp4):
        print(" ", line)

    n = args.cols * args.rows
    tw, th = args.tile, int(args.tile * 1080 / 1920)
    sheet = Image.new("RGB", (tw * args.cols, th * args.rows), (0, 0, 0))
    for i in range(n):
        t = dur * (i + 0.5) / n
        raw = subprocess.run(
            [exe, "-v", "error", "-ss", f"{t:.3f}", "-i", str(mp4), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True).stdout
        if len(raw) < 1920 * 1080 * 3:
            print(f"  missing frame at {t:.2f}s")
            continue
        im = Image.frombytes("RGB", (1920, 1080), raw[:1920 * 1080 * 3])
        sheet.paste(im.resize((tw, th), Image.LANCZOS),
                    ((i % args.cols) * tw, (i // args.cols) * th))
    dst = OUT / "previews" / f"{args.rep}_film_contact_sheet.png"
    sheet.save(dst)
    print("wrote", dst, f"({n} frames across {dur:.1f} s)")


if __name__ == "__main__":
    main()
