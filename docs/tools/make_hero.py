#!/usr/bin/env python
"""Draw the header image of the README (docs/img/hero.jpg) from a finished run.

    python docs/tools/make_hero.py --config run.yaml [--out docs/img/hero.jpg]

The picture on the right is one frame of the film (`gcalign animate`): all
sections of the run in one coordinate system, with only the section names
drawn. The text on the left names the package, says what it does, and says what
the picture shows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from gcalign import theme
from gcalign.film.hud import Pen
from gcalign.film.story import Film
from gcalign.runio import Run

W, H = 2400, 1000
STILL = (1400, 1000)          # the part of the canvas that holds the picture
ZOOM = 1.24                   # the frame is rendered larger and its centre is used


class Still(Film):
    """The film without its text; the positions of the section names are kept."""

    def draw_hud(self, canvas, t, shot, tt, info) -> None:
        self.section_labels = list(info.get("labels", []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "img" / "hero.jpg")
    parser.add_argument("--tissue", default="Mouse spleen", help="tissue named in the text")
    args = parser.parse_args()

    run = Run.open(args.config, output_dir=args.output_dir)
    size = (round(STILL[0] * ZOOM), round(STILL[1] * ZOOM))
    film = Still(run, size=size, preset="full", upstream=None, log=None)
    shot = [s for s in film.shots if s.act == 4][0]
    frame = film.render(shot.start + shot.length * 0.5)
    crop_x, crop_y = (size[0] - STILL[0]) // 2 - 40, (size[1] - STILL[1]) // 2 + 20
    still = frame.crop((crop_x, crop_y, crop_x + STILL[0], crop_y + STILL[1]))

    canvas = Image.new("RGB", (W, H), theme.BG)
    left = W - STILL[0]
    canvas.paste(still, (left, 0))
    d = ImageDraw.Draw(canvas)
    # section names beside the stack, blue for IF and pink for ST
    for x, y, name, colour, _ in film.section_labels:
        d.text((left + x - crop_x - 8, y - crop_y), name, font=theme.pil_font(30, "medium"),
               fill=colour, anchor="rm")

    x = 110
    d.text((x - 8, 170), "gcalign", font=theme.pil_font(200, "semibold"), fill=theme.INK)
    for row, line in enumerate(("Registers immunofluorescence sections",
                                "to spatial-transcriptomics sections",
                                "of the same tissue block.")):
        d.text((x, 440 + row * 62), line, font=theme.pil_font(46, "regular"), fill=theme.INK_2)

    n_if, n_st = len(run.if_names), len(run.st_names)
    y = 690
    for swatch, text in ((theme.IF, f"{n_if} immunofluorescence (IF) sections"),
                         (theme.ST, f"{n_st} spatial-transcriptomics (ST) sections")):
        d.ellipse((x, y + 13, x + 22, y + 35), fill=swatch)
        d.text((x + 42, y), text, font=theme.pil_font(36, "medium"), fill=theme.INK)
        y += 60
    d.text((x, y + 14), f"{args.tissue} · {run.cfg.n_planes} serial sections of "
                        f"{run.cfg.section_thickness_um:g} µm in one coordinate system",
           font=theme.pil_font(30, "regular"), fill=theme.MUTED)
    d.text((x, y + 58), f"Depth is stretched ×{film.A.z_exag:g} in this picture.",
           font=theme.pil_font(30, "regular"), fill=theme.MUTED)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, quality=92, optimize=True, subsampling=0)
    print(f"wrote {args.out}  {canvas.size[0]} x {canvas.size[1]}  "
          f"{args.out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
