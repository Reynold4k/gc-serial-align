#!/usr/bin/env python
"""Draw docs/img/theme.png from gcalign.theme, so the card can never drift from the code."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from gcalign import theme as T

W, H = 1800, 760
PAD = 72


def main(out: Path) -> None:
    im = Image.new("RGB", (W, H), T.BG)
    d = ImageDraw.Draw(im)

    def label(xy, text, size=17, weight="regular", fill=T.MUTED):
        d.text(xy, text, font=T.pil_font(size, weight), fill=fill)

    # ---- meaning ----------------------------------------------------------
    label((PAD, 56), "MEANING", 15, "medium")
    x = PAD
    for name, text in (("IF", "immunofluorescence"), ("ST", "spatial transcriptomics"),
                       ("AGREE", "where they agree"), ("GOOD", "good"), ("WARN", "look"),
                       ("BAD", "stop")):
        colour = getattr(T, name)
        d.rounded_rectangle((x, 92, x + 232, 172), radius=14, fill=colour)
        label((x, 184), text, 17, "medium", T.INK_2)
        label((x, 208), colour, 15, "regular", T.MUTED)
        x += 276

    # ---- surface ----------------------------------------------------------
    label((PAD, 272), "SURFACE", 15, "medium")
    x = PAD
    for name, text in (("BG", "page"), ("SURFACE", "panel"), ("SURFACE_2", "control"),
                       ("LINE", "hairline"), ("TISSUE", "section body"),
                       ("INK", "text"), ("INK_2", "secondary"), ("MUTED", "tertiary")):
        colour = getattr(T, name)
        d.rounded_rectangle((x, 308, x + 176, 364), radius=12, fill=colour, outline=T.LINE_2)
        label((x, 376), text, 16, "medium", T.INK_2)
        label((x, 398), colour, 14, "regular", T.MUTED)
        x += 207

    # ---- identity ---------------------------------------------------------
    label((PAD, 462), "ONE COLOUR PER GERMINAL CENTRE, THE SAME IN EVERY VIEW", 15, "medium")
    ids = list(range(1, 73))
    palette = T.identity_colours(ids)
    step = (W - 2 * PAD) / len(ids)
    for k, gid in enumerate(ids):
        cx = PAD + k * step
        d.rounded_rectangle((cx, 498, cx + step - 5, 566), radius=7, fill=palette[gid])

    # ---- type -------------------------------------------------------------
    label((PAD, 618), "TYPE", 15, "medium")
    x = PAD
    for weight in ("light", "regular", "medium", "semibold"):
        d.text((x, 646), "Inter", font=T.pil_font(46, weight), fill=T.INK)
        label((x + 148, 672), weight, 16, "regular", T.MUTED)
        x += 300
    d.text((W - PAD, 646), "0123456789 µm", font=T.pil_font(46, "light"), fill=T.INK_2,
           anchor="ra")

    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, optimize=True)
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1] / "img" / "theme.png")
