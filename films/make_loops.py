#!/usr/bin/env python3
"""The README loops, cut from rendered films.

    python films/make_loops.py --short rep1_short.mp4 rep2_short.mp4 \\
        --names rep1_film rep2_film --closeup rep1.mp4 --out docs/img

--short FILM ...   each short edit becomes one loop, the whole clip
--names NAME ...   output names for the short loops (default: the film stems)
--closeup FILM     a full film: the top-view matching close-up of one IF section
                   becomes ``gc_matching_topview``; the interval (end of the dive,
                   the stored ICP iterations, the warp taking over) is read from
                   the film's validation JSON
--section NAME     the section for the close-up (default: the film's featured one)
--format ...       webp and/or gif (default both); each is fitted under --max-mb 5

The same as ``python -m gcalign.film loops``. Animated WebP is several times
smaller than GIF at the same look; GitHub shows both in an <img> tag.
"""
import sys

from gcalign.film.cli import main

if __name__ == "__main__":
    sys.exit(main(["loops", *sys.argv[1:]]))
