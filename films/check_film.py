#!/usr/bin/env python3
"""Check a rendered film as a whole.

Reads what the container says - duration, frame count, rate - and compares the
frame count with the validation JSON written next to the film (the only way to
be sure the encode kept every frame that was rendered), lists where each IF
section's ICP steps came from, and tiles a 5 x 5 contact sheet of frames pulled
back out of the MP4.

    python films/check_film.py path/to/film.mp4 [--sheet sheet.png] [--cols 5 --rows 5]

The same as ``python -m gcalign.film check``. Exit status 1 if the frame
counts disagree.
"""
import sys

from gcalign.film.cli import main

if __name__ == "__main__":
    sys.exit(main(["check", *sys.argv[1:]]))
