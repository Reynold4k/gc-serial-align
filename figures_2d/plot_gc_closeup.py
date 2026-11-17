#!/usr/bin/env python
"""Draw one germinal centre on every section of a finished run (a per-GC close-up).

This script is a command-line wrapper around `gcalign.figures.gc_closeups`, which
replaced the close-up code that used to live here. It writes one figure per GC and
``gc_closeups.csv`` with the per-section numbers of those GCs (area, equivalent
radius, centroid, and what occupies the GC's footprint on sections where it is
absent). The same figures come from ``gcalign figures --config X --gc 3,19``.

    python figures_2d/plot_gc_closeup.py --config rep1.yaml --gc 19 --output-dir figures/rep1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gcalign import figures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw germinal centres on every section of a finished gcalign run, one "
                    "figure per GC, plus gc_closeups.csv with their per-section numbers.")
    parser.add_argument("--config", required=True, type=Path,
                        help="the experiment YAML of a finished gcalign run")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="the run folder, when it is not the output_dir written in the config")
    parser.add_argument("--gc", required=True,
                        help="GC ids to draw, comma-separated (for example 3,19), or 'all'")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="folder the figures and gc_closeups.csv are written to")
    parser.add_argument("--dpi", type=float, default=220, help="output resolution. Default: %(default)s")
    args = parser.parse_args(argv)
    run = figures.as_run(args.config, output_dir=args.run_dir)
    gcs = "all" if args.gc == "all" else args.gc
    written = figures.gc_closeups(run, gcs, args.output_dir, dpi=args.dpi)
    for key, path in written.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
