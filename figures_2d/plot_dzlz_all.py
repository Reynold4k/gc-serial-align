#!/usr/bin/env python
"""Draw the dark-zone / light-zone figure of every germinal centre of a run, plus a manifest.

This runs `plot_dzlz_split.py` over all GCs of one run (or the ids given with ``--gc``)
and writes ``manifest.csv`` next to the figures. The manifest has one row per GC:
the figure file, the number of sections the GC is present on, the number of IF
sections on which the GC was large enough to be split into zones, and its area in
grid pixels on every section. Sort the manifest by those columns before looking
through the figures: a GC present on two sections gives a mostly empty figure.

A whole run (about 60-80 GCs) takes a few minutes of CPU; on a cluster, submit it
as a batch job.

    python figures_2d/plot_dzlz_all.py --config rep1.yaml --output-dir figures/rep1/dzlz \\
        --weights /path/to/rctd/weights --table-pattern '*{source}_rctd8.csv.gz' \\
        --zone-channel CD4_AF594
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_dzlz_split as split  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw the dark-zone / light-zone figure of every GC of a finished gcalign "
                    "run and write manifest.csv next to the figures.")
    split.add_common_arguments(parser)
    parser.add_argument("--gc", default="all",
                        help="GC ids to draw, comma-separated, or 'all'. Default: %(default)s")
    args = parser.parse_args(argv)
    ctx = split.context_from(args)
    run = ctx.run
    ids = run.gc_ids if args.gc == "all" else [int(g.upper().replace("GC", ""))
                                               for g in args.gc.replace(",", " ").split()]
    rows = []
    print(f"[figures_2d] {run.cfg.name}: {len(ids)} GCs", flush=True)
    for gid in ids:
        started = time.time()
        px = {n: int((run.labels(n) == gid).sum()) for n in run.names}
        out = split.draw_gc(ctx, gid, args.output_dir / f"GC{gid}_zones.png", dpi=args.dpi)
        rows.append({"gc_id": gid, "file": out.name,
                     "sections_present": sum(1 for v in px.values() if v > 0),
                     "if_sections_split": sum(1 for n in run.if_names if px[n] >= ctx.min_gc_px),
                     "total_px": sum(px.values()), **{f"px_{n}": px[n] for n in run.names}})
        print(f"  GC{gid}: {rows[-1]['sections_present']}/{len(run.names)} sections, "
              f"{rows[-1]['if_sections_split']} IF sections split [{time.time() - started:.1f} s]", flush=True)
    table = pd.DataFrame(rows)
    path = args.output_dir / "manifest.csv"
    table.to_csv(path, index=False)
    print(f"[figures_2d] wrote {len(table)} figures and {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
