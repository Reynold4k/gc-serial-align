#!/usr/bin/env python3
"""Build the B-cell follicle points of every section (05_viewer/follicle_points.npz).

The viewer draws these points faintly behind the germinal centres (GCs). A GC
can then be judged against the B-cell follicle it sits in: a shift between
sections is visibly either the GC moving or the registration drifting.

Immunofluorescence (IF) sections: the pixels where the follicle marker
(--follicle-channel, for example IgD) is above its Otsu threshold inside the
tissue, after a small closing and removal of specks.
Spatial-transcriptomics (ST) sections: the bins whose dominant RCTD cell type
(the `first_type` column) is one of --follicle-types. Requiring the dominant
type, rather than summing weights, keeps red-pulp bins with a minor B-cell
share out of the follicle.
Each section keeps at most --max-points points, drawn at random with a fixed seed.

Output: one float32 array of (x, y) frame positions in micrometres per section,
keyed by section name, and a report next to it (`follicle_points_report.json`).

Example:
    python viewer_layers/build_follicle_layer.py --config my_experiment.yaml \\
        --weights rctd_weights/ --follicle-channel IgD_AF647
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage import filters, morphology

import layer_common as common


def thin(points: np.ndarray, rng: np.random.Generator, cap: int) -> np.ndarray:
    """At most `cap` points, drawn without replacement."""
    if len(points) <= cap:
        return points
    return points[rng.choice(len(points), cap, replace=False)]


def main(argv=None) -> int:
    ap = common.parser(__doc__.split("\n")[0])
    common.add_weights(ap)
    ap.add_argument("--follicle-channel", required=True,
                    help="IF channel that marks the B-cell follicle, named as in the config (for example IgD_AF647)")
    ap.add_argument("--follicle-types", default="B naive,B memory,B MZ",
                    help="comma-separated RCTD first_type values that count as follicular B cells. "
                         "Default: %(default)s")
    ap.add_argument("--max-points", type=int, default=6000,
                    help="largest number of points kept per section. Default: %(default)s")
    ap.add_argument("--min-blob-px", type=int, default=25,
                    help="IF sections: smallest follicle patch kept, in grid pixels. Default: %(default)s")
    ap.add_argument("--min-matched", type=float, default=0.8,
                    help="ST sections: stop if fewer than this fraction of the follicle bins in the RCTD table "
                         "are found in the scaffold. Default: %(default)s")
    ap.add_argument("--seed", type=int, default=20260917, help="random seed for the thinning. Default: %(default)s")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file. Default: <run>/05_viewer/follicle_points.npz")
    args = ap.parse_args(argv)

    run = common.open_run(args)
    channel = common.require_channel(run, args.follicle_channel, "--follicle-channel")
    types = [t.strip() for t in args.follicle_types.split(",") if t.strip()]
    tissue_channels = [c for c in run.cfg.if_images.tissue_channels if c in run.channel_names] or run.channel_names
    out = args.out or run.root / "05_viewer" / "follicle_points.npz"
    tables = common.resolve_tables(run, args.weights, args.weights_pattern)
    rng = np.random.default_rng(args.seed)
    arrays, report = {}, []

    for plane in run.planes:                                       # cutting order: fixes the random draws
        name = plane.name
        if plane.kind == "IF":
            marker = run.channel(name, channel).astype(np.float32)
            tissue = np.zeros(marker.shape, bool)
            for other in tissue_channels:
                tissue |= run.channel(name, other) > 0
            threshold = float(filters.threshold_otsu(marker[tissue]))
            mask = (marker > threshold) & tissue
            mask = morphology.binary_closing(mask, morphology.disk(2))
            mask = morphology.remove_small_objects(mask, args.min_blob_px)
            overlap = float((mask & (run.labels(name) > 0)).sum() / max(mask.sum(), 1))
            rows, cols = np.nonzero(mask)
            points = run.grid.to_xy(rows, cols).astype(np.float32)
            source, extra = channel, dict(otsu=round(threshold, 1), gc_overlap=round(overlap, 4))
        else:
            path = tables[name]
            table = common.rctd_table(path)
            candidates = table[table.first_type.isin(types)]
            joined = common.on_bins(run, name, candidates)
            points = np.column_stack([joined.wx.values, joined.wy.values]).astype(np.float32)
            source, extra = "RCTD first_type", dict(matched=int(len(joined)), candidates=int(len(candidates)),
                                                    table=path.name)
            if len(joined) < args.min_matched * len(candidates):
                raise SystemExit(f"{name}: only {len(joined)} of {len(candidates)} follicle bins in "
                                 f"{path.name} were found in the scaffold; the coordinates do not agree")
        kept = thin(points, rng, args.max_points)
        arrays[name] = kept
        report.append(dict(section=plane.index, plane=name, kind=plane.kind, source=source,
                           found=int(len(points)), kept=int(len(kept)), **extra))
        print(f"  {name:<10} {plane.kind}  {source:<16} found {len(points):>7,}  kept {len(kept):>6,}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    out.with_name(out.stem + "_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
