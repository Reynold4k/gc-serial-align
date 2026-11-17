#!/usr/bin/env python3
"""Build the dark-zone and light-zone points of every section (05_viewer/dzlz_points.npz).

A germinal centre (GC) has a dark zone (DZ), where B cells divide, and a light
zone (LZ), where they meet T follicular helper cells. These points show the two
zones inside each registered GC; the viewer draws them in its DZ / LZ view and
filters them by GC when one is selected.

Immunofluorescence (IF) sections: inside each GC, the pixels at or above that
GC's own median of the light-zone marker (--zone-channel, for example CD4, which
is high where T follicular helper cells gather) are light zone and the rest are
dark zone. The split is unsmoothed and at the median: smoothing cost up to 15
points of agreement with the neighbouring ST sections, and the split point
changed almost nothing.
Spatial-transcriptomics (ST) sections: bins inside a GC whose normalised RCTD
weight for the zone (--dz-column, --lz-column) is at least --min-weight.
The two zones are thresholded independently here, so the point clouds can
overlap; build_zone_blocks.py makes solids that tile each GC instead.

Output: "<section>|DZ" and "<section>|LZ" float32 arrays of (x, y) frame
positions in micrometres, each with a matching "<section>|<zone>|gc" array of
GC ids, and a report next to it (`dzlz_points_report.json`).

Example:
    python viewer_layers/build_dzlz_layer.py --config my_experiment.yaml \\
        --weights rctd_weights/ --zone-channel CD4_AF594
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import layer_common as common


def thin_idx(n: int, rng: np.random.Generator, cap: int) -> np.ndarray:
    """Indices to keep, so points and their GC ids are thinned together."""
    return np.arange(n) if n <= cap else rng.choice(n, cap, replace=False)


def main(argv=None) -> int:
    ap = common.parser(__doc__.split("\n")[0])
    common.add_weights(ap)
    ap.add_argument("--zone-channel", required=True,
                    help="IF channel that is high in the light zone, named as in the config (for example CD4_AF594)")
    ap.add_argument("--dz-column", default="DZ", help="RCTD weight column of the dark zone. Default: %(default)s")
    ap.add_argument("--lz-column", default="LZ", help="RCTD weight column of the light zone. Default: %(default)s")
    ap.add_argument("--min-weight", type=float, default=0.15,
                    help="ST sections: smallest normalised RCTD weight that counts as the zone. Default: %(default)s")
    ap.add_argument("--min-gc-px", type=int, default=60,
                    help="IF sections: smallest GC cross-section split into zones, in grid pixels. Default: %(default)s")
    ap.add_argument("--max-points", type=int, default=5000,
                    help="largest number of points kept per zone per section. Default: %(default)s")
    ap.add_argument("--seed", type=int, default=20260917, help="random seed for the thinning. Default: %(default)s")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file. Default: <run>/05_viewer/dzlz_points.npz")
    args = ap.parse_args(argv)

    run = common.open_run(args)
    channel = common.require_channel(run, args.zone_channel, "--zone-channel")
    columns = {"DZ": args.dz_column, "LZ": args.lz_column}
    out = args.out or run.root / "05_viewer" / "dzlz_points.npz"
    tables = common.resolve_tables(run, args.weights, args.weights_pattern)
    rng = np.random.default_rng(args.seed)
    arrays, report = {}, []

    for plane in run.planes:                                       # cutting order: fixes the random draws
        name = plane.name
        if plane.kind == "IF":
            marker = run.channel(name, channel).astype(np.float32)    # unsmoothed: see the docstring
            labels = run.labels(name)
            ids = np.unique(labels)
            ids = ids[ids > 0]
            picked = {"LZ": ([], [], []), "DZ": ([], [], [])}
            used = 0
            for gid in ids:
                inside = labels == gid
                if inside.sum() < args.min_gc_px:
                    continue
                values = marker[inside]
                if values.max() - values.min() < 1e-6:
                    continue
                rows, cols = np.nonzero(inside)
                high = values >= np.median(values)
                for cls, keep in (("LZ", high), ("DZ", ~high)):
                    picked[cls][0].append(rows[keep])
                    picked[cls][1].append(cols[keep])
                    picked[cls][2].append(np.full(int(keep.sum()), gid, np.int32))
                used += 1
            for cls in ("LZ", "DZ"):
                rows, cols, gcs = picked[cls]
                if not rows:
                    continue
                r, c, g = np.concatenate(rows), np.concatenate(cols), np.concatenate(gcs)
                points = run.grid.to_xy(r, c).astype(np.float32)
                k = thin_idx(len(points), rng, args.max_points)
                arrays[f"{name}|{cls}"] = points[k]
                arrays[f"{name}|{cls}|gc"] = g[k]
            report.append(dict(section=plane.index, plane=name, kind="IF", source=channel,
                               rule=f"{channel} >= the GC's own median, inside the registered GC",
                               gcs_used=used,
                               LZ=int(len(arrays.get(f"{name}|LZ", []))),
                               DZ=int(len(arrays.get(f"{name}|DZ", [])))))
        else:
            path = tables[name]
            table = common.rctd_table(path)
            common.check_columns(table, columns.values(), path)
            joined = common.on_bins(run, name, table)
            joined = joined[joined.gc > 0]
            for cls in ("LZ", "DZ"):
                chosen = joined[joined[columns[cls]] >= args.min_weight]
                points = np.column_stack([chosen.wx.values, chosen.wy.values]).astype(np.float32)
                k = thin_idx(len(points), rng, args.max_points)
                arrays[f"{name}|{cls}"] = points[k]
                arrays[f"{name}|{cls}|gc"] = chosen.gc.values.astype(np.int32)[k]
            report.append(dict(section=plane.index, plane=name, kind="ST", source="RCTD", table=path.name,
                               rule=f"normalised weight >= {args.min_weight}, inside a GC",
                               LZ=int(len(arrays[f"{name}|LZ"])), DZ=int(len(arrays[f"{name}|DZ"]))))
        entry = report[-1]
        used = f"  ({entry['gcs_used']} GCs)" if "gcs_used" in entry else ""
        print(f"  {name:<10} {plane.kind}  LZ {entry['LZ']:>5,}  DZ {entry['DZ']:>5,}{used}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    out.with_name(out.stem + "_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
