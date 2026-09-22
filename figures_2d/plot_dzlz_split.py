#!/usr/bin/env python
"""3x3 close-up of one GC across the nine aligned sections, with the DZ/LZ split
drawn on top of what it was derived from.

The point is to be able to judge the split by eye, so the underlying signal is
never hidden: the IF panels keep the stain at full strength and the two zones go
over it as a light tint, with the dividing line reading as the colour change.
Red is CD4 in the composite, so on an IF panel the light zone should track the
red - that is the whole claim being checked here, and it is visible or it is not.

Rules reproduced exactly from build_dzlz_layer.py:
  IF  - inside each registered GC, CD4 is split at that GC's own median; the
        Tfh-rich half is the light zone. Unsmoothed, no tuned cut point.
  ST  - RCTD DZ / LZ, normalised weight >= 0.15, inside the curated GC region.

The ST panels are the transcriptional call on the neighbouring sections, i.e. the
evidence the IF rule was validated against, which is why they are in the same
figure rather than a separate one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
matplotlib.rcParams["hatch.linewidth"] = 1.35
import numpy as np
import pandas as pd

BASE = Path("/path/to/nine_section_output")
WEIGHTS = Path("/path/to/rctd_work/weights")
REPS = {
    "rep1": dict(root=BASE / "rep1",
                 planes=[("IF1", "IF"), ("D1", "ST"), ("IF3", "IF"), ("E3", "ST"), ("IF5", "IF"),
                         ("G4", "ST"), ("IF7", "IF"), ("G5", "ST"), ("IF9", "IF")],
                 chips={"D1": "B06289D1", "E3": "B06289E3", "G4": "B06289G4", "G5": "B06289G5"}),
    "rep2": dict(root=BASE / "rep2",
                 planes=[("IF1", "IF"), ("A4", "ST"), ("IF3", "IF"), ("C1", "ST"), ("IF5", "IF"),
                         ("C4", "ST"), ("IF7", "IF"), ("C5", "ST"), ("IF9", "IF")],
                 chips={"A4": "B06286A4", "C1": "B06286C1", "C4": "B06286C4", "C5": "B06286C5"}),
}
GRID_UM, DNB_UM, MIN_GC_PX, W_THRESH = 10.0, 0.5, 60, 0.15
BG, PANEL = "#05070d", "#0d1322"
# the dark-zone green is lifted off GL7's own green so the hatch reads
# against the stain; the Stereo-seq panels use the same two colours
DZ_COL, LZ_COL = "#5BFF6E", "#FFFFFF"
RGB_CHANNELS = [("CD4_AF594", 0), ("GL7_AF488", 1), ("IgD_AF647", 2)]


def square_window(union, shape):
    ys, xs = np.nonzero(union)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    spread = max(ys.ptp(), xs.ptp())
    half = spread / 2 + max(14, 0.30 * spread)
    r0 = int(max(cy - half, 0)); r1 = int(min(cy + half, shape[0]))
    c0 = int(max(cx - half, 0)); c1 = int(min(cx + half, shape[1]))
    side = min(r1 - r0, c1 - c0)
    return r0, r0 + side, c0, c0 + side


_CACHE = {}


def rep_context(rep):
    """Load a replicate's plane labels, IF channels and RCTD calls once.

    Every GC re-reads the same three npz files and four gzipped weight tables,
    which is most of the cost of a batch run, so the join to the scaffold is done
    once per replicate and the per-GC call is a filter on the result.
    """
    if rep in _CACHE:
        return _CACHE[rep]
    spec = REPS[rep]
    store = np.load(spec["root"] / "03_gc_match/plane_gc_labels.npz")
    channels = np.load(spec["root"] / "03_gc_match/if_channels_warped.npz")
    scaffold = np.load(spec["root"] / "00_scaffold/st_scaffold.npz")
    st = {}
    for short, chip in spec["chips"].items():
        d = pd.read_csv(WEIGHTS / f"{rep}_{chip}_rctd8.csv.gz",
                        usecols=["x", "y", "DZ", "LZ"])
        m = scaffold["slice_short"] == short
        key = pd.DataFrame({"x": np.rint(scaffold["chip_um"][m, 0] / DNB_UM).astype(np.int32),
                            "y": np.rint(scaffold["chip_um"][m, 1] / DNB_UM).astype(np.int32),
                            "wx": scaffold["world_um"][m, 0], "wy": scaffold["world_um"][m, 1],
                            "gc": scaffold["gc_id"][m]})
        j = d.merge(key, on=["x", "y"], how="inner")
        st[short] = j[j.gc > 0][["wx", "wy", "gc", "DZ", "LZ"]].reset_index(drop=True)
    _CACHE[rep] = dict(spec=spec, store=store, channels=channels, st=st,
                       labels={n: store[n] for n, _ in spec["planes"]})
    return _CACHE[rep]


def st_zone_points(ctx, short, gid):
    """RCTD DZ/LZ spots of one GC on one Stereo-seq section, in world microns."""
    j = ctx["st"][short]
    j = j[j.gc == gid]
    return {cls: j[j[cls] >= W_THRESH][["wx", "wy"]].to_numpy() for cls in ("DZ", "LZ")}


def draw_gc(rep, gid, out, dpi=230):
    ctx = rep_context(rep)
    spec, store, channels = ctx["spec"], ctx["store"], ctx["channels"]
    x0, y0, _, _ = (float(v) for v in store["grid"])

    masks = {n: store[n] == gid for n, _ in spec["planes"]}
    union = np.logical_or.reduce(list(masks.values()))
    r0, r1, c0, c1 = square_window(union, union.shape)
    extent = [x0 + c0 * GRID_UM, x0 + c1 * GRID_UM, y0 + r1 * GRID_UM, y0 + r0 * GRID_UM]

    if_names = [n for n, k in spec["planes"] if k == "IF"]
    limits = {}
    for name, _ in RGB_CHANNELS:
        pool = np.concatenate([channels[f"{p}|{name}"][r0:r1, c0:c1].ravel()
                               for p in if_names if f"{p}|{name}" in channels])
        limits[name] = (float(np.percentile(pool, 2)), float(np.percentile(pool, 99.6)))

    fig, axes = plt.subplots(3, 3, figsize=(15.0, 15.0), facecolor=BG)
    for index, (name, kind) in enumerate(spec["planes"]):
        ax = axes[index // 3][index % 3]
        ax.set_facecolor(PANEL)
        sub_gc = masks[name][r0:r1, c0:c1]

        if kind == "IF":
            rgb = np.zeros((r1 - r0, c1 - c0, 3), np.float32)
            for chan, band in RGB_CHANNELS:
                key = f"{name}|{chan}"
                if key in channels:
                    lo, hi = limits[chan]
                    a = channels[key][r0:r1, c0:c1].astype(np.float32)
                    rgb[..., band] = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
            ax.imshow(rgb, extent=extent, interpolation="bilinear", origin="upper")

            full = masks[name]
            if full.sum() >= MIN_GC_PX:
                cd4 = channels[f"{name}|CD4_AF594"].astype(np.float32)
                v = cd4[full]
                if v.max() - v.min() > 1e-6:
                    lz_full = np.zeros_like(full)
                    lz_full[full] = v >= np.median(v)      # CD4-high half = light zone
                    lz = (lz_full[r0:r1, c0:c1]) & sub_gc
                    dz = sub_gc & ~lz
                    # hatch rather than fill: the whole point is to judge whether the
                    # split follows CD4, and a wash over the stain hides exactly that
                    for m, colour, pattern in ((dz, DZ_COL, "\\\\"), (lz, LZ_COL, "////")):
                        ax.contourf(np.flipud(m.astype(float)), levels=[0.5, 1.5],
                                    colors="none", hatches=[pattern], extent=extent,
                                    origin="lower")
                        for coll in ax.collections[-1:]:
                            coll.set_edgecolor(colour)
                            coll.set_linewidth(0.0)
                    ax.contour(np.flipud(lz), levels=[0.5], colors="#00e5ff",
                               linewidths=1.6, extent=extent, origin="lower")
        else:
            canvas = np.zeros((*sub_gc.shape, 4))
            canvas[sub_gc] = mcolors.to_rgba("#2c3550", 1.0)
            ax.imshow(canvas, extent=extent, interpolation="nearest", origin="upper")
            pts = st_zone_points(ctx, name, gid)
            for cls, colour in (("DZ", DZ_COL), ("LZ", LZ_COL)):
                p = pts[cls]
                if len(p):
                    ax.scatter(p[:, 0], p[:, 1], s=13, marker="s", c=colour,
                               linewidths=0, alpha=0.95)

        if sub_gc.any():
            ax.contour(np.flipud(sub_gc), levels=[0.5], colors="#ffffff",
                       linewidths=2.0, extent=extent, origin="lower")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#1e2a45")

    fig.subplots_adjust(left=0.006, right=0.994, top=0.994, bottom=0.006,
                        wspace=0.02, hspace=0.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=sorted(REPS))
    ap.add_argument("--gc", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    print(draw_gc(a.rep, a.gc, a.out))
