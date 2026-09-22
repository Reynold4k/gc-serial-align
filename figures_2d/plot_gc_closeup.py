#!/usr/bin/env python
"""3x3 close-up of one germinal centre across all nine aligned sections.

Everything is drawn from the aligned label maps, so the nine panels share one
coordinate frame: a shift between panels is a real shift of the GC, not a
registration artefact. The previous section's outline is ghosted onto each panel
so growth and shrinkage read directly, and where the GC is absent the panel says
what occupies its footprint instead - "nothing" and "the neighbouring GC" are
very different problems.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

REPS = {
    "rep1": dict(root=Path("/path/to/nine_section_output/rep1"),
                 planes=[("IF1", "IF"), ("D1", "ST"), ("IF3", "IF"), ("E3", "ST"), ("IF5", "IF"),
                         ("G4", "ST"), ("IF7", "IF"), ("G5", "ST"), ("IF9", "IF")],
                 accent="#00e5d0"),
    "rep2": dict(root=Path("/path/to/nine_section_output/rep2"),
                 planes=[("IF1", "IF"), ("A4", "ST"), ("IF3", "IF"), ("C1", "ST"), ("IF5", "IF"),
                         ("C4", "ST"), ("IF7", "IF"), ("C5", "ST"), ("IF9", "IF")],
                 accent="#ff5a3c"),
}
GRID_UM = 10.0
BG, PANEL, DIM, GHOST, WARN = "#05070d", "#0d1322", "#2c3550", "#93a7cc", "#ffb066"

# IF panels show the stain itself, in the conventional fluorophore colours:
# AF594 red, AF488 green, AF647 blue. Stereo-seq panels stay as flat GC regions -
# there is no image to show there, only the transcriptional call.
RGB_CHANNELS = [("CD4_AF594", "red"), ("GL7_AF488", "green"), ("IgD_AF647", "blue")]
LEGEND = "IF sections show the stain: red CD4 (AF594) · green GL7 (AF488) · blue IgD (AF647)"


def square_window(union: np.ndarray, shape) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(union)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.ptp(), xs.ptp()) / 2 + max(16, 0.42 * max(ys.ptp(), xs.ptp()))
    r0 = int(max(cy - half, 0)); r1 = int(min(cy + half, shape[0]))
    c0 = int(max(cx - half, 0)); c1 = int(min(cx + half, shape[1]))
    side = min(r1 - r0, c1 - c0)
    return r0, r0 + side, c0, c0 + side


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=sorted(REPS))
    ap.add_argument("--gc", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=240)
    args = ap.parse_args()

    spec = REPS[args.rep]
    root, planes, accent = spec["root"], spec["planes"], spec["accent"]
    store = np.load(root / "03_gc_match/plane_gc_labels.npz")
    channels = np.load(root / "03_gc_match/if_channels_warped.npz")
    x0, y0, _, _ = (float(v) for v in store["grid"])
    metrics = pd.read_csv(root / "04_columns/gc_column_metrics.csv").set_index("merged_gc_id")

    masks = {name: store[name] == args.gc for name, _ in planes}
    present = [n for n, _ in planes if masks[n].sum() > 0]
    if not present:
        raise SystemExit(f"GC{args.gc} does not appear on any plane of {args.rep}")

    union = np.logical_or.reduce([masks[n] for n, _ in planes])
    r0, r1, c0, c1 = square_window(union, union.shape)
    extent = [x0 + c0 * GRID_UM, x0 + c1 * GRID_UM, y0 + r1 * GRID_UM, y0 + r0 * GRID_UM]

    if_names = [n for n, k in planes if k == "IF"]
    limits = {}
    for name, _ in RGB_CHANNELS:
        pool = np.concatenate([channels[f"{p}|{name}"][r0:r1, c0:c1].ravel()
                               for p in if_names if f"{p}|{name}" in channels])
        limits[name] = (float(np.percentile(pool, 2)), float(np.percentile(pool, 99.6)))

    def composite(plane_name):
        rgb = np.zeros((r1 - r0, c1 - c0, 3), np.float32)
        for (name, band), index in zip(RGB_CHANNELS, range(3)):
            key = f"{plane_name}|{name}"
            if key not in channels:
                continue
            a = channels[key][r0:r1, c0:c1].astype(np.float32)
            lo, hi = limits[name]
            rgb[..., index] = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        return rgb

    fig, axes = plt.subplots(3, 3, figsize=(16.0, 17.4), facecolor=BG)
    row = metrics.loc[args.gc] if args.gc in metrics.index else None
    head = (f"{args.rep}    GC{args.gc}    {len(present)}/9 sections"
            + (f"    {row.volume_um3/1e6:.2f} nL    max radius {row.max_equiv_radius_um:.0f} µm"
               f"    radius CV {row.radius_cv:.3f}    centroid drift {row.centroid_drift_um:.0f} µm"
               if row is not None else ""))
    fig.suptitle(head, color="#e6ecfa", fontsize=18, fontweight="bold", y=0.989)
    fig.text(0.5, 0.947,
             "one aligned coordinate frame · sections 10 µm apart, cutting order · "
             "white outline = this GC · dashed = this GC on the previous section\n" + LEGEND,
             ha="center", color="#8fa3c8", fontsize=12)

    previous = None
    for index, (name, kind) in enumerate(planes):
        ax = axes[index // 3][index % 3]
        ax.set_facecolor(PANEL)
        sub_all = store[name][r0:r1, c0:c1]
        sub_gc = masks[name][r0:r1, c0:c1]

        if kind == "IF":
            ax.imshow(composite(name), extent=extent, interpolation="bilinear",
                      origin="upper")
            others = (sub_all > 0) & ~sub_gc
            if others.any():
                ax.contour(np.flipud(others), levels=[0.5], colors=GHOST,
                           linewidths=0.8, extent=extent, origin="lower", alpha=0.45)
        else:
            canvas = np.zeros((*sub_all.shape, 4))
            canvas[(sub_all > 0) & ~sub_gc] = mcolors.to_rgba(DIM, 1.0)
            canvas[sub_gc] = mcolors.to_rgba(accent, 1.0)
            ax.imshow(canvas, extent=extent, interpolation="nearest", origin="upper")

        if sub_gc.any():
            ax.contour(np.flipud(sub_gc), levels=[0.5], colors="#ffffff",
                       linewidths=1.9, extent=extent, origin="lower")
        if previous is not None:
            ax.contour(np.flipud(previous), levels=[0.5], colors=GHOST, linewidths=1.2,
                       linestyles="--", extent=extent, origin="lower", alpha=0.9)

        area_px = int(sub_gc.sum())
        ax.set_title(f"{name}   ·   {kind}   ·   z = {index * 10} µm",
                     color=accent if kind == "IF" else "#e6ecfa",
                     fontsize=15, fontweight="600", pad=10)
        if area_px:
            area = area_px * GRID_UM ** 2
            caption = f"{area/1000:.1f}×10³ µm²    r ≈ {np.sqrt(area/np.pi):.0f} µm"
            colour = "#cfe0ff"
            previous = sub_gc.copy()
        else:
            foot = store[name][union][:]            # what sits in the GC's footprint here
            vals, counts = np.unique(foot, return_counts=True)
            order = np.argsort(-counts)
            parts = []
            for v, c in zip(vals[order][:2], counts[order][:2]):
                parts.append(f"{'no GC' if v == 0 else f'GC{int(v)}'} {100*c/len(foot):.0f}%")
            caption = "absent here — footprint holds " + ", ".join(parts)
            colour = WARN
        ax.text(0.5, -0.062, caption, transform=ax.transAxes, ha="center",
                color=colour, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#1e2a45")

    # scale bar in axes coordinates: the panel is `span` micrometres wide, so the
    # bar is a known fraction of it regardless of how the axis is oriented
    ax = axes[2][2]
    span = extent[1] - extent[0]
    frac = 100.0 / span
    x, y = 0.055, 0.055
    ax.add_patch(Rectangle((x - 0.03, y - 0.03), frac + 0.06, 0.115,
                           transform=ax.transAxes, facecolor="#05070d",
                           edgecolor="none", alpha=0.85, zorder=4))
    ax.add_patch(Rectangle((x, y), frac, 0.013, transform=ax.transAxes,
                           facecolor="#ffffff", edgecolor="none", zorder=5))
    ax.text(x + frac / 2, y + 0.028, "100 µm", transform=ax.transAxes, color="#ffffff",
            fontsize=12, ha="center", va="bottom", zorder=5)

    fig.subplots_adjust(left=0.012, right=0.988, top=0.918, bottom=0.022,
                        wspace=0.05, hspace=0.20)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=BG)
    seq = " → ".join(f"{int(masks[n].sum())*100/1000:.0f}" if masks[n].sum() else "—"
                     for n, _ in planes)
    print(f"{args.rep} GC{args.gc}: {len(present)}/9 · area ×10³µm² {seq}")
    print(f"   -> {args.out}  ({int(16.0*args.dpi)}×{int(17.4*args.dpi)} px)")


if __name__ == "__main__":
    main()
