#!/usr/bin/env python3
"""Compute a B-follicle point cloud for all nine planes of one replicate.

IF planes  -> IgD (AF647), Otsu-thresholded inside the tissue, cleaned.
             IgD marks the mantle/follicle and is naturally low inside the GC,
             which is exactly the backdrop effect wanted.
ST planes  -> RCTD first_type in {B naive, B memory, B MZ}, the rule that gives a
             follicle rather than a haze (summed doublet-mode weights do not).

Output: 05_viewer/follicle_points.npz with one <f4 (n,2) world-um array per plane.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology

GRID_UM = 10.0
DNB_UM = 0.5
MAX_POINTS = 6000          # per plane; a backdrop does not need every pixel
MIN_BLOB_PX = 25

REPS = {
    "rep1": dict(root="/path/to/nine_section_output/rep1",
                 st=["D1", "E3", "G4", "G5"],
                 chips=["B06289D1", "B06289E3", "B06289G4", "B06289G5"]),
    "rep2": dict(root="/path/to/nine_section_output/rep2",
                 st=["A4", "C1", "C4", "C5"],
                 chips=["B06286A4", "B06286C1", "B06286C4", "B06286C5"]),
}
ORDER = ["IF1", None, "IF3", None, "IF5", None, "IF7", None, "IF9"]   # ST slots filled per rep


def thin(points, rng, cap=MAX_POINTS):
    if len(points) <= cap:
        return points
    return points[rng.choice(len(points), cap, replace=False)]


def main(rep, weights_dir, out):
    cfg = REPS[rep]
    root = Path(cfg["root"])
    ch = np.load(root / "03_gc_match/if_channels_warped.npz")
    lab = np.load(root / "03_gc_match/plane_gc_labels.npz")
    x0, y0, h, w = [float(v) for v in lab["grid"]]
    rng = np.random.default_rng(20260917)
    out_arrays, report = {}, []

    planes = list(ORDER)
    for i, s in enumerate(cfg["st"]):
        planes[1 + 2 * i] = s

    for section, name in enumerate(planes, start=1):
        if name.startswith("IF"):
            igd = ch[f"{name}|IgD_AF647"].astype(np.float32)
            tissue = sum(ch[f"{name}|{c}"].astype(np.float32)
                         for c in ("IgD_AF647", "GL7_AF488", "CD4_AF594")) > 0
            vals = igd[tissue]
            thr = float(filters.threshold_otsu(vals))
            mask = (igd > thr) & tissue
            mask = morphology.binary_closing(mask, morphology.disk(2))
            mask = morphology.remove_small_objects(mask, MIN_BLOB_PX)
            gc = lab[name] > 0
            overlap = float((mask & gc).sum() / max(mask.sum(), 1))
            rows, cols = np.nonzero(mask)
            pts = np.column_stack([x0 + cols * GRID_UM, y0 + rows * GRID_UM]).astype(np.float32)
            src, extra = "IgD_AF647", dict(otsu=round(thr, 1), gc_overlap=round(overlap, 4))
        else:
            chip = cfg["chips"][cfg["st"].index(name)]
            d = pd.read_csv(Path(weights_dir) / f"{rep}_{chip}_rctd8.csv.gz")
            d = d[d.first_type.isin(["B naive", "B memory", "B MZ"])]
            # RCTD x/y are chip units; the scaffold carries the aligned world position
            sc = np.load(root / "00_scaffold/st_scaffold.npz")
            m = sc["slice_short"] == name
            key = pd.DataFrame({"sx": np.rint(sc["chip_um"][m, 0] / DNB_UM).astype(np.int32),
                                "sy": np.rint(sc["chip_um"][m, 1] / DNB_UM).astype(np.int32),
                                "wx": sc["world_um"][m, 0], "wy": sc["world_um"][m, 1]})
            j = d.rename(columns={"x": "sx", "y": "sy"}).merge(key, on=["sx", "sy"], how="inner")
            pts = np.column_stack([j.wx.values, j.wy.values]).astype(np.float32)
            src, extra = "RCTD first_type", dict(matched=int(len(j)), candidates=int(len(d)))
            if len(j) < 0.8 * len(d):
                raise SystemExit(f"{name}: only {len(j)}/{len(d)} follicle spots matched the scaffold")
        kept = thin(pts, rng)
        out_arrays[name] = kept
        report.append(dict(section=section, plane=name, source=src,
                           found=int(len(pts)), kept=int(len(kept)), **extra))
        print(f"  {name:<4} {src:<16} found {len(pts):>7,}  kept {len(kept):>6,}  {extra}")

    np.savez_compressed(out, **out_arrays)
    Path(str(out).replace(".npz", "_report.json")).write_text(json.dumps(report, indent=1))
    print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=list(REPS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.rep, a.weights, a.out)
