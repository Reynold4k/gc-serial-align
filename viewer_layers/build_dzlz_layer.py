#!/usr/bin/env python3
"""Dark-zone / light-zone point clouds for all nine planes.

IF planes  -> inside each registered GC, the area is split on CD4 at that GC's own
              median: the Tfh-rich half is the light zone, the Tfh-poor half the dark
              zone. No smoothing and no tuned cut point — a sweep over sigma
              (0 / 1.2 / 2.5 px) and split (50/50 … 80/20) showed smoothing costs up to
              15 points of agreement while the split point changes almost nothing, so
              the least-parameterised choice wins. Validated against the RCTD DZ/LZ axis
              on the neighbouring Stereo-seq sections: the two agree in sign for 73%
              (rep1) and 83% (rep2) of GCs, against 50% for a coin.
ST planes  -> RCTD DZ / LZ, normalised weight >= 0.15, inside the curated GC region.

Output: 05_viewer/dzlz_points.npz with "<plane>|DZ" / "<plane>|LZ" world-um arrays
and a matching "<plane>|<cls>|gc" curated GC id per point.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi

GRID_UM, DNB_UM, MAX_PER_CLASS, MIN_GC_PX = 10.0, 0.5, 5000, 60
REPS = {
    "rep1": dict(root="/path/to/nine_section_output/rep1",
                 st=["D1","E3","G4","G5"], chips=["B06289D1","B06289E3","B06289G4","B06289G5"]),
    "rep2": dict(root="/path/to/nine_section_output/rep2",
                 st=["A4","C1","C4","C5"], chips=["B06286A4","B06286C1","B06286C4","B06286C5"]),
}


def thin_idx(n, rng, cap=MAX_PER_CLASS):
    """Indices to keep, so the points and their GC ids are thinned together."""
    return np.arange(n) if n <= cap else rng.choice(n, cap, replace=False)


def main(rep, weights_dir, out):
    cfg = REPS[rep]; root = Path(cfg["root"])
    ch  = np.load(root / "03_gc_match/if_channels_warped.npz")
    lab = np.load(root / "03_gc_match/plane_gc_labels.npz")
    sc  = np.load(root / "00_scaffold/st_scaffold.npz")
    x0, y0, H, W = [float(v) for v in lab["grid"]]
    rng = np.random.default_rng(20260917)
    arrays, report = {}, []

    planes = ["IF1", None, "IF3", None, "IF5", None, "IF7", None, "IF9"]
    for i, s in enumerate(cfg["st"]):
        planes[1 + 2 * i] = s

    for section, name in enumerate(planes, start=1):
        if name.startswith("IF"):
            cd4 = ch[f"{name}|CD4_AF594"].astype(np.float32)   # unsmoothed: see the docstring
            ids = np.unique(lab[name]); ids = ids[ids > 0]
            lz_r, lz_c, lz_g, dz_r, dz_c, dz_g, used = [], [], [], [], [], [], 0
            for gid in ids:
                m = lab[name] == gid
                if m.sum() < MIN_GC_PX:
                    continue
                v = cd4[m]
                if v.max() - v.min() < 1e-6:
                    continue
                rr, cc = np.nonzero(m); hi = v >= np.median(v)
                lz_r.append(rr[hi]); lz_c.append(cc[hi])
                lz_g.append(np.full(int(hi.sum()), gid, np.int32))
                dz_r.append(rr[~hi]); dz_c.append(cc[~hi])
                dz_g.append(np.full(int((~hi).sum()), gid, np.int32)); used += 1
            for cls, R, C, G in (("LZ", lz_r, lz_c, lz_g), ("DZ", dz_r, dz_c, dz_g)):
                if not R:
                    continue
                r = np.concatenate(R); c = np.concatenate(C); g = np.concatenate(G)
                pts = np.column_stack([x0 + c * GRID_UM, y0 + r * GRID_UM]).astype(np.float32)
                k = thin_idx(len(pts), rng)
                arrays[f"{name}|{cls}"] = pts[k]
                # the GC each point belongs to, so selecting a GC filters the cloud
                arrays[f"{name}|{cls}|gc"] = g[k]
            report.append(dict(section=section, plane=name, source="IgD/GL7/CD4 IF",
                               rule="CD4 >= that GC's own median, inside the registered GC",
                               gcs_used=used,
                               LZ=int(len(arrays.get(f"{name}|LZ", []))),
                               DZ=int(len(arrays.get(f"{name}|DZ", [])))))
        else:
            chip = cfg["chips"][cfg["st"].index(name)]
            d = pd.read_csv(Path(weights_dir) / f"{rep}_{chip}_rctd8.csv.gz")
            m = sc["slice_short"] == name
            key = pd.DataFrame({"x": np.rint(sc["chip_um"][m, 0] / DNB_UM).astype(np.int32),
                                "y": np.rint(sc["chip_um"][m, 1] / DNB_UM).astype(np.int32),
                                "wx": sc["world_um"][m, 0], "wy": sc["world_um"][m, 1],
                                "gc": sc["gc_id"][m]})
            j = d.merge(key, on=["x", "y"], how="inner")
            j = j[j.gc > 0]
            for cls in ("LZ", "DZ"):
                sel = j[j[cls] >= 0.15]
                pts = np.column_stack([sel.wx.values, sel.wy.values]).astype(np.float32)
                k = thin_idx(len(pts), rng)
                arrays[f"{name}|{cls}"] = pts[k]
                arrays[f"{name}|{cls}|gc"] = sel.gc.values.astype(np.int32)[k]
            report.append(dict(section=section, plane=name, source="RCTD",
                               rule="normalised weight >= 0.15, inside the curated GC region",
                               LZ=int(len(arrays[f"{name}|LZ"])), DZ=int(len(arrays[f"{name}|DZ"]))))
        r = report[-1]
        print(f"  {name:<4} {r['source']:<15} LZ {r['LZ']:>5,}  DZ {r['DZ']:>5,}"
              f"{'  (' + str(r.get('gcs_used')) + ' GCs)' if 'gcs_used' in r else ''}")

    np.savez_compressed(out, **arrays)
    Path(str(out).replace(".npz", "_report.json")).write_text(json.dumps(report, indent=1))
    print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=list(REPS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.rep, a.weights, a.out)
