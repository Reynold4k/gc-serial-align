#!/usr/bin/env python3
"""Full-resolution true-colour IF composites for the nine-slice viewer panel.

Red = CD4 (AF594), green = GL7 (AF488), blue = IgD (AF647) — the ordinary dye
colours, and the same convention as the single-GC close-up figures. Contrast
limits are shared across the five IF sections of a replicate so brightness is
comparable between them.

Also emits, per plane, the outline of every registered GC in image pixel
coordinates so the panel can draw boundaries and ids over the photograph.
"""
from __future__ import annotations
import argparse, base64, io, json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import measure

CH = [("CD4_AF594", "R"), ("GL7_AF488", "G"), ("IgD_AF647", "B")]
LO_P, HI_P, JPEG_Q = 1.0, 99.7, 88
REPS = {"rep1": "/path/to/nine_section_output/rep1",
        "rep2": "/path/to/nine_section_output/rep2"}
IF_PLANES = ["IF1", "IF3", "IF5", "IF7", "IF9"]


def main(rep, out):
    root = Path(REPS[rep])
    ch = np.load(root / "03_gc_match/if_channels_warped.npz")
    lab = np.load(root / "03_gc_match/plane_gc_labels.npz")
    x0, y0, H, W = [float(v) for v in lab["grid"]]

    # shared contrast limits per channel across the five sections
    lim = {}
    for name, _ in CH:
        pool = np.concatenate([ch[f"{p}|{name}"][ch[f"{p}|{name}"] > 0].ravel()[::7]
                               for p in IF_PLANES])
        lim[name] = (float(np.percentile(pool, LO_P)), float(np.percentile(pool, HI_P)))
        print(f"  {name}: shared limits {lim[name][0]:.0f} – {lim[name][1]:.0f}")

    payload, total = [], 0
    for p in IF_PLANES:
        planes = []
        for name, _ in CH:
            a = ch[f"{p}|{name}"].astype(np.float32)
            lo, hi = lim[name]
            planes.append(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1))
        rgb = (np.dstack(planes) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=JPEG_Q, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        total += len(b64)

        # GC outlines in image pixel coordinates
        L = lab[p]
        outlines = []
        for gid in np.unique(L):
            if gid <= 0:
                continue
            m = L == gid
            if m.sum() < 25:
                continue
            for c in measure.find_contours(m.astype(np.float32), 0.5):
                if len(c) < 8:
                    continue
                step = max(1, len(c) // 120)          # <=120 points is plenty at this scale
                pts = c[::step][:, ::-1]               # (row,col) -> (x,y)
                outlines.append(dict(gc=int(gid),
                                     xy=[[round(float(a), 1), round(float(b), 1)] for a, b in pts]))
            rr, cc = np.nonzero(m)
        labels = []
        for gid in np.unique(L):
            if gid <= 0:
                continue
            m = L == gid
            if m.sum() < 25:
                continue
            rr, cc = np.nonzero(m)
            labels.append(dict(gc=int(gid), x=round(float(cc.mean()), 1), y=round(float(rr.mean()), 1)))
        payload.append(dict(plane=p, w=int(L.shape[1]), h=int(L.shape[0]),
                            x0=x0, y0=y0, um=10.0, jpeg=b64,
                            outlines=outlines, labels=labels))
        print(f"  {p}: {L.shape[1]}x{L.shape[0]}  jpeg {len(b64)/1e3:.0f} kB b64  "
              f"{len(outlines)} outline rings, {len(labels)} GC labels")

    Path(out).write_text(json.dumps(dict(rep=rep, channels="R=CD4 · G=GL7 · B=IgD",
                                         limits={k: list(v) for k, v in lim.items()},
                                         planes=payload), separators=(",", ":")),
                         encoding="utf-8")
    print(f"  wrote {out}  ({Path(out).stat().st_size/1e6:.2f} MB, images {total/1e6:.2f} MB b64)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=list(REPS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.rep, a.out)
