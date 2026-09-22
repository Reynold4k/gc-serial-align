#!/usr/bin/env python
"""Every GC of both replicates, as a 3x3 DZ/LZ close-up.

Same figure as plot_dzlz_split.py, run over the whole set rather than a sample.
GCs that are absent from some planes are still drawn - the panel simply has no
outline there, which is itself worth seeing - so the manifest records how many
planes each GC actually appears on and on how many IF planes the median split
could be made at all, and that is what to sort by when looking through them.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_dzlz_split as P

OUT = Path(__file__).resolve().parent / "DZLZ"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep, spec in P.REPS.items():
        ctx = P.rep_context(rep)
        names = [n for n, _ in spec["planes"]]
        if_names = [n for n, k in spec["planes"] if k == "IF"]
        store = ctx["store"]
        ids = sorted({int(v) for n in names for v in np.unique(store[n]) if v > 0})
        print(f"{rep}: {len(ids)} GCs", flush=True)
        for gid in ids:
            t0 = time.time()
            px = {n: int((store[n] == gid).sum()) for n in names}
            out = OUT / f"{rep}_GC{gid}_DZLZ_nine_sections.png"
            P.draw_gc(rep, gid, out)
            rows.append(dict(
                rep=rep, gc_id=gid, file=out.name,
                planes_present=sum(1 for n in names if px[n] > 0),
                if_planes_with_split=sum(1 for n in if_names if px[n] >= P.MIN_GC_PX),
                total_bins=sum(px.values()),
                max_plane_bins=max(px.values()),
                mb=round(out.stat().st_size / 1e6, 2),
                **{f"px_{n}": px[n] for n in names}))
            print(f"  {out.name}  {px and sum(1 for n in names if px[n]>0)}/9 planes  "
                  f"[{time.time()-t0:.1f}s]", flush=True)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "manifest.csv", index=False)
    print(f"\n{len(m)} figures, {m.mb.sum()/1000:.2f} GB")
    print(m.groupby("rep")[["planes_present", "if_planes_with_split"]]
           .apply(lambda g: pd.Series({"on all nine": int((g.planes_present == 9).sum()),
                                       "IF split on all five": int((g.if_planes_with_split == 5).sum()),
                                       "n": len(g)})).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
