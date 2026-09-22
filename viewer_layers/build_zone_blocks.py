#!/usr/bin/env python3
"""One green DZ block and one white LZ block per germinal centre.

The existing DZ/LZ layer is two independent point clouds: each spot is kept if it
passes a threshold, so the two classes overlap in places and leave gaps in others.
That cannot be smoothed into two solid blocks. This builder instead makes DZ and LZ
a *partition* of the curated GC region, and does the smoothing where the column
builder does it - in the voxel volume, not on the mesh.

Three things have to be true for two clean blocks to come out:

1. The two classes must tile the GC. So every GC pixel is assigned, by splitting a
   continuous "LZ-ness" field rather than by thresholding DZ and LZ separately.
   IF planes: CD4 (Tfh-rich half = light zone) - the rule already validated against
   the neighbouring Stereo-seq sections at 73% (rep1) / 83% (rep2) sign agreement.
   ST planes: the RCTD LZ - DZ weight difference. Inside a GC this field has a local
   autocorrelation of 0.45-0.67 against its own 25 um neighbourhood mean, so it
   carries real spatial structure and not just doublet-mode noise.

2. The two modalities must be on the same scale, or a bright IF plane would come out
   all light zone. Each plane's field is z-scored inside that GC before anything else.

3. The polarity must not flip between neighbouring planes, or the blocks interleave
   and smoothing turns them to mush. The IF-vs-ST validation says the axis does
   disagree on roughly a quarter of GCs. So the field is assembled as a 9-plane
   volume and Gaussian-smoothed *in 3D including z* before the split, and the split
   uses one threshold for the whole volume instead of one per plane. A plane whose
   own data is noisy then borrows its polarity from its neighbours, while a boundary
   the planes agree on keeps its real shape.

The split point is the GC's own light-zone fraction, measured from the RCTD
first_type calls on its Stereo-seq planes (clamped to 0.25-0.75 so neither block
degenerates to a sliver). So the two blocks carry the measured DZ:LZ ratio of that
GC, not a forced 50/50.

Output: 04_columns/zone_blocks.npz, keyed DZ_v<gid>/DZ_f<gid>/DZ_sv/DZ_sf and the
same for LZ, plus a per-GC metrics table.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi

PKG = Path("/path/to/gc-serial-align")
sys.path.insert(0, str(PKG))
from gcalign import config as gcfg                       # noqa: E402
from gcalign.columns import build_column                 # noqa: E402

GRID_UM, DNB_UM = 10.0, 0.5
SIGMA_Z, SIGMA_XY = 1.5, 2.5      # planes, px - the 3D pre-smoothing of the field
FRAC_CLAMP = (0.25, 0.75)
MIN_PX_PER_PLANE, MIN_PLANES, MIN_BLOCK_PX = 8, 2, 50   # match the column view, then let
                                                       # build_column reject what it cannot march

REPS = {
    "rep1": dict(root=PKG.parent / "rep1", yaml=PKG / "configs/rep1_mouse_spleen.yaml",
                 st=["D1", "E3", "G4", "G5"],
                 chips=["B06289D1", "B06289E3", "B06289G4", "B06289G5"]),
    "rep2": dict(root=PKG.parent / "rep2", yaml=PKG / "configs/rep2_mouse_spleen.yaml",
                 st=["A4", "C1", "C4", "C5"],
                 chips=["B06286A4", "B06286C1", "B06286C4", "B06286C5"]),
}


def zscore(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)


def smooth_masked(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalised convolution: blur inside the mask without bleeding in the zeros
    outside it, which would otherwise drag every GC edge towards the dark zone."""
    num = ndi.gaussian_filter(field * mask, (SIGMA_Z, SIGMA_XY, SIGMA_XY))
    den = ndi.gaussian_filter(mask.astype(np.float32), (SIGMA_Z, SIGMA_XY, SIGMA_XY))
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def main(rep: str, weights_dir: Path, out_dir: Path) -> None:
    cfg_spec = REPS[rep]
    cfg = gcfg.load(cfg_spec["yaml"])
    cfg.output_dir = cfg_spec["root"]                    # rep1's yaml has a relative path
    root = cfg_spec["root"]

    lab = np.load(root / "03_gc_match/plane_gc_labels.npz")
    ch = np.load(root / "03_gc_match/if_channels_warped.npz")
    sc = np.load(root / "00_scaffold/st_scaffold.npz")
    x0, y0 = float(lab["grid"][0]), float(lab["grid"][1])
    planes = [(n, k, s) for n, k, s in cfg.planes]
    names = [n for n, _, _ in planes]

    # ---- the ST field, once per Stereo-seq plane ---------------------------
    st_field, st_call = {}, {}
    for short, chip in zip(cfg_spec["st"], cfg_spec["chips"]):
        d = pd.read_csv(Path(weights_dir) / f"{rep}_{chip}_rctd8.csv.gz")
        m = sc["slice_short"] == short
        key = pd.DataFrame({"x": np.rint(sc["chip_um"][m, 0] / DNB_UM).astype(np.int32),
                            "y": np.rint(sc["chip_um"][m, 1] / DNB_UM).astype(np.int32),
                            "wx": sc["world_um"][m, 0], "wy": sc["world_um"][m, 1],
                            "gc": sc["gc_id"][m]})
        j = d.merge(key, on=["x", "y"], how="inner")
        j = j[j.gc > 0]
        col = np.rint((j.wx.values - x0) / GRID_UM).astype(np.int32)
        row = np.rint((j.wy.values - y0) / GRID_UM).astype(np.int32)
        H, W = lab[short].shape
        ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
        grid = np.zeros((H, W), np.float32)
        grid[row[ok], col[ok]] = (j.LZ.values - j.DZ.values)[ok]
        st_field[short] = grid
        st_call[short] = pd.DataFrame({"gc": j.gc.values, "ft": j.first_type.values})
        print(f"  {short}: {len(j):,} GC spots gridded")

    # ---- per-GC light-zone fraction from the RCTD dominant calls -----------
    allcalls = pd.concat(st_call.values())
    frac = {}
    for gid, g in allcalls.groupby("gc"):
        n_lz, n_dz = int((g.ft == "LZ").sum()), int((g.ft == "DZ").sum())
        frac[int(gid)] = (n_lz / (n_lz + n_dz)) if (n_lz + n_dz) >= 20 else 0.5
    raw_frac = dict(frac)
    for k in frac:
        frac[k] = float(np.clip(frac[k], *FRAC_CLAMP))

    gc_ids = sorted({int(v) for n in names for v in np.unique(lab[n]) if v > 0})
    print(f"{len(gc_ids)} GCs on the stack; LZ fraction "
          f"median {np.median([raw_frac.get(g, .5) for g in gc_ids]):.2f}, "
          f"clamped on {sum(1 for g in gc_ids if raw_frac.get(g, .5) != frac.get(g, .5))}")

    store, rows = {}, []
    kept = []
    for gid in gc_ids:
        masks = np.stack([lab[n] == gid for n in names])
        present = [i for i in range(len(names)) if masks[i].sum() >= MIN_PX_PER_PLANE]
        if len(present) < MIN_PLANES:
            continue
        ys, xs = np.nonzero(masks.any(axis=0))
        pad = 4
        r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, masks.shape[1])
        c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, masks.shape[2])
        sub = masks[:, r0:r1, c0:c1]

        field = np.zeros_like(sub, np.float32)
        for i, (name, kind, _) in enumerate(planes):
            if not sub[i].any():
                continue
            raw = (ch[f"{name}|CD4_AF594"] if kind == "IF" else st_field[name])
            v = raw[r0:r1, c0:c1].astype(np.float32)
            field[i][sub[i]] = zscore(v[sub[i]])          # same scale for IF and ST

        sm = smooth_masked(field, sub)
        vals = sm[sub]
        p = frac.get(gid, 0.5)
        thr = float(np.quantile(vals, 1.0 - p))           # one threshold for the volume
        lz = sub & (sm >= thr)
        dz = sub & ~lz
        if lz.sum() < MIN_BLOCK_PX or dz.sum() < MIN_BLOCK_PX:
            continue

        built = {}
        for cls, vol in (("DZ", dz), ("LZ", lz)):
            b = build_column(cfg, vol, x0 + c0 * GRID_UM, y0 + r0 * GRID_UM, 0.0)
            if b is not None:
                built[cls] = b
        if len(built) < 2:
            continue
        for cls, (verts, faces, vol_um3, open_edges, sv, sf) in built.items():
            store[f"{cls}_v{gid}"] = verts
            store[f"{cls}_f{gid}"] = faces
            store[f"{cls}_sv{gid}"] = sv
            store[f"{cls}_sf{gid}"] = sf
        kept.append(gid)
        rows.append(dict(merged_gc_id=gid, planes=len(present),
                         lz_fraction_measured=round(raw_frac.get(gid, 0.5), 3),
                         lz_fraction_used=round(p, 3),
                         dz_px=int(dz.sum()), lz_px=int(lz.sum()),
                         dz_volume_um3=round(built["DZ"][2], 1),
                         lz_volume_um3=round(built["LZ"][2], 1),
                         dz_open_edges=built["DZ"][3], lz_open_edges=built["LZ"][3]))

    store["gc_ids"] = np.asarray(kept, np.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "zone_blocks.npz", **store)
    met = pd.DataFrame(rows)
    met.to_csv(out_dir / "zone_block_metrics.csv", index=False)
    (out_dir / "zone_block_report.json").write_text(json.dumps(dict(
        rep=rep, gcs_built=len(kept), gcs_on_stack=len(gc_ids),
        field="IF = CD4 z-scored in-GC; ST = RCTD (LZ - DZ) weight z-scored in-GC",
        smoothing=f"3D Gaussian sigma_z={SIGMA_Z} planes, sigma_xy={SIGMA_XY} px, "
                  "normalised convolution inside the GC mask",
        split="single volume-wide quantile at the GC's measured LZ fraction "
              f"(clamped {FRAC_CLAMP})",
        watertight=int((met.dz_open_edges == 0).sum() + (met.lz_open_edges == 0).sum()),
        meshes=int(2 * len(kept)),
    ), indent=1))
    print(f"  built {len(kept)}/{len(gc_ids)} GCs -> {2*len(kept)} meshes, "
          f"{int((met.dz_open_edges==0).sum()+(met.lz_open_edges==0).sum())} watertight")
    print(f"  wrote {out_dir/'zone_blocks.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=list(REPS))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.rep, Path(a.weights), Path(a.out))
