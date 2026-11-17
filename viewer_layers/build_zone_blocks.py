#!/usr/bin/env python3
"""Build one dark-zone solid and one light-zone solid per germinal centre (04_columns/zone_blocks.npz).

A germinal centre (GC) has a dark zone (DZ) and a light zone (LZ). This builder
cuts each GC into one DZ solid and one LZ solid that together fill the GC
exactly, for the viewer's DZ / LZ view. (build_dzlz_layer.py instead keeps the
measured points, which can overlap and leave gaps.)

How each GC is cut:
1. A continuous "light-zone score" is set on every pixel of the GC on every
   section. Immunofluorescence (IF) sections: the light-zone marker
   (--zone-channel, for example CD4). Spatial-transcriptomics (ST) sections: the
   RCTD light-zone weight minus the dark-zone weight.
2. The score is z-scored inside the GC on each section, so IF and ST sections
   are on one scale.
3. The scores of all sections are stacked into one volume at the true section
   spacing and smoothed in 3D, including across sections. A section with noisy
   data then takes its polarity from its neighbours.
4. One threshold splits the whole volume. It is placed so that the light-zone
   share equals the GC's measured light-zone fraction: the share of its ST bins
   whose dominant RCTD type is LZ rather than DZ (needs 20 such bins, otherwise
   0.5; clamped to 0.25-0.75 so neither solid degenerates to a sliver).
5. Each zone is meshed with the same code as the GC columns
   (`gcalign.columns.build_column`), so the solids line up with the columns.
   If sections were skipped, the gaps are bridged the same way the columns
   bridge them (`gcalign.columns.stack_layout` and `fill_unmeasured`).

Output, in <run>/04_columns: zone_blocks.npz (DZ_v<id>, DZ_f<id>, DZ_sv<id>,
DZ_sf<id> and the same for LZ, plus gc_ids), zone_block_metrics.csv (one row per
GC) and zone_block_report.json.

Example:
    python viewer_layers/build_zone_blocks.py --config my_experiment.yaml \\
        --weights rctd_weights/ --zone-channel CD4_AF594
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import layer_common as common
from gcalign.columns import build_column, fill_unmeasured, stack_layout

SIGMA_Z, SIGMA_XY = 1.5, 2.5          # 3D smoothing of the score: slices, grid pixels
FRAC_CLAMP = (0.25, 0.75)
MIN_CALLS = 20                        # dominant-type ST bins needed to measure a GC's LZ fraction
PAD = 4                               # grid pixels of margin round each GC


def zscore(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)


def smooth_masked(field: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Normalised convolution: blur the score where it was measured, without
    bleeding in the zeros outside the GC (which would drag every edge towards
    the dark zone). Slices with no measurement are filled from their neighbours."""
    weight = weight.astype(np.float32)
    num = ndi.gaussian_filter(field * weight, (SIGMA_Z, SIGMA_XY, SIGMA_XY))
    den = ndi.gaussian_filter(weight, (SIGMA_Z, SIGMA_XY, SIGMA_XY))
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def main(argv=None) -> int:
    ap = common.parser(__doc__.split("\n")[0])
    common.add_weights(ap)
    ap.add_argument("--zone-channel", required=True,
                    help="IF channel that is high in the light zone, named as in the config (for example CD4_AF594)")
    ap.add_argument("--dz-column", default="DZ", help="RCTD weight column of the dark zone. Default: %(default)s")
    ap.add_argument("--lz-column", default="LZ", help="RCTD weight column of the light zone. Default: %(default)s")
    ap.add_argument("--dz-type", default=None,
                    help="RCTD first_type value of a dark-zone bin. Default: the --dz-column name")
    ap.add_argument("--lz-type", default=None,
                    help="RCTD first_type value of a light-zone bin. Default: the --lz-column name")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the three files. Default: <run>/04_columns")
    args = ap.parse_args(argv)

    run = common.open_run(args)
    cfg = run.cfg
    channel = common.require_channel(run, args.zone_channel, "--zone-channel")
    dz_type, lz_type = args.dz_type or args.dz_column, args.lz_type or args.lz_column
    out_dir = args.out_dir or run.root / "04_columns"
    grid = run.grid
    names = run.names
    planes = run.planes
    slots, pitch_um = stack_layout(cfg)
    if slots[-1] + 1 != len(slots):
        print(f"sections were skipped: {slots[-1] + 1 - len(slots)} slice position(s) are bridged as in the columns")

    tables = common.resolve_tables(run, args.weights, args.weights_pattern)

    # ---- the ST score, once per ST section ------------------------------------
    st_score, st_calls = {}, {}
    for plane in planes:
        if plane.kind != "ST":
            continue
        path = tables[plane.name]
        table = common.rctd_table(path)
        common.check_columns(table, (args.dz_column, args.lz_column), path)
        joined = common.on_bins(run, plane.name, table)
        joined = joined[joined.gc > 0]
        col = np.rint((joined.wx.values - grid.x0) / grid.um).astype(np.int32)
        row = np.rint((joined.wy.values - grid.y0) / grid.um).astype(np.int32)
        ok = grid.inside(row, col)
        raster = np.zeros(grid.shape, np.float32)
        raster[row[ok], col[ok]] = (joined[args.lz_column].values - joined[args.dz_column].values)[ok]
        st_score[plane.name] = raster
        st_calls[plane.name] = pd.DataFrame({"gc": joined.gc.values, "ft": joined.first_type.values})
        print(f"  {plane.name}: {len(joined):,} GC bins from {path.name}")
    if not st_score:
        raise SystemExit("the run has no ST section, so there is no RCTD score to cut the GCs with")

    # ---- each GC's light-zone fraction, from the dominant RCTD calls -------------
    calls = pd.concat(st_calls.values())
    fraction = {}
    for gid, group in calls.groupby("gc"):
        n_lz, n_dz = int((group.ft == lz_type).sum()), int((group.ft == dz_type).sum())
        fraction[int(gid)] = (n_lz / (n_lz + n_dz)) if (n_lz + n_dz) >= MIN_CALLS else 0.5
    measured = dict(fraction)
    fraction = {k: float(np.clip(v, *FRAC_CLAMP)) for k, v in fraction.items()}

    gc_ids = run.gc_ids
    print(f"{len(gc_ids)} GCs on the stack; light-zone fraction median "
          f"{np.median([measured.get(g, .5) for g in gc_ids]):.2f}, clamped on "
          f"{sum(1 for g in gc_ids if measured.get(g, .5) != fraction.get(g, .5))}")

    labels = np.stack([run.labels(n) for n in names])
    store, rows, kept = {}, [], []
    for gid in gc_ids:
        masks = labels == gid
        present = [i for i in range(len(names)) if masks[i].sum() >= cfg.columns.min_px_per_plane]
        if len(present) < cfg.columns.min_planes:
            continue
        ys, xs = np.nonzero(masks.any(axis=0))
        r0, r1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, masks.shape[1])
        c0, c1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, masks.shape[2])
        sub = masks[:, r0:r1, c0:c1]

        score = np.zeros(sub.shape, np.float32)
        for i, plane in enumerate(planes):
            if not sub[i].any():
                continue
            source = run.channel(plane.name, channel) if plane.kind == "IF" else st_score[plane.name]
            values = source[r0:r1, c0:c1].astype(np.float32)
            score[i][sub[i]] = zscore(values[sub[i]])            # one scale for IF and ST

        # lay the sections out at their true spacing, exactly as the columns are
        volume, _ = fill_unmeasured(sub, slots)
        field = np.zeros(volume.shape, np.float32)
        weight = np.zeros(volume.shape, bool)
        for i, slot in enumerate(slots):
            field[slot] = score[i]
            weight[slot] = sub[i]
        smoothed = smooth_masked(field, weight)
        share = fraction.get(gid, 0.5)
        threshold = float(np.quantile(smoothed[volume], 1.0 - share))   # one threshold for the volume
        lz = volume & (smoothed >= threshold)
        dz = volume & ~lz
        if lz.sum() < 50 or dz.sum() < 50:
            continue

        built = {}
        for cls, part in (("DZ", dz), ("LZ", lz)):
            mesh = build_column(cfg, part, grid.x0 + c0 * grid.um, grid.y0 + r0 * grid.um, 0.0,
                                z_pitch_um=pitch_um)
            if mesh is not None:
                built[cls] = mesh
        if len(built) < 2:
            continue
        for cls, (verts, faces, vol_um3, open_edges, shell_v, shell_f) in built.items():
            store[f"{cls}_v{gid}"] = verts
            store[f"{cls}_f{gid}"] = faces
            store[f"{cls}_sv{gid}"] = shell_v
            store[f"{cls}_sf{gid}"] = shell_f
        kept.append(gid)
        rows.append(dict(merged_gc_id=gid, planes=len(present),
                         lz_fraction_measured=round(measured.get(gid, 0.5), 3),
                         lz_fraction_used=round(share, 3),
                         dz_px=int(dz.sum()), lz_px=int(lz.sum()),
                         dz_volume_um3=round(built["DZ"][2], 1),
                         lz_volume_um3=round(built["LZ"][2], 1),
                         dz_open_edges=built["DZ"][3], lz_open_edges=built["LZ"][3]))

    store["gc_ids"] = np.asarray(kept, np.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "zone_blocks.npz", **store)
    metrics = pd.DataFrame(rows, columns=["merged_gc_id", "planes", "lz_fraction_measured", "lz_fraction_used",
                                          "dz_px", "lz_px", "dz_volume_um3", "lz_volume_um3",
                                          "dz_open_edges", "lz_open_edges"])
    metrics.to_csv(out_dir / "zone_block_metrics.csv", index=False)
    watertight = int((metrics.dz_open_edges == 0).sum() + (metrics.lz_open_edges == 0).sum())
    (out_dir / "zone_block_report.json").write_text(json.dumps(dict(
        run=cfg.name, gcs_built=len(kept), gcs_on_stack=len(gc_ids),
        field=f"IF = {channel} z-scored inside the GC; ST = RCTD ({args.lz_column} - {args.dz_column}) "
              "weight z-scored inside the GC",
        smoothing=f"3D Gaussian sigma_z={SIGMA_Z} slices, sigma_xy={SIGMA_XY} px, "
                  "normalised convolution inside the GC mask",
        split=f"one volume-wide quantile at the GC's measured light-zone fraction (clamped {FRAC_CLAMP})",
        slice_pitch_um=pitch_um, watertight=watertight, meshes=int(2 * len(kept)),
    ), indent=1), encoding="utf-8")
    print(f"built {len(kept)} of {len(gc_ids)} GCs -> {2 * len(kept)} meshes, {watertight} watertight")
    print(f"wrote {out_dir / 'zone_blocks.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
