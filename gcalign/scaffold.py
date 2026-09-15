"""Stage 1 - read the frozen Stereo-seq scaffold.

Nothing here moves any coordinate. The spatial transcriptomics sections were
already registered to each other before this pipeline starts, and that result is
treated as ground truth for the rest of the run.
"""
from __future__ import annotations

import json

import h5py
import numpy as np

from .config import Config, stage_dir

def categorical(handle: h5py.File, name: str) -> np.ndarray:
    node = handle["obs"][name]
    if isinstance(node, h5py.Group):
        codes = node["codes"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]])
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], "NA")
    return node[:]


def run(cfg: Config) -> None:
    OUT = stage_dir(cfg, "00_scaffold")
    V4 = cfg.st_scaffold.h5ad
    DNB_UM = cfg.st_scaffold.dnb_um
    SLICE_ORDER = cfg.st_names
    SECTION_OF_SLICE = {s.name: s.index for s in cfg.st_sections}
    # The AnnData labels its sections however it likes ("D1", or "B06286A4").
    # Translate them once, here, into the display names used by the config and
    # everything downstream, so long chip ids do not have to leak into the UI.
    label_of = {s.source: s.name for s in cfg.st_sections}
    with h5py.File(V4, "r") as handle:
        raw_labels = categorical(handle, cfg.st_scaffold.slice_col)
        unknown = sorted(set(raw_labels) - set(label_of))
        missing = sorted(set(label_of) - set(raw_labels))
        if missing:
            raise ValueError(
                f"config names Stereo-seq sections {missing} that are not in "
                f"obs['{cfg.st_scaffold.slice_col}'] (which holds {sorted(set(raw_labels))})")
        if unknown:
            print(f"  note: ignoring {len(unknown)} section(s) present in the AnnData but "
                  f"not listed in the config: {unknown}")
        slice_short = np.array([label_of.get(v, v) for v in raw_labels])
        leiden = (categorical(handle, cfg.st_scaffold.cluster_col)
                  if cfg.st_scaffold.cluster_col else
                  np.full(len(slice_short), "NA"))
        gc_id = handle["obs"][cfg.st_scaffold.gc_id_col][:].astype(np.int32)
        deformable = handle["obsm"][cfg.st_scaffold.coord_key][:].astype(np.float64)
        spatial = handle["obsm"]["spatial"][:].astype(np.float64)

    world = deformable * DNB_UM                      # micrometres, frozen scaffold
    np.savez_compressed(
        OUT / "st_scaffold.npz",
        slice_short=slice_short,      # never truncate: chip ids can be long
        leiden=leiden.astype(str),
        gc_id=gc_id,
        world_um=world.astype(np.float32),
        chip_um=(spatial * DNB_UM).astype(np.float32),
    )

    summary = {"dnb_um": DNB_UM, "section_of_slice": SECTION_OF_SLICE, "slices": {}}
    for name in SLICE_ORDER:
        keep = slice_short == name
        gc = gc_id[keep]
        summary["slices"][name] = {
            "n_bins": int(keep.sum()),
            "n_gc_bins": int((gc > 0).sum()),
            "n_merged_gc": int(len(np.unique(gc[gc > 0]))),
            "world_x_um": [float(world[keep, 0].min()), float(world[keep, 0].max())],
            "world_y_um": [float(world[keep, 1].min()), float(world[keep, 1].max())],
        }
        print(f"{name}: {keep.sum():,} bins, {(gc > 0).sum():,} GC bins, "
              f"{summary['slices'][name]['n_merged_gc']} merged GCs")

    # merged-GC centroid per slice and the four-slice consensus
    rows = []
    for merged in sorted({int(v) for v in np.unique(gc_id) if v > 0}):
        per_slice = {}
        for name in SLICE_ORDER:
            keep = (slice_short == name) & (gc_id == merged)
            if keep.sum() == 0:
                continue
            per_slice[name] = (float(world[keep, 0].mean()), float(world[keep, 1].mean()),
                               int(keep.sum()))
        if not per_slice:
            continue
        xs = [v[0] for v in per_slice.values()]
        ys = [v[1] for v in per_slice.values()]
        rows.append({
            "merged_gc_id": merged,
            "n_slices": len(per_slice),
            "consensus_x_um": float(np.mean(xs)),
            "consensus_y_um": float(np.mean(ys)),
            "total_bins": int(sum(v[2] for v in per_slice.values())),
            "per_slice": {k: {"x_um": v[0], "y_um": v[1], "bins": v[2]} for k, v in per_slice.items()},
        })
    summary["merged_gc"] = rows
    (OUT / "st_scaffold_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(rows)} merged GCs, consensus centroids written")

