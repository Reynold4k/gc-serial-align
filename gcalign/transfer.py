"""Read the IF stain at every spatial-transcriptomics bin.

Once the sections share a frame, each transcriptomics bin has a position on the
IF sections cut immediately before and after it. This writes, for every bin,
the fluorescence of each channel at that position - protein signal next to the
transcriptome, without imputing either from the other.

What it is and is not:

  - the two modalities come from different sections, 1 section thickness apart
    in z, so a value is the stain *above or below* a bin, not in it;
  - values are the stage-2 intensities (detector units after the stitching
    ripple is divided out), sampled through the same transform the rasters were
    warped with; `scale` in the output turns them into the 0-1 display scale;
  - a bin outside an IF section's tissue outline gets NaN for that section.

    gcalign transfer --config my_experiment.yaml

writes ``06_transfer/st_bins_if.npz`` (and a CSV on request), one row per bin in
the row order of the AnnData the scaffold was read from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .config import Config, stage_dir
from .match_warp import channel_top
from .runio import Run
from .transforms import Transforms


def flanking_if(cfg: Config) -> dict[str, dict[str, tuple[str, float] | None]]:
    """For each transcriptomics section, the nearest IF section cut before and after it.

    Returns {st_name: {"before": (if_name, distance_um) | None, "after": ...}}.
    """
    ordered = sorted(cfg.sections, key=lambda s: s.index)
    out = {}
    for position, section in enumerate(ordered):
        if section.kind != "ST":
            continue
        entry: dict[str, tuple[str, float] | None] = {"before": None, "after": None}
        for side, step in (("before", -1), ("after", 1)):
            cursor = position + step
            while 0 <= cursor < len(ordered):
                if ordered[cursor].kind == "IF":
                    distance = abs(ordered[cursor].index - section.index) * cfg.section_thickness_um
                    entry[side] = (ordered[cursor].name, distance)
                    break
                cursor += step
        out[section.name] = entry
    return out


def obs_names(cfg: Config, n_expected: int) -> np.ndarray | None:
    """The AnnData's row names, if the file is still where the config says."""
    import h5py

    path = cfg.st_scaffold.h5ad
    if not Path(path).exists():
        return None
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        if not isinstance(obs, h5py.Group):               # a very old AnnData layout
            return None
        key = obs.attrs.get("_index", "_index")
        if isinstance(key, bytes):
            key = key.decode()
        if key not in obs:
            return None
        names = obs[key][:]
    names = np.array([n.decode() if isinstance(n, bytes) else str(n) for n in names])
    return names if len(names) == n_expected else None


def sample_plane(plane: np.ndarray, raster_xy: np.ndarray, order: int = 1) -> np.ndarray:
    """Sample a raster at fractional (col, row) positions; NaN outside it."""
    coords = np.array([raster_xy[:, 1], raster_xy[:, 0]])
    return ndi.map_coordinates(plane.astype(np.float32), coords, order=order,
                               mode="constant", cval=np.nan, prefilter=False)


def run(cfg: Config, out: str | Path | None = None, fmt: str = "npz") -> Path:
    if fmt not in ("npz", "csv"):
        raise ValueError("format must be npz or csv")
    run_ = Run(cfg)
    transforms = Transforms.load(cfg)
    grid = run_.grid
    scaffold = run_.scaffold
    section_of_bin = scaffold["slice_short"]
    world = scaffold["world_um"].astype(np.float64)
    n = len(world)
    channels = run_.channel_names
    flank = flanking_if(cfg)

    columns: dict[str, np.ndarray] = {
        "section": section_of_bin.astype(str),
        "x_um": world[:, 0].astype(np.float32),
        "y_um": world[:, 1].astype(np.float32),
        "gc_id": scaffold["gc_id"].astype(np.int32),
        "if_before": np.full(n, "", dtype="U32"),
        "if_after": np.full(n, "", dtype="U32"),
        "covered_before": np.zeros(n, bool),
        "covered_after": np.zeros(n, bool),
        "if_gc_before": np.zeros(n, np.int32),
        "if_gc_after": np.zeros(n, np.int32),
    }
    for channel in channels:
        for suffix in ("", "_before", "_after"):
            columns[channel + suffix] = np.full(n, np.nan, np.float32)

    # one scale per section and channel: the value the display treats as white
    scale: dict[str, dict[str, float]] = {}
    local_cache: dict[str, dict] = {}

    def local(name: str) -> dict:
        if name not in local_cache:
            data = run_.local(name)
            planes = {c: np.asarray(data[c], np.float32) for c in channels}
            scale[name] = {}
            for c, plane in planes.items():
                scale[name][c] = channel_top(plane)
            local_cache[name] = {"planes": planes, "tissue": np.asarray(data["tissue"], bool)}
        return local_cache[name]

    summary = {}
    for st_name, sides in flank.items():
        rows = np.flatnonzero(section_of_bin == st_name)
        if len(rows) == 0:
            continue
        xy = world[rows]
        weights = np.zeros((2, len(rows)), np.float32)
        used = []
        for side_index, side in enumerate(("before", "after")):
            if sides[side] is None:
                continue
            if_name, distance = sides[side]
            used.append(if_name)
            item = local(if_name)
            t = transforms[if_name]
            raster_xy = t.local_to_raster_px(t.from_frame(xy))
            inside = sample_plane(item["tissue"].astype(np.float32), raster_xy, order=0) > 0.5
            columns[f"if_{side}"][rows] = if_name
            columns[f"covered_{side}"][rows] = inside
            r, c = grid.to_rc(xy)
            ok = grid.inside(r, c)
            gc_here = np.zeros(len(rows), np.int32)
            gc_here[ok] = run_.labels(if_name)[r[ok], c[ok]]
            columns[f"if_gc_{side}"][rows] = gc_here
            for channel in channels:
                values = sample_plane(item["planes"][channel], raster_xy)
                values[~inside] = np.nan
                columns[f"{channel}_{side}"][rows] = values
            # nearer sections count for more; equal weights in an alternating stack
            weights[side_index] = np.where(inside, 1.0 / max(distance, 1e-9), 0.0)
        total = weights.sum(axis=0)
        for channel in channels:
            before = np.nan_to_num(columns[f"{channel}_before"][rows], nan=0.0)
            after = np.nan_to_num(columns[f"{channel}_after"][rows], nan=0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                blend = (weights[0] * before + weights[1] * after) / total
            columns[channel][rows] = np.where(total > 0, blend, np.nan)
        covered = int((total > 0).sum())
        summary[st_name] = {"bins": int(len(rows)), "if_sections": used,
                            "bins_under_if_tissue": covered,
                            "pct_under_if_tissue": round(100.0 * covered / len(rows), 1)}
        print(f"{st_name}: {len(rows):,} bins, read from {' + '.join(used) or 'no IF section'}, "
              f"{summary[st_name]['pct_under_if_tissue']}% under IF tissue", flush=True)

    names = obs_names(cfg, n)
    target_dir = stage_dir(cfg, "06_transfer") if out is None else None
    path = Path(out) if out is not None else target_dir / f"st_bins_if.{fmt}"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment": cfg.name,
        "channels": channels,
        "units": "stage-2 intensity (detector units, stitching ripple divided out)",
        "scale": scale,
        "scale_note": "divide a section's values by its scale to get the 0-1 display range",
        "combined": "mean of the before/after values, weighted by 1/z-distance, over the "
                    "sections whose tissue covers the bin",
        "transforms": transforms.origin,
        "row_order": "the AnnData the scaffold was read from" if names is not None
                     else "00_scaffold/st_scaffold.npz",
        "sections": summary,
    }
    if fmt == "npz":
        payload = dict(columns)
        if names is not None:
            payload["obs_names"] = names
        np.savez_compressed(path, **payload)
    else:
        import pandas as pd
        frame = pd.DataFrame(columns)
        if names is not None:
            frame.insert(0, "obs_name", names)
        frame.to_csv(path, index=False, float_format="%.6g")
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB) and {path.with_suffix('.json').name}")
    return path
