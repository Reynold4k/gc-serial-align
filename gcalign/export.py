"""Write an IF section as a registered image in the transcriptomics frame.

The pipeline itself works on the common grid (10 um by default), which is the
right scale for finding and matching structures but too coarse to look at
cells. The registration, though, is a smooth map and can be applied at any
resolution: this reads the original image again at the resolution asked for and
resamples it through the section's transform, giving a multi-channel OME-TIFF
whose pixel (0, 0) sits at a known position of the frame.

    gcalign export --config my_experiment.yaml --section IF3 --um-per-px 2

Next to the TIFF a small JSON gives the frame origin and pixel size, so the
image can be laid over the transcriptomics coordinates in any viewer:

    x_frame_um = x0_um + column * um_per_px        (and likewise for y / rows)
    x_in_anndata_units = x_frame_um / dnb_um

The accuracy of the result is the accuracy of the registration (see the
leave-one-out error in the validation report), not of the output pixel size.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

from .config import Config, stage_dir
from .if_sections import reader_for, tiff_channels
from .transforms import Transforms


def read_at(cfg: Config, path: Path, um_per_px: float) -> tuple[np.ndarray, list[str]]:
    """The section's channels at roughly `um_per_px`, as (channel, row, col) float32.

    Never finer than the image itself. Returns the array and the channel names.
    """
    native = cfg.if_images.um_per_px
    names = [cfg.if_images.channels[i] for i in sorted(cfg.if_images.channels)]
    indices = sorted(cfg.if_images.channels)
    scale = min(1.0, native / um_per_px)
    if reader_for(cfg, path) == "czi":
        from aicspylibczi import CziFile
        czi = CziFile(path)
        planes = [np.squeeze(czi.read_mosaic(C=i, scale_factor=scale)).astype(np.float32)
                  for i in indices]
        return np.stack(planes), names
    from skimage.transform import resize
    data, level_um, (full_y, full_x) = tiff_channels(path, native, um_per_px)
    shape = (max(1, int(round(full_y * scale))), max(1, int(round(full_x * scale))))
    planes = [resize(data[i].astype(np.float32), shape, order=1, mode="reflect",
                     anti_aliasing=level_um < um_per_px, preserve_range=True).astype(np.float32)
              for i in indices]
    return np.stack(planes), names


def frame_window(transform, frame: dict, um_per_px: float, margin_um: float = 300.0):
    """The part of the frame this section lands on, snapped to the output pixel grid."""
    height, width = transform.image["raster_shape"]
    step = transform.local_um_per_px
    edge = np.concatenate([
        np.column_stack([np.arange(width) * step, np.zeros(width)]),
        np.column_stack([np.arange(width) * step, np.full(width, (height - 1) * step)]),
        np.column_stack([np.zeros(height), np.arange(height) * step]),
        np.column_stack([np.full(height, (width - 1) * step), np.arange(height) * step]),
    ])
    placed = transform.to_frame(edge, warp=False)
    x_lo, y_lo = placed.min(axis=0) - margin_um
    x_hi, y_hi = placed.max(axis=0) + margin_um
    # never larger than the frame the registration was made in
    fx0, fy0 = frame["x0_um"], frame["y0_um"]
    fx1 = fx0 + frame["width"] * frame["um_per_px"]
    fy1 = fy0 + frame["height"] * frame["um_per_px"]
    x_lo, y_lo, x_hi, y_hi = max(x_lo, fx0), max(y_lo, fy0), min(x_hi, fx1), min(y_hi, fy1)
    x0 = fx0 + np.floor((x_lo - fx0) / um_per_px) * um_per_px
    y0 = fy0 + np.floor((y_lo - fy0) / um_per_px) * um_per_px
    n_cols = int(np.ceil((x_hi - x0) / um_per_px))
    n_rows = int(np.ceil((y_hi - y0) / um_per_px))
    return float(x0), float(y0), n_rows, n_cols


def run(cfg: Config, section: str, um_per_px: float = 2.0, out: str | Path | None = None,
        channels: list[str] | None = None, warp: bool = True, rows_per_block: int = 512) -> Path:
    """Export one IF section, registered. Returns the path of the TIFF."""
    transforms = Transforms.load(cfg)
    t = transforms[section]
    if not t.image:
        raise RuntimeError(f"{section}: the run has no image geometry for this section "
                           "(01_if_gc/if_gc_manifest.json); re-run stage `if`")
    native = cfg.if_images.um_per_px
    if um_per_px < native:
        print(f"note: {um_per_px} um/px is finer than the image ({native:.4f} um/px); "
              "the output is interpolated, it holds no extra detail")
    path = cfg.if_path(cfg.section(section))
    print(f"{section}: reading {path.name} at ~{max(um_per_px, native):.3g} um/px", flush=True)
    source, names = read_at(cfg, path, um_per_px)
    if channels:
        missing = [c for c in channels if c not in names]
        if missing:
            raise ValueError(f"unknown channel(s) {missing}; the image has {names}")
        keep = [names.index(c) for c in channels]
        source, names = source[keep], [names[i] for i in keep]
    # full-resolution size from the raster the registration was made on
    raster_h, raster_w = t.image["raster_shape"]
    ratio = t._image_ratio()                                # raster px per image px (x, y)
    full_w, full_h = raster_w / ratio[0], raster_h / ratio[1]
    shrink = np.array([source.shape[2] / full_w, source.shape[1] / full_h])

    x0, y0, n_rows, n_cols = frame_window(t, transforms.frame, um_per_px)
    top = float(np.iinfo(np.uint16).max)
    peak = max(float(source.max()), 1.0)
    as_uint16 = peak <= top
    out_array = np.zeros((len(names), n_rows, n_cols), np.uint16 if as_uint16 else np.float32)
    cols = x0 + np.arange(n_cols) * um_per_px
    for start in range(0, n_rows, rows_per_block):
        stop = min(start + rows_per_block, n_rows)
        yy = y0 + np.arange(start, stop) * um_per_px
        frame_xy = np.stack(np.meshgrid(cols, yy), axis=-1).reshape(-1, 2)
        image_px = t.frame_to_image_px(frame_xy, warp=warp)
        # image pixels -> pixels of the array that was read (centres at integers)
        src = (image_px + 0.5) * shrink - 0.5
        coords = np.array([src[:, 1], src[:, 0]])
        for index in range(len(names)):
            values = ndi.map_coordinates(source[index], coords, order=1, mode="constant",
                                         cval=0.0, prefilter=False)
            block = values.reshape(stop - start, n_cols)
            out_array[index, start:stop] = np.clip(np.rint(block), 0, top) if as_uint16 else block
        print(f"  rows {stop}/{n_rows}", end="\r", flush=True)
    print()

    target = Path(out) if out is not None else (
        stage_dir(cfg, "06_registered_images") / f"{section}_registered_{um_per_px:g}um.ome.tif")
    target.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        target, out_array, ome=True, photometric="minisblack", compression="zlib",
        metadata={"axes": "CYX", "Channel": {"Name": names},
                  "PhysicalSizeX": um_per_px, "PhysicalSizeXUnit": "µm",
                  "PhysicalSizeY": um_per_px, "PhysicalSizeYUnit": "µm"})
    sidecar = {
        "experiment": cfg.name, "section": section, "z_um": t.z_um,
        "channels": names,
        "um_per_px": um_per_px, "x0_um": x0, "y0_um": y0,
        "shape": [int(n_rows), int(n_cols)],
        "pixel_to_frame": "x_um = x0_um + col * um_per_px ; y_um = y0_um + row * um_per_px",
        "frame_um_per_anndata_unit": cfg.st_scaffold.dnb_um,
        "anndata_coord_key": cfg.st_scaffold.coord_key,
        "registration": "similarity + thin-plate spline" if warp else "similarity only",
        "transforms": transforms.origin,
        "one_sided": t.one_sided,
        "source_image": str(path),
        "source_um_per_px": native,
    }
    target.with_suffix("").with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {target}  ({target.stat().st_size / 1e6:.1f} MB, "
          f"{len(names)} channels, {n_rows} x {n_cols} px at {um_per_px:g} um)")
    return target
