"""Stage 2 - turn each immunofluorescence section into a germinal-centre mask.

The image is read straight onto the spatial-transcriptomics grid (10 um by
default) so both modalities live on the same ruler from the very beginning. The
stitching ripple is divided out, the tissue outline is cut, and the GC marker
channel is thresholded into discrete GC objects.

Two kinds of image are read: a Zeiss CZI mosaic, and a multi-channel TIFF
(plain or OME, pyramidal or not).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from . import masks
from .config import Config, stage_dir
from .parallel import run_sections


def reader_for(cfg: Config, path: Path) -> str:
    """"czi" or "tiff", from the config or the file extension."""
    if cfg.if_images.reader != "auto":
        return cfg.if_images.reader
    suffix = path.suffix.lower()
    if suffix == ".czi":
        return "czi"
    if suffix in (".tif", ".tiff"):
        return "tiff"
    raise ValueError(f"cannot tell how to read {path.name}: set if_images.reader to czi or tiff")


def read_czi(cfg: Config, path: Path):
    try:
        from aicspylibczi import CziFile
    except ImportError as error:                              # pragma: no cover
        raise ImportError("reading CZI files needs aicspylibczi (pip install aicspylibczi)") from error
    czi = CziFile(path)
    scale = cfg.if_images.um_per_px / cfg.grid_um
    planes = {}
    for index, name in cfg.if_images.channels.items():
        planes[name] = np.squeeze(czi.read_mosaic(C=index, scale_factor=scale)).astype(np.float32)
    box = czi.get_mosaic_bounding_box()
    first = next(iter(cfg.if_images.channels.values()))
    height, width = planes[first].shape
    return (planes, box.w * cfg.if_images.um_per_px / width,
            box.h * cfg.if_images.um_per_px / height)


def tiff_channels(path: Path, um_per_px: float, target_um: float):
    """A TIFF as (channel, row, col), from the pyramid level that suits `target_um`.

    Returns the array, the pixel size of the level that was read, and the
    full-resolution (height, width). A pyramidal file is read at its coarsest
    level that is still at least twice as fine as the target, so a whole-slide
    image does not have to be loaded at full resolution to be put on a 10 um grid.
    """
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = list(getattr(series, "levels", [series]))
        full = levels[0]
        full_y, full_x = full.shape[full.axes.index("Y")], full.shape[full.axes.index("X")]
        chosen = full
        for level in levels[1:]:
            size = um_per_px * full_x / level.shape[level.axes.index("X")]
            if size <= target_um / 2:
                chosen = level
        axes = chosen.axes
        data = chosen.asarray()
    # drop axes of length one (a single Z plane, a single time point), keep Y and X
    for axis in reversed(range(data.ndim)):
        if data.shape[axis] == 1 and axes[axis] not in "YX":
            data = np.squeeze(data, axis=axis)
            axes = axes[:axis] + axes[axis + 1:]
    if data.ndim == 2:
        data = data[None]
    elif data.ndim == 3:
        other = next(i for i, a in enumerate(axes) if a not in "YX")
        data = np.moveaxis(data, other, 0)
    else:
        raise ValueError(f"{path.name} has axes {axes}; expected one 2D plane per channel "
                         "(a Z stack or time series has to be projected first)")
    level_um = um_per_px * full_x / data.shape[2]
    return data, level_um, (full_y, full_x)


def read_tiff(cfg: Config, path: Path):
    from skimage.transform import resize

    data, level_um, (full_y, full_x) = tiff_channels(path, cfg.if_images.um_per_px, cfg.grid_um)
    needed = max(cfg.if_images.channels) + 1
    if data.shape[0] < needed:
        raise ValueError(f"{path.name} has {data.shape[0]} channel(s), but the config names "
                         f"channel index {needed - 1}")
    scale = cfg.if_images.um_per_px / cfg.grid_um
    shape = (max(1, int(round(full_y * scale))), max(1, int(round(full_x * scale))))
    shrinking = level_um < cfg.grid_um
    planes = {}
    for index, name in cfg.if_images.channels.items():
        planes[name] = resize(data[index].astype(np.float32), shape, order=1, mode="reflect",
                              anti_aliasing=shrinking, preserve_range=True).astype(np.float32)
    return (planes, full_x * cfg.if_images.um_per_px / shape[1],
            full_y * cfg.if_images.um_per_px / shape[0])


def read_section(cfg: Config, path: Path):
    """One IF section on the common grid: {channel: plane}, and the true pixel
    pitch of that raster in micrometres (x, y)."""
    if not path.exists():
        raise FileNotFoundError(f"no image at {path} (if_images.dir + filename_template)")
    return (read_czi if reader_for(cfg, path) == "czi" else read_tiff)(cfg, path)


def _combined(cfg: Config, planes: dict[str, np.ndarray], sigma: float) -> np.ndarray:
    combined = np.zeros_like(next(iter(planes.values())))
    for name, weight in cfg.if_images.tissue_channels.items():
        smoothed = ndi.gaussian_filter(planes[name], sigma)
        low, high = np.percentile(smoothed, [1, 99.5])
        combined += weight * np.clip((smoothed - low) / max(high - low, 1.0), 0, 1)
    return combined


def tissue_mask(cfg: Config, planes: dict[str, np.ndarray], sigma: float = 6.0):
    """One connected section, smoothed across the ~77 px stitching ripple."""
    combined = _combined(cfg, planes, sigma)
    mask = combined > cfg.if_images.tissue_threshold_scale * filters.threshold_otsu(combined)
    mask = masks.closing(mask, morphology.disk(8))
    mask = ndi.binary_fill_holes(mask)
    mask = masks.remove_small(mask, 5000)
    labels = measure.label(mask)
    if labels.max() > 1:
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        mask = labels == int(areas.argmax())
    return ndi.binary_fill_holes(mask)


def destripe(plane: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    """Divide out the ~774 um stitching ripple along both axes."""
    out = plane.copy()
    for axis in (0, 1):
        level = np.full(plane.shape[axis ^ 1], np.nan, dtype=np.float32)
        for index in range(plane.shape[axis ^ 1]):
            column = plane[:, index] if axis == 0 else plane[index, :]
            keep = tissue[:, index] if axis == 0 else tissue[index, :]
            if keep.sum() > 50:
                level[index] = np.median(column[keep])
        valid = np.isfinite(level)
        if valid.sum() < 20:
            continue
        level = np.interp(np.arange(level.size), np.flatnonzero(valid), level[valid])
        trend = ndi.uniform_filter1d(level, 121, mode="nearest")
        gain = np.clip(trend / np.maximum(level, 1.0), 0.5, 2.0)
        out = out * (gain[None, :] if axis == 0 else gain[:, None])
    return out


def segment_gc(cfg: Config, gl7: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    smoothed = ndi.gaussian_filter(gl7, 1.5)
    inside = smoothed[tissue]
    high = filters.threshold_otsu(inside)
    low = np.percentile(inside, 50) + 0.65 * (high - np.percentile(inside, 50))
    seeds = (smoothed > high) & tissue
    seeds = masks.remove_small(seeds, cfg.if_images.min_gc_px // 2)
    grown = (smoothed > low) & tissue
    labels = measure.label(grown)
    keep = np.unique(labels[seeds])
    mask = np.isin(labels, keep[keep > 0])
    mask = masks.closing(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    return masks.remove_small(mask, cfg.if_images.min_gc_px)


def segment_section(cfg: Config, name: str) -> dict:
    """Read and segment one IF section. Independent of the other sections."""
    OUT = cfg.output_dir / "01_if_gc"
    GRID_UM = cfg.grid_um
    entry = cfg.section(name)
    path = cfg.if_path(entry)
    planes, um_x, um_y = read_section(cfg, path)
    # destripe against a rough mask first, then cut the real tissue outline
    rough = tissue_mask(cfg, planes, sigma=10.0)
    corrected = {channel: destripe(plane, rough) for channel, plane in planes.items()}
    tissue = tissue_mask(cfg, corrected, sigma=6.0)
    gc = segment_gc(cfg, corrected[cfg.if_images.gc_channel], tissue)

    labels = measure.label(gc)
    objects = []
    for region in measure.regionprops(labels):
        cy, cx = region.centroid
        objects.append({
            "label": int(region.label),
            "area_px": int(region.area),
            "area_um2": float(region.area * GRID_UM ** 2),
            "equiv_radius_um": float(np.sqrt(region.area * GRID_UM ** 2 / np.pi)),
            "centroid_x_um": float(cx * um_x),
            "centroid_y_um": float(cy * um_y),
            "eccentricity": float(region.eccentricity),
        })

    np.savez_compressed(
        OUT / f"{name}_gc.npz",
        tissue=tissue, gc=gc, gc_labels=labels.astype(np.int32),
        **{channel: plane.astype(np.float32) for channel, plane in corrected.items()},
    )
    tifffile.imwrite(OUT / f"{name}_gc_mask.tif", gc.astype(np.uint8) * 255)
    item = {
        "section": entry.index, "name": name, "source": str(path),
        "shape": list(map(int, tissue.shape)),
        "um_per_px_x": um_x, "um_per_px_y": um_y,
        "tissue_mm2": float(tissue.sum() * GRID_UM ** 2 / 1e6),
        "n_gc": len(objects),
        "gc_area_mm2": float(gc.sum() * GRID_UM ** 2 / 1e6),
        "median_equiv_radius_um": (float(np.median([o["equiv_radius_um"] for o in objects]))
                                   if objects else None),
        "objects": objects,
    }
    radius = f"{item['median_equiv_radius_um']:.0f} um" if objects else "-"
    print(f"{name}: tissue {item['tissue_mm2']:.2f} mm2  nGC {len(objects)}  "
          f"GC {item['gc_area_mm2']:.3f} mm2  median r {radius}", flush=True)
    return item


def run(cfg: Config, jobs: int = 1) -> None:
    OUT = stage_dir(cfg, "01_if_gc")
    manifest = run_sections(segment_section, [(cfg, name) for name in cfg.if_names], jobs)
    (OUT / "if_gc_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", OUT / "if_gc_manifest.json")

