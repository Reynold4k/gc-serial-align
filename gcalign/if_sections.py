"""Stage 2 - turn each immunofluorescence section into a germinal-centre mask.

The CZI is read straight onto the spatial-transcriptomics grid (10 um by
default) so both modalities live on the same ruler from the very beginning. The
stitching ripple is divided out, the tissue outline is cut, and the GC marker
channel is thresholded into discrete GC objects.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
from aicspylibczi import CziFile
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from .config import Config, stage_dir

def read_section(cfg: Config, path: Path):
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
    mask = combined > filters.threshold_otsu(combined)
    mask = morphology.binary_closing(mask, morphology.disk(8))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, 5000)
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
    seeds = morphology.remove_small_objects(seeds, cfg.if_images.min_gc_px // 2)
    grown = (smoothed > low) & tissue
    labels = measure.label(grown)
    keep = np.unique(labels[seeds])
    mask = np.isin(labels, keep[keep > 0])
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    return morphology.remove_small_objects(mask, cfg.if_images.min_gc_px)


def run(cfg: Config) -> None:
    OUT = stage_dir(cfg, "01_if_gc")
    GRID_UM = cfg.grid_um
    manifest = []
    for entry in cfg.if_sections:
        section = entry.index
        path = cfg.if_images.dir / cfg.if_images.filename_template.format(index=section)
        planes, um_x, um_y = read_section(cfg, path)
        # destripe against a rough mask first, then cut the real tissue outline
        rough = tissue_mask(cfg, planes, sigma=10.0)
        corrected = {name: destripe(plane, rough) for name, plane in planes.items()}
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
            OUT / f"IF{section}_gc.npz",
            tissue=tissue, gc=gc, gc_labels=labels.astype(np.int32),
            **{name: plane.astype(np.float32) for name, plane in corrected.items()},
        )
        tifffile.imwrite(OUT / f"IF{section}_gc_mask.tif", gc.astype(np.uint8) * 255)
        manifest.append({
            "section": section, "source": str(path),
            "shape": list(map(int, tissue.shape)),
            "um_per_px_x": um_x, "um_per_px_y": um_y,
            "tissue_mm2": float(tissue.sum() * GRID_UM ** 2 / 1e6),
            "n_gc": len(objects),
            "gc_area_mm2": float(gc.sum() * GRID_UM ** 2 / 1e6),
            "median_equiv_radius_um": float(np.median([o["equiv_radius_um"] for o in objects])),
            "objects": objects,
        })
        print(f"IF{section}: tissue {manifest[-1]['tissue_mm2']:.2f} mm2  "
              f"nGC {len(objects)}  GC {manifest[-1]['gc_area_mm2']:.3f} mm2  "
              f"median r {manifest[-1]['median_equiv_radius_um']:.0f} um", flush=True)

    (OUT / "if_gc_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", OUT / "if_gc_manifest.json")

