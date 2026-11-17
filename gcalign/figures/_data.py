"""Data the figures need beyond what `Run` returns directly: display outlines, display limits, windows."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..runio import Run


def display_tissue(run: Run, name: str) -> np.ndarray:
    """The tissue outline a section is cut to in a figure: `Run.tissue` as a boolean mask."""
    return np.asarray(run.tissue(name), bool)


STAIN_LOW, STAIN_HIGH = 25.0, 99.6


def stain_limits(run: Run, window: tuple[int, int, int, int] | None = None,
                 low: float = STAIN_LOW, high: float = STAIN_HIGH) -> dict[str, tuple[float, float]]:
    """Display limits per channel, shared by every IF plane, from tissue pixels.

    The black point is the lower quartile of the tissue, not of the whole
    imaged field: a broadly expressed marker (CD4 here) otherwise lays a haze
    of its own colour over the entire section. Pooling over all IF planes keeps
    brightness comparable from one section to the next.
    """
    r0, r1, c0, c1 = window or run.grid.full
    limits = {}
    for channel in run.cfg.rgb_channels:
        if channel is None:
            continue
        pool = []
        for name in run.if_names:
            piece = run.channel(name, channel)[r0:r1, c0:c1]
            inside = display_tissue(run, name)[r0:r1, c0:c1] & (piece > 0)
            pool.append(piece[inside] if inside.any() else piece.ravel())
        pool = np.concatenate(pool) if pool else np.zeros(1)
        lo, hi = np.percentile(pool, [low, high]) if pool.size else (0.0, 1.0)
        limits[channel] = (float(lo), float(max(hi, lo + 1.0)))
    return limits


def stack_window(run: Run, pad: int = 6) -> tuple[int, int, int, int]:
    """The smallest window holding every plane's tissue, plus a margin."""
    from ._base import bbox
    union = np.zeros(run.grid.shape, bool)
    for name in run.names:
        union |= display_tissue(run, name)
    for name in run.names:
        union |= run.labels(name) != 0
    return bbox(union, pad=pad)


def feather(mask: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    """A mask with a sub-pixel soft edge, so a cut-out stain has no jagged rim."""
    soft = ndi.gaussian_filter(mask.astype(np.float32), sigma)
    soft[~ndi.binary_dilation(mask, iterations=1)] = 0.0
    return np.clip(soft, 0, 1)


def gc_areas(run: Run) -> "dict[int, dict[str, int]]":
    """Pixels of every GC on every plane, straight from the label rasters."""
    out: dict[int, dict[str, int]] = {}
    for name in run.names:
        labels = run.labels(name)
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        for gid, count in zip(ids.tolist(), counts.tolist()):
            out.setdefault(int(gid), {})[name] = int(count)
    return out
