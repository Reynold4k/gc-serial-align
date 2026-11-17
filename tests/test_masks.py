"""The two mask operations of gcalign.masks, each against its definition written out in full."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from scipy import ndimage as ndi

from gcalign import masks


def disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return xx ** 2 + yy ** 2 <= radius ** 2


def closing_by_definition(mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Dilation with background outside the image, then erosion with foreground outside."""
    radius = footprint.shape[0] // 2
    offsets = list(zip(*np.nonzero(footprint)))
    h, w = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    grown = np.zeros_like(mask)
    for dy, dx in offsets:
        grown |= padded[dy:dy + h, dx:dx + w]
    padded = np.pad(grown, radius, constant_values=True)
    out = np.ones_like(mask)
    for dy, dx in offsets:
        out &= padded[dy:dy + h, dx:dx + w]
    return out


def remove_small_by_definition(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Flood fill over the four direct neighbours; keep objects of at least `min_size` pixels."""
    out = np.zeros_like(mask)
    seen = np.zeros_like(mask)
    h, w = mask.shape
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        todo, pixels = deque([(r0, c0)]), []
        while todo:
            r, c = todo.popleft()
            pixels.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        if len(pixels) >= min_size:
            rows, cols = zip(*pixels)
            out[rows, cols] = True
    return out


def random_masks():
    rng = np.random.default_rng(11)
    for trial in range(24):
        h, w = (int(v) for v in rng.integers(12, 48, 2))
        if trial % 3 == 0:
            yield rng.random((h, w)) < rng.uniform(0.05, 0.8)                 # scattered pixels
        elif trial % 3 == 1:                                                  # blobs, many on the border
            yield ndi.gaussian_filter(rng.standard_normal((h, w)), 2.0) > 0
        else:                                                                 # blobs with holes and gaps
            yield (ndi.gaussian_filter(rng.standard_normal((h, w)), 3.0) > 0) & (rng.random((h, w)) > 0.2)


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_closing_follows_its_definition(radius):
    footprint = disk(radius)
    for mask in random_masks():
        closed = masks.closing(mask, footprint)
        assert closed.dtype == bool
        assert np.array_equal(closed, closing_by_definition(mask, footprint))
        assert not (mask & ~closed).any()                 # closing never removes a pixel


def test_closing_fills_a_narrow_gap_and_keeps_the_image_border():
    mask = np.zeros((9, 12), bool)
    mask[2:7, 1:5] = True
    mask[2:7, 6:11] = True                                # two blocks, one empty column between
    closed = masks.closing(mask, disk(1))
    assert closed[2:7, 5].any()                           # the gap is bridged
    full = np.ones((7, 7), bool)
    assert masks.closing(full, disk(2)).all()             # a full image is not eroded at its border


def test_small_objects_are_removed_by_pixel_count():
    for mask in random_masks():
        for min_size in (0, 1, 2, 4, 15, 200):
            kept = masks.remove_small(mask, min_size)
            assert kept.dtype == bool
            assert np.array_equal(kept, remove_small_by_definition(mask, min_size))


def test_diagonal_pixels_are_separate_objects():
    mask = np.zeros((5, 5), bool)
    mask[1, 1] = mask[2, 2] = True                        # touch at a corner only
    mask[3, 0] = mask[4, 0] = True                        # share an edge
    kept = masks.remove_small(mask, 2)
    assert not kept[1, 1] and not kept[2, 2]
    assert kept[3, 0] and kept[4, 0]
    assert not masks.remove_small(np.zeros((4, 4), bool), 3).any()
