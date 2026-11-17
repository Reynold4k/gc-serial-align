"""Two operations on boolean masks: morphological closing and removal of small objects.

Both give the same result as the scikit-image functions that gcalign 0.1 used
(`skimage.morphology.binary_closing` and `skimage.morphology.remove_small_objects`
with `min_size`). They are written with SciPy because scikit-image 0.26
deprecated the first function and the `min_size` parameter of the second, and the
SciPy functions used here are the same in every SciPy version.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def closing(mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Close a boolean mask: dilate it with `footprint`, then erode it with `footprint`.

    Closing fills gaps and notches that are narrower than the footprint.
    Pixels outside the image count as background during the dilation and as
    foreground during the erosion, so a mask that touches the image border is
    not eroded along that border.
    """
    mask = np.asarray(mask, bool)
    footprint = np.asarray(footprint, bool)
    grown = ndi.binary_dilation(mask, structure=footprint)
    return ndi.binary_erosion(grown, structure=footprint, border_value=True)


def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove every object of a boolean mask that has fewer than `min_size` pixels.

    An object is a group of pixels connected through shared edges (in 2D: the
    four direct neighbours; diagonal neighbours do not connect).
    """
    mask = np.asarray(mask, bool)
    labels, _ = ndi.label(mask)                 # default structure: shared edges only
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False                             # label 0 is the background
    return keep[labels]
