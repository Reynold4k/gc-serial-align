"""Layout, type and raster helpers shared by every static figure.

Every figure is laid out in device pixels. A raster - a stain, a label map, a
rendered column - is pasted with ``figimage`` at an integer magnification, so
one grid pixel is always exactly k x k output pixels: nothing is resampled by
the plotting library, and a 10 um bin looks like a 10 um bin. Everything
vector (outlines, scale bars, labels, chart marks) is drawn on top in the same
pixel coordinates, so the two layers cannot drift apart.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import numpy as np

from .. import theme
from ..runio import Run

# --------------------------------------------------------------------------
# one set of sizes for the whole figure set (points; 72 pt = 1 inch)
# --------------------------------------------------------------------------
LABEL_PT = 11.0         # plane names, GC ids - the key labels
SMALL_PT = 8.5          # tick labels, colour-bar units, row ids
SCALE_PT = 9.5          # the length printed above a scale bar
AXIS_PT = 9.0           # axis titles on the quantitative plots
MARGIN_PT = 18.0        # outer margin of every figure
GAP_PT = 10.0           # gap between panels
BAND_PT = 21.0          # label band above a spatial panel
LABEL_GAP_PT = 6.0      # from a panel's top edge up to its label's baseline
FOOT_PT = 34.0          # band under the panels that holds the scale bar
FOOT_BAR_PT = 14.0      # scale bar height above the bottom edge
FIGURE_WIDTH_IN = 10.5  # every multi-panel figure is about this wide
LINE_PT = 0.9           # hairlines of the charts
OUTLINE_PT = 1.15       # the white outline of "this GC"
GHOST_PT = 0.7          # the dashed outline of "this GC on the previous plane"
BAR_PT = 1.4            # scale-bar thickness

FORMATS = ("png", "pdf", "svg")


def as_run(source, output_dir=None) -> Run:
    """A `Run` from a Run, a Config, or a path to a config file."""
    if isinstance(source, Run):
        return source
    from ..config import Config
    if isinstance(source, Config):
        if output_dir is not None:
            import copy
            source = copy.copy(source)
            source.output_dir = Path(output_dir)
        return Run(source)
    return Run.open(source, output_dir=output_dir)


@contextlib.contextmanager
def styled():
    """The black theme for the duration of one figure; rcParams restored after."""
    import matplotlib as mpl
    with mpl.rc_context():
        theme.use_mpl(font_size=SMALL_PT)
        yield


def font(weight: str = "regular", size: float = LABEL_PT):
    return theme.mpl_font(weight, size)


def kind_colour(kind: str) -> str:
    return theme.KIND.get(kind, theme.INK)


# --------------------------------------------------------------------------
# the canvas
# --------------------------------------------------------------------------
class Canvas:
    """A figure of exactly ``width x height`` device pixels.

    ``ax`` is a transparent axes spanning the whole figure with its data
    coordinates equal to pixel coordinates from the top-left corner, so every
    vector element is placed in the same units as the rasters.
    """

    def __init__(self, width: float, height: float, dpi: float):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.width, self.height, self.dpi = int(round(width)), int(round(height)), float(dpi)
        self.fig = Figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi,
                          facecolor=theme.BG, edgecolor=theme.BG)
        FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes((0, 0, 1, 1), zorder=10)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(self.height, 0)
        self.ax.set_axis_off()
        self.ax.patch.set_alpha(0.0)

    # units ---------------------------------------------------------------
    def px(self, points: float) -> float:
        """Points to device pixels at this canvas's dpi."""
        return points * self.dpi / 72.0

    # rasters -------------------------------------------------------------
    def raster(self, image: np.ndarray, x: float, y: float) -> None:
        """Paste an image with its top-left corner at pixel (x, y), unresampled."""
        image = to_uint8(image)
        self.fig.figimage(image, xo=int(round(x)), yo=int(self.height - round(y) - image.shape[0]),
                          origin="upper", zorder=1)

    # vector --------------------------------------------------------------
    def text(self, x: float, y: float, text: str, *, size: float = LABEL_PT,
             colour: str = theme.INK, weight: str = "medium", ha: str = "left",
             va: str = "baseline", **kwargs):
        return self.ax.text(x, y, text, fontproperties=font(weight, size), color=colour,
                            ha=ha, va=va, **kwargs)

    def polyline(self, points: np.ndarray, *, colour: str = theme.INK, width: float = OUTLINE_PT,
                 dashes: tuple[float, float] | None = None, alpha: float = 1.0, closed: bool = True):
        points = np.asarray(points, float)
        if closed and len(points) and not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[:1]])
        (line,) = self.ax.plot(points[:, 0], points[:, 1], color=colour, lw=width, alpha=alpha,
                               solid_joinstyle="round", solid_capstyle="round",
                               dash_capstyle="round", dash_joinstyle="round")
        if dashes is not None:
            line.set_dashes(dashes)
        return line

    def rect(self, x: float, y: float, w: float, h: float, colour: str, alpha: float = 1.0):
        from matplotlib.patches import Rectangle
        patch = Rectangle((x, y), w, h, facecolor=colour, edgecolor="none", alpha=alpha, lw=0)
        self.ax.add_patch(patch)
        return patch

    def text_width(self, text: str, size: float = LABEL_PT, weight: str = "medium") -> float:
        """Rendered width of a string in pixels."""
        from matplotlib.textpath import TextPath
        path = TextPath((0, 0), text, prop=font(weight, size))
        extent = path.get_extents()
        return float(extent.width) * self.dpi / 72.0

    def chart_axes(self, x: float, y: float, w: float, h: float):
        """A normal matplotlib axes occupying a pixel box (for quantitative plots)."""
        ax = self.fig.add_axes((x / self.width, 1 - (y + h) / self.height,
                                w / self.width, h / self.height), zorder=5)
        ax.set_facecolor("none")
        return ax

    # output --------------------------------------------------------------
    def save(self, path: str | Path, fmt: str | None = None) -> Path:
        path = Path(path)
        fmt = (fmt or path.suffix.lstrip(".") or "png").lower()
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; choose one of {FORMATS}")
        path = path.with_suffix("." + fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"dpi": self.dpi, "facecolor": theme.BG, "edgecolor": theme.BG}
        if fmt == "pdf":
            kwargs["metadata"] = {"Creator": "gcalign"}
        import matplotlib as mpl
        # SVG text is written as outlines: the file then looks the same on a
        # machine without Inter, and matplotlib's SVG text writer cannot handle
        # the numeric font weight the theme sets
        with mpl.rc_context({"svg.fonttype": "path"}):
            self.fig.savefig(path, format=fmt, **kwargs)
        return path


# --------------------------------------------------------------------------
# rasters
# --------------------------------------------------------------------------
def to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def magnify(image: np.ndarray, k: int, smooth: bool = False) -> np.ndarray:
    """Enlarge by an integer factor; nearest (label maps) or bilinear (a stain)."""
    k = int(k)
    if k <= 1:
        return image
    if not smooth:
        return np.repeat(np.repeat(image, k, axis=0), k, axis=1)
    from scipy import ndimage as ndi
    image = np.asarray(image, np.float32)
    zoom = (k, k) + (1,) * (image.ndim - 2)
    return np.clip(ndi.zoom(image, zoom, order=1, grid_mode=True, mode="nearest"), 0, 1)


def integer_scale(n_pixels: int, target_px: float) -> int:
    """The whole-number magnification that brings n grid pixels near target_px.

    The nearest whole number, raised by one when it would leave the panel under
    three quarters of the target, so every panel ends up between 0.75 and 1.5
    times its intended size.
    """
    n_pixels = max(n_pixels, 1)
    k = max(1, int(round(target_px / n_pixels)))
    if n_pixels * k < 0.75 * target_px:
        k += 1
    return k


def solid(rgb_hex: str) -> np.ndarray:
    return np.asarray(theme.rgb01(rgb_hex), np.float32)


def colour_table(colours: dict[int, str], size: int) -> np.ndarray:
    """A lookup table id -> RGB for non-negative ids below `size`."""
    table = np.zeros((size, 3), np.float32)
    for gid, value in colours.items():
        if 0 <= gid < size:
            table[gid] = theme.rgb01(value)
    return table


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def bbox(mask: np.ndarray, pad: int = 0, shape=None) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("empty mask")
    shape = shape or mask.shape
    return (max(int(ys.min()) - pad, 0), min(int(ys.max()) + 1 + pad, shape[0]),
            max(int(xs.min()) - pad, 0), min(int(xs.max()) + 1 + pad, shape[1]))


def outlines(mask: np.ndarray, min_pixels: int = 1) -> list[np.ndarray]:
    """Closed outlines of a mask as (x, y) arrays in pixel-edge coordinates.

    Pixel (r, c) covers [c, c + 1] x [r, r + 1], so an outline drawn at k x
    magnification lines up with the pasted raster at any k.
    """
    from skimage import measure
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    out = []
    padded = np.pad(mask, 1).astype(np.float32)
    for contour in measure.find_contours(padded, 0.5):
        if len(contour) < 3:
            continue
        rows = contour[:, 0] - 1.0 + 0.5
        cols = contour[:, 1] - 1.0 + 0.5
        # area check drops specks that are only a pixel or two
        area = 0.5 * abs(np.dot(cols, np.roll(rows, 1)) - np.dot(rows, np.roll(cols, 1)))
        if area < min_pixels * 0.5:
            continue
        out.append(np.column_stack([cols, rows]))
    return out


def grid_shape(n: int) -> tuple[int, int]:
    """(columns, rows) for n panels in reading order."""
    if n <= 3:
        return n, 1
    cols = int(math.ceil(math.sqrt(n)))
    return cols, int(math.ceil(n / cols))


# --------------------------------------------------------------------------
# scale bars
# --------------------------------------------------------------------------
def nice_length(max_um: float) -> float:
    """The largest 1-2-5 length not above max_um."""
    if max_um <= 0:
        return 0.0
    exponent = math.floor(math.log10(max_um))
    best = 10.0 ** exponent
    for step in (1.0, 2.0, 5.0, 10.0):
        value = step * 10.0 ** exponent
        if value <= max_um * (1 + 1e-9):
            best = value
    return best


def length_label(um: float) -> str:
    if um >= 1000:
        value = um / 1000.0
        return f"{value:g} mm"
    return f"{um:g} µm"


def scale_bar(canvas: Canvas, x_right: float, y_bar: float, px_per_um: float,
              max_px: float, *, colour: str = theme.INK, label_colour: str = theme.INK_2,
              align: str = "right") -> tuple[float, float]:
    """A thin bar of a round length ending at x_right (or starting there if
    align='left'), its label above it. Returns (bar length in px, length in um)."""
    um = nice_length(max_px / px_per_um)
    length = um * px_per_um
    thickness = max(canvas.px(BAR_PT), 2.0)
    x0 = x_right - length if align == "right" else x_right
    canvas.rect(x0, y_bar - thickness / 2, length, thickness, colour)
    canvas.text(x0 + length / 2, y_bar - thickness / 2 - canvas.px(3.0), length_label(um),
                size=SCALE_PT, colour=label_colour, weight="regular", ha="center", va="baseline")
    return length, um
