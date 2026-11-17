"""Colours, fonts and palettes for all output of gcalign.

The HTML viewer, the figures, the report and the film read every colour and
font from this module, so all outputs look the same.

Rules:

  background   pure black (`BG`, #000000) in every output; no tint, no gradient
  text, lines  neutral greys (`INK`, `INK_2`, `MUTED`, `LINE`)
  IF           blue (`IF`): immunofluorescence
  ST           pink (`ST`): spatial transcriptomics
  overlap      white (`AGREE`): where IF and ST regions coincide
  GC colours   one colour per germinal-centre id, the same in every output
               (`identity_colours`)
  font         Inter; the font files are in gcalign/assets/fonts (SIL Open Font
               License), so the output does not depend on installed fonts

This module does not import matplotlib, PIL or plotly when it is imported. Each
user calls the helper it needs: `use_mpl` (matplotlib), `pil_font` (PIL),
`css_vars` and `as_dict` (HTML).
"""
from __future__ import annotations

import colorsys
import math
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# surface
# --------------------------------------------------------------------------
BG = "#000000"          # every canvas, frame and page
SURFACE = "#0A0A0B"     # a raised panel: viewer sidebar, report card
SURFACE_2 = "#131315"   # a control sitting on a panel
LINE = "#222225"        # hairline
LINE_2 = "#34343A"      # hairline under hover or emphasis
INK = "#F5F5F7"         # primary text
INK_2 = "#A1A1AA"       # secondary text
MUTED = "#7C7C86"       # tertiary text, tick labels (5.0:1 on BG, 4.7:1 on SURFACE)
FAINT = "#3A3A42"       # ghost outlines and disabled controls; too dark for text

# --------------------------------------------------------------------------
# meaning
# --------------------------------------------------------------------------
IF = "#4D8DFF"          # immunofluorescence
ST = "#FF4D9D"          # spatial transcriptomics
AGREE = "#FFFFFF"       # where the two modalities overlap
GOOD = "#4ADE9B"
WARN = "#FFC857"
BAD = "#FF6B6B"
DZ = "#3DDC84"          # dark zone
LZ = "#F5F5F7"          # light zone
TISSUE = "#161618"      # body of a section that has no image to show
TISSUE_EDGE = "#2A2A30"
FOLLICLE = "#A9C6FB"    # B-cell follicles drawn behind the GCs (optional viewer layer)

KIND = {"IF": IF, "ST": ST}

# the three display channels of the IF stain are shown as pure red, green and blue
STAIN_RGB = ("#FF0000", "#00FF00", "#0000FF")

# family bases for the per-GC colour families of the 3D viewer; a config can
# override them with the colours its own atlas uses
ST_FAMILY_HEX = "#FF1493"
IF_FAMILY_HEX = "#0F52BA"

PALETTES = ("aurora", "hsv")


# --------------------------------------------------------------------------
# colour helpers
# --------------------------------------------------------------------------
def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{int(min(max(round(c), 0), 255)):02X}" for c in rgb)


def rgb01(value: str) -> tuple[float, float, float]:
    return tuple(c / 255.0 for c in hex_to_rgb(value))


def rgba(value: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    """A colour for PIL: 0-255 integers with alpha."""
    return (*hex_to_rgb(value), int(round(min(max(alpha, 0.0), 1.0) * 255)))


def mix(a: str, b: str, t: float) -> str:
    """Blend two hex colours; t = 0 gives `a`, t = 1 gives `b`."""
    ra, rb = hex_to_rgb(a), hex_to_rgb(b)
    return rgb_to_hex([x + (y - x) * t for x, y in zip(ra, rb)])


def _oklch_to_linear(L: float, C: float, h_deg: float) -> tuple[float, float, float]:
    a = C * math.cos(math.radians(h_deg))
    b = C * math.sin(math.radians(h_deg))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


def _encode(channel: float) -> float:
    channel = min(max(channel, 0.0), 1.0)
    return 12.92 * channel if channel <= 0.0031308 else 1.055 * channel ** (1 / 2.4) - 0.055


def oklch(L: float, C: float, h_deg: float) -> str:
    """An OKLCH colour as sRGB hex.

    If the requested chroma is outside the sRGB gamut it is reduced until the
    colour fits, keeping lightness and hue - so asking for an even lightness
    across hues gives an even lightness, never a clipped, darker blue.
    """
    lo, hi = 0.0, C
    rgb = _oklch_to_linear(L, C, h_deg)
    if not all(-1e-4 <= c <= 1 + 1e-4 for c in rgb):
        for _ in range(24):
            mid = (lo + hi) / 2
            rgb = _oklch_to_linear(L, mid, h_deg)
            if all(-1e-4 <= c <= 1 + 1e-4 for c in rgb):
                lo = mid
            else:
                hi = mid
        rgb = _oklch_to_linear(L, lo, h_deg)
    return rgb_to_hex([_encode(c) * 255 for c in rgb])


# --------------------------------------------------------------------------
# palettes
# --------------------------------------------------------------------------
_GOLDEN_ANGLE = 137.50776405003785
# three lightness / chroma tiers, cycled, so two ids whose hues land close
# together still differ in weight
_AURORA_TIERS = ((0.80, 0.150), (0.70, 0.170), (0.86, 0.130))


def identity_colours(ids, style: str = "aurora") -> dict[int, str]:
    """One colour per germinal-centre id, the same in every view.

    ``aurora`` (default) walks the hue circle by the golden angle at an even
    perceived lightness, so no colour shouts over its neighbours on black and
    consecutive ids never sit on consecutive hues. ``hsv`` reproduces the
    saturated palette of gcalign 0.1 and of the four-section atlas, for when
    colours have to match figures made with those.
    """
    ordered = sorted(int(i) for i in ids)
    if style == "hsv":
        out = {}
        for index, gid in enumerate(ordered):
            hue = index / len(ordered)
            sat = 0.6 + (index % 3) * 0.15
            val = 0.7 + (index % 2) * 0.2
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            out[gid] = f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
        return out
    if style != "aurora":
        raise ValueError(f"unknown palette {style!r}; choose one of {PALETTES}")
    out = {}
    for index, gid in enumerate(ordered):
        L, C = _AURORA_TIERS[index % len(_AURORA_TIERS)]
        out[gid] = oklch(L, C, (28.0 + index * _GOLDEN_ANGLE) % 360.0)
    return out


def family_colours(ids, base_hex: str, hue_spread: float = 0.08,
                   sat_range: tuple[float, float] = (0.60, 1.0),
                   val_range: tuple[float, float] = (0.70, 1.0)) -> dict[int, str]:
    """Shades of one base colour, one per id.

    Used where a whole section is drawn in its modality's colour - blue for IF,
    pink for spatial transcriptomics - and neighbouring germinal centres still
    have to be told apart.
    """
    base = rgb01(base_hex)
    h0, _, _ = colorsys.rgb_to_hsv(*base)
    ordered = list(ids)
    out = {}
    for index, gid in enumerate(ordered):
        t = index / max(len(ordered) - 1, 1)
        hue = (h0 + hue_spread * (t - 0.5)) % 1.0
        sat = sat_range[0] + (sat_range[1] - sat_range[0]) * ((index * 7) % 5) / 4
        val = val_range[0] + (val_range[1] - val_range[0]) * ((index * 3) % 4) / 3
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        out[int(gid)] = f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
    return out


# shades of the ST and IF colour, one per GC, where a whole section is drawn in
# the colour of its kind (the 3D viewer's section views, the film's last act)
_KIND_SHADES = {
    "ST": {"hue_spread": 0.085, "sat_range": (0.62, 1.0), "val_range": (0.80, 1.0)},
    "IF": {"hue_spread": 0.075, "sat_range": (0.50, 0.92), "val_range": (0.84, 1.0)},
}


def kind_colours(ids, kind: str, base_hex: str | None = None) -> dict[int, str]:
    """One shade of the section kind's colour per GC id.

    `kind` is "ST" or "IF". `base_hex` is the colour the shades are built
    around: `viewer.st_hex` or `viewer.if_hex` from the config (default: the
    theme's family colour for that kind). The shades only separate neighbouring
    GCs; they carry no other information.
    """
    if kind not in _KIND_SHADES:
        raise ValueError(f"kind must be ST or IF, not {kind!r}")
    base = base_hex or (ST_FAMILY_HEX if kind == "ST" else IF_FAMILY_HEX)
    return family_colours(ids, base, **_KIND_SHADES[kind])


# --------------------------------------------------------------------------
# type
# --------------------------------------------------------------------------
FONT_DIR = Path(__file__).parent / "assets" / "fonts"
FONT_FAMILY = "Inter"
_FONT_FILES = {"light": "Inter-Light.ttf", "regular": "Inter-Regular.ttf",
               "medium": "Inter-Medium.ttf", "semibold": "Inter-SemiBold.ttf"}
_WEIGHT_NUMBER = {"light": 300, "regular": 400, "medium": 500, "semibold": 600}
# what a browser falls back to when Inter is not installed on the viewing machine
CSS_FONT_STACK = ('Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
                  '"Helvetica Neue", Arial, sans-serif')
CSS_MONO_STACK = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def font_path(weight: str = "regular") -> Path:
    try:
        return FONT_DIR / _FONT_FILES[weight]
    except KeyError:
        raise ValueError(f"unknown weight {weight!r}; choose one of {sorted(_FONT_FILES)}") from None


@lru_cache(maxsize=256)
def pil_font(size: int, weight: str = "regular"):
    """The theme typeface as a PIL font (cached)."""
    from PIL import ImageFont
    return ImageFont.truetype(str(font_path(weight)), int(size))


def mpl_font(weight: str = "regular", size: float | None = None):
    """The theme typeface as matplotlib FontProperties, by file.

    Selecting by file rather than by family name does not depend on how the
    running matplotlib resolves weights, so "medium" is always Inter Medium.
    """
    from matplotlib.font_manager import FontProperties
    props = FontProperties(fname=str(font_path(weight)))
    if size is not None:
        props.set_size(size)
    return props


def use_mpl(font_size: float = 10.0) -> None:
    """Switch matplotlib to the black theme. Safe to call more than once."""
    import matplotlib as mpl
    from matplotlib import font_manager

    known = {f.fname for f in font_manager.fontManager.ttflist}
    for name in _FONT_FILES.values():
        path = str(FONT_DIR / name)
        if path not in known:
            font_manager.fontManager.addfont(path)
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": font_size,
        "font.weight": "normal",
        "text.color": INK,
        "figure.facecolor": BG,
        "figure.edgecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": LINE_2,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": 500,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": LINE,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
        "legend.frameon": False,
        "legend.fontsize": font_size - 1,
        "image.interpolation": "nearest",
        "savefig.dpi": 220,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# characters every HTML page of the package may use: Latin, Latin-1 and Latin
# Extended-A, and the signs that appear in units and labels
_HTML_CHARACTERS = (list(range(0x20, 0x7F)) + list(range(0xA0, 0x180))
                    + [0x03BC, 0x2009, 0x200A, 0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D,
                       0x2022, 0x2026, 0x2070, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079,
                       0x207B, 0x2190, 0x2191, 0x2192, 0x2193, 0x2212, 0x2248, 0x2264, 0x2265,
                       0x2318, 0x21E7])


def font_face_css(extra_text: str = "",
                  weights: tuple[str, ...] = ("regular", "medium", "semibold")) -> str:
    """CSS `@font-face` rules that embed the theme font in an HTML page.

    Each weight is cut down to the characters an HTML page of the package can
    show (plus those in `extra_text`, for example a run's title and section
    names) and embedded as WOFF, about 35 kB per weight. The page then shows
    the same font on every computer and needs no network.

    Returns "" when fontTools is not installed; the page then uses the system
    fonts listed in `CSS_FONT_STACK`. The font's own timestamp is kept, so the
    same input always gives the same bytes.
    """
    import base64
    import io

    try:
        from fontTools import subset
        from fontTools.ttLib import TTFont
    except Exception:                                   # pragma: no cover
        return ""
    wanted = sorted(set(_HTML_CHARACTERS) | {ord(c) for c in extra_text if ord(c) >= 0x20})
    rules = []
    for weight in weights:
        try:
            options = subset.Options()
            options.flavor = "woff"
            options.layout_features = ["kern", "liga", "calt", "tnum", "case", "cv11", "ss01"]
            options.name_IDs = ["*"]                    # keep the licence text in the font
            options.hinting = False
            options.desubroutinize = True
            font = TTFont(str(font_path(weight)), recalcTimestamp=False)
            subsetter = subset.Subsetter(options)
            subsetter.populate(unicodes=wanted)
            subsetter.subset(font)
            buffer = io.BytesIO()
            font.flavor = "woff"
            font.save(buffer)
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
        except Exception as error:                      # pragma: no cover
            print(f"  font: {FONT_FAMILY} {weight} not embedded ({error})")
            continue
        rules.append(f"@font-face{{font-family:{FONT_FAMILY};font-style:normal;"
                     f"font-weight:{_WEIGHT_NUMBER[weight]};font-display:block;"
                     f"src:url(data:font/woff;base64,{data}) format('woff')}}")
    return "\n".join(rules)


# --------------------------------------------------------------------------
# hand-offs to the HTML side
# --------------------------------------------------------------------------
def as_dict() -> dict[str, str]:
    """Every named colour, for a JSON payload."""
    return {"bg": BG, "surface": SURFACE, "surface2": SURFACE_2, "line": LINE,
            "line2": LINE_2, "ink": INK, "ink2": INK_2, "muted": MUTED, "faint": FAINT,
            "if": IF, "st": ST, "agree": AGREE, "good": GOOD, "warn": WARN, "bad": BAD,
            "dz": DZ, "lz": LZ, "tissue": TISSUE, "tissueEdge": TISSUE_EDGE,
            "follicle": FOLLICLE}


def css_vars() -> str:
    """The theme as CSS custom properties, to drop inside ``:root{ ... }``."""
    names = {"bg": BG, "surface": SURFACE, "surface-2": SURFACE_2, "line": LINE,
             "line-2": LINE_2, "ink": INK, "ink-2": INK_2, "muted": MUTED, "faint": FAINT,
             "if": IF, "st": ST, "agree": AGREE, "good": GOOD, "warn": WARN, "bad": BAD,
             "dz": DZ, "lz": LZ, "tissue": TISSUE, "tissue-edge": TISSUE_EDGE,
             "follicle": FOLLICLE}
    parts = [f"--{key}:{value}" for key, value in names.items()]
    parts.append(f"--font:{CSS_FONT_STACK}")
    parts.append(f"--mono:{CSS_MONO_STACK}")
    return ";".join(parts) + ";"
