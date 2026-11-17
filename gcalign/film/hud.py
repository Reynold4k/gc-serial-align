"""The film's on-screen text: a title, one live counter, keys and a progress line.

Every label on screen says what it shows: a counter carries its name and its
total ("Spateo iteration 412 / 1100"), a measurement carries its name and unit
("mean distance between matched GCs 53 µm"), and every colour that encodes
something has a key. Everything else is left off the picture.

All sizes are set on a 1080-line design grid and scaled to the frame, so a
480 x 270 test frame and the 1920 x 1080 master are the same picture. Colours
and type come from `gcalign.theme`. Numbers that change every frame are set in
tabular figures (Inter's ``tnum`` feature), so a counter never shifts sideways
while its digit count stays the same.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from .. import theme
from .core import Vectors

_NICE_UM = (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000)
_DIGITS = "0123456789"


def _has_raqm() -> bool:
    """Font features such as ``tnum`` need Pillow built with libraqm."""
    try:
        from PIL import features
        return bool(features.check("raqm"))
    except Exception:                                            # pragma: no cover
        return False


RAQM = _has_raqm()


def _features(tnum: bool):
    return ["tnum"] if (tnum and RAQM) else None


def _cells(text: str, tnum: bool) -> bool:
    """True when digits must be set in fixed-width cells by hand: tabular
    figures were asked for, but this Pillow cannot switch the font feature on."""
    return tnum and not RAQM and any(ch in _DIGITS for ch in text)


class Pen:
    """The frame being annotated and a drawer for it.

    Text is not drawn with ``ImageDraw.text``, because Pillow ignores the alpha
    of a text colour: a fading label would stay at full strength. Each string is
    rendered into a coverage mask scaled by its alpha, and its colour is pasted
    through that mask instead (see `Hud.text`)."""

    def __init__(self, image: Image.Image):
        if image.mode != "RGB":
            raise ValueError("annotate the RGB frame: Pillow blends translucent shapes only there")
        self.image = image
        self.draw = ImageDraw.Draw(image, "RGBA")


class Hud:
    """Layout and type for one frame size."""

    def __init__(self, W: int, H: int):
        self.W, self.H = int(W), int(H)
        self.s = self.H / 1080.0
        self.margin = round(72 * self.s)
        self.top = round(58 * self.s)
        self.key_base = self.H - round(66 * self.s)      # baseline of the lowest key row
        self.key_step = round(27 * self.s)

    # ---- type -------------------------------------------------------------
    def px(self, value: float) -> int:
        return max(1, int(round(value * self.s)))

    def font(self, size: float, weight: str = "regular"):
        return theme.pil_font(self.px(size), weight)

    def width(self, text: str, size: float, weight: str = "regular",
              tnum: bool = False, tracking: float = 0.0) -> float:
        font = self.font(size, weight)
        if _cells(text, tnum):
            cell = max(font.getlength(d) for d in _DIGITS)
            return sum(cell if ch in _DIGITS else font.getlength(ch) for ch in text)
        if tracking:
            return sum(font.getlength(ch) for ch in text) + tracking * self.px(size) * (len(text) - 1)
        return font.getlength(text, features=_features(tnum))

    def text(self, d: Pen, xy, text: str, size: float, weight: str = "regular",
             fill=theme.INK, alpha: float = 1.0, anchor: str = "ls", tnum: bool = False,
             tracking: float = 0.0) -> None:
        """Draw a string at ``alpha``. ``tracking`` is extra letter spacing in
        em, for the small upper-case labels."""
        if alpha <= 0.004 or not text:
            return
        colour = theme.hex_to_rgb(fill) if isinstance(fill, str) else tuple(int(c) for c in fill[:3])
        font = self.font(size, weight)
        if _cells(text, tnum) and not tracking:
            # no libraqm: centre each digit in a cell as wide as the widest digit
            total = self.width(text, size, weight, tnum=True)
            x, y = xy
            if anchor[0] == "r":
                x -= total
            elif anchor[0] == "m":
                x -= total / 2
            cell = max(font.getlength(d) for d in _DIGITS)
            for ch in text:
                w = cell if ch in _DIGITS else font.getlength(ch)
                self._paste_text(d, (x + w / 2, y), ch, font, colour, alpha, "m" + anchor[1], None)
                x += w
            return
        if not tracking:
            self._paste_text(d, xy, text, font, colour, alpha, anchor, _features(tnum))
            return
        total = self.width(text, size, weight, tracking=tracking)
        x, y = xy
        if anchor[0] == "r":
            x -= total
        elif anchor[0] == "m":
            x -= total / 2
        gap = tracking * self.px(size)
        for ch in text:
            self._paste_text(d, (x, y), ch, font, colour, alpha, "l" + anchor[1], None)
            x += font.getlength(ch) + gap

    @staticmethod
    def _paste_text(d: Pen, xy, text, font, colour, alpha, anchor, features) -> None:
        """Render ``text`` into a mask of coverage x alpha and paste ``colour``
        through it, so translucent text blends with whatever is behind it."""
        x, y = int(round(xy[0])), int(round(xy[1]))
        left, top, right, bottom = font.getbbox(text, anchor=anchor, features=features)
        w, h = int(right - left) + 4, int(bottom - top) + 4
        if w <= 4 or h <= 4:
            return
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).text((2 - left, 2 - top), text, font=font, anchor=anchor,
                                  fill=int(round(255 * min(alpha, 1.0))), features=features)
        d.image.paste(colour, (x + int(left) - 2, y + int(top) - 2, x + int(left) - 2 + w,
                               y + int(top) - 2 + h), mask)

    # ---- pieces -----------------------------------------------------------
    def title(self, d, text: str, alpha: float = 1.0) -> None:
        """What the act shows, top left."""
        self.text(d, (self.margin, self.top + self.px(24)), text, 25, "medium", theme.INK, alpha)

    def tag(self, d, items) -> None:
        """Small right-aligned tags in the top-right corner: [(text, colour, alpha)]."""
        x = self.W - self.margin
        y = self.top + self.px(20)
        for text, colour, alpha in reversed(items):
            self.text(d, (x, y), text, 14, "medium", colour, alpha, anchor="rs", tracking=0.04)
            x -= self.width(text, 14, "medium", tracking=0.04) + self.px(24)

    def counter(self, d, label: str, value: str, total: str = "", sub: str = "",
                alpha: float = 1.0) -> None:
        """The one live counter, bottom left: its name, the number, its total."""
        x = self.margin
        base = self.H - self.px(112)
        self.text(d, (x, base - self.px(80)), label.upper(), 12.5, "semibold", theme.MUTED,
                  alpha, tracking=0.16)
        self.text(d, (x - self.px(3), base), value, 84, "light", theme.INK, alpha, tnum=True)
        if total:
            vw = self.width(value, 84, "light", tnum=True)
            self.text(d, (x + vw + self.px(12), base), total, 28, "light", theme.MUTED, alpha,
                      tnum=True)
        if sub:
            self.text(d, (x, base + self.px(36)), sub, 16, "regular", theme.INK_2, alpha, tnum=True)

    def progress(self, d, frac: float, segments) -> None:
        """A hairline across the foot of the frame, one segment per act."""
        y = self.H - self.px(40)
        h = max(1, self.px(2))
        x0, x1 = self.margin, self.W - self.margin
        gap = self.px(6)
        for a, b in segments:
            sa = x0 + (x1 - x0) * a + (gap / 2 if a > 0 else 0)
            sb = x0 + (x1 - x0) * b - (gap / 2 if b < 1 else 0)
            if sb <= sa:
                continue
            d.draw.rectangle((sa, y, sb, y + h - 1), fill=theme.rgba(theme.LINE_2, 1.0))
            fill_to = min(sb, x0 + (x1 - x0) * frac)
            if fill_to > sa:
                d.draw.rectangle((sa, y, fill_to, y + h - 1), fill=theme.rgba(theme.INK_2, 1.0))

    def label(self, d, xy, text: str, colour: str, alpha: float = 1.0,
              anchor: str = "rm") -> None:
        """A section's name beside it, in its kind's colour."""
        self.text(d, xy, text, 14, "medium", colour, alpha, anchor=anchor, tracking=0.04)

    # ---- keys ---------------------------------------------------------------
    def _item_width(self, glyph, colour, text, size) -> float:
        strong, rest = text if isinstance(text, tuple) else ("", text)
        w = 0.0
        if strong:
            w += self.width(strong, size, "semibold") + self.px(5)
        w += self.width(rest, size, "regular")
        g = {"dot": 7.2, "ring": 8.4, "line": 16.0}.get(glyph, 0.0)
        if glyph == "dots":
            g = 7.2 * len(colour) + 2.5 * (len(colour) - 1)
        return w + self.px(g) + self.px(7)

    def key_row(self, ov: Vectors, d, xy, items, alpha: float = 1.0, align: str = "right",
                size: float = 13) -> None:
        """One row of colour keys. Each item is (glyph, colour, text): glyph is
        "dot", "ring", "line" or "dots" (then colour is a list); text is a
        string, or (abbreviation, meaning) to set the abbreviation in bold."""
        if alpha <= 0.004 or not items:
            return
        gap = self.px(24)
        widths = [self._item_width(g, c, t, size) for g, c, t in items]
        x, y = xy
        if align == "right":
            x -= sum(widths) + gap * (len(items) - 1)
        cy = y - self.px(4.6)
        r = self.px(3.6)
        for (glyph, colour, text), w in zip(items, widths):
            if glyph == "dots":
                cx = x
                for c in colour:
                    ov.dot((cx + r, cy), r, c, alpha)
                    cx += 2 * r + self.px(2.5)
                tx = cx - self.px(2.5) + self.px(7)
            elif glyph == "dot":
                ov.dot((x + r, cy), r, colour, alpha)
                tx = x + 2 * r + self.px(7)
            elif glyph == "ring":
                ov.ring((x + r + 0.6, cy), r + 0.6, colour, alpha, 1.5 * self.s)
                tx = x + 2 * r + self.px(1.2) + self.px(7)
            elif glyph == "line":
                ov.line([(x, cy), (x + self.px(16), cy)], colour, alpha, 1.5 * self.s)
                tx = x + self.px(16) + self.px(7)
            else:
                tx = x
            strong, rest = text if isinstance(text, tuple) else ("", text)
            if strong:
                self.text(d, (tx, y), strong, size, "semibold", theme.INK, alpha)
                tx += self.width(strong, size, "semibold") + self.px(5)
            self.text(d, (tx, y), rest, size, "regular", theme.INK_2, alpha)
            x += w + gap

    def keys(self, canvas, d: Pen, rows, alpha: float = 1.0) -> float:
        """Rows of keys at the bottom right, the first row lowest. Returns the
        y above the block, for whatever sits on top of it."""
        y = self.key_base
        live = [r for r in rows if r]
        if alpha > 0.004 and live:
            ov = Vectors(self.W, self.H, region=(self.W * 0.30, self.H * 0.55, self.W, self.H))
            for row in live:
                self.key_row(ov, d, (self.W - self.margin, y), row, alpha)
                y -= self.key_step
            ov.composite(canvas)
        return y

    def scale_bar(self, d, px_per_um: float, alpha: float = 1.0, y: float | None = None) -> None:
        """A scale bar at the right - only meaningful in a top view."""
        if alpha <= 0.004 or px_per_um <= 0:
            return
        lo, hi = 70 * self.s, 190 * self.s
        length = next((L for L in _NICE_UM if lo <= L * px_per_um <= hi), None)
        if length is None:
            length = min(_NICE_UM, key=lambda L: abs(math.log(L * px_per_um / (120 * self.s))))
        bar = length * px_per_um
        x1 = self.W - self.margin
        y = (self.key_base if y is None else y) - self.px(14)
        h = max(1, self.px(2))
        d.draw.rectangle((x1 - bar, y, x1, y + h - 1), fill=theme.rgba(theme.INK_2, alpha))
        text = f"{length / 1000:g} mm" if length >= 1000 else f"{length} µm"
        self.text(d, (x1, y - self.px(10)), text, 13, "medium", theme.INK_2, alpha,
                  anchor="rs", tnum=True, tracking=0.04)

    def stats(self, d, x0: float, x1: float, y: float, rows, alpha: float = 1.0) -> float:
        """Label-value rows; values right-aligned in tabular figures."""
        step = self.px(34)
        for label, value, value_alpha in rows:
            self.text(d, (x0, y), label, 14, "regular", theme.INK_2, alpha)
            self.text(d, (x1, y), value, 20, "medium", theme.INK, alpha * value_alpha,
                      anchor="rs", tnum=True)
            y += step
        return y
