from __future__ import annotations

import re

import pytest

from gcalign import theme

HEX = re.compile(r"^#[0-9A-F]{6}$")


def luminance(value: str) -> float:
    def channel(c: int) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = theme.hex_to_rgb(value)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_on_black(value: str) -> float:
    return (luminance(value) + 0.05) / 0.05


def test_the_page_is_black():
    assert theme.BG == "#000000"


@pytest.mark.parametrize("name, minimum", [("INK", 15), ("INK_2", 7), ("MUTED", 4.5), ("IF", 4.5), ("ST", 4.5)])
def test_text_colours_are_readable_on_black(name, minimum):
    assert contrast_on_black(getattr(theme, name)) >= minimum


@pytest.mark.parametrize("style", theme.PALETTES)
def test_identity_palette_is_stable_and_distinct(style):
    ids = list(range(1, 81))
    palette = theme.identity_colours(ids, style)
    assert list(palette) == ids
    assert all(HEX.match(c) for c in palette.values())
    assert palette == theme.identity_colours(reversed(ids), style)     # order of asking is irrelevant
    assert len(set(palette.values())) == len(ids)


def test_aurora_keeps_an_even_lightness():
    values = [luminance(c) for c in theme.identity_colours(range(1, 61)).values()]
    assert min(values) > 0.25 and max(values) < 0.85      # nothing murky, nothing white


def test_oklch_stays_in_gamut():
    for hue in range(0, 360, 15):
        assert HEX.match(theme.oklch(0.8, 0.4, hue))       # impossible chroma is reduced, not clipped


def test_fonts_ship_with_the_package():
    for weight in ("light", "regular", "medium", "semibold"):
        assert theme.font_path(weight).exists()
    assert (theme.FONT_DIR / "OFL.txt").exists()
    with pytest.raises(ValueError):
        theme.font_path("black")


def test_css_and_json_carry_the_same_colours():
    css, data = theme.css_vars(), theme.as_dict()
    assert f"--bg:{theme.BG}" in css and data["bg"] == theme.BG
    assert f"--if:{theme.IF}" in css and data["st"] == theme.ST


def test_text_colours_are_readable_on_a_panel():
    """Text also sits on the raised panel colour, which is slightly lighter than the page."""
    panel = luminance(theme.SURFACE)
    for name in ("INK", "INK_2", "MUTED"):
        assert (luminance(getattr(theme, name)) + 0.05) / (panel + 0.05) >= 4.5, name


def test_the_theme_font_can_be_embedded_in_html():
    css = theme.font_face_css("µ α")
    if not css:
        pytest.skip("fontTools is not installed")
    assert css.count("@font-face") == 3 and "font-family:Inter" in css
    assert css == theme.font_face_css("µ α")            # the same input gives the same bytes


def test_kind_colours_follow_the_base_colour():
    ids = list(range(1, 30))
    st, default = theme.kind_colours(ids, "ST", "#FF4500"), theme.kind_colours(ids, "ST")
    assert set(st) == set(ids) and st != default
    with pytest.raises(ValueError):
        theme.kind_colours(ids, "XX")
