"""The quantitative figures: warp error per IF section, orientation candidates, and GC presence per section."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import theme
from . import _base as B


def _px(points: float, dpi: float) -> float:
    return points * dpi / 72.0


def _note(message: str) -> None:
    print(f"[gcalign.figures] {message}", flush=True)


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    for step in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if step * 10 ** exponent >= value:
            return step * 10 ** exponent
    return 10.0 ** (exponent + 1)


def _axis_style(ax, *, x_label: str | None = None, y_label: str | None = None,
                grid_x: bool = False, grid_y: bool = False, left: bool = False) -> None:
    """Hairline axes in the theme: only the bottom (and optionally left) spine."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(theme.LINE_2)
        ax.spines[side].set_linewidth(B.LINE_PT)
    ax.tick_params(axis="both", colors=theme.MUTED, labelcolor=theme.INK_2,
                   labelsize=B.SMALL_PT, length=3.0, width=B.LINE_PT, pad=3.0)
    if not left:
        ax.tick_params(axis="y", length=0)
    if grid_x:
        ax.grid(axis="x", color=theme.LINE, lw=B.LINE_PT * 0.8, zorder=0)
    if grid_y:
        ax.grid(axis="y", color=theme.LINE, lw=B.LINE_PT * 0.8, zorder=0)
    ax.set_axisbelow(True)
    if x_label:
        ax.set_xlabel(x_label, fontproperties=B.font("regular", B.AXIS_PT), color=theme.INK_2,
                      labelpad=6.0)
    if y_label:
        ax.set_ylabel(y_label, fontproperties=B.font("regular", B.AXIS_PT), color=theme.INK_2,
                      labelpad=6.0)


def _tick_fonts(ax) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(B.font("regular", B.SMALL_PT))


# --------------------------------------------------------------------------
# 3. alignment
# --------------------------------------------------------------------------
def alignment_table(run) -> list[dict]:
    """The numbers the alignment figure plots, one dict per IF section in cutting order."""
    rows = []
    for name in run.if_names:
        match = run.match.get(name)
        if match is None:
            continue
        place = run.transforms.get(name, {})
        two = match.get("two_sided_disagreement")
        rows.append({
            "plane": name,
            "neighbours": run.cfg.neighbours.get(run.plane(name).index, []),
            "before_um": float(match["centroid_error_pre_um"]),
            "after_um": float(match["tps_residual_median_um"]),
            "held_out_um": float(match["tps_loo_median_um"]),
            "two_sided_um": None if not two else float(two["median_um"]),
            "anchors": int(match["n_anchor_pairs"]),
            "if_objects": int(match.get("n_if_objects", place.get("n_if_gc", 0))),
            "stage_a_matched": place.get("n_matched"),
            "stage_a_within_close": place.get("n_matched_under_150um"),
            "stage_a_median_um": place.get("match_median_um"),
        })
    return rows


def alignment(run, out=None, *, dpi: float = 220, fmt: str = "png") -> Path | None:
    """Draw the alignment error of each IF section, one row per section in cutting order.

    Panel 1: distance between matched IF and ST GC centres before the elastic
    warp (ring) and after it (arrow head), in um. Panel 2: the leave-one-out
    (leave-one-out) error of the warp, in um. Panel 3: the two-sided disagreement,
    the distance between the warps to the ST section before and the ST section
    after the IF section, in um; a section at an end of the stack has one ST
    neighbour and is marked "one-sided". Panel 4: IF GCs used as warp anchors
    (blue) out of all GC objects found on the IF section (grey).
    """
    from matplotlib.ticker import MaxNLocator

    run = B.as_run(run)
    out = Path(out) if out is not None else run.root / "06_figures" / f"alignment.{fmt}"
    rows = alignment_table(run)
    if not rows:
        _note("skipped the alignment figure: 03_gc_match/match_report.json has no IF section")
        return None
    with B.styled():
        n = len(rows)
        margin = _px(B.MARGIN_PT, dpi)
        row_h = _px(26, dpi)
        names_w = max(B.Canvas(10, 10, dpi).text_width(r["plane"], B.LABEL_PT) for r in rows) + _px(14, dpi)
        col_w, col_gap = _px(150, dpi), _px(26, dpi)
        plot_h = n * row_h
        x_axis_h = _px(40, dpi)
        width = 2 * margin + names_w + 4 * col_w + 3 * col_gap
        height = margin + plot_h + x_axis_h + margin * 0.5
        canvas = B.Canvas(width, height, dpi)

        microns = [r[k] for r in rows for k in ("before_um", "after_um", "held_out_um", "two_sided_um")
                   if r[k] is not None]
        x_max = _nice_ceiling(max(microns) * 1.06)
        ys = np.arange(n)
        x0 = margin + names_w
        axes = []
        labels = ("centroid error (µm)", "leave-one-out error (µm)",
                  "two-sided disagreement (µm)", "matched GCs")
        for index, label in enumerate(labels):
            ax = canvas.chart_axes(x0 + index * (col_w + col_gap), margin, col_w, plot_h)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_yticks([])
            _axis_style(ax, x_label=label, grid_x=True)
            axes.append(ax)
        for ax in axes[:3]:
            ax.set_xlim(0, x_max)
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))

        dot = dict(ms=6.2, mec=theme.BG, mew=1.2, zorder=4)
        for y, r in zip(ys, rows):
            # before -> after
            axes[0].plot([r["before_um"]], [y], "o", ms=5.6, mfc=theme.BG, mec=theme.INK_2,
                         mew=1.1, zorder=4)
            axes[0].annotate("", xy=(r["after_um"], y), xytext=(r["before_um"], y),
                             arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.28",
                                             color=theme.IF, lw=1.5, shrinkA=3.2, shrinkB=0.0),
                             zorder=3)
            axes[1].plot([r["held_out_um"]], [y], "o", color=theme.IF, **dot)
            if r["two_sided_um"] is not None:
                axes[2].plot([r["two_sided_um"]], [y], "o", color=theme.IF, **dot)
            else:
                # an end section has one flank, so there is nothing to disagree with
                text = "one-sided" if len(r["neighbours"]) < 2 else "no estimate"
                axes[2].text(x_max * 0.02, y, text, color=theme.MUTED, va="center", ha="left",
                             fontproperties=B.font("regular", B.SMALL_PT))
            axes[3].barh([y], [r["if_objects"]], height=0.34, color=theme.LINE_2, zorder=2)
            axes[3].barh([y], [r["anchors"]], height=0.34, color=theme.IF, zorder=3)
        counts_max = max(r["if_objects"] for r in rows)
        axes[3].set_xlim(0, _nice_ceiling(counts_max * 1.04))
        axes[3].xaxis.set_major_locator(MaxNLocator(4, integer=True, steps=[1, 2, 5, 10]))
        for ax in axes:
            _tick_fonts(ax)
        for y, r in zip(ys, rows):
            y_px = margin + (y + 0.5) * row_h
            canvas.text(margin, y_px, r["plane"], colour=B.kind_colour("IF"), va="center_baseline")
        return canvas.save(out, fmt)


# --------------------------------------------------------------------------
# 4. orientation
# --------------------------------------------------------------------------
def orientation(run, out=None, *, dpi: float = 220, fmt: str = "png",
                search: dict | None = None) -> Path | None:
    """Draw every orientation candidate of every IF section, one panel per section.

    x: Dice overlap of the IF and ST tissue outlines for the candidate. y:
    number of IF GC centres within the close-match radius of an ST GC centre.
    The kept orientation is a large blue dot labelled with its rotation; other
    candidates are grey dots (not mirrored) and grey rings (mirrored). Needs
    ``02_global_align/orientation_search.json``, which gcalign 0.2 and later
    write; for older runs the figure is skipped.
    """
    run = B.as_run(run)
    out = Path(out) if out is not None else run.root / "06_figures" / f"orientation.{fmt}"
    search = search if search is not None else run.orientation_search
    if not search or not search.get("sections"):
        _note("skipped the orientation figure: this run has no 02_global_align/orientation_search.json "
              "(written by gcalign 0.2 and later)")
        return None
    from matplotlib.ticker import MaxNLocator

    sections = [n for n in run.if_names if n in search["sections"]]
    sections += [n for n in search["sections"] if n not in sections]
    radius = float(search.get("close_match_um", run.cfg.align.close_match_um))
    with B.styled():
        n = len(sections)
        margin = _px(B.MARGIN_PT, dpi)
        panel, gap = _px(118, dpi), _px(14, dpi)
        band = _px(B.BAND_PT, dpi)
        y_axis_w, x_axis_h = _px(44, dpi), _px(40, dpi)
        width = 2 * margin + y_axis_w + n * panel + (n - 1) * gap
        height = margin * 0.6 + band + panel + x_axis_h + margin * 0.4
        canvas = B.Canvas(width, height, dpi)

        dice = [c["tissue_dice"] for s in sections for c in search["sections"][s]["candidates"]]
        close = [c["n_close"] for s in sections for c in search["sections"][s]["candidates"]]
        span = max(max(dice) - min(dice), 0.02)
        x_lo, x_hi = min(dice) - 0.12 * span, max(dice) + 0.12 * span
        y_hi = _nice_ceiling(max(close) * 1.2)           # room for the winner's label
        top = margin * 0.6 + band
        for index, name in enumerate(sections):
            x = margin + y_axis_w + index * (panel + gap)
            ax = canvas.chart_axes(x, top, panel, panel)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(0, y_hi)
            _axis_style(ax, left=True, grid_y=True)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
            if index:
                ax.tick_params(axis="y", labelleft=False)
            entry = search["sections"][name]
            winner = int(entry.get("winner", 0))
            mirrored = []
            for position, cand in enumerate(entry["candidates"]):
                if position == winner:
                    continue
                if cand.get("flip"):
                    mirrored.append(cand)
                    ax.plot([cand["tissue_dice"]], [cand["n_close"]], "o", ms=6.0, mfc="none",
                            mec=theme.INK_2, mew=1.2, zorder=3)
                else:
                    ax.plot([cand["tissue_dice"]], [cand["n_close"]], "o", ms=6.0,
                            color=theme.INK_2, mec=theme.BG, mew=1.0, zorder=3)
            best = entry["candidates"][winner]
            ax.plot([best["tissue_dice"]], [best["n_close"]], "o", ms=8.5,
                    mfc=theme.IF if not best.get("flip") else theme.BG, mec=theme.IF,
                    mew=1.5, zorder=4)
            angle = f"{best.get('angle_deg', 0):.0f}°" + (" mirrored" if best.get("flip") else "")
            ax.annotate(angle, (best["tissue_dice"], best["n_close"]), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom", color=theme.INK_2,
                        fontproperties=B.font("regular", B.SMALL_PT))
            if index == 0 and mirrored:
                # one label teaches the ring: the strongest mirrored candidate
                ring = max(mirrored, key=lambda c: (c["n_close"], c["tissue_dice"]))
                ax.annotate("mirrored", (ring["tissue_dice"], ring["n_close"]), xytext=(0, 7),
                            textcoords="offset points", ha="center", va="bottom", color=theme.MUTED,
                            fontproperties=B.font("regular", B.SMALL_PT))
            _tick_fonts(ax)
            canvas.text(x, top - _px(B.LABEL_GAP_PT, dpi), name, colour=B.kind_colour("IF"))
        x_mid = margin + y_axis_w + (n * panel + (n - 1) * gap) / 2
        canvas.text(x_mid, height - margin * 0.4 - _px(4, dpi), "tissue overlap (Dice)",
                    size=B.AXIS_PT, weight="regular", colour=theme.INK_2, ha="center")
        canvas.text(margin * 0.75, top + panel / 2, f"GCs within {radius:g} µm",
                    size=B.AXIS_PT, weight="regular", colour=theme.INK_2, ha="center",
                    va="center", rotation=90)
        return canvas.save(out, fmt)


# --------------------------------------------------------------------------
# 5. presence
# --------------------------------------------------------------------------
def presence_matrix(run):
    """GC ids and a GC x section matrix of areas in um^2, from the label rasters; rows by total area, largest first."""
    from ._data import gc_areas
    areas = gc_areas(run)
    ids = sorted(areas, key=lambda g: (-sum(areas[g].values()), g))
    matrix = np.array([[areas[g].get(n, 0) for n in run.names] for g in ids], float)
    return ids, matrix * run.grid.um ** 2


def presence(run, out=None, *, dpi: float = 220, fmt: str = "png") -> Path | None:
    """Draw which GC is on which section: one row per GC, one column per section.

    Rows are ordered by the GC's area summed over all sections, largest first.
    Columns are the sections in cutting order. A cell is blue on an IF section
    and pink on an ST section, brighter for a larger cross-section area; a
    black cell means the GC is absent from that section.
    """
    run = B.as_run(run)
    out = Path(out) if out is not None else run.root / "06_figures" / f"presence.{fmt}"
    ids, area = presence_matrix(run)
    if not ids:
        _note("skipped the presence figure: no section has a GC")
        return None
    with B.styled():
        names = run.names
        n_rows, n_cols = len(ids), len(names)
        margin = _px(B.MARGIN_PT, dpi)
        cell_w, cell_h, gap = _px(30, dpi), _px(6.6, dpi), max(_px(0.9, dpi), 2.0)
        probe = B.Canvas(10, 10, dpi)
        id_w = max(probe.text_width(str(g), B.SMALL_PT, "regular") for g in ids) + _px(8, dpi)
        band = _px(B.BAND_PT, dpi)
        key_h = _px(46, dpi)
        grid_w = n_cols * cell_w + (n_cols - 1) * gap
        grid_h = n_rows * cell_h + (n_rows - 1) * gap
        width = 2 * margin + id_w + grid_w
        height = margin * 0.6 + band + grid_h + key_h + margin * 0.4
        canvas = B.Canvas(width, height, dpi)

        top = margin * 0.6 + band
        left = margin + id_w
        peak = float(area.max())
        floor = 0.22

        def shade(kind: str, fraction):
            base = B.solid(B.kind_colour(kind))
            t = floor + (1 - floor) * np.clip(fraction, 0, 1)
            return np.asarray(t)[..., None] * base

        image = np.zeros((int(round(grid_h)), int(round(grid_w)), 3), np.float32)
        ys = [int(round(r * (cell_h + gap))) for r in range(n_rows)]
        xs = [int(round(c * (cell_w + gap))) for c in range(n_cols)]
        ch, cw = int(round(cell_h)), int(round(cell_w))
        for c, name in enumerate(names):
            kind = run.plane(name).kind
            for r in range(n_rows):
                if area[r, c] > 0:
                    image[ys[r]:ys[r] + ch, xs[c]:xs[c] + cw] = shade(kind, area[r, c] / peak)
        canvas.raster(image, left, top)

        for c, name in enumerate(names):
            canvas.text(left + xs[c] + cw / 2, top - _px(B.LABEL_GAP_PT, dpi), name,
                        colour=B.kind_colour(run.plane(name).kind), ha="center")
        for r, gid in enumerate(ids):
            canvas.text(left - _px(6, dpi), top + ys[r] + ch / 2, str(gid), size=B.SMALL_PT,
                        weight="regular", colour=theme.MUTED, ha="right", va="center_baseline")

        # key: the two ramps, 0 -> largest cross-section, in 10^3 um^2
        ramp_w, ramp_h = min(grid_w * 0.42, _px(120, dpi)), max(_px(4.2, dpi), 6.0)
        y_ramp = top + grid_h + _px(16, dpi)
        steps = np.linspace(0, 1, int(round(ramp_w)))
        kinds = [k for k in ("IF", "ST") if any(run.plane(n).kind == k for n in names)]
        for row, kind in enumerate(kinds):
            strip = np.repeat(shade(kind, steps)[None, :, :], int(round(ramp_h)), axis=0)
            canvas.raster(strip, left, y_ramp + row * (ramp_h + _px(2.5, dpi)))
        y_text = y_ramp + len(kinds) * (ramp_h + _px(2.5, dpi)) + _px(9, dpi)
        top_value = peak / 1e3
        canvas.text(left, y_text, "0", size=B.SMALL_PT, weight="regular", colour=theme.INK_2)
        canvas.text(left + ramp_w, y_text, f"{top_value:.0f}", size=B.SMALL_PT, weight="regular",
                    colour=theme.INK_2, ha="right")
        canvas.text(left + ramp_w + _px(8, dpi), y_ramp + ramp_h, "10³ µm²",
                    size=B.SMALL_PT, weight="regular", colour=theme.INK_2, va="center_baseline")
        return canvas.save(out, fmt)
