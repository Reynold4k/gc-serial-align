"""The figures drawn on the common grid: all sections, the GC mask overlay, and per-GC close-ups.

All three paste rasters at a whole-number magnification of the grid (see
`_base.Canvas`). A label map is enlarged by repeating each pixel and is never
interpolated. Only the fluorescence stain is enlarged bilinearly, because the
stain is a sampled continuous image and a label map is a set of labelled bins.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .. import theme
from . import _base as B
from ._data import display_tissue, feather, stack_window, stain_limits

FIGURE_WIDTH_IN = B.FIGURE_WIDTH_IN


def _px(points: float, dpi: float) -> float:
    return points * dpi / 72.0


def _note(message: str) -> None:
    print(f"[gcalign.figures] {message}", flush=True)


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
def _stain_panel(run, name, window, limits, tissue=None) -> np.ndarray:
    rgb = run.rgb(name, window, limits)
    if tissue is not None:
        r0, r1, c0, c1 = window
        rgb = rgb * feather(tissue[r0:r1, c0:c1])[..., None]
    return rgb


def _st_panel(run, name, window, colours, tissue, *, focus: int | None = None) -> np.ndarray:
    """An ST section as an image: the section outline in dark grey and its GC regions.

    Without `focus`, every GC takes its colour from `colours`. With `focus`,
    only that GC takes its colour and every other GC is drawn in grey; the
    close-ups use this to show one GC among its neighbours.
    """
    r0, r1, c0, c1 = window
    labels = run.labels(name)[r0:r1, c0:c1]
    out = np.zeros(labels.shape + (3,), np.float32)
    out[tissue[r0:r1, c0:c1]] = B.solid(theme.TISSUE)
    gc = labels > 0
    if focus is not None:
        out[gc] = B.solid(theme.FAINT)
        out[labels == focus] = B.solid(colours.get(focus, theme.INK))
    elif isinstance(colours, str):
        out[gc] = B.solid(colours)
    else:
        table = B.colour_table(colours, int(labels.max()) + 1)
        out[gc] = table[labels[gc]]
    return out


def gc_display_colour(run) -> str:
    """The colour of the GC marker channel in the IF composite (pure green when the marker is shown as green)."""
    channels = run.cfg.rgb_channels
    gc = run.cfg.if_images.gc_channel
    if gc in channels:
        rgb = [0, 0, 0]
        rgb[channels.index(gc)] = 255
        return theme.rgb_to_hex(rgb)
    return theme.AGREE


# --------------------------------------------------------------------------
# 1. sections
# --------------------------------------------------------------------------
def sections(run, out=None, *, dpi: float = 220, fmt: str = "png",
             gc_colour: str = "identity", if_outlines: bool = False) -> Path | None:
    """Draw every section of the stack in cutting order on the common grid.

    IF sections show their stain, cut to the tissue outline, with one set of
    display limits shared by all IF sections. ST sections show the section
    outline in dark grey with the GC regions on it. `gc_colour` sets the colour
    of the ST GC regions: "identity" (default: each GC in its own colour, the
    same colour it has in every other gcalign view), "stain" (the colour of
    the GC marker channel in the IF composite) or one hex colour for all GCs.
    `if_outlines` also draws each IF GC's outline in its identity colour.
    """
    run = B.as_run(run)
    out = Path(out) if out is not None else run.root / "06_figures" / f"sections.{fmt}"
    names = run.names
    with B.styled():
        tissue = {n: display_tissue(run, n) for n in names}
        window = stack_window(run)
        r0, r1, c0, c1 = window
        h, w = r1 - r0, c1 - c0
        ncols, nrows = B.grid_shape(len(names))
        margin, gap, band = _px(B.MARGIN_PT, dpi), _px(B.GAP_PT * 1.4, dpi), _px(B.BAND_PT, dpi)
        target = (FIGURE_WIDTH_IN * dpi - 2 * margin - (ncols - 1) * gap) / ncols
        k = B.integer_scale(max(w, h), target)
        pw, ph = w * k, h * k
        top, foot = margin * 0.6, _px(B.FOOT_PT, dpi)
        width = 2 * margin + ncols * pw + (ncols - 1) * gap
        height = top + nrows * (band + ph) + (nrows - 1) * gap + foot
        canvas = B.Canvas(width, height, dpi)

        limits = stain_limits(run) if run.if_names else {}
        if gc_colour == "identity":
            colours = run.identity
        elif gc_colour == "stain":
            colours = gc_display_colour(run)
        else:
            colours = gc_colour
        for index, name in enumerate(names):
            col, row = index % ncols, index // ncols
            x = margin + col * (pw + gap)
            y = top + row * (band + ph + gap) + band
            kind = run.plane(name).kind
            if kind == "IF":
                image = B.magnify(_stain_panel(run, name, window, limits, tissue[name]), k, smooth=True)
            else:
                image = B.magnify(_st_panel(run, name, window, colours, tissue[name]), k)
            canvas.raster(image, x, y)
            if kind == "IF" and if_outlines:
                labels = run.labels(name)[r0:r1, c0:c1]
                for gid in np.unique(labels[labels != 0]):
                    colour = run.identity.get(int(gid), theme.MUTED) if gid > 0 else theme.MUTED
                    for line in B.outlines(labels == gid):
                        canvas.polyline(line * k + (x, y), colour=colour, width=0.6)
            canvas.text(x, y - _px(B.LABEL_GAP_PT, dpi), name, colour=B.kind_colour(kind))

        B.scale_bar(canvas, width - margin, height - _px(B.FOOT_BAR_PT, dpi),
                    px_per_um=k / run.grid.um, max_px=0.22 * pw)
        return canvas.save(out, fmt)


# --------------------------------------------------------------------------
# 2. overlay
# --------------------------------------------------------------------------
def agreement_rgb(st_fraction: np.ndarray, if_fraction: np.ndarray,
                  base: np.ndarray | None = None) -> np.ndarray:
    """Colour each pixel by how many ST and IF sections call it GC; full agreement is white.

    The colour is interpolated bilinearly between four corners: nothing (the
    base colour), ST only (pink), IF only (blue) and both (white). A pixel
    called GC on every section of both kinds is exactly white. A pixel called
    GC by only some sections takes the colour of the modality that calls it,
    brighter the larger the fraction of that modality's sections.
    """
    s = np.clip(st_fraction, 0, 1)[..., None]
    i = np.clip(if_fraction, 0, 1)[..., None]
    zero = np.zeros(st_fraction.shape + (3,), np.float32) if base is None else base
    return ((1 - s) * (1 - i) * zero + s * (1 - i) * B.solid(theme.ST)
            + (1 - s) * i * B.solid(theme.IF) + s * i * B.solid(theme.AGREE)).astype(np.float32)


def densest_square(weight: np.ndarray, side: int) -> tuple[int, int, int, int]:
    """The square window of the given side, fully inside the image, with the largest summed weight."""
    side = int(min(side, *weight.shape))
    summed = ndi.uniform_filter(weight.astype(np.float64), size=side, mode="constant")
    half = side // 2
    valid = np.full(weight.shape, -np.inf)
    valid[half:weight.shape[0] - (side - half) + 1, half:weight.shape[1] - (side - half) + 1] = 0.0
    r, c = np.unravel_index(int(np.argmax(summed + valid)), weight.shape)
    r0, c0 = int(r - half), int(c - half)
    return r0, r0 + side, c0, c0 + side


def _corner_brackets(canvas, x, y, w, h, *, arm: float, colour=theme.INK_2, width=0.9):
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        canvas.polyline(np.array([[cx + dx * arm, cy], [cx, cy], [cx, cy + dy * arm]]),
                        colour=colour, width=width, closed=False)


def overlay(run, out=None, *, dpi: float = 220, fmt: str = "png", inset: bool = True,
            inset_fraction: float = 0.4, tissue: bool = False) -> Path | None:
    """Overlay the GC masks of all sections in one panel: ST pink, IF blue, both white.

    The colour of each pixel is set by the fraction of ST sections and the
    fraction of IF sections that call the pixel GC (see `agreement_rgb`). With
    `inset`, a second panel enlarges the square that holds the most GC area
    summed over all sections; its side is `inset_fraction` of the longer side
    of the first panel, and corner brackets mark it in the first panel.
    `tissue` draws the ST section outline underneath the masks.
    """
    run = B.as_run(run)
    out = Path(out) if out is not None else run.root / "06_figures" / f"overlay.{fmt}"
    if not run.st_names or not run.if_names:
        _note("skipped the overlay figure: the run needs at least one IF section and one ST section")
        return None
    with B.styled():
        st_hits = sum((run.labels(n) > 0).astype(np.float32) for n in run.st_names)
        if_hits = sum((run.labels(n) != 0).astype(np.float32) for n in run.if_names)
        if not (st_hits.any() or if_hits.any()):
            _note("skipped the overlay figure: no section has a GC")
            return None
        # without the silhouette there is nothing to show outside the GCs, so
        # the panel is cut to their extent and they are drawn larger
        window = stack_window(run) if tissue else B.bbox((st_hits + if_hits) > 0, pad=12)
        r0, r1, c0, c1 = window
        base = None
        if tissue:
            body = np.logical_or.reduce([display_tissue(run, n) for n in run.st_names])
            base = np.zeros(body.shape + (3,), np.float32)
            base[body] = B.solid(theme.TISSUE)
        full = agreement_rgb(st_hits / len(run.st_names), if_hits / len(run.if_names), base)
        main = full[r0:r1, c0:c1]
        h, w = main.shape[:2]

        margin, gap = _px(B.MARGIN_PT, dpi), _px(B.GAP_PT * 2.4, dpi)
        top, foot = margin, _px(B.FOOT_PT, dpi)
        zoom = None
        if inset:
            side = max(int(round(inset_fraction * max(h, w))), 8)
            weight = (st_hits + if_hits)[r0:r1, c0:c1]
            zr0, zr1, zc0, zc1 = densest_square(weight, side)
            zoom = (zr0, zr1, zc0, zc1)
        target_main = (FIGURE_WIDTH_IN * dpi - 2 * margin - (gap if inset else 0)) / (2 if inset else 1.4)
        k = B.integer_scale(max(h, w), target_main)
        pw, ph = w * k, h * k
        if zoom is not None:
            zs = zoom[1] - zoom[0]
            kz = B.integer_scale(zs, ph)
            zw = zs * kz
        else:
            kz, zw = 0, 0
        width = 2 * margin + pw + ((gap + zw) if zoom is not None else 0)
        height = top + max(ph, zw) + foot
        canvas = B.Canvas(width, height, dpi)

        y_main = top + (max(ph, zw) - ph) / 2
        canvas.raster(B.magnify(main, k), margin, y_main)
        bar_y = height - _px(B.FOOT_BAR_PT, dpi)
        B.scale_bar(canvas, margin + pw, bar_y, px_per_um=k / run.grid.um, max_px=0.22 * pw)
        if zoom is not None:
            zr0, zr1, zc0, zc1 = zoom
            _corner_brackets(canvas, margin + zc0 * k, y_main + zr0 * k, (zc1 - zc0) * k,
                             (zr1 - zr0) * k, arm=_px(7, dpi))
            x_zoom = margin + pw + gap
            y_zoom = top + (max(ph, zw) - zw) / 2
            canvas.raster(B.magnify(main[zr0:zr1, zc0:zc1], kz), x_zoom, y_zoom)
            B.scale_bar(canvas, x_zoom + zw, bar_y, px_per_um=kz / run.grid.um, max_px=0.22 * zw)
        return canvas.save(out, fmt)


# --------------------------------------------------------------------------
# 7. one GC on every plane
# --------------------------------------------------------------------------
def gc_window(run, gid: int) -> tuple[int, int, int, int]:
    footprint = run.footprint(gid)
    if not footprint.any():
        raise KeyError(f"GC{gid} is on no plane of this run")
    return run.square_window(footprint)


def gc_plane_rows(run, gid: int) -> list[dict]:
    """Per-section numbers of one GC, one dict per section: the rows of gc_closeups.csv.

    Each row has the GC's area, equivalent radius and centroid on the section,
    and what fills the GC's footprint there (the footprint is the union of the
    GC's masks over all sections): this GC, no GC, or another GC.
    """
    footprint = run.footprint(gid)
    metrics = None
    if run.metrics is not None:
        table = run.metrics.set_index("merged_gc_id")
        metrics = table.loc[gid] if gid in table.index else None
    present = [n for n in run.names if (run.labels(n) == gid).any()]
    rows = []
    for plane in run.planes:
        labels = run.labels(plane.name)
        mask = labels == gid
        area_px = int(mask.sum())
        row = {"gc_id": gid, "plane": plane.name, "kind": plane.kind, "z_um": plane.z_um,
               "present": int(area_px > 0), "area_px": area_px,
               "area_um2": area_px * run.grid.um ** 2,
               "equiv_radius_um": round(float(np.sqrt(area_px * run.grid.um ** 2 / np.pi)), 1)}
        if area_px:
            cy, cx = ndi.center_of_mass(mask)
            row["centroid_x_um"] = round(run.grid.x0 + cx * run.grid.um, 1)
            row["centroid_y_um"] = round(run.grid.y0 + cy * run.grid.um, 1)
        else:
            row["centroid_x_um"] = row["centroid_y_um"] = None
        # what occupies this GC's footprint here: nothing, or a neighbour
        under = labels[footprint]
        row["footprint_px"] = int(footprint.sum())
        row["footprint_this_gc_pct"] = round(100.0 * float((under == gid).mean()), 1)
        row["footprint_no_gc_pct"] = round(100.0 * float((under == 0).mean()), 1)
        others = under[(under != gid) & (under != 0)]
        if others.size:
            values, counts = np.unique(others, return_counts=True)
            top = int(values[np.argmax(counts)])
            row["footprint_other_gc_pct"] = round(100.0 * others.size / under.size, 1)
            row["footprint_top_other_gc"] = top if top > 0 else f"IF-only{top}"
            row["footprint_top_other_gc_pct"] = round(100.0 * float(counts.max()) / under.size, 1)
        else:
            row["footprint_other_gc_pct"] = 0.0
            row["footprint_top_other_gc"] = None
            row["footprint_top_other_gc_pct"] = 0.0
        row["n_planes_present"] = len(present)
        for key, column in (("volume_nL", "volume_um3"), ("max_equiv_radius_um", "max_equiv_radius_um"),
                            ("radius_cv", "radius_cv"), ("centroid_drift_um", "centroid_drift_um")):
            value = None if metrics is None else float(metrics[column])
            if value is not None and key == "volume_nL":
                value = value / 1e6
            row[key] = None if value is None else round(value, 4)
        rows.append(row)
    return rows


def gc_closeup(run, gid: int, out=None, *, dpi: float = 220, fmt: str = "png") -> Path:
    """Draw one GC on every section, in cutting order, in one shared square window.

    IF sections show the stain; the display limits come from this window and
    are shared by all IF sections. ST sections show the section with this GC
    in its identity colour and the other GCs in grey. The white outline is this
    GC on the section; the dashed outline is this GC on the previous section.
    """
    run = B.as_run(run)
    gid = int(gid)
    out = Path(out) if out is not None else run.root / "06_figures" / "gc_closeups" / f"GC{gid}.{fmt}"
    with B.styled():
        window = gc_window(run, gid)
        r0, r1, c0, c1 = window
        side = r1 - r0
        names = run.names
        ncols, nrows = B.grid_shape(len(names))
        margin, gap, band = _px(B.MARGIN_PT, dpi), _px(B.GAP_PT, dpi), _px(B.BAND_PT, dpi)
        target = (FIGURE_WIDTH_IN * dpi - 2 * margin - (ncols - 1) * gap) / ncols
        k = B.integer_scale(side, target)
        pw = side * k
        top, foot = margin * 0.6, _px(B.FOOT_PT, dpi)
        width = 2 * margin + ncols * pw + (ncols - 1) * gap
        height = top + nrows * (band + pw) + (nrows - 1) * gap + foot
        canvas = B.Canvas(width, height, dpi)

        limits = stain_limits(run, window) if run.if_names else {}
        previous = None
        for index, name in enumerate(names):
            col, row = index % ncols, index // ncols
            x = margin + col * (pw + gap)
            y = top + row * (band + pw + gap) + band
            kind = run.plane(name).kind
            if kind == "IF":
                image = B.magnify(_stain_panel(run, name, window, limits), k, smooth=True)
            else:
                image = B.magnify(_st_panel(run, name, window, run.identity, display_tissue(run, name),
                                            focus=gid), k)
            canvas.raster(image, x, y)
            here = run.labels(name)[r0:r1, c0:c1] == gid
            if previous is not None and previous.any():
                for line in B.outlines(previous):
                    canvas.polyline(line * k + (x, y), colour=theme.INK, width=B.GHOST_PT,
                                    alpha=0.75, dashes=(3.2, 2.6))
            for line in B.outlines(here):
                canvas.polyline(line * k + (x, y), colour=theme.AGREE, width=B.OUTLINE_PT)
            previous = here
            canvas.text(x, y - _px(B.LABEL_GAP_PT, dpi), name, colour=B.kind_colour(kind))

        foot_y = height - _px(B.FOOT_BAR_PT, dpi)
        canvas.text(margin, foot_y + _px(1.0, dpi), f"GC{gid}", colour=theme.INK, weight="semibold")
        B.scale_bar(canvas, width - margin, foot_y, px_per_um=k / run.grid.um, max_px=0.3 * pw)
        return canvas.save(out, fmt)


def resolve_gcs(run, gcs) -> list[int]:
    if gcs is None:
        return []
    if isinstance(gcs, str):
        if gcs.lower() == "all":
            return list(run.gc_ids)
        gcs = [g for g in gcs.replace(",", " ").split() if g]
    chosen = []
    for item in gcs:
        gid = int(str(item).upper().replace("GC", ""))
        if gid not in run.gc_ids:
            _note(f"skipped GC{gid}: this run has no GC with that id (its GC ids run from {run.gc_ids[0]} to {run.gc_ids[-1]})")
            continue
        chosen.append(gid)
    return chosen


def gc_closeups(run, gcs="all", out_dir=None, *, dpi: float = 220, fmt: str = "png") -> dict:
    """Draw the close-ups of several GCs and write ``gc_closeups.csv`` with their per-section numbers."""
    run = B.as_run(run)
    out_dir = Path(out_dir) if out_dir is not None else run.root / "06_figures" / "gc_closeups"
    written: dict = {}
    rows: list[dict] = []
    for gid in resolve_gcs(run, gcs):
        written[f"gc{gid}"] = gc_closeup(run, gid, out_dir / f"GC{gid}.{fmt}", dpi=dpi, fmt=fmt)
        rows.extend(gc_plane_rows(run, gid))
    if rows:
        path = out_dir / "gc_closeups.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in rows:
                writer.writerow({key: ("" if item.get(key) is None else item.get(key)) for key in fields})
        written["gc_closeups_csv"] = path
    return written
