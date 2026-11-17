#!/usr/bin/env python
"""Draw the dark-zone / light-zone split of one germinal centre on every section of a run.

The figure is the per-GC close-up of `gcalign figures` (one shared window, sections in
cutting order) with the two zones of the GC marked on every section:

  IF sections   The GC is split at the median of one IF channel inside the GC on that
                section (``--zone-channel``, for example CD4). Pixels at or above the
                median are the light zone (LZ); pixels below it are the dark zone (DZ).
                The split is drawn as hatching over the untouched stain, so a reader can
                judge whether the split follows the channel it was computed from:
                DZ hatched down-right in the DZ colour, LZ hatched up-right in the LZ
                colour. A GC smaller than ``--min-gc-px`` pixels on a section is not split.
  ST sections   Each spatial-transcriptomics bin inside the GC is marked DZ or LZ from an
                RCTD weight table (``--weights``): a bin is a zone when that zone's weight
                is at least ``--min-weight``; when both weights pass, the larger one wins.

The RCTD table of an ST section is the one file in ``--weights`` whose name matches
``--table-pattern`` with ``{source}`` replaced by the section's ``source`` in the config
(the chip id). The table needs columns x and y (chip coordinates in DNB units, as in the
run's scaffold) and one column per zone.

Example (mouse spleen, replicate 1):

    python figures_2d/plot_dzlz_split.py --config rep1.yaml --gc 19 \\
        --weights /path/to/rctd/weights --table-pattern '*{source}_rctd8.csv.gz' \\
        --zone-channel CD4_AF594 --output-dir figures/rep1/dzlz
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gcalign import theme
from gcalign.figures import _base as B
from gcalign.figures._data import display_tissue, stain_limits
from gcalign.figures.spatial import gc_window
from gcalign.runio import Run

DEFAULT_PATTERN = "*{source}*.csv*"
HATCH_PERIOD_PT, HATCH_WIDTH_PT = 3.2, 0.6      # hatch spacing and line width, in points


def _px(points: float, dpi: float) -> float:
    return points * dpi / 72.0


def find_table(weights: Path, source: str, pattern: str) -> Path:
    """The one RCTD table of an ST section, found by its chip id."""
    wanted = pattern.format(source=source)
    hits = sorted(p for p in weights.iterdir() if fnmatch.fnmatch(p.name, wanted))
    if len(hits) != 1:
        found = ", ".join(p.name for p in hits) or "none"
        raise FileNotFoundError(
            f"expected one RCTD table matching {wanted!r} in {weights}, found {len(hits)} ({found}); "
            "set --table-pattern so that exactly one file matches each section")
    return hits[0]


class ZoneContext:
    """Everything the zone figures of one run share: the run and the ST zone calls."""

    def __init__(self, run: Run, weights: Path | None, pattern: str = DEFAULT_PATTERN,
                 zones: tuple[str, str] = ("DZ", "LZ"), min_weight: float = 0.15,
                 zone_channel: str | None = None, min_gc_px: int = 60):
        self.run, self.zones, self.min_weight = run, zones, float(min_weight)
        self.min_gc_px = int(min_gc_px)
        channels = run.channel_names
        if zone_channel is None or zone_channel not in channels:
            raise ValueError(f"--zone-channel must be one of the IF channels {channels}")
        self.zone_channel = zone_channel
        self.st: dict[str, pd.DataFrame] = {}
        if weights is None:
            print("[figures_2d] no --weights folder given: ST sections are drawn without zones", flush=True)
            return
        scaffold = run.scaffold
        dnb = float(run.cfg.st_scaffold.dnb_um)
        for name in run.st_names:
            source = run.cfg.section(name).source
            try:
                table = find_table(Path(weights), source, pattern)
            except FileNotFoundError as error:
                print(f"[figures_2d] {name}: {error}", flush=True)
                continue
            calls = pd.read_csv(table, usecols=["x", "y", *zones])
            rows = scaffold["slice_short"] == name
            key = pd.DataFrame({"x": np.rint(scaffold["chip_um"][rows, 0] / dnb).astype(np.int64),
                                "y": np.rint(scaffold["chip_um"][rows, 1] / dnb).astype(np.int64),
                                "wx": scaffold["world_um"][rows, 0], "wy": scaffold["world_um"][rows, 1],
                                "gc": scaffold["gc_id"][rows]})
            joined = calls.merge(key, on=["x", "y"], how="inner")
            self.st[name] = joined[joined["gc"] > 0].reset_index(drop=True)
            print(f"[figures_2d] {name}: zone calls from {table.name} "
                  f"({len(self.st[name]):,} bins inside GCs)", flush=True)

    def st_zones(self, name: str, gid: int, window) -> np.ndarray:
        """Zone per grid pixel of one ST section in a window: 0 none, 1 DZ, 2 LZ."""
        r0, r1, c0, c1 = window
        out = np.zeros((r1 - r0, c1 - c0), np.uint8)
        table = self.st.get(name)
        if table is None:
            return out
        table = table[table["gc"] == gid]
        first, second = (table[z].to_numpy(float) for z in self.zones)
        zone = np.where((first >= self.min_weight) | (second >= self.min_weight),
                        np.where(second > first, 2, 1), 0)
        rows, cols = self.run.grid.to_rc(table[["wx", "wy"]].to_numpy(float))
        keep = (zone > 0) & (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
        out[rows[keep] - r0, cols[keep] - c0] = zone[keep]
        return out

    def if_zones(self, name: str, gid: int, window) -> np.ndarray | None:
        """Zone per grid pixel of one IF section: the median split of the zone channel."""
        mask = self.run.labels(name) == gid
        if mask.sum() < self.min_gc_px:
            return None
        values = self.run.channel(name, self.zone_channel)[mask].astype(np.float32)
        if values.max() - values.min() < 1e-6:
            return None
        full = np.zeros(mask.shape, np.uint8)
        full[mask] = np.where(values >= np.median(values), 2, 1)
        r0, r1, c0, c1 = window
        return full[r0:r1, c0:c1]


def _hatch(shape, period: float, width: float, rising: bool, grow: float = 0.0) -> np.ndarray:
    """Diagonal lines on the pixel grid; `grow` widens them on both sides (for a halo)."""
    rows, cols = np.indices(shape)
    t = (cols - rows) if rising else (cols + rows)
    step = max(int(round(period)), 4)
    return ((t + int(round(grow))) % step) < max(width, 1.0) + 2 * int(round(grow))


def draw_gc(ctx: ZoneContext, gid: int, out: Path, dpi: float = 220, fmt: str = "png") -> Path:
    """One GC on every section with its two zones marked; returns the written path."""
    run = ctx.run
    dz, lz = B.solid(theme.DZ), B.solid(theme.LZ)
    with B.styled():
        window = gc_window(run, gid)
        r0, r1, c0, c1 = window
        side = r1 - r0
        ncols, nrows = B.grid_shape(len(run.names))
        margin, gap, band = _px(B.MARGIN_PT, dpi), _px(B.GAP_PT, dpi), _px(B.BAND_PT, dpi)
        target = (B.FIGURE_WIDTH_IN * dpi - 2 * margin - (ncols - 1) * gap) / ncols
        k = B.integer_scale(side, target)
        pw = side * k
        top, foot = margin * 0.6, _px(B.FOOT_PT, dpi)
        width = 2 * margin + ncols * pw + (ncols - 1) * gap
        height = top + nrows * (band + pw) + (nrows - 1) * gap + foot
        canvas = B.Canvas(width, height, dpi)
        limits = stain_limits(run, window) if run.if_names else {}
        period, line = _px(HATCH_PERIOD_PT, dpi), _px(HATCH_WIDTH_PT, dpi)
        for index, name in enumerate(run.names):
            col, row = index % ncols, index // ncols
            x = margin + col * (pw + gap)
            y = top + row * (band + pw + gap) + band
            here = run.labels(name)[r0:r1, c0:c1] == gid
            if run.plane(name).kind == "IF":
                image = B.magnify(run.rgb(name, window, limits), k, smooth=True)
                zones = ctx.if_zones(name, gid, window)
                if zones is not None:
                    big = B.magnify(zones, k)
                    for value, colour, rising in ((1, dz, False), (2, lz, True)):
                        # a dark halo under each line keeps it legible on a stain of any
                        # colour, including a DZ line on a green GC marker
                        halo = (big == value) & _hatch(big.shape, period, line, rising, grow=1)
                        image[halo] *= 0.25
                        stroke = (big == value) & _hatch(big.shape, period, line, rising)
                        image[stroke] = colour
            else:
                labels = run.labels(name)[r0:r1, c0:c1]
                base = np.zeros(labels.shape + (3,), np.float32)
                base[display_tissue(run, name)[r0:r1, c0:c1]] = B.solid(theme.TISSUE)
                base[labels > 0] = B.solid(theme.FAINT)
                image = B.magnify(base, k)
                zones = B.magnify(ctx.st_zones(name, gid, window), k)
                inset = max(int(round(k * 0.14)), 1) if k >= 4 else 0
                if inset:                                 # each bin a square bead with a hairline gap
                    rows_, cols_ = np.indices(zones.shape)
                    inner = ((rows_ % k) >= inset) & ((rows_ % k) < k - inset) \
                        & ((cols_ % k) >= inset) & ((cols_ % k) < k - inset)
                    zones = np.where(inner, zones, 0)
                image[zones == 1] = dz
                image[zones == 2] = lz
            canvas.raster(image, x, y)
            for outline in B.outlines(here):
                canvas.polyline(outline * k + (x, y), colour=theme.AGREE, width=B.OUTLINE_PT)
            canvas.text(x, y - _px(B.LABEL_GAP_PT, dpi), name, colour=B.kind_colour(run.plane(name).kind))
        foot_y = height - _px(B.FOOT_BAR_PT, dpi)
        canvas.text(margin, foot_y + _px(1.0, dpi), f"GC{gid}", colour=theme.INK, weight="semibold")
        B.scale_bar(canvas, width - margin, foot_y, px_per_um=k / run.grid.um, max_px=0.3 * pw)
        return canvas.save(out, fmt)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path,
                        help="the experiment YAML of a finished gcalign run")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="the run folder, when it is not the output_dir written in the config")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="folder the figures are written to")
    parser.add_argument("--weights", type=Path, default=None,
                        help="folder of RCTD weight tables, one per ST section (without it, ST "
                             "sections are drawn without zones)")
    parser.add_argument("--table-pattern", default=DEFAULT_PATTERN,
                        help="file-name pattern of one ST section's table; {source} is replaced by "
                             "the section's source (chip id) from the config. Default: %(default)s")
    parser.add_argument("--zone-channel", required=True,
                        help="IF channel whose median splits each GC: at or above the median is the "
                             "light zone (for example CD4_AF594)")
    parser.add_argument("--zones", default="DZ,LZ",
                        help="the two zone columns of the RCTD tables, dark zone first. Default: %(default)s")
    parser.add_argument("--min-weight", type=float, default=0.15,
                        help="RCTD weight at or above which a bin belongs to a zone. Default: %(default)s")
    parser.add_argument("--min-gc-px", type=int, default=60,
                        help="smallest GC, in grid pixels, that is split on an IF section. Default: %(default)s")
    parser.add_argument("--dpi", type=float, default=220, help="output resolution. Default: %(default)s")


def context_from(args) -> ZoneContext:
    run = Run.open(args.config, output_dir=args.run_dir)
    zones = tuple(z.strip() for z in args.zones.split(","))
    if len(zones) != 2:
        raise SystemExit("--zones takes two column names, dark zone first, for example DZ,LZ")
    return ZoneContext(run, args.weights, args.table_pattern, zones, args.min_weight,
                       args.zone_channel, args.min_gc_px)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw the dark-zone / light-zone split of germinal centres on every section "
                    "of a finished gcalign run, one figure per GC.")
    add_common_arguments(parser)
    parser.add_argument("--gc", required=True,
                        help="GC ids to draw, comma-separated (for example 3,19)")
    args = parser.parse_args(argv)
    ctx = context_from(args)
    for item in args.gc.replace(",", " ").split():
        gid = int(item.upper().replace("GC", ""))
        if gid not in ctx.run.gc_ids:
            print(f"[figures_2d] GC{gid} is not in this run; skipped", flush=True)
            continue
        path = draw_gc(ctx, gid, args.output_dir / f"GC{gid}_zones.png", dpi=args.dpi)
        print(f"[figures_2d] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
