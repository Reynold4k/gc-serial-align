"""Static figures of a finished gcalign run, drawn in the package's black theme.

    from gcalign import config, figures
    cfg = config.load("my_experiment.yaml")
    figures.run(cfg)                          # every whole-run figure -> <output_dir>/06_figures
    figures.run(cfg, gcs=[3, 12])             # ... plus the per-GC figures of GCs 3 and 12
    figures.sections(cfg, "sections.png")     # any single figure, from Python

Abbreviations: IF = immunofluorescence, ST = spatial transcriptomics, GC = germinal centre.

Whole-run figures

  sections     every section in cutting order on the common grid: IF sections show their
               stain, ST sections show the section outline with each GC in its identity colour
  overlay      the GC masks of all sections in one panel: pink where only ST sections call a
               pixel GC, blue where only IF sections do, white where both do; a second panel
               enlarges the square with the most GC area
  alignment    per IF section: GC centre error before and after the elastic warp,
               leave-one-out error, two-sided disagreement, and GCs used as warp anchors
  orientation  every orientation candidate of every IF section: tissue overlap (Dice) against
               the number of GC centres within the close-match radius (runs made with
               gcalign 0.2 and later)
  presence     which GC is on which section: one row per GC, one column per section, brightness
               = cross-section area
  columns      the largest GC columns in 3D, all at one scale

Per-GC figures, drawn for the GC ids passed as `gcs`

  gc           one GC on every section in one shared window, plus gc_closeups.csv
  column       one GC's measured cross-sections beside the 3D surface drawn through them

Every figure follows the same rules: a pure black background; the Inter typeface; key labels
only (section names, GC ids, scale bars, and axis labels on the quantitative figures), with no
titles, captions or legends; and rasters pasted at a whole-number magnification of the grid, so
a 10 um bin is never resampled.
"""
from __future__ import annotations

from pathlib import Path

from ._base import as_run
from .charts import alignment, alignment_table, orientation, presence, presence_matrix
from .columns3d import column, columns, renderer_available
from .spatial import gc_closeup, gc_closeups, gc_plane_rows, overlay, resolve_gcs, sections

WHOLE_RUN = ("sections", "overlay", "alignment", "orientation", "presence", "columns")
PER_GC = ("gc", "column")
NAMES = WHOLE_RUN + PER_GC

_WHOLE = {"sections": sections, "overlay": overlay, "alignment": alignment,
          "orientation": orientation, "presence": presence, "columns": columns}

__all__ = ["run", "WHOLE_RUN", "PER_GC", "NAMES", "sections", "overlay", "alignment",
           "orientation", "presence", "columns", "column", "gc_closeup", "gc_closeups",
           "gc_plane_rows", "alignment_table", "presence_matrix", "renderer_available", "as_run"]


def _say(message: str) -> None:
    print(f"[gcalign.figures] {message}", flush=True)


def _selection(which, gcs) -> list[str]:
    if which is None:
        chosen = list(WHOLE_RUN) + (list(PER_GC) if gcs else [])
    else:
        if isinstance(which, str):
            which = [w for w in which.replace(",", " ").split() if w]
        chosen = [str(w).strip().lower() for w in which]
    unknown = [w for w in chosen if w not in NAMES]
    if unknown:
        raise ValueError(f"unknown figure name(s) {unknown}; the figure names are {list(NAMES)}")
    return chosen


def run(cfg, out_dir=None, which=None, gcs=None, dpi: float = 220, fmt: str = "png") -> dict[str, Path]:
    """Draw the figures of a finished run and return {figure name: written file}.

    `cfg` is a `gcalign.config.Config` (a `Run` or a config path also work).
    `out_dir` defaults to ``cfg.output_dir / "06_figures"``. `which` selects
    figures by name; the default is every whole-run figure, plus the per-GC
    figures when `gcs` is given. `gcs` is a list of GC ids, "all", or None.
    `fmt` is "png" (default), "pdf" or "svg". A figure whose input file is
    missing from the run is skipped with a printed note and left out of the result.
    """
    data = as_run(cfg)
    out = Path(out_dir) if out_dir is not None else Path(data.cfg.output_dir) / "06_figures"
    chosen = _selection(which, gcs)
    written: dict[str, Path] = {}
    for name in chosen:
        if name in _WHOLE:
            try:
                path = _WHOLE[name](data, out / f"{name}.{fmt}", dpi=dpi, fmt=fmt)
            except FileNotFoundError as error:
                _say(f"skipped the {name} figure: a run file is missing ({error})")
                path = None
            if path is not None:
                written[name] = path
                _say(f"wrote the {name} figure: {path}")
    ids = resolve_gcs(data, gcs)
    if any(name in PER_GC for name in chosen) and not ids:
        _say("skipped the per-GC figures: no GC ids were given (pass gcs=[...] or gcs='all')")
    if "gc" in chosen and ids:
        made = gc_closeups(data, ids, out / "gc_closeups", dpi=dpi, fmt=fmt)
        written.update(made)
        _say(f"wrote {len(ids)} GC close-up figure(s) and gc_closeups.csv to {out / 'gc_closeups'}")
    if "column" in chosen and ids:
        count = 0
        for gid in ids:
            path = column(data, gid, out / "columns" / f"GC{gid}.{fmt}", dpi=dpi, fmt=fmt)
            if path is not None:
                written[f"column{gid}"] = path
                count += 1
        if count:
            _say(f"wrote {count} single-GC column figure(s) to {out / 'columns'}")
    return written
