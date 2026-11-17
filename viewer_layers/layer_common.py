"""Shared code for the viewer-layer builders in this folder.

Each builder reads one finished gcalign run through `gcalign.config` and
`gcalign.runio.Run`: the sections and their kinds, the germinal-centre (GC)
label rasters, the warped immunofluorescence (IF) channels and the frozen
spatial-transcriptomics (ST) bins. No builder names a section, a chip or a
channel of a particular experiment; those come from the config and the command
line.

RCTD cell-type weights (one table per ST section, written by `extract_rctd.R`)
are found by the section's `source` in the config, which for an ST section is
its chip or slice id.
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
try:                                    # an installed gcalign wins; a plain checkout works too
    import gcalign  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

from gcalign import config as gconfig   # noqa: E402
from gcalign.runio import Run           # noqa: E402

DEFAULT_PATTERN = "*{source}*.csv*"


def parser(description: str) -> argparse.ArgumentParser:
    """The arguments every builder takes: the config, and where the run is."""
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path,
                    help="the experiment YAML file the run was made with")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="the run directory to read and write; default: output_dir in the config")
    return ap


def add_weights(ap: argparse.ArgumentParser) -> None:
    """The arguments of a builder that reads RCTD weights."""
    ap.add_argument("--weights", required=True, type=Path,
                    help="directory of RCTD weight tables, one CSV(.gz) per ST section (see extract_rctd.R)")
    ap.add_argument("--weights-pattern", default=DEFAULT_PATTERN,
                    help="file-name pattern of one ST section's table; {source} is replaced by the "
                         "section's source in the config (its chip id). Default: %(default)s")


def open_run(args) -> Run:
    """The finished run named by --config and --output-dir."""
    cfg = gconfig.load(args.config, output_dir=args.output_dir)
    return Run(cfg)


def require_channel(run: Run, channel: str, flag: str) -> str:
    """Stop with a clear message if `channel` is not an IF channel of this run."""
    if channel not in run.channel_names:
        raise SystemExit(f"{flag} {channel!r} is not an IF channel of this run; "
                         f"the config lists {run.channel_names}")
    return channel


def weights_file(weights_dir: Path, pattern: str, source: str) -> Path:
    """The one RCTD table whose file name matches `pattern` for this section's source."""
    want = pattern.replace("{source}", str(source))
    folder = Path(weights_dir)
    if not folder.is_dir():
        raise SystemExit(f"--weights {folder} is not a directory")
    hits = sorted(p for p in folder.iterdir() if p.is_file() and fnmatch.fnmatch(p.name, want))
    if not hits:
        raise SystemExit(f"no RCTD table for source {source!r}: no file in {folder} matches {want!r}")
    if len(hits) > 1:
        raise SystemExit(f"{len(hits)} RCTD tables match {want!r} in {folder}: {[h.name for h in hits]}. "
                         "Pass a narrower --weights-pattern, for example '*{source}_rctd8.csv.gz'.")
    return hits[0]


def resolve_tables(run: Run, weights_dir: Path, pattern: str) -> dict[str, Path]:
    """The RCTD table of every ST section, found before any work starts, so a
    missing or ambiguous table stops the builder at once."""
    return {p.name: weights_file(weights_dir, pattern, run.cfg.section(p.name).source)
            for p in run.planes if p.kind == "ST"}


def st_bins(run: Run, name: str) -> pd.DataFrame:
    """The scaffold bins of one ST section: chip-grid x/y, frame position (um), GC id."""
    scaffold = run.scaffold
    mask = scaffold["slice_short"] == name
    if not mask.any():
        raise SystemExit(f"the scaffold has no bins for ST section {name!r}")
    dnb = float(run.cfg.st_scaffold.dnb_um)
    return pd.DataFrame({"x": np.rint(scaffold["chip_um"][mask, 0] / dnb).astype(np.int32),
                         "y": np.rint(scaffold["chip_um"][mask, 1] / dnb).astype(np.int32),
                         "wx": scaffold["world_um"][mask, 0], "wy": scaffold["world_um"][mask, 1],
                         "gc": scaffold["gc_id"][mask]})


def rctd_table(path: Path) -> pd.DataFrame:
    """One ST section's RCTD weights."""
    table = pd.read_csv(path)
    missing = {"x", "y", "first_type"} - set(table.columns)
    if missing:
        raise SystemExit(f"{path.name} has no column(s) {sorted(missing)}")
    return table


def on_bins(run: Run, name: str, table: pd.DataFrame) -> pd.DataFrame:
    """RCTD rows joined to the section's scaffold bins on the chip grid (inner join)."""
    return table.merge(st_bins(run, name), on=["x", "y"], how="inner")


def check_columns(table: pd.DataFrame, columns, path: Path) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise SystemExit(f"{path.name} has no column(s) {missing}; it has {list(table.columns)}")
