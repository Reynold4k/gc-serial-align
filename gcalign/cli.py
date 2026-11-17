"""The `gcalign` command: every subcommand, its options, and what it calls.

Run `gcalign <command> --help` for the options of one command; docs/commands.md
describes each command and its output files.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from . import config as config_module

# name, what it does, module, runs one process per IF section
STAGES = [
    ("scaffold", "read the registered ST coordinates and GC ids (never modified)", "scaffold", False),
    ("if", "read each IF image; find the tissue outline and the GCs", "if_sections", True),
    ("align", "find mirror state, rotation, scale and shift of each IF section", "global_align", True),
    ("match", "match GCs between IF and ST, fit the elastic warp, give each GC one id", "match_warp", True),
    ("columns", "stack the outlines of each GC into a closed 3D surface", "columns", False),
    ("viewer", "write the interactive HTML viewer", "viewer", False),
    ("validate", "write the quality report and the checksums", "validate", False),
]
NAMES = [name for name, _, _, _ in STAGES]

# what each stage reads from the stages before it, so `--from` can say what is missing
NEEDS = {
    "align": ["00_scaffold/st_scaffold.npz", "01_if_gc/if_gc_manifest.json"],
    "match": ["00_scaffold/st_scaffold.npz", "02_global_align/global_transforms.json"],
    "columns": ["03_gc_match/plane_gc_labels.npz"],
    "viewer": ["04_columns/gc_columns.npz", "03_gc_match/match_report.json"],
    "validate": ["04_columns/gc_column_metrics.csv", "03_gc_match/match_report.json",
                 "02_global_align/global_transforms.json", "01_if_gc/if_gc_manifest.json"],
}

AFTER_A_RUN = [
    ("figures", "draw the result as PNG figures (black background, no captions)"),
    ("report", "write one HTML page with the quality numbers and the figures"),
    ("animate", "render a film (MP4) of the registration"),
    ("transfer", "read the IF intensity of each channel at every ST bin"),
    ("export", "write an IF section as a registered OME-TIFF image at a chosen pixel size"),
    ("map", "convert point coordinates between an IF image and the registered frame"),
]


def _stage(module: str):
    from importlib import import_module
    return import_module(f".{module}", __package__).run


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path,
                        help="the experiment description (YAML)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="use this run directory instead of the one written in the config")


def _ids(text: str | None):
    if text is None:
        return None
    if text.strip().lower() == "all":
        return "all"
    return [int(v) for v in text.replace(",", " ").split()]


def _names(text: str | None):
    return None if text is None else [v for v in text.replace(",", " ").split() if v]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcalign",
        description="Register serial immunofluorescence (IF) sections to the "
                    "spatial-transcriptomics (ST) sections of the same tissue block, using "
                    "germinal centres (GCs) as landmarks, and reconstruct each GC in 3D.",
        epilog="Documentation: docs/commands.md (commands), docs/outputs.md (output files), "
               "docs/method.md (method).")
    parser.add_argument("--version", action="version", version=f"gcalign {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    run = sub.add_parser("run", help="run the registration pipeline (seven stages)")
    _add_config(run)
    run.add_argument("--from", dest="start", choices=NAMES, default=NAMES[0],
                     help="resume at this stage (earlier outputs must already exist)")
    run.add_argument("--to", dest="stop", choices=NAMES, default=NAMES[-1],
                     help="last stage to run")
    run.add_argument("--only", choices=NAMES, help="run exactly one stage")
    run.add_argument("--jobs", type=int, default=1,
                     help="number of processes for the stages if, align and match, which handle "
                          "each IF section separately (0 = all CPUs); the result is identical "
                          "for every value")

    show = sub.add_parser("show", help="print the section layout derived from the config")
    _add_config(show)

    sub.add_parser("stages", help="list the pipeline stages and the commands for a finished run")

    check = sub.add_parser("check", help="test the input files and report problems before a run")
    _add_config(check)

    figures = sub.add_parser("figures", help="draw the result of a finished run as PNG figures")
    _add_config(figures)
    figures.add_argument("--out", type=Path, default=None, help="where to write (default 06_figures/)")
    figures.add_argument("--only", default=None, help="figure names, comma separated (default: all)")
    figures.add_argument("--gc", default=None,
                         help="also draw close-ups of these GC ids (comma separated, or `all`)")
    figures.add_argument("--dpi", type=int, default=220, help="resolution of the PNG files")

    report = sub.add_parser("report", help="write one HTML page with quality numbers and figures")
    _add_config(report)
    report.add_argument("--out", type=Path, default=None,
                        help="output folder (default 06_figures/)")

    # `gcalign animate` is the film renderer's own command (gcalign/film/cli.py)
    from .film.cli import add_animate_parser
    add_animate_parser(sub).description = (
        "Render a film (MP4) that replays the registration from the files of a finished run. "
        "docs/commands.md and films/README.md describe the film.")

    transfer = sub.add_parser("transfer", help="read the IF intensity of each channel at every ST bin")
    _add_config(transfer)
    transfer.add_argument("--out", type=Path, default=None,
                          help="output file (default 06_transfer/st_bins_if.<format>)")
    transfer.add_argument("--format", dest="fmt", choices=["npz", "csv"], default="npz",
                          help="npz: NumPy archive, one array per column; csv: one table")

    export = sub.add_parser("export", help="write an IF section as a registered OME-TIFF image "
                                             "at a chosen pixel size")
    _add_config(export)
    export.add_argument("--section", required=True,
                        help="the IF section to export, or `all`")
    export.add_argument("--um-per-px", type=float, default=2.0,
                        help="pixel size of the output image in micrometres (default 2)")
    export.add_argument("--channels", default=None, help="channel names, comma separated (default: all)")
    export.add_argument("--no-warp", action="store_true",
                        help="apply the similarity placement only, without the elastic warp")
    export.add_argument("--out", type=Path, default=None,
                        help="output file (default 06_registered_images/<section>_registered_<P>um.ome.tif)")

    points = sub.add_parser("map", help="convert point coordinates between an IF image and the "
                                          "registered frame")
    _add_config(points)
    points.add_argument("--section", required=True, help="name of the IF section")
    points.add_argument("--points", required=True, type=Path,
                        help="CSV with columns x and y")
    points.add_argument("--from", dest="source", choices=["image", "frame"], default="image",
                        help="image: pixels of the original IF image; frame: micrometres of the "
                             "registered frame")
    points.add_argument("--out", type=Path, default=None,
                        help="output CSV (default <points>_<section>_mapped.csv)")
    return parser


def cmd_show(cfg) -> int:
    print(f"experiment         {cfg.name}")
    print(f"run directory      {cfg.output_dir}")
    print(f"section thickness  {cfg.section_thickness_um:g} um")
    print(f"stack depth        {cfg.z_span_um + cfg.section_thickness_um:.0f} um of tissue "
          f"({cfg.n_planes} sections; their depths z run from 0 to {cfg.z_span_um:.0f} um)")
    red, green, blue = (c or "none" for c in cfg.rgb_channels)
    print(f"IF display colours red = {red}, green = {green}, blue = {blue}")
    print("\nsections in cutting order "
          "(IF = immunofluorescence, ST = spatial transcriptomics):")
    width = max(len(name) for name in cfg.plane_names)
    print(f"  {'no.':>3}  kind {'name':<{width}}  depth z   registered to")
    for name, kind, index in cfg.planes:
        note = ""
        if kind == "IF":
            neighbours = cfg.neighbours[index]
            note = " + ".join(neighbours) if neighbours else "no ST section"
            if len(neighbours) < 2:
                note += "   (only one ST neighbour: no two-sided error estimate)"
        print(f"  {index:>3}  {kind:<4} {name:<{width}}  {cfg.z_um(index):>5.0f} um   {note}")
    return 0


def cmd_run(cfg, args) -> int:
    selected = [args.only] if args.only else NAMES[NAMES.index(args.start):NAMES.index(args.stop) + 1]
    if NAMES.index(args.start) > NAMES.index(args.stop) and not args.only:
        raise SystemExit(f"--from {args.start} comes after --to {args.stop}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in NEEDS.get(selected[0], []) if not (cfg.output_dir / f).exists()]
    if missing:
        earlier = NAMES[NAMES.index(selected[0]) - 1]
        raise SystemExit(
            f"stage `{selected[0]}` needs output that is not in {cfg.output_dir}:\n  "
            + "\n  ".join(missing)
            + f"\nRun the earlier stages first (for example `--from {NAMES[0]} --to {earlier}`), "
              "or point --output-dir at the run that has them.")
    for name, blurb, module, per_section in STAGES:
        if name not in selected:
            continue
        print(f"\n=== {name}: {blurb} ===", flush=True)
        started = time.time()
        if per_section:
            _stage(module)(cfg, jobs=args.jobs)
        else:
            _stage(module)(cfg)
        print(f"--- {name} finished in {time.time() - started:.0f} s", flush=True)
    viewer = cfg.output_dir / "05_viewer" / cfg.viewer.filename
    if viewer.exists():
        print(f"\nDone. Open the viewer in a browser: {viewer}")
        print("Quality numbers: " + str(cfg.output_dir / "validation_report.json"))
        print("Further commands for this run (same --config): gcalign figures, report, "
              "animate, transfer, export, map")
    else:
        print("\nDone.")
    return 0


def cmd_map(cfg, args) -> int:
    import pandas as pd

    from .transforms import Transforms

    transforms = Transforms.load(cfg)
    t = transforms[args.section]
    table = pd.read_csv(args.points)
    missing = [c for c in ("x", "y") if c not in table.columns]
    if missing:
        raise SystemExit(f"{args.points} needs columns x and y (missing: {missing})")
    xy = table[["x", "y"]].to_numpy(float)
    if args.source == "image":
        frame = t.image_px_to_frame(xy)
        table["x_frame_um"], table["y_frame_um"] = frame[:, 0], frame[:, 1]
        units = transforms.frame_um_to_st_units(frame)
        table["x_anndata"], table["y_anndata"] = units[:, 0], units[:, 1]
    else:
        pixels = t.frame_to_image_px(xy)
        table["x_image_px"], table["y_image_px"] = pixels[:, 0], pixels[:, 1]
    out = args.out or args.points.with_name(args.points.stem + f"_{args.section}_mapped.csv")
    table.to_csv(out, index=False)
    print(f"wrote {out}  ({len(table):,} points, {args.section}, "
          f"{'image px -> frame um' if args.source == 'image' else 'frame um -> image px'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "stages":
        print("Stages of `gcalign run`, in order ('*' = processes each IF section "
              "separately; use --jobs N to run them in parallel):")
        for name, blurb, _, per_section in STAGES:
            print(f"  {name:<9}{'*' if per_section else ' '} {blurb}")
        print("\nCommands for a finished run:")
        for name, blurb in AFTER_A_RUN:
            print(f"  {name:<9}  {blurb}")
        return 0

    if args.command == "animate":
        return args.handler(args)

    cfg = config_module.load(args.config, output_dir=args.output_dir)

    if args.command == "show":
        return cmd_show(cfg)
    if args.command == "check":
        from . import check
        return 1 if check.run(cfg) else 0
    if args.command == "run":
        return cmd_run(cfg, args)
    if args.command == "figures":
        from . import figures
        written = figures.run(cfg, out_dir=args.out, which=_names(args.only), gcs=_ids(args.gc),
                              dpi=args.dpi)
        print(f"{len(written)} figure(s) written")
        return 0
    if args.command == "report":
        from . import report
        print("wrote", report.run(cfg, out_dir=args.out))
        return 0
    if args.command == "transfer":
        from . import transfer
        transfer.run(cfg, out=args.out, fmt=args.fmt)
        return 0
    if args.command == "export":
        from . import export
        sections = cfg.if_names if args.section.lower() == "all" else [args.section]
        if args.out is not None and len(sections) > 1:
            raise SystemExit("--out names one file; leave it out when exporting every section")
        for name in sections:
            export.run(cfg, name, um_per_px=args.um_per_px, out=args.out,
                       channels=_names(args.channels), warp=not args.no_warp)
        return 0
    if args.command == "map":
        return cmd_map(cfg, args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
