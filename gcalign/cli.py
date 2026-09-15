"""Command line entry point: `gcalign run --config my_experiment.yaml`."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import config as config_module
from . import columns, global_align, if_sections, match_warp, scaffold, validate, viewer

STAGES = [
    ("scaffold", "read the frozen Stereo-seq scaffold", scaffold.run),
    ("if", "segment germinal centres on the IF sections", if_sections.run),
    ("align", "place each IF section in the Stereo-seq frame", global_align.run),
    ("match", "match GCs, warp, assign one identity per GC", match_warp.run),
    ("columns", "stack each GC into a smooth closed surface", columns.run),
    ("viewer", "write the standalone HTML viewer", viewer.run),
    ("validate", "collect QC numbers and hash the outputs", validate.run),
]
NAMES = [name for name, _, _ in STAGES]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gcalign",
        description="Align serial IF sections into a spatial-transcriptomics frame "
                    "and reconstruct germinal centres in 3D.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--from", dest="start", choices=NAMES, default=NAMES[0],
                     help="resume at this stage (earlier outputs must already exist)")
    run.add_argument("--to", dest="stop", choices=NAMES, default=NAMES[-1])
    run.add_argument("--only", choices=NAMES, help="run exactly one stage")

    show = sub.add_parser("show", help="print the derived experiment layout and exit")
    show.add_argument("--config", required=True, type=Path)

    sub.add_parser("stages", help="list the pipeline stages")

    args = parser.parse_args(argv)

    if args.command == "stages":
        for name, blurb, _ in STAGES:
            print(f"  {name:<9} {blurb}")
        return 0

    cfg = config_module.load(args.config)

    if args.command == "show":
        print(f"experiment   {cfg.name}")
        print(f"output       {cfg.output_dir}")
        print(f"thickness    {cfg.section_thickness_um} um per section")
        print(f"z span       {cfg.z_um(max(s.index for s in cfg.sections)):.0f} um "
              f"over {len(cfg.sections)} sections")
        print("sections (cutting order):")
        for name, kind, index in cfg.planes:
            flag = ""
            if kind == "IF":
                neighbours = cfg.neighbours[index]
                flag = f"  <- anchored to {' + '.join(neighbours)}"
                if len(neighbours) < 2:
                    flag += "   (one-sided)"
            print(f"  {index:>3}  {kind:<3} {name:<6} z = {cfg.z_um(index):>5.0f} um{flag}")
        return 0

    selected = [args.only] if args.only else NAMES[NAMES.index(args.start):NAMES.index(args.stop) + 1]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for name, blurb, fn in STAGES:
        if name not in selected:
            continue
        print(f"\n=== {name}: {blurb} ===", flush=True)
        started = time.time()
        fn(cfg)
        print(f"--- {name} finished in {time.time() - started:.0f} s", flush=True)
    print(f"\nDone. Viewer: {cfg.output_dir / '05_viewer' / cfg.viewer.filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
