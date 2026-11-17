"""The film's command line: ``python -m gcalign.film {render,check,loop,loops,compact}``.

``render`` draws a finished run as a film; ``check`` compares a rendered film
with its validation JSON and writes a contact sheet; ``loop`` and ``loops`` cut
README animations from rendered films; ``compact`` writes a smaller copy.

``gcalign animate`` is the same command as ``render``. `add_animate_parser`
and `render_from_args` let the main gcalign command line offer it without
repeating the options.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _size(text: str):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size is WIDTHxHEIGHT, e.g. 1920x1080") from None
    return (w, h)


def _acts(text: str):
    return sorted({int(v) for v in text.split(",") if v.strip()})


def add_render_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The options of the render command (also used by `gcalign animate`)."""
    p.add_argument("--config", required=True, type=Path,
                   help="the experiment YAML file of a finished gcalign run")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="read the run from this folder instead of the output_dir in the config")
    p.add_argument("--out", type=Path, default=None,
                   help="the MP4 file to write (default: <output_dir>/07_film/<name>.mp4, "
                        "or <name>_short.mp4 for the short edit)")
    p.add_argument("--preset", choices=["full", "short"], default="full",
                   help="full: the complete film, about 86 s with act 1; short: an edit of "
                        "about 20 s that still shows every act")
    p.add_argument("--spateo-dir", type=Path, default=None,
                   help="folder of the Spateo run that aligned the ST sections, with its stored "
                        "iterations; act 1 needs it (default: the GCALIGN_SPATEO_DIR environment "
                        "variable; without a Spateo run the film starts at act 2)")
    p.add_argument("--spateo-pattern", default=None,
                   help="file name of each Spateo file in --spateo-dir; {source} is replaced by "
                        "the ST section's source in the config (default: "
                        "{source}_gc_bin20_aligned_all_stages.h5ad)")
    p.add_argument("--no-act1", action="store_true",
                   help="leave out act 1 even when a Spateo run is given")
    p.add_argument("--acts", type=_acts, default=None,
                   help="render only these acts, for example 3 or 2,3")
    p.add_argument("--preview", action="store_true",
                   help="write key still images (PNG) and storyboard.json instead of the film")
    p.add_argument("--only", default="",
                   help="with --preview: the still names or times in seconds, comma-separated")
    p.add_argument("--size", type=_size, default=(1920, 1080),
                   help="frame size in pixels as WIDTHxHEIGHT (default 1920x1080)")
    p.add_argument("--fps", type=float, default=60.0, help="frames per second (default 60)")
    p.add_argument("--quality", choices=["master", "draft"], default="master",
                   help="master: x264 preset slow at CRF 16, textures at three times the grid "
                        "resolution; draft: faster to render and softer")
    p.add_argument("--label", default=None,
                   help="a short label shown at the top right of every frame, for example rep1")
    p.add_argument("--z-exaggeration", type=float, default=None,
                   help="the factor by which depth is stretched (default: the factor that makes "
                        "the stack about 0.44 times as tall as the tissue is wide)")
    p.add_argument("--workers", type=int, default=None,
                   help="number of processes that render frames (default: the cores of the "
                        "allocation minus two)")
    return p


def render_from_args(args) -> int:
    from .story import run
    from .upstream import Upstream

    if args.no_act1:
        upstream = False
    elif args.spateo_dir is not None:
        upstream = Upstream(args.spateo_dir, args.spateo_pattern) if args.spateo_pattern \
            else Upstream(args.spateo_dir)
    else:
        upstream = None
    preview = False
    if args.preview:
        preview = True
        if args.only:
            preview = [float(v) if v.replace(".", "", 1).isdigit() else v.strip()
                       for v in args.only.split(",") if v.strip()]
    path = run(args.config, args.out, preset=args.preset, fps=args.fps, size=args.size,
               acts=args.acts, preview=preview, upstream=upstream, quality=args.quality,
               label=args.label, z_exaggeration=args.z_exaggeration, workers=args.workers,
               output_dir=args.output_dir)
    print(path)
    return 0


def add_animate_parser(subparsers, name: str = "animate"):
    """Mount the render command on another CLI: ``gcalign animate``."""
    p = subparsers.add_parser(name, help="render a film (MP4) of the registration from a finished run")
    add_render_arguments(p)
    p.set_defaults(handler=render_from_args)
    return p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gcalign.film",
                                     description="Render, check and loop the reconstruction film.")
    sub = parser.add_subparsers(dest="command", required=True)
    add_render_arguments(sub.add_parser("render", help="render the film (or preview stills)"))

    loop = sub.add_parser("loop", help="a small looping GIF / WebP from a rendered MP4")
    loop.add_argument("mp4", type=Path)
    loop.add_argument("out", type=Path, help=".gif or .webp")
    loop.add_argument("--width", type=int, default=800)
    loop.add_argument("--fps", type=float, default=15.0)
    loop.add_argument("--start", type=float, default=None)
    loop.add_argument("--duration", type=float, default=None)
    loop.add_argument("--max-mb", type=float, default=5.0)

    check = sub.add_parser("check", help="container duration/frames/fps and a contact sheet")
    check.add_argument("mp4", type=Path)
    check.add_argument("--sheet", type=Path, default=None,
                       help="PNG (default <film>_contact_sheet.png next to it)")
    check.add_argument("--cols", type=int, default=5)
    check.add_argument("--rows", type=int, default=5)
    check.add_argument("--tile", type=int, default=384)

    small = sub.add_parser("compact", help="the same frames at a cheaper rate, for slides")
    small.add_argument("mp4", type=Path)
    small.add_argument("out", type=Path)
    small.add_argument("--crf", type=int, default=26)

    loops = sub.add_parser("loops", help="the README loops: short edits and the top-view close-up")
    loops.add_argument("--short", nargs="*", type=Path, default=[],
                       help="short edits; each becomes one loop")
    loops.add_argument("--names", nargs="*", default=None, help="output names for --short")
    loops.add_argument("--closeup", type=Path, default=None,
                       help="a full film: its top-view matching close-up -> gc_matching_topview")
    loops.add_argument("--section", default=None, help="IF section for the close-up")
    loops.add_argument("--out", type=Path, required=True)
    loops.add_argument("--format", nargs="+", default=["webp", "gif"], choices=["webp", "gif"])
    loops.add_argument("--width", type=int, default=800)
    loops.add_argument("--fps", type=float, default=15.0)
    loops.add_argument("--max-mb", type=float, default=5.0)

    args = parser.parse_args(argv)
    if args.command == "render":
        return render_from_args(args)
    if args.command == "loops":
        from .tools import readme_loops
        results = readme_loops(args.out, args.short, args.names, args.closeup, args.section,
                               args.format, args.width, args.fps, args.max_mb)
        for r in results:
            flag = "" if r["fits"] else "   OVER BUDGET"
            print(f"  {r['path']}  {r['mb']:.2f} MB  {r['width']} px  {r['fps']} fps{flag}")
        return 0 if all(r["fits"] for r in results) else 1
    if args.command == "compact":
        from .tools import compact
        out = compact(args.mp4, args.out, crf=args.crf)
        print(f"{out}  {args.mp4.stat().st_size / 1e6:.1f} MB -> {out.stat().st_size / 1e6:.1f} MB")
        return 0
    if args.command == "loop":
        from .tools import make_loop
        result = make_loop(args.mp4, args.out, width=args.width, fps=args.fps, start=args.start,
                           duration=args.duration, max_mb=args.max_mb)
        print(json.dumps(result, indent=2))
        return 0 if result["fits"] else 1
    if args.command == "check":
        return check_film(args.mp4, args.sheet, args.cols, args.rows, args.tile)
    return 2


def check_film(mp4: Path, sheet: Path | None = None, cols: int = 5, rows: int = 5,
               tile: int = 384) -> int:
    """Compare the container with the validation JSON and tile a contact sheet."""
    from .tools import contact_sheet, probe

    info = probe(mp4)
    for line in info["banner"]:
        print(" ", line)
    print(f"  duration {info.get('duration_s')} s, {info.get('frames')} frames, "
          f"{info.get('fps')} fps, {info['bytes'] / 1e6:.1f} MB")
    status = 0
    record = mp4.with_name(mp4.stem + "_validation.json")
    if record.exists():
        rec = json.loads(record.read_text(encoding="utf-8"))
        expected = (rec.get("film") or {}).get("frames")
        if expected is not None and info.get("frames") != expected:
            print(f"  MISMATCH: the container holds {info.get('frames')} frames, "
                  f"{expected} were rendered")
            status = 1
        else:
            print(f"  frames match the validation record ({expected})")
        for name, icp in (rec.get("icp") or {}).items():
            print(f"  ICP {name}: {icp['iterations']} iterations, {icp['source']}")
    sheet = sheet or mp4.with_name(mp4.stem + "_contact_sheet.png")
    contact_sheet(mp4, sheet, cols=cols, rows=rows, tile=tile)
    print("  wrote", sheet)
    return status


if __name__ == "__main__":
    sys.exit(main())
