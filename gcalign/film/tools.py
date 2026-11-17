"""Everything done to a film after it is rendered: encode, check, cut loops.

``find_ffmpeg``    the encoder: ``GCALIGN_FFMPEG``, then ``ffmpeg`` on PATH, then
                   the binary imageio-ffmpeg ships
``probe``          what the container says - duration, frame count, rate
``contact_sheet``  frames pulled back out of the MP4 and tiled, to check the
                   cut as a whole
``make_loop``      a small looping GIF or animated WebP for a README, cut from a
                   rendered MP4 and fitted under a size budget
``compact``        a cheaper-rate copy of a film for slides
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from .. import theme


def find_ffmpeg() -> str:
    """The encoder: GCALIGN_FFMPEG, then PATH, then imageio-ffmpeg."""
    exe = os.environ.get("GCALIGN_FFMPEG", "").strip()
    if exe:
        return exe
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError as exc:                                     # pragma: no cover
        raise RuntimeError("no ffmpeg found: set GCALIGN_FFMPEG, put ffmpeg on PATH, "
                           "or install imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path, exe: str | None = None) -> dict:
    """Duration, frame count, rate and size as the container reports them.

    ffprobe is not shipped with imageio-ffmpeg, so this reads ffmpeg's banner
    and counts frames by decoding (``-c copy`` would count packets only)."""
    exe = exe or find_ffmpeg()
    path = str(path)
    banner = subprocess.run([exe, "-hide_banner", "-i", path],
                            capture_output=True, text=True).stderr
    out = {"path": path, "bytes": Path(path).stat().st_size}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", banner)
    if m:
        out["duration_s"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Stream #\S+: Video: (\w+).*?, (\d+)x(\d+)", banner)
    if m:
        out["codec"], out["width"], out["height"] = m.group(1), int(m.group(2)), int(m.group(3))
    m = re.search(r"([\d.]+) fps", banner)
    if m:
        out["fps"] = float(m.group(1))
    counted = subprocess.run([exe, "-hide_banner", "-nostats", "-i", path, "-map", "0:v:0",
                              "-f", "null", "-", "-progress", "pipe:1"],
                             capture_output=True, text=True).stdout
    frames = re.findall(r"frame=(\d+)", counted)
    if frames:
        out["frames"] = int(frames[-1])
    out["banner"] = [ln.strip() for ln in banner.splitlines() if "Duration" in ln or "Stream #" in ln]
    return out


def grab(path, t: float, size=None, exe: str | None = None) -> Image.Image | None:
    """One decoded frame at ``t`` seconds."""
    exe = exe or find_ffmpeg()
    if size is None:
        info = probe(path, exe)
        size = (info["width"], info["height"])
    w, h = size
    raw = subprocess.run([exe, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    if len(raw) < w * h * 3:
        return None
    return Image.frombytes("RGB", (w, h), raw[:w * h * 3])


def contact_sheet(path, out, cols: int = 5, rows: int = 5, tile: int = 384,
                  start: float = 0.0, end: float | None = None, exe: str | None = None) -> Path:
    """Tile frames evenly spaced through the film (or through [start, end])."""
    exe = exe or find_ffmpeg()
    info = probe(path, exe)
    w, h = info["width"], info["height"]
    end = info.get("duration_s", 0.0) if end is None else end
    n = cols * rows
    th = int(round(tile * h / w))
    gap = 4
    sheet = Image.new("RGB", (cols * tile + (cols - 1) * gap, rows * th + (rows - 1) * gap),
                      theme.hex_to_rgb(theme.LINE))
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n
        frame = grab(path, t, (w, h), exe)
        if frame is None:
            continue
        sheet.paste(frame.resize((tile, th), Image.LANCZOS),
                    ((i % cols) * (tile + gap), (i // cols) * (th + gap)))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")


def make_loop(path, out, width: int = 800, fps: float = 15.0, start: float | None = None,
              duration: float | None = None, max_mb: float = 5.0, exe: str | None = None,
              log=print) -> dict:
    """A small looping animation for a README, from a rendered MP4.

    ``out`` ending in ``.gif`` gives a palette-optimised GIF (one palette built
    from the whole clip, ordered dither, only changed rectangles stored);
    ``.webp`` gives an animated WebP, several times smaller at the same look.
    If the result is over ``max_mb`` the clip is re-encoded with a lower rate
    or quality, then narrower, until it fits."""
    exe = exe or find_ffmpeg()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cut = []
    if start is not None:
        cut += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cut += ["-t", f"{duration:.3f}"]
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    attempts = []
    if out.suffix.lower() == ".gif":
        ladder = [(width, fps, 256), (width, fps * 0.8, 192), (width, fps * 0.8, 128),
                  (int(width * 0.85), fps * 0.8, 128), (int(width * 0.75), fps * 0.67, 128),
                  (int(width * 0.65), fps * 0.67, 96)]
        for w, f, colours in ladder:
            vf = (f"fps={f:.3f},scale={w}:-2:flags=lanczos,split[a][b];"
                  f"[a]palettegen=max_colors={colours}:stats_mode=full:reserve_transparent=0[p];"
                  f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
            _run([exe, "-y", "-v", "error", *cut, "-i", str(path), "-filter_complex", vf,
                  "-loop", "0", str(tmp)])
            size = tmp.stat().st_size / 1e6
            attempts.append({"width": w, "fps": round(f, 2), "colours": colours, "mb": round(size, 2)})
            log(f"  gif {w}px {f:.1f} fps {colours} colours: {size:.2f} MB", flush=True)
            if size <= max_mb:
                break
    elif out.suffix.lower() == ".webp":
        ladder = [(width, fps, 72), (width, fps, 62), (width, fps, 52), (width, fps * 0.8, 50),
                  (int(width * 0.85), fps * 0.8, 50), (int(width * 0.75), fps * 0.67, 45)]
        for w, f, q in ladder:
            vf = f"fps={f:.3f},scale={w}:-2:flags=lanczos"
            _run([exe, "-y", "-v", "error", *cut, "-i", str(path), "-vf", vf,
                  "-c:v", "libwebp_anim", "-lossless", "0", "-quality", str(q),
                  "-compression_level", "4", "-loop", "0", "-an", str(tmp)])
            size = tmp.stat().st_size / 1e6
            attempts.append({"width": w, "fps": round(f, 2), "quality": q, "mb": round(size, 2)})
            log(f"  webp {w}px {f:.1f} fps q{q}: {size:.2f} MB", flush=True)
            if size <= max_mb:
                break
    else:
        raise ValueError("make_loop writes .gif or .webp")
    tmp.replace(out)
    final = attempts[-1]
    return {"path": str(out), "fits": final["mb"] <= max_mb, **final, "attempts": attempts}


def closeup_window(film, section: str | None = None, lead: float = 0.9, tail: float = 0.5):
    """(section, start, length) of one IF section's top-view matching close-up
    in a rendered film, read from the validation JSON next to it: the end of
    the dive, the stored ICP iterations, and the warp taking over."""
    import json
    film = Path(film)
    record = json.loads(film.with_name(film.stem + "_validation.json").read_text(encoding="utf-8"))
    section = section or record.get("short_edit_featured_section")
    for shot in record["shots"]:
        if shot.get("kind") == "insert" and shot.get("section") == section:
            start = max(shot["start_s"], shot["icp_start_s"] - lead)
            end = min(shot["end_s"], shot["tps_end_s"] + tail)
            return section, start, end - start
    raise ValueError(f"{film.name}: no insertion of {section!r} in its validation JSON")


def readme_loops(out_dir, short=(), names=None, closeup=None, section=None,
                 formats=("webp", "gif"), width: int = 800, fps: float = 15.0,
                 max_mb: float = 5.0, log=print) -> list[dict]:
    """The README loops: each short edit as a whole, and the top-view matching
    close-up (``gc_matching_topview``) cut from a full film."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = [Path(p) for p in short]
    names = list(names) if names else [p.stem for p in short]
    if len(names) != len(short):
        raise ValueError("one name per short film")
    jobs = []
    for film, name in zip(short, names):
        for fmt in formats:
            log(f"{name}.{fmt}  <- {film.name}", flush=True)
            jobs.append(dict(path=film, out=out_dir / f"{name}.{fmt}"))
    if closeup is not None:
        sec, start, length = closeup_window(closeup, section)
        for fmt in formats:
            log(f"gc_matching_topview.{fmt}  <- {Path(closeup).name} {sec} "
                f"{start:.2f}-{start + length:.2f} s", flush=True)
            jobs.append(dict(path=closeup, out=out_dir / f"gc_matching_topview.{fmt}",
                             start=start, duration=length))
    # every loop is its own ffmpeg process, so they encode side by side
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = [pool.submit(make_loop, width=width, fps=fps, max_mb=max_mb, log=log, **job)
                   for job in jobs]
        return [f.result() for f in futures]


def compact(path, out, crf: int = 26, exe: str | None = None) -> Path:
    """Write the same frames at a lower bit rate, for slides (about a quarter of the size at CRF 26)."""
    exe = exe or find_ffmpeg()
    out = Path(out)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    _run([exe, "-y", "-v", "error", "-i", str(path), "-c:v", "libx264", "-preset", "slow",
          "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)])
    tmp.replace(out)
    return out
