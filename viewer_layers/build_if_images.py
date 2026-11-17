#!/usr/bin/env python3
"""Build the immunofluorescence images for the viewer's image panel (05_viewer/if_images.json).

For each immunofluorescence (IF) section, this writes a true-colour JPEG of the
section as warped onto the common grid, plus the outline and the id of every
registered germinal centre (GC) in image pixel coordinates. The viewer shows the
image with the outlines drawn on top.

The red, green and blue bands are the config's display channels
(`if_images.display_rgb`, or the fluorophores in the channel names when that is
unset; see `gcalign.config.Config.rgb_channels`). The contrast limits of each
channel are one pair of percentiles shared by all IF sections, so brightness can
be compared from one section to the next.

Example:
    python viewer_layers/build_if_images.py --config my_experiment.yaml
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure

import layer_common as common


def main(argv=None) -> int:
    ap = common.parser(__doc__.split("\n")[0])
    ap.add_argument("--low", type=float, default=1.0,
                    help="lower contrast limit of each channel, as a percentile of its non-zero pixels. Default: %(default)s")
    ap.add_argument("--high", type=float, default=99.7,
                    help="upper contrast limit of each channel, as a percentile. Default: %(default)s")
    ap.add_argument("--stride", type=int, default=7,
                    help="use every n-th pixel when computing the limits. Default: %(default)s")
    ap.add_argument("--quality", type=int, default=88, help="JPEG quality, 1-95. Default: %(default)s")
    ap.add_argument("--min-gc-px", type=int, default=25,
                    help="smallest GC cross-section that gets an outline and an id, in grid pixels. Default: %(default)s")
    ap.add_argument("--out", type=Path, default=None, help="output file. Default: <run>/05_viewer/if_images.json")
    args = ap.parse_args(argv)

    run = common.open_run(args)
    cfg = run.cfg
    names = run.if_names
    if not names:
        raise SystemExit("the run has no IF section")
    rgb = list(cfg.rgb_channels)                                   # [red, green, blue]; None = black band
    out = args.out or run.root / "05_viewer" / "if_images.json"

    limits = {}
    for channel in rgb:
        if channel is None:
            continue
        pieces = []
        for n in names:
            values = run.channel(n, channel)                       # each read decompresses; read once
            pieces.append(values[values > 0].ravel()[::args.stride])
        pool = np.concatenate(pieces)
        if pool.size == 0:
            pool = np.zeros(1)
        limits[channel] = (float(np.percentile(pool, args.low)), float(np.percentile(pool, args.high)))
        print(f"  {channel}: contrast limits {limits[channel][0]:.0f} - {limits[channel][1]:.0f}, shared by "
              f"{len(names)} sections")

    planes, total = [], 0
    for name in names:
        bands = []
        for channel in rgb:
            if channel is None:
                bands.append(np.zeros(run.grid.shape, np.float32))
                continue
            values = run.channel(name, channel).astype(np.float32)
            lo, hi = limits[channel]
            bands.append(np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1))
        image = (np.dstack(bands) * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, "JPEG", quality=args.quality, optimize=True)
        jpeg = base64.b64encode(buffer.getvalue()).decode("ascii")
        total += len(jpeg)

        labels = run.labels(name)
        outlines, ids = [], []
        for gid in np.unique(labels):
            if gid <= 0:
                continue
            mask = labels == gid
            if mask.sum() < args.min_gc_px:
                continue
            for contour in measure.find_contours(mask.astype(np.float32), 0.5):
                if len(contour) < 8:
                    continue
                step = max(1, len(contour) // 120)             # 120 points is plenty at this scale
                xy = contour[::step][:, ::-1]                 # (row, col) -> (x, y)
                outlines.append(dict(gc=int(gid), xy=[[round(float(a), 1), round(float(b), 1)] for a, b in xy]))
            rows, cols = np.nonzero(mask)
            ids.append(dict(gc=int(gid), x=round(float(cols.mean()), 1), y=round(float(rows.mean()), 1)))
        planes.append(dict(plane=name, w=int(labels.shape[1]), h=int(labels.shape[0]),
                           x0=run.grid.x0, y0=run.grid.y0, um=run.grid.um, jpeg=jpeg,
                           outlines=outlines, labels=ids))
        print(f"  {name}: {labels.shape[1]} x {labels.shape[0]} px, JPEG {len(jpeg) / 1e3:.0f} kB as base64, "
              f"{len(outlines)} outlines, {len(ids)} GC ids")

    channels = " · ".join(f"{band}={channel}" for band, channel in zip("RGB", rgb) if channel)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(run=cfg.name, rgb=rgb, channels=channels,
                                   limits={k: list(v) for k, v in limits.items()}, planes=planes),
                              separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, images {total / 1e6:.2f} MB as base64)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
