#!/usr/bin/env python3
"""One continuous film: four Stereo-seq sections become a frame, five IF
sections are matched into it, and the finished nine-section stack turns once.

Five acts, one camera, no cuts:

  1  the Stereo-seq skeleton   all 1,100 stored Spateo iterations, top view,
                               one iteration per frame at 60 fps - exactly
                               twice the pace of the previous cut, nothing
                               dropped
  2  the frame becomes 3-D     the same four sections tilt out of the top view
                               and separate along the cutting axis
  3  five insertions           each IF section flies in, the camera drops to a
                               top view, and the 12 saved Stage-A ICP
                               iterations run for real; then the TPS result
                               takes over and the section settles into its slot
  4  nine sections             the finished stack
  5  the viewer view           re-coloured to the HTML viewer's own palette,
                               blue IF alternating with pink Stereo-seq, one
                               full turn

Everything moving on screen is a transform the pipeline actually stored.  The
two things that are not measurements are labelled on screen: the z axis is
exaggerated, and the sideways fly-in is staging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import film_paths as P                                               # noqa: E402
import story_assets as SA                                            # noqa: E402
import story_core as S                                               # noqa: E402
from story_assets import Assets, coords_at                           # noqa: E402
from story_core import (Camera, backdrop, chip, clamp01, draw_slab,   # noqa: E402
                        ease_out, interp_camera, lerp, progress, scrim,
                        smooth, smoother, splat, text, tw, vignette)

OUT = P.FILM_OUT
FPS = 60
W, H = 1920, 1080

# ---------------------------------------------------------------- timeline --
A1_IN, A1_ITERS, A1_OUT = 1.20, 1100 / FPS, 1.50      # act 1: 21.03 s
A2 = 4.60                                              # tilt into 3-D
SEG = 8.20                                             # one IF insertion
A4 = 4.00                                              # finished stack
A5_FADE, A5_ORBIT, A5_OUT = 1.80, 11.00, 2.40          # viewer palette + turn

T1 = A1_IN + A1_ITERS + A1_OUT
T2 = T1 + A2
T3 = T2 + 5 * SEG
T4 = T3 + A4
T5 = T4 + A5_FADE + A5_ORBIT + A5_OUT

ACT_BOUNDS = [0.0, T1, T2, T3, T4, T5]
ACT_TITLES = [
    "Four Stereo-seq sections find one common frame",
    "The frame lifts into the cutting axis",
    "Five IF sections are matched into the frame",
    "Nine sections, one coordinate system",
    "Blue IF, pink Stereo-seq - one turn",
]
CAPTIONS = [
    "All 1,100 stored Spateo iterations in their saved order, one per frame. Reference section D1 never moves.",
    "z exaggerated x45; the true section pitch is 10 um. Slabs are drawn thinner than the pitch so all nine stay visible.",
    "The 12 Stage-A ICP iterations are gcalign's own, replayed and asserted identical to the saved transform. The sideways entry is staging.",
    "Same colour = the same matched GC identity across sections, not the same cells observed twice.",
    "Viewer palette: Stereo-seq GC #FF1493, IF GC #0F52BA. Section order IF1-D1-IF3-E3-IF5-G4-IF7-G5-IF9.",
]

STAGE_LABEL = {"iter_rigid": "RIGID", "iter_nonrigid": "NONRIGID",
               "iter_deformable": "DEFORMABLE"}
TISSUE_RGB = np.array([104, 124, 146], np.float32)
ZOOM_HALF_UM = 1150.0        # half-width of the act-3 close-up window


# ============================================================== scene helper =
class Film:
    def __init__(self, rep, verbose=True):
        self.rep = rep
        self.A = Assets(rep, verbose=verbose)
        A = self.A
        self.bg = backdrop(W, H)
        self.vig = vignette(W, H)
        self.scrim = scrim(W, H)
        self.slices, self.seq, self.groups = A.load_trajectory(tissue_stride=5, gc_stride=1)
        self.steps = [(g, k) for g in self.groups for k in self.seq[g]]
        self.n_iter = len(self.steps)
        self.stage_start = {}
        n = 0
        for g in self.groups:
            self.stage_start[g] = n
            n += len(self.seq[g])

        # act 1 framing: wide enough for the unregistered start, closing gently
        # onto the grid frame as the sections converge
        pts = np.vstack([np.vstack([s["start"], s["final"]]) for s in self.slices])
        wxy = A.world_xy(pts)
        self.a1_half = float(max(wxy[:, 0].ptp(), wxy[:, 1].ptp()) / 2) * 1.06
        self.a1_centre = np.array([wxy[:, 0].mean(), wxy[:, 1].mean()])

        self.gc_cols = []
        # bin20 bins sit on a regular lattice; projected straight onto pixels
        # that lattice beats against the pixel grid and the tissue fills with
        # moire.  One fixed sub-pixel offset per point dithers it away and,
        # being fixed, never shimmers between frames.
        rng = np.random.default_rng(20260923)
        for s in self.slices:
            gid = s["gid"][s["gsel"]]
            self.gc_cols.append(np.array([A.pal.get(int(g), (150, 150, 150)) for g in gid],
                                         np.float32))
            s["jitter"] = rng.uniform(-0.5, 0.5, (len(s["start"]), 2)).astype(np.float32)

        # act 3 close-up: the window holding the most matched GC, so the
        # refinement iterations are watched where there is something to watch
        self.zoom = {}
        for k in A.if_names:
            tgt = np.asarray(A.if_local[k]["steps"][-1]["target_points_um"], float)
            best, count = tgt.mean(axis=0), -1
            for p in tgt:
                n = int((np.abs(tgt - p).max(axis=1) < ZOOM_HALF_UM).sum())
                if n > count:
                    best, count = p, n
            self.zoom[k] = (best, count)
        self.z_full = {k: A.z(k) for k in A.order}
        self.zmid = (A.z(A.order[0]) + A.z(A.order[-1])) / 2

    # -- cameras ----------------------------------------------------------
    def cam_top(self, centre, half, frac, fov=15.0, shift=(0, 0), z=0.0):
        c = Camera(W, H, [centre[0], centre[1], z], 270.0, 90.0, 1.0, fov, shift)
        c.distance = c.distance_for_width(half, frac)
        return c

    def cam_oblique(self, az=270.0, el=20.0, frac=0.50, fov=33.0, z=None):
        A = self.A
        c = Camera(W, H, [A.centre[0], A.centre[1], self.zmid if z is None else z],
                   az, el, 1.0, fov)
        c.distance = c.distance_for_width(A.half_width, frac)
        return c

    # -- stack ------------------------------------------------------------
    def draw_stack(self, canvas, cam, states):
        """Painter's algorithm over whatever sections are on screen."""
        live = [(k, st) for k, st in states.items() if st["alpha"] > 0.004]
        if not live:
            return
        mids = np.array([st["corners"].mean(axis=0) for _, st in live])
        _, depth = cam.project(mids)
        for i in np.argsort(-depth):
            key, st = live[i]
            draw_slab(canvas, cam, st["corners"], st["tex"],
                      thickness=self.A.slab, shadow=st.get("shadow"),
                      alpha=st["alpha"])

    def base_states(self, alpha=1.0, viewer=False, z=None):
        A = self.A
        out = {}
        for k in A.order:
            zz = A.z(k) if z is None else z[k]
            out[k] = {"corners": S.grid_corners(A.bbox, zz),
                      "tex": A.view_tex[k] if viewer else A.tex[k],
                      "shadow": None if viewer else A.shadow[k], "alpha": alpha}
        return out


# =================================================================== act 1 ==
def act1(film, canvas, t):
    A = film.A
    if t < A1_IN:
        stage, key, idx = "start", None, 0
        u = 0.0
    elif t < A1_IN + A1_ITERS:
        idx = min(film.n_iter - 1, int((t - A1_IN) * FPS))
        stage, key = film.steps[idx]
        idx += 1
        u = idx / film.n_iter
    else:
        stage, key, idx = "final", None, film.n_iter
        u = 1.0

    half = lerp(film.a1_half, A.half_width * 1.02, smooth(u))
    centre = lerp(film.a1_centre, A.centre, smooth(u))
    cam = film.cam_top(centre, half, 0.60)

    tissue, gcs = [], []
    for s, cols in zip(film.slices, film.gc_cols):
        xy = coords_at(s, stage, key)
        w = A.world_xy(xy)
        p3 = np.column_stack([w, np.zeros(len(w))])
        scr, _ = cam.project(p3)
        scr += s["jitter"]
        tissue.append((scr[s["tsel"]], TISSUE_RGB, 0.55))
        gcs.append((scr[s["gsel"]], cols, 1.0))
    canvas.alpha_composite(splat(W, H, tissue, gain=0.70))
    canvas.alpha_composite(splat(W, H, gcs, gain=0.80))

    if stage in ("start", "final"):
        big, sub = ("0000", "before alignment") if stage == "start" else \
                   (f"{film.n_iter}", "saved final coordinates")
        seg = None
    else:
        big = f"{idx}"
        local = idx - film.stage_start[stage]
        sub = f"{STAGE_LABEL[stage]}  {local} / {len(film.seq[stage])}"
        seg = stage
    rows = [(s["short"], "reference" if not s["moving"] else "moving") for s in film.slices]
    return {"big": big, "big_sub": sub, "big_label": "SPATEO ITERATION",
            "big_total": f"/ {film.n_iter}", "rows": rows, "stage": seg,
            "stage_progress": u}


# =================================================================== act 2 ==
def act2(film, canvas, t):
    A = film.A
    u = t / A2
    move = smoother(clamp01(u / 0.86))
    fade = smooth(clamp01((u - 0.40) / 0.52))          # points -> slabs

    start = film.cam_top(A.centre, A.half_width * 1.02, 0.60)
    end = film.cam_oblique()
    cam = interp_camera(start, end, move)

    z = {k: A.z(k) * move for k in A.order}
    if fade < 1:
        tissue, gcs = [], []
        for s, cols, name in zip(film.slices, film.gc_cols,
                                 [s["short"] for s in film.slices]):
            w = A.world_xy(s["final"])
            p3 = np.column_stack([w, np.full(len(w), z[name])])
            scr, _ = cam.project(p3)
            scr += s["jitter"]
            tissue.append((scr[s["tsel"]], TISSUE_RGB, 0.55))
            gcs.append((scr[s["gsel"]], cols, 1.0))
        a = 1 - fade
        for layer, gain in ((tissue, 0.70), (gcs, 0.80)):
            img = splat(W, H, layer, gain=gain)
            img.putalpha(img.getchannel("A").point(lambda v: int(v * a)))
            canvas.alpha_composite(img)

    states = {}
    for k in A.st_names:
        states[k] = {"corners": S.grid_corners(A.bbox, z[k]), "tex": A.tex[k],
                     "shadow": A.shadow[k], "alpha": fade}
    film.draw_stack(canvas, cam, states)

    return {"big": f"{film.n_iter}", "big_sub": "frame frozen", "big_label": "SPATEO ITERATION",
            "big_total": f"/ {film.n_iter}",
            "rows": [(s["short"], f"section {A.section[s['short']]}") for s in film.slices],
            "note": ["Stereo-seq coordinates are now fixed.",
                     "Nothing below moves them again."]}


# =================================================================== act 3 ==
def act3(film, canvas, t):
    A = film.A
    slot = min(4, int(t // SEG))
    tt = t - slot * SEG
    key = A.if_names[slot]
    L = A.if_local[key]
    nb = A.neighbours[key]
    steps = L["steps"]
    n_icp = len(steps) - 1

    # ---- sub-timeline --------------------------------------------------
    FLY, DIVE, ICP, TPS, RISE = 1.05, 1.15, 3.40, 0.95, 1.20
    t_fly, t_dive = FLY, FLY + DIVE
    t_icp, t_tps = t_dive + ICP, t_dive + ICP + TPS
    t_rise = t_tps + RISE

    dive = smoother(clamp01((tt - t_fly) / DIVE))
    rise = smoother(clamp01((tt - t_tps) / RISE))
    close = max(dive - rise, 0.0)                       # 0 oblique, 1 top view

    # which ICP iteration, and how far into the move toward it
    if tt < t_dive:
        step_i, blend = 0, 0.0
    elif tt < t_icp:
        f = (tt - t_dive) / ICP * n_icp
        step_i = min(n_icp - 1, int(f))
        blend = S.ease_out(min(1.0, (f - step_i) / 0.72))
    else:
        step_i, blend = n_icp - 1, 1.0
    cur = steps[step_i]
    nxt = steps[min(step_i + 1, n_icp)]
    lin = np.array(cur["linear"]) + (np.array(nxt["linear"]) - np.array(cur["linear"])) * blend
    off = np.array(cur["offset"]) + (np.array(nxt["offset"]) - np.array(cur["offset"])) * blend
    shown_iter = step_i + (1 if blend > 0.5 else 0)

    tps = smooth(clamp01((tt - t_icp) / TPS))           # ICP frame -> TPS result

    # ---- camera ---------------------------------------------------------
    # The first three ICP iterations do nearly all the work and are obvious at
    # section scale; everything after that is tens of micrometres, which is
    # under a pixel from up here.  So the top view pushes in on the densest GC
    # neighbourhood once the big move is over, and the refinement - and the TPS
    # that follows it - happen at a scale where they can actually be seen.
    zoom_in = smoother(clamp01((tt - (t_dive + 0.95)) / 0.85))
    zoom_out = smoother(clamp01((tt - t_tps) / (RISE * 0.55)))
    zoomu = max(0.0, zoom_in - zoom_out)
    zc = A.z(key)
    zcentre, zcount = film.zoom[key]
    half = lerp(A.half_width * 1.04, ZOOM_HALF_UM, zoomu)
    centre = lerp(A.centre, np.array([zcentre[0], -zcentre[1]]), zoomu)

    oblique = film.cam_oblique()
    top = film.cam_top(centre, half, 0.545, fov=15.0, shift=(-150, 0), z=zc)
    cam = interp_camera(oblique, top, close)

    # ---- section placement ---------------------------------------------
    # In the top view the pair collapses onto one z, so perspective cannot make
    # the two sections differ in scale - xy is all the match ever uses - and
    # the Stereo-seq contour is lifted just above the IF so it reads as an
    # overlay rather than disappearing beneath it.
    lift = 0.45 * A.slab
    states = {}
    for k in A.order:
        if k == key:
            continue
        if k.startswith("IF") and A.if_names.index(k) >= slot:
            continue                                    # not inserted yet
        involved = k in nb
        overlay = involved and close > 0.5
        states[k] = {
            "corners": S.grid_corners(A.bbox, lerp(A.z(k), zc + lift, close)
                                      if involved else A.z(k)),
            "tex": A.overlay_tex[k] if overlay else A.tex[k],
            "shadow": None if (overlay or close > 0.5) else A.shadow[k],
            "alpha": lerp(1.0, 1.0 if involved else 0.07, close)}

    fly = smoother(clamp01(tt / FLY))
    glide = (1 - fly) * (A.bbox[1] - A.bbox[0]) * 1.25
    arc = math.sin(math.pi * fly) * 520.0
    moving = S.local_corners(L["shape"], L["um_per_px"], lin, off, zc + arc)
    moving[:, 0] += glide
    states["_moving"] = {"corners": moving, "tex": L["tex"], "shadow": L["shadow"],
                         "alpha": min(1.0, fly * 1.6) * (1 - tps)}
    if tps > 0:
        states["_warped"] = {"corners": S.grid_corners(A.bbox, zc + 0.03 * A.slab),
                             "tex": A.tex[key], "shadow": A.shadow[key], "alpha": tps}
    film.draw_stack(canvas, cam, states)

    # ---- match overlay ---------------------------------------------------
    panel = None
    if close > 0.25:
        d = ImageDraw.Draw(canvas, "RGBA")
        vis = clamp01((close - 0.25) / 0.45) * (1 - 0.75 * tps)
        moved = L["centroids_um"] @ lin.T + off
        p_all, _ = cam.project(np.column_stack([moved[:, 0], -moved[:, 1],
                                                np.full(len(moved), zc + lift)]))
        rows = np.asarray(nxt.get("matched_if_index", []), int)
        tgt = np.asarray(nxt.get("target_points_um", []), float)
        if len(tgt):
            p_t, _ = cam.project(np.column_stack([tgt[:, 0], -tgt[:, 1],
                                                  np.full(len(tgt), zc + lift)]))
            for j, r in enumerate(rows):
                a, b = p_all[r], p_t[j]
                d.line([tuple(a), tuple(b)], fill=(255, 255, 255, int(165 * vis)), width=3)
                d.line([(b[0] - 7, b[1]), (b[0] + 7, b[1])],
                       fill=(*S.ST_ACCENT, int(240 * vis)), width=2)
                d.line([(b[0], b[1] - 7), (b[0], b[1] + 7)],
                       fill=(*S.ST_ACCENT, int(240 * vis)), width=2)
        r_px = 7 + 5 * zoomu
        for j, p in enumerate(p_all):
            col = S.IF_ACCENT if L["matched"][j] else (150, 160, 172)
            d.ellipse((p[0] - r_px, p[1] - r_px, p[0] + r_px, p[1] + r_px),
                      outline=(*col, int(235 * vis)), width=2)
        panel = vis
        if zoomu > 0.03:
            scale = A.half_width * 1.04 / half
            text(d, (64, 128), f"ZOOM ×{scale:.1f}   {zcount} paired GC in view",
                 15, (*S.DIM, int(255 * vis * zoomu)), True, condensed=True)

    # ---- readout ---------------------------------------------------------
    hist = L["history"]
    if tt < t_dive:
        big, sub, label = "0", "Stage-A placement, before ICP", "ICP ITERATION"
    elif tt < t_icp:
        big = f"{shown_iter}"
        sub = f"{hist[max(shown_iter - 1, 0)][0]} GC paired   mean {hist[max(shown_iter - 1, 0)][1]:.0f} um"
        label = "ICP ITERATION"
    else:
        big = f"{n_icp}"
        sub = "converged   thin-plate spline takes over"
        label = "ICP ITERATION"

    m = A.match[key]
    stats = [
        ("section", f"{key}   slot {A.section[key]} of 9"),
        ("neighbours", " + ".join(nb)),
        ("anchor pairs", f"{m['n_anchor_pairs']}"),
        ("centroid error before TPS", f"{m['centroid_error_pre_um']:.0f} um"),
        ("TPS residual (median)", f"{m['tps_residual_median_um']:.0f} um"),
        ("leave-one-out (median)", f"{m['tps_loo_median_um']:.0f} um"),
    ]
    return {"big": big, "big_sub": sub, "big_label": label, "big_total": f"/ {n_icp}",
            "panel": panel, "panel_key": key, "panel_hist": hist,
            "panel_iter": shown_iter if tt >= t_dive else 0,
            "panel_stats": stats, "panel_tps": tps,
            "inserted": slot + (1 if tt > t_tps else 0)}


# =================================================================== act 4 ==
def act4(film, canvas, t):
    A = film.A
    u = t / A4
    cam = film.cam_oblique(az=270.0 + 9.0 * math.sin(math.pi * u),
                           el=20.0 + 2.5 * math.sin(math.pi * u))
    film.draw_stack(canvas, cam, film.base_states())
    pres = A.validation.get("geometry", {})
    return {"big": "9", "big_sub": "sections in one frame", "big_label": "SECTIONS PLACED",
            "big_total": "/ 9", "ladder": True,
            "note": [f"{len(A.ids)} matched GC identities",
                     "4 Stereo-seq + 5 immunofluorescence",
                     "true pitch 10 um, 80 um end to end"]}


# =================================================================== act 5 ==
def act5(film, canvas, t):
    A = film.A
    fade = smooth(clamp01(t / A5_FADE))
    spin = clamp01((t - A5_FADE) / A5_ORBIT)
    az = 270.0 + 360.0 * smoother(spin) if spin < 1 else 270.0
    el = 20.0 + 5.0 * math.sin(math.pi * 2 * spin)
    cam = film.cam_oblique(az=az, el=el, frac=lerp(0.50, 0.45, fade))

    if fade < 1:
        film.draw_stack(canvas, cam, film.base_states(alpha=1.0))
    if fade > 0:
        film.draw_stack(canvas, cam, film.base_states(alpha=fade, viewer=True))

    deg = int(round((az - 270.0) % 360))
    return {"big": f"{deg}°", "big_sub": "one full turn", "big_label": "ORBIT",
            "big_total": "/ 360°",
            "legend": [("Stereo-seq GC", S.ST_VIEWER_HEX), ("IF GC", S.IF_VIEWER_HEX)],
            "note": ["four ST + five IF", "alternating, 10 um apart"]}


# ==================================================================== hud ===
def act_index(t):
    for i in range(5):
        if t < ACT_BOUNDS[i + 1] or i == 4:
            return i
    return 4


def hud(film, canvas, t, info):
    A = film.A
    d = ImageDraw.Draw(canvas, "RGBA")
    ai = act_index(t)

    # ---- title -----------------------------------------------------------
    text(d, (64, 40), f"{film.rep.upper()}   ·   MOUSE SPLEEN   ·   "
                      "NINE CONSECUTIVE 10 µm SECTIONS",
         15, S.DIM, True, condensed=True)
    text(d, (64, 62), ACT_TITLES[ai], 33, S.INK, True)
    text(d, (W - 64, 40), f"ACT {ai + 1} / 5", 15, S.DIM, True, anchor="ra", condensed=True)
    lo, hi = ACT_BOUNDS[ai], ACT_BOUNDS[ai + 1]
    d.line((W - 64 - 108, 66, W - 64, 66), fill=(*S.RULE, 255), width=3)
    d.line((W - 64 - 108, 66, W - 64 - 108 + 108 * clamp01((t - lo) / (hi - lo)), 66),
           fill=(*S.IF_ACCENT, 255), width=3)

    # ---- big iteration readout ------------------------------------------
    text(d, (64, 872), info.get("big_label", ""), 14, S.DIM, True, condensed=True)
    big = info.get("big", "")
    text(d, (62, 888), big, 72, S.INK, True)
    bw = tw(big, 72, True)
    text(d, (72 + bw, 940), info.get("big_total", ""), 22, S.FAINT, True)
    text(d, (64, 976), info.get("big_sub", ""), 20, S.DIM)

    # ---- act 1 stage bar --------------------------------------------------
    if ai == 0 and info.get("stage"):
        x = 64
        for g in film.groups:
            n = len(film.seq[g])
            w = 300 * n / film.n_iter * 3.0
            on = g == info["stage"]
            d.rounded_rectangle((x, 826, x + w, 832), radius=3,
                                fill=(*(S.IF_ACCENT if on else S.RULE), 255))
            text(d, (x, 802), f"{STAGE_LABEL[g]} {n}", 13,
                 S.INK if on else S.FAINT, on, condensed=True)
            x += w + 10

    # ---- right-hand lists -------------------------------------------------
    base = 992
    if info.get("legend"):
        base -= 30 * len(info["legend"])
        y = base
        for name, hexcol in info["legend"]:
            col = tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
            d.rounded_rectangle((W - 64 - 16, y + 4, W - 64, y + 20), radius=4,
                                fill=(*col, 255))
            text(d, (W - 64 - 28, y + 2), name, 19, S.INK, anchor="ra")
            y += 30
        base -= 16
    if info.get("rows"):
        base -= 27 * len(info["rows"])
        y = base
        for name, state in info["rows"]:
            text(d, (W - 64, y), name, 19, S.INK, True, anchor="ra")
            text(d, (W - 64 - 120, y + 2), state, 15, S.FAINT, anchor="ra", condensed=True)
            y += 27
        base -= 16
    if info.get("note"):
        base -= 28 * len(info["note"])
        y = base
        for line in info["note"]:
            text(d, (W - 64, y), line, 19, S.DIM, anchor="ra")
            y += 28

    # ---- act 3 panel ------------------------------------------------------
    if info.get("panel"):
        a = info["panel"]
        x0, y0, x1, y1 = W - 64 - 392, 150, W - 64, 606
        d.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=(12, 18, 26, int(216 * a)),
                            outline=(*S.RULE, int(255 * a)), width=1)
        text(d, (x0 + 22, y0 + 20), "GC MATCHING  ·  TOP VIEW", 14,
             (*S.DIM, int(255 * a)), True, condensed=True)
        text(d, (x0 + 22, y0 + 42), f"{info['panel_key']}  ↔  "
             + " + ".join(A.neighbours[info["panel_key"]]), 26, (*S.INK, int(255 * a)), True)
        # convergence curve: mean paired distance per saved ICP iteration
        hist = info["panel_hist"]
        cx0, cy0, cx1, cy1 = x0 + 22, y0 + 92, x1 - 22, y0 + 212
        vals = [v[1] for v in hist]
        vmax = max(vals) * 1.12
        d.line((cx0, cy1, cx1, cy1), fill=(*S.RULE, int(255 * a)), width=1)
        pts = [(cx0 + (cx1 - cx0) * i / max(len(vals) - 1, 1),
                cy1 - (cy1 - cy0) * (v / vmax)) for i, v in enumerate(vals)]
        shown = max(1, min(len(pts), info["panel_iter"]))
        if shown > 1:
            d.line(pts[:shown], fill=(*S.IF_ACCENT, int(255 * a)), width=3, joint="curve")
        p = pts[shown - 1]
        d.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill=(*S.GOOD, int(255 * a)))
        text(d, (cx0, cy0 - 22), "mean paired GC distance", 13,
             (*S.DIM, int(255 * a)), condensed=True)
        text(d, (cx1, cy0 - 22), f"{vals[shown - 1]:.0f} µm", 13,
             (*S.GOOD, int(255 * a)), True, anchor="ra", condensed=True)
        y = y0 + 234
        for label, value in info["panel_stats"]:
            text(d, (x0 + 22, y), label, 15, (*S.FAINT, int(255 * a)), condensed=True)
            text(d, (x1 - 22, y - 2), value, 17, (*S.INK, int(255 * a)), True, anchor="ra")
            y += 29
        if info.get("panel_tps", 0) > 0.02:
            b = int(255 * a * info["panel_tps"])
            text(d, (x0 + 22, y1 - 40), "thin-plate spline applied", 17, (*S.GOOD, b), True)

    # ---- nine-slot ladder --------------------------------------------------
    if ai in (2, 3):
        full = bool(info.get("ladder"))
        x = 64
        for k in A.order:
            is_if = k.startswith("IF")
            on = full or not is_if or A.if_names.index(k) < info.get("inserted", 0)
            col = S.IF_ACCENT if is_if else S.ST_ACCENT
            d.rounded_rectangle((x, 826, x + 52, 832), radius=3,
                                fill=(*(col if on else S.RULE), 255))
            if full:
                text(d, (x, 802), k, 14, col, True, condensed=True)
            x += 58

    # ---- footer -----------------------------------------------------------
    marks = [b / T5 for b in ACT_BOUNDS[1:-1]]
    progress(d, (64, 1040, W - 64, 1046), t / T5, marks)
    text(d, (64, 1056), CAPTIONS[ai], 14, S.FAINT)
    text(d, (W - 64, 1056),
         "gcalign nine-section pipeline  ·  no registration recomputed here",
         14, S.FAINT, anchor="ra")


# ================================================================== driver ==
def render(film, t):
    canvas = film.bg.copy()
    if t < T1:
        info = act1(film, canvas, t)
    elif t < T2:
        info = act2(film, canvas, t - T1)
    elif t < T3:
        info = act3(film, canvas, t - T2)
    elif t < T4:
        info = act4(film, canvas, t - T3)
    else:
        info = act5(film, canvas, t - T4)
    canvas.alpha_composite(film.vig)
    canvas.alpha_composite(film.scrim)
    hud(film, canvas, t, info)
    return canvas.convert("RGB")


PREVIEW_TIMES = {
    "a1_start": 0.6, "a1_rigid": 3.0, "a1_nonrigid": 7.5, "a1_deform": 15.0,
    "a1_final": T1 - 0.4, "a2_tilt": T1 + 2.3, "a2_done": T2 - 0.2,
    "a3_fly": T2 + 0.7, "a3_dive": T2 + 2.0, "a3_icp": T2 + 3.6,
    "a3_tps": T2 + 5.9, "a3_settle": T2 + 7.3, "a3_second": T2 + SEG + 3.6,
    "a3_last": T2 + 4 * SEG + 3.6, "a4": T3 + 2.0,
    "a5_fade": T4 + 1.2, "a5_quarter": T4 + A5_FADE + A5_ORBIT * 0.25,
    "a5_half": T4 + A5_FADE + A5_ORBIT * 0.55, "a5_end": T5 - 1.0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=["rep1", "rep2"])
    ap.add_argument("--preview", action="store_true", help="key frames only, no encode")
    ap.add_argument("--only", default="", help="comma-separated preview keys")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=T5)
    args = ap.parse_args()

    t0 = time.time()
    film = Film(args.rep)
    print(f"assets ready in {time.time() - t0:.0f} s; film length {T5:.2f} s "
          f"= {int(round(T5 * FPS))} frames", flush=True)

    prev = OUT / "previews"
    prev.mkdir(parents=True, exist_ok=True)
    if args.preview:
        keys = args.only.split(",") if args.only else list(PREVIEW_TIMES)
        for k in keys:
            t = PREVIEW_TIMES[k.strip()]
            s = time.time()
            render(film, t).save(prev / f"{args.rep}_{k.strip()}.png")
            print(f"  {k:12s} t={t:6.2f}  {time.time() - s:.2f} s", flush=True)
        return

    exe = P.ffmpeg()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{args.rep}_nine_section_reconstruction.mp4"
    tmp = target.with_suffix(".render.mp4")
    cmd = [exe, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
           "-threads", "4", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n0, n1 = int(args.start * FPS), int(round(min(args.end, T5) * FPS))
    marks = {}
    try:
        for i in range(n0, n1):
            frame = render(film, i / FPS)
            proc.stdin.write(frame.tobytes())
            if i % 180 == 0:
                el = time.time() - t0
                print(f"  frame {i}/{n1}  {el:.0f} s elapsed "
                      f"({(i - n0 + 1) / max(el, 1e-6):.2f} fps)", flush=True)
        proc.stdin.close()
        assert proc.wait() == 0, "ffmpeg failed"
    except BaseException:
        proc.kill()
        raise
    tmp.replace(target)

    for k, t in PREVIEW_TIMES.items():
        render(film, t).save(prev / f"{args.rep}_{k}.png")
        marks[k] = round(t, 3)

    A = film.A
    audit = {
        "rep": args.rep, "output": target.name,
        "dimensions": [W, H], "fps": FPS,
        "frames": n1 - n0, "duration_seconds": round((n1 - n0) / FPS, 3),
        "speed_note": "act 1 plays the same 1,100 saved Spateo iterations as "
                      "rep_four_slice_spateo_alignment.mp4 at one iteration per "
                      "frame, 60 fps instead of 30 - exactly twice the pace, no "
                      "iteration dropped",
        "acts": [{"index": i + 1, "title": ACT_TITLES[i],
                  "start_s": round(ACT_BOUNDS[i], 3), "end_s": round(ACT_BOUNDS[i + 1], 3)}
                 for i in range(5)],
        "spateo_iterations": film.n_iter,
        "spateo_stage_counts": {g: len(film.seq[g]) for g in film.groups},
        "icp_iterations_per_IF": {k: len(A.if_local[k]["steps"]) - 1 for k in A.if_names},
        "icp_source": f"{args.rep}_verified_icp_replay.json (verified against the "
                      "saved global_transforms.json)",
        "z_exaggeration": SA.Z_EXAG,
        "section_pitch_um": SA.SECTION_UM,
        "slab_fraction_of_pitch": SA.SLAB_FRACTION,
        "projection": "perspective pinhole, look-at; act 1 fov 15 deg, "
                      "3-D acts fov 33 deg",
        "viewer_palette": {"ST": S.ST_VIEWER_HEX, "IF": S.IF_VIEWER_HEX},
        "gc_ids": len(A.ids),
        "plane_order": A.order,
        "neighbours": A.neighbours,
        "approximations_on_screen": [
            "z axis exaggerated; true section pitch is 10 um",
            "the sideways entry of each IF section is staging, not a stored transform",
            "the warped IF silhouette is re-cut from the warped channels with "
            "gcalign's own tissue recipe, because no warped tissue mask was stored",
        ],
        "source_sha256": {
            "plane_gc_labels.npz": hashlib.sha256(
                (A.root / "03_gc_match/plane_gc_labels.npz").read_bytes()).hexdigest(),
            "global_transforms.json": hashlib.sha256(
                (A.root / "02_global_align/global_transforms.json").read_bytes()).hexdigest(),
            "verified_icp_replay.json": hashlib.sha256(
                (P.ICP_REPLAY_DIR / f"{args.rep}_verified_icp_replay.json").read_bytes()).hexdigest(),
        },
        "preview_frames": marks,
    }
    (OUT / f"{args.rep}_story_validation.json").write_text(json.dumps(audit, indent=2))
    mb = target.stat().st_size / 1e6
    print(f"DONE {target.name}  {mb:.1f} MB  {audit['duration_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
