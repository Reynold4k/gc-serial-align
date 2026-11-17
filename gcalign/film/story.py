"""The film's storyboard, its frames, its encoding and its validation record.

The film is one continuous camera shot in five acts. IF is immunofluorescence,
ST is spatial transcriptomics, GC is germinal centre, ICP is iterative closest
point and TPS is thin-plate spline.

  act 1   Spateo aligns the ST sections: the ST bins converge in a top view,
          one stored Spateo iteration per frame. Act 1 needs the Spateo run.
  act 2   The aligned ST sections tilt out of the top view and separate along
          the cutting axis; their bins become sections with a thickness.
  act 3   Each IF section slides in to its stored Stage-A placement. The camera
          drops to a top view, the stored ICP iterations run, the view pushes
          in on the window with the most matched GCs, the TPS result replaces
          the last ICP position, and the section settles into its slot.
  act 4   All sections in one coordinate frame.
  act 5   One full turn of the stack, each GC coloured by its section kind.

`Film` builds the storyboard (a list of `Shot`) from a `Run` and renders any
frame by time. `encode` pipes every frame through x264. `run` is the entry
point: it renders the film, or key stills, and writes the validation JSON.
`Film.staging` lists everything in the film that is staging and not
measurement; that list is written into the validation JSON.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .. import __version__, theme
from ..config import Config
from ..runio import Run
from . import core as C
from .assets import Assets
from .core import Vectors
from .hud import Hud, Pen
from .tools import find_ffmpeg
from .upstream import SpateoReplay, Upstream

# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------
PRESETS = {
    # the 86-second film: every stored Spateo iteration, every ICP iteration of
    # every IF section, one full turn
    "full": dict(
        spateo_in=1.2, spateo_seconds=None, spateo_out=1.5,
        opening=1.4, lift=4.6,
        insert=dict(fly=1.05, dive=1.15, icp_per_iteration=0.283, icp_min=0.9, icp_max=3.4,
                    tps=0.95, rise=1.2, hold=0.45, zoom=0.85),
        act3_budget=41.0, drop=None,
        stack=4.0, turn=(1.8, 11.0, 2.4),
        fade_in=0.7, fade_out=1.0,
    ),
    # about 18 s: the Spateo run sampled, one IF section matched in full and the
    # others dropped into place already registered, a quick turn
    "short": dict(
        spateo_in=0.5, spateo_seconds=3.0, spateo_out=0.5,
        opening=1.0, lift=2.3,
        insert=dict(fly=0.6, dive=0.7, icp_per_iteration=0.15, icp_min=0.6, icp_max=1.8,
                    tps=0.6, rise=0.8, hold=0.15, zoom=0.55),
        act3_budget=None, drop=0.62,
        stack=1.3, turn=(0.6, 3.8, 0.6),
        fade_in=0.4, fade_out=0.5,
    ),
}
QUALITY = {
    "master": dict(scale=3, x264_preset="slow", crf=16),
    "draft": dict(scale=2, x264_preset="veryfast", crf=22),
}
# the stain is shown as an RGB composite (`run.rgb`): these are its primaries
RGB_PRIMARIES = tuple(theme.hex_to_rgb(colour) for colour in theme.STAIN_RGB)
STAGE_NAMES = {"rigid": "rigid", "nonrigid": "non-rigid", "deformable": "deformable"}
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve"]
OBLIQUE = dict(az=270.0, el=20.0, frac=0.50, fov=33.0)
TOP_FOV = 15.0
ZOOM_FRACTION = 0.28          # close-up half-width / tissue half-width (x3.6)


@dataclass
class Shot:
    act: int
    kind: str
    start: float
    length: float
    params: dict = field(default_factory=dict)

    @property
    def end(self) -> float:
        return self.start + self.length


def _scrim(W: int, H: int, side: str) -> Image.Image:
    """A black gradient behind a block of type, for the close-up only."""
    a = np.zeros((H, W), np.float32)
    if side == "right":
        x = np.linspace(0, 1, W)[None, :]
        a[:] = C.smooth_array(np.clip((x - 0.47) / 0.24, 0, 1)) ** 1.2 * 0.94
    else:                                       # bottom left, the foot and the head
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot(xx / (0.30 * W), (H - yy) / (0.30 * H))
        corner = C.smooth_array(np.clip((1.25 - r) / 0.6, 0, 1)) * 0.93
        foot = C.smooth_array(np.clip((yy / H - 0.90) / 0.08, 0, 1)) * 0.80
        head = C.smooth_array(np.clip((0.15 - yy / H) / 0.13, 0, 1)) * 0.80
        a = np.maximum(np.maximum(corner, foot), head)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(rgba)                          # uint8, four channels: Pillow mode "RGBA"


# --------------------------------------------------------------------------
class Film:
    """One run, staged."""

    def __init__(self, run: Run, size=(1920, 1080), fps: float = 60, preset: str = "full",
                 upstream: Upstream | None = None, acts=None, quality: str = "master",
                 label: str | None = None, z_exaggeration: float | None = None, log=print):
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}")
        if quality not in QUALITY:
            raise ValueError(f"quality must be one of {sorted(QUALITY)}")
        log = log or _quiet
        self.run = run
        self.cfg = run.cfg
        self.W, self.H = int(size[0]), int(size[1])
        self.fps = float(fps)
        self.preset = preset
        self.P = PRESETS[preset]
        self.quality = quality
        self.label = label
        self.log = log
        t0 = time.time()
        self.A = A = Assets(run, scale=QUALITY[quality]["scale"], z_exaggeration=z_exaggeration,
                            log=log)
        self.hud = Hud(self.W, self.H)
        self.acts = set(acts) if acts else {1, 2, 3, 4, 5}
        # everything a frame needs is read here, before any worker is forked:
        # nothing below touches the run's open files again
        self.match = {n: dict(run.match.get(n, {})) for n in A.if_names}
        self.spateo = None
        if upstream is not None and 1 in self.acts:
            self.spateo = SpateoReplay(run, upstream, log=log)
            pts = np.vstack([np.vstack([s["start"], s["final"]]) for s in self.spateo.slices])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            self.a1_frame = (np.array([(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2]),
                             (hi[0] - lo[0]) / 2 * 1.04, (hi[1] - lo[1]) / 2 * 1.04)
        self.cloud = self._cloud()
        self.zoom_half = max(ZOOM_FRACTION * max(A.half_w, A.half_h), 4 * A.grid.um)
        self.zoom = {n: A.zoom_window(n, self.zoom_half) for n in A.if_names}
        self.featured = self._featured()
        self.scrim_right = _scrim(self.W, self.H, "right")
        self.scrim_left = _scrim(self.W, self.H, "left")
        self.ring = self._ring()
        self.keyrows = self._keyrows()
        self.shots = self._timeline()
        self.duration = self.shots[-1].end if self.shots else 0.0
        self.n_frames = int(round(self.duration * self.fps))
        self.build_seconds = time.time() - t0
        self._act_bounds()

    # ---- inputs ------------------------------------------------------------
    def _cloud(self) -> list[dict]:
        """The registered transcriptomics bins, from the run's frozen scaffold:
        the state act 1 ends in and act 2 starts from. One fixed sub-pixel
        offset per bin breaks up the moire a regular bin lattice makes against
        the pixel grid; being fixed, it never shimmers."""
        A = self.A
        sc = self.run.scaffold
        slice_short = np.asarray(sc["slice_short"])
        gid = np.asarray(sc["gc_id"])
        world = np.asarray(sc["world_um"], np.float32)
        rng = np.random.default_rng(20260923)
        out = []
        for name in A.st_names:
            rows = np.flatnonzero(slice_short == name)
            g = gid[rows]
            gsel = np.flatnonzero(g > 0)
            tsel = np.flatnonzero(g <= 0)[::5]
            out.append({
                "name": name, "final": world[rows], "tsel": tsel, "gsel": gsel,
                "gcol": np.array([A.identity.get(int(v), A.other) for v in g[gsel]], np.float32),
                "jitter": rng.uniform(-0.5, 0.5, (len(rows), 2)).astype(np.float32),
            })
        if self.spateo is not None:
            for c, s in zip(out, self.spateo.slices):
                assert c["name"] == s["name"] and len(c["tsel"]) == len(s["tsel"])
        return out

    def _featured(self) -> str | None:
        """The section the short edit matches in full: two-sided, most pairs."""
        A = self.A
        best, key = None, None
        for n in A.if_names:
            steps = A.local[n]["steps"]
            pairs = len(steps[-1].get("matched_if_index", [])) if len(steps) > 1 else 0
            k = (len(A.neighbours[n]) >= 2, pairs, -abs(A.if_names.index(n) - len(A.if_names) / 2))
            if key is None or k > key:
                best, key = n, k
        return best

    def _keyrows(self) -> dict:
        """The colour keys: what each colour on screen stands for."""
        A = self.A
        sample = [A.identity_hex[i] for i in A.ids[:4]] or [theme.INK_2]
        channels = [("dot", primary, name) for primary, name in
                    zip(RGB_PRIMARIES, self.cfg.rgb_channels) if name is not None]
        return {
            "gc": [("dots", sample, "one colour per germinal centre (GC)")],
            "st": [("dot", theme.ST, ("ST", "spatial transcriptomics"))],
            "kinds": [("dot", theme.IF, ("IF", "immunofluorescence")),
                      ("dot", theme.ST, ("ST", "spatial transcriptomics"))],
            "channels": channels if A.if_names else [],
            "families": [("dot", A.family_base["IF"], ("IF", "germinal centres")),
                         ("dot", A.family_base["ST"], ("ST", "germinal centres"))],
        }

    def _ring(self) -> np.ndarray:
        """Points round the tissue, for placing section labels beside the stack."""
        A = self.A
        th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        return np.column_stack([A.centre[0] + A.half_w * np.cos(th),
                                A.centre[1] + A.half_h * np.sin(th)])

    # ---- timeline ------------------------------------------------------------
    def _insert_phases(self, name: str, factor: float) -> dict:
        P = self.P["insert"]
        n = self.A.local[name]["n_icp"]
        icp = min(max(n * P["icp_per_iteration"], P["icp_min"]), P["icp_max"])
        ph = {k: P[k] * factor for k in ("fly", "dive", "tps", "rise", "hold", "zoom")}
        ph["icp"] = icp * factor
        # push in once the large first move is over: after the iteration from
        # which the mean paired distance stays within 25 % of its final value
        hist = [d for _, d in self.A.local[name]["history"]]
        k = 1
        if hist:
            final = hist[-1]
            for i in range(len(hist)):
                if all(d <= 1.25 * final + 1e-9 for d in hist[i:]):
                    k = max(i, 1)
                    break
        per = ph["icp"] / max(n, 1)
        ph["zoom_at"] = ph["fly"] + ph["dive"] + min(per * (k + 0.6), ph["icp"] * 0.45)
        return ph

    def _timeline(self) -> list[Shot]:
        P, A, fps = self.P, self.A, self.fps
        shots: list[Shot] = []
        t = 0.0

        def add(act, kind, length, **params):
            nonlocal t
            if act in self.acts:
                shots.append(Shot(act, kind, t, length, params))
                t += length

        if self.spateo is not None:
            n = self.spateo.n_iter
            frames = n if P["spateo_seconds"] is None else min(n, int(round(P["spateo_seconds"] * fps)))
            sample = np.round(np.linspace(0, n - 1, frames)).astype(int)
            add(1, "spateo", P["spateo_in"] + frames / fps + P["spateo_out"],
                frames=frames, sample=sample, hold_in=P["spateo_in"])
        opening = 0.0 if (self.spateo is not None and 1 in self.acts) else P["opening"]
        add(2, "lift", opening + P["lift"], opening=opening)

        n_if = len(A.if_names)
        factor = 1.0
        if P["act3_budget"] and n_if:
            nominal = sum(sum(v for k, v in self._insert_phases(n, 1.0).items() if k not in ("zoom", "zoom_at"))
                          for n in A.if_names)
            if nominal > P["act3_budget"] * 1.2:
                factor = max(0.6, P["act3_budget"] / nominal)
        placed: list[str] = []
        for name in A.if_names:
            if P["drop"] and name != self.featured:
                add(3, "drop", P["drop"], name=name, placed=list(placed))
            else:
                ph = self._insert_phases(name, factor)
                length = sum(v for k, v in ph.items() if k not in ("zoom", "zoom_at"))
                add(3, "insert", length, name=name, placed=list(placed), phases=ph)
            placed.append(name)
        add(4, "stack", P["stack"])
        fade, orbit, out = P["turn"]
        add(5, "turn", fade + orbit + out, fade=fade, orbit=orbit)
        return shots

    def _act_bounds(self) -> None:
        bounds = {}
        for s in self.shots:
            lo, hi = bounds.get(s.act, (s.start, s.end))
            bounds[s.act] = (min(lo, s.start), max(hi, s.end))
        self.act_bounds = dict(sorted(bounds.items()))

    def shot_at(self, t: float) -> Shot:
        for s in self.shots:
            if t < s.end:
                return s
        return self.shots[-1]

    # ---- cameras ---------------------------------------------------------------
    def cam_top(self, centre, half_w, half_h, frac_w=0.60, frac_h=0.80, z=0.0,
                shift=(0.0, 0.0), fov=TOP_FOV) -> C.Camera:
        c = C.Camera(self.W, self.H, [centre[0], centre[1], z], 270.0, 90.0, 1.0, fov, shift)
        c.distance = max(c.distance_for(half_w, frac_w, "w"), c.distance_for(half_h, frac_h, "h"))
        return c

    def cam_full_top(self, z=0.0, shift=(0.0, 0.0)) -> C.Camera:
        A = self.A
        return self.cam_top(A.centre, A.half_w * 1.03, A.half_h * 1.03, z=z, shift=shift)

    def cam_oblique(self, az=None, el=None, frac=None, fov=None) -> C.Camera:
        A = self.A
        o = OBLIQUE
        c = C.Camera(self.W, self.H, [A.centre[0], A.centre[1], A.z_mid],
                     o["az"] if az is None else az, o["el"] if el is None else el, 1.0,
                     o["fov"] if fov is None else fov)
        c.distance = c.distance_for(A.half, o["frac"] if frac is None else frac, "w")
        return c

    # ---- drawing helpers ---------------------------------------------------------
    def draw_stack(self, canvas, cam, states) -> None:
        """Painter's algorithm over whatever sections are on screen.
        ``states``: list of (corners, texture, edge, alpha)."""
        live = [s for s in states if s[3] > 0.004]
        if not live:
            return
        mids = np.array([s[0].mean(axis=0) for s in live])
        _, depth = cam.project(mids)
        for i in np.argsort(-depth):
            corners, tex, edge, alpha = live[i]
            C.draw_slab(canvas, cam, corners, tex, edge, self.A.slab if edge is not None else 0.0,
                        alpha)

    def splat_cloud(self, canvas, cam, coords, z, alpha=1.0) -> None:
        """The transcriptomics bins as points: grey tissue, GCs in identity colour
        with a soft glow."""
        if alpha <= 0.004:
            return
        tissue, gcs = [], []
        grey = np.asarray(theme.hex_to_rgb(theme.INK_2), np.float32)
        for c, xy, zz in zip(self.cloud, coords, z):
            sel = np.concatenate([c["tsel"], c["gsel"]])
            scr, _ = cam.project(C.to_world(xy[sel], zz))
            scr += c["jitter"][sel]
            nt = len(c["tsel"])
            tissue.append((scr[:nt], grey, 1.0))
            gcs.append((scr[nt:], c["gcol"], 1.0))
        k = self.H / 1080.0
        t_img = C.splat(self.W, self.H, tissue, gain=0.30)
        g_img = C.splat(self.W, self.H, gcs, gain=0.95)
        halo = (g_img.convert("RGBa").reduce(4).filter(_blur(2.2 * k))
                .resize((self.W, self.H), Image.BILINEAR).convert("RGBA"))
        for img, a in ((t_img, alpha), (halo, alpha * 0.85), (g_img, alpha)):
            canvas.alpha_composite(C.scale_alpha(img, a) if a < 0.999 else img)

    def section_labels(self, cam, names_z_alpha) -> list:
        """(x, y, name, colour, alpha): one column left of the stack, each name
        level with its own section. Left, because IF sections enter from the
        right."""
        out = []
        xs = []
        for name, z, alpha in names_z_alpha:
            if alpha <= 0.01:
                continue
            pts, _ = cam.project(np.column_stack([self.ring, np.full(len(self.ring), z)]))
            i = int(np.argmin(pts[:, 0]))
            xs.append(pts[i, 0])
            out.append([pts[i, 0], pts[i, 1], name, theme.KIND[self.A.kind[name]], alpha])
        if out:
            x = max(min(xs) - self.hud.px(30), self.hud.margin + self.hud.px(40))
            for row in out:
                row[0] = x
        return out

    # ==================================================================== acts
    def act_spateo(self, canvas, shot, tt, ov, info):
        sp = self.spateo
        p = shot.params
        run_len = p["frames"] / self.fps
        if tt < p["hold_in"]:
            step, u = "start", 0.0
        elif tt < p["hold_in"] + run_len:
            k = min(p["frames"] - 1, int((tt - p["hold_in"]) * self.fps + 1e-6))
            step = int(p["sample"][k])
            u = (step + 1) / sp.n_iter
        else:
            step, u = "final", 1.0
        coords = [sp.coords(s, step) for s in sp.slices]
        c0, hw0, hh0 = self.a1_frame
        e = C.smooth(u)
        cam_a = self.cam_top(c0, hw0, hh0)
        cam_b = self.cam_full_top()
        cam = C.interp_camera(cam_a, cam_b, e)
        self.splat_cloud(canvas, cam, coords, [0.0] * len(coords))
        if step == "start":
            value, sub = "0", "before alignment"
        elif step == "final":
            value, sub = f"{sp.n_iter}", "aligned"
        else:
            stage, i, n = sp.stage_of(step)
            value, sub = f"{step + 1}", f"{STAGE_NAMES.get(stage, stage)} stage: {i} / {n}"
        info.update(counter=("Spateo iteration", value, f"/ {sp.n_iter}", sub),
                    scale=(cam.px_per_um(), 1.0), keys=[self.keyrows["gc"]])

    def act_lift(self, canvas, shot, tt, ov, info):
        A = self.A
        opening = shot.params["opening"]
        L = shot.length - opening
        u = C.window(tt, opening, L)
        move = C.smoother(C.clamp01(u / 0.86))
        fade = C.smooth(C.window(u, 0.40, 0.52))
        appear = C.smooth(C.window(tt, 0.0, opening * 0.8)) if opening > 0 else 1.0
        cam = C.interp_camera(self.cam_full_top(), self.cam_oblique(), move)
        z = {k: A.z[k] * move for k in A.st_names}
        if fade < 1:
            coords = [c["final"] for c in self.cloud]
            self.splat_cloud(canvas, cam, coords, [z[c["name"]] for c in self.cloud],
                             alpha=(1 - fade) * appear)
        states = [(A.corners(k, z[k]), A.tex[k], A.edge[k], fade) for k in A.st_names]
        self.draw_stack(canvas, cam, states)
        info.update(
            counter=("Sections placed", f"{len(A.st_names)}", f"/ {len(A.names)}", ""),
            labels=self.section_labels(cam, [(k, z[k], fade) for k in A.st_names]),
            ztag=move, scale=(cam.px_per_um(), 1 - C.smooth(C.clamp01(move * 4))),
            keys=[self.keyrows["st"], self.keyrows["gc"]])

    def act_insert(self, canvas, shot, tt, ov, info):
        A = self.A
        p = shot.params
        name, ph = p["name"], p["phases"]
        L = A.local[name]
        t_fly = ph["fly"]
        t_dive = t_fly + ph["dive"]
        t_icp = t_dive + ph["icp"]
        t_tps = t_icp + ph["tps"]
        dive = C.smoother(C.window(tt, t_fly, ph["dive"]))
        rise = C.smoother(C.window(tt, t_tps, ph["rise"]))
        close = max(dive - rise, 0.0)

        steps, n = L["steps"], L["n_icp"]
        if n == 0 or tt < t_dive:
            step_i, blend = 0, 0.0
        elif tt < t_icp:
            f = (tt - t_dive) / ph["icp"] * n
            step_i = min(n - 1, int(f))
            blend = C.ease_out(min(1.0, (f - step_i) / 0.72))
        else:
            step_i, blend = n - 1, 1.0
        cur = steps[step_i]
        nxt = steps[min(step_i + 1, n)]
        lin = C.lerp(np.asarray(cur["linear"], float), np.asarray(nxt["linear"], float), blend)
        off = C.lerp(np.asarray(cur["offset"], float), np.asarray(nxt["offset"], float), blend)
        shown = 0 if tt < t_dive or n == 0 else step_i + (1 if blend > 0.5 else 0)
        tps = C.smooth(C.window(tt, t_icp, ph["tps"]))

        zoom_in = C.smoother(C.window(tt, ph["zoom_at"], ph["zoom"]))
        zoom_out = C.smoother(C.window(tt, t_tps, ph["rise"] * 0.55))
        centre_zoom, _ = self.zoom[name]
        zoomu = max(0.0, zoom_in - zoom_out) if centre_zoom is not None else 0.0
        zc = A.z[name]
        shift = (-0.085 * self.W, 0.0)
        top_full = self.cam_top(A.centre, A.half_w * 1.04, A.half_h * 1.04, frac_w=0.56,
                                frac_h=0.80, z=zc, shift=shift)
        if centre_zoom is not None:
            cz = np.array([centre_zoom[0], -centre_zoom[1]])
            top_zoom = self.cam_top(cz, self.zoom_half, self.zoom_half * 0.3, frac_w=0.56,
                                    z=zc, shift=shift)
            top = C.interp_camera(top_full, top_zoom, zoomu)
        else:
            top = top_full
        cam = C.interp_camera(self.cam_oblique(), top, close)

        # ---- sections ----------------------------------------------------
        lift = 0.45 * A.slab
        swap = C.smooth(C.window(close, 0.15, 0.30))
        away = 1 - C.smooth(C.window(close, 0.0, 0.45))     # sections in the way fade out
        label_alpha = 1 - C.smooth(C.window(close, 0.0, 0.3))
        states, labels = [], []
        for k in A.names:
            if k == name or (A.kind[k] == "IF" and k not in p["placed"]):
                continue
            involved = k in A.neighbours[name]
            z = C.lerp(A.z[k], zc + lift, close) if involved else A.z[k]
            corners = A.corners(k, z)
            if involved:
                if swap < 1:
                    states.append((corners, A.tex[k], A.edge[k], 1 - swap))
                if swap > 0:
                    states.append((corners, A.overlay[k], None, swap))
            else:
                states.append((corners, A.tex[k], A.edge[k], away))
            labels.append((k, z, label_alpha))
        fly = C.smoother(C.window(tt, 0.0, ph["fly"]))
        glide = (1 - fly) * 2.6 * A.half_w
        arc = math.sin(math.pi * fly) * 1.2 * A.slab
        moving = C.local_corners(L["shape"], A.grid.um, lin, off, zc + arc)
        moving[:, 0] += glide
        states.append((moving, L["tex"], L["edge"], min(1.0, fly * 1.6) * (1 - tps)))
        if tps > 0:
            states.append((A.corners(name, zc + 0.02 * A.slab), A.tex[name], A.edge[name], tps))
        self.draw_stack(canvas, cam, states)
        labels.append((name, zc + arc, label_alpha * min(1.0, fly * 1.6)))

        # ---- the match, drawn on the stored pairs ---------------------------
        panel = 0.0
        if close > 0.25 and n > 0:
            vis = C.window(close, 0.25, 0.45) * (1 - 0.85 * tps)
            moved = C.place(L["centroids_um"], lin, off)
            p_all, _ = cam.project(C.to_world(moved, zc + lift))
            rows = np.asarray(nxt.get("matched_if_index", []), int)
            tgt = np.asarray(nxt.get("target_points_um", []), float).reshape(-1, 2)
            k = self.H / 1080.0 * (1 + 0.5 * zoomu)
            if len(tgt) and len(rows):
                p_t, _ = cam.project(C.to_world(tgt, zc + lift))
                for j, r in enumerate(rows):
                    ov.line([p_all[r], p_t[j]], theme.AGREE, 0.62 * vis, 1.4 * k)
                for q in p_t:
                    ov.ring(q, 4.2 * k, theme.ST, 0.95 * vis, 1.5 * k)
            for j, q in enumerate(p_all):
                colour = theme.IF if L["matched"][j] else theme.MUTED
                ov.dot(q, 3.2 * k, colour, 0.95 * vis)
            panel = C.window(close, 0.25, 0.45)
        elif close > 0.25:
            panel = C.window(close, 0.25, 0.45)

        m = self.match.get(name, {})
        hist = L["history"]
        if shown > 0 and hist:
            pairs, dist = hist[shown - 1]
            pairs_s, dist_s = f"{pairs}", f"{dist:.0f} µm"
        else:
            pairs_s, dist_s = "–", "–"
        tps_res = m.get("tps_residual_median_um")
        tps_s = f"{tps_res:.0f} µm" if (tps_res is not None and tps > 0.5) else "–"
        info.update(
            counter=("ICP iteration", f"{shown}", f"/ {n}", ""),
            labels=self.section_labels(cam, labels),
            ztag=1.0,
            scale=(cam.px_per_um(), C.smooth(C.window(close, 0.85, 0.15))),
            scrim=close,
            keys=[self.keyrows["kinds"], self.keyrows["channels"]],
            panel=dict(alpha=panel, name=name, neighbours=A.neighbours[name],
                       curve=[d for _, d in hist], shown=shown,
                       rows=[("matched GC pairs", pairs_s, 1.0),
                             ("mean distance between matched GCs", dist_s, 1.0),
                             ("median distance after thin-plate spline", tps_s, 1.0)]))

    def act_drop(self, canvas, shot, tt, ov, info):
        A = self.A
        name = shot.params["name"]
        u = C.smoother(C.window(tt, 0.0, shot.length * 0.9))
        cam = self.cam_oblique()
        states, labels = [], []
        for k in A.names:
            if k == name or (A.kind[k] == "IF" and k not in shot.params["placed"]):
                continue
            states.append((A.corners(k), A.tex[k], A.edge[k], 1.0))
            labels.append((k, A.z[k], 1.0))
        z = A.z[name] + (1 - u) * 5.0 * A.slab
        a = C.smooth(C.window(tt, 0.0, shot.length * 0.55))
        states.append((A.corners(name, z), A.tex[name], A.edge[name], a))
        labels.append((name, z, a))
        self.draw_stack(canvas, cam, states)
        done = len(A.st_names) + len(shot.params["placed"]) + (1 if u > 0.5 else 0)
        info.update(counter=("Sections placed", f"{done}", f"/ {len(A.names)}", ""),
                    labels=self.section_labels(cam, labels), ztag=1.0,
                    keys=[self.keyrows["kinds"], self.keyrows["channels"]])

    def act_stack(self, canvas, shot, tt, ov, info):
        A = self.A
        u = tt / shot.length
        cam = self.cam_oblique(az=OBLIQUE["az"] + 9.0 * math.sin(math.pi * u),
                               el=OBLIQUE["el"] + 2.5 * math.sin(math.pi * u))
        states = [(A.corners(k), A.tex[k], A.edge[k], 1.0) for k in A.names]
        self.draw_stack(canvas, cam, states)
        info.update(counter=("Sections placed", f"{len(A.names)}", f"/ {len(A.names)}", ""),
                    labels=self.section_labels(cam, [(k, A.z[k], 1.0) for k in A.names]),
                    ztag=1.0, keys=[self.keyrows["kinds"], self.keyrows["channels"],
                                    self.keyrows["gc"]])

    def act_turn(self, canvas, shot, tt, ov, info):
        A = self.A
        fade_len, orbit = shot.params["fade"], shot.params["orbit"]
        fade = C.smooth(C.window(tt, 0.0, fade_len))
        spin = C.window(tt, fade_len, orbit)
        az = OBLIQUE["az"] + 360.0 * C.smoother(spin)
        el = OBLIQUE["el"] + 5.0 * math.sin(math.pi * 2 * spin)
        cam = self.cam_oblique(az=az, el=el, frac=C.lerp(OBLIQUE["frac"], 0.46, fade))
        if fade < 1:
            self.draw_stack(canvas, cam, [(A.corners(k), A.tex[k], A.edge[k], 1 - fade)
                                          for k in A.names])
        if fade > 0:
            self.draw_stack(canvas, cam, [(A.corners(k), A.view[k], None, fade) for k in A.names])
        deg = int(round(360.0 * C.smoother(spin)))
        before = [self.keyrows["kinds"], self.keyrows["channels"], self.keyrows["gc"]]
        info.update(counter=("Rotation", f"{deg}°", "/ 360°", ""),
                    labels=self.section_labels(cam, [(k, A.z[k], 1 - fade) for k in A.names]),
                    ztag=1.0, keys=before if fade < 0.5 else [self.keyrows["families"]],
                    keys_alpha=abs(1 - 2 * fade) ** 0.7)

    # ==================================================================== frame
    def title(self, act: int) -> str:
        """What the act shows, in a few words."""
        if act == 1:
            return "Spateo aligns the spatial transcriptomics (ST) sections"
        if act == 2:
            return "ST sections stacked in cutting order"
        if act == 3:
            return "Immunofluorescence (IF) sections matched to the ST sections"
        if act == 4:
            n = len(self.A.names)
            word = NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)
            return f"{word.capitalize()} sections in one coordinate frame"
        return "One full turn of the stack"

    def render(self, t: float) -> Image.Image:
        W, H = self.W, self.H
        canvas = Image.new("RGBA", (W, H), theme.rgba(theme.BG, 1.0))
        ov = Vectors(W, H)
        shot = self.shot_at(t)
        tt = t - shot.start
        info: dict = {}
        getattr(self, f"act_{shot.kind}")(canvas, shot, tt, ov, info)
        ov.composite(canvas)
        if info.get("scrim", 0) > 0.01:
            a = C.smooth(C.clamp01((info["scrim"] - 0.3) / 0.5))
            if a > 0.01:
                canvas.alpha_composite(C.scale_alpha(self.scrim_right, a))
                canvas.alpha_composite(C.scale_alpha(self.scrim_left, a))
        # the text goes onto the RGB frame: Pillow blends a translucent ink only
        # when it draws onto RGB (onto RGBA it overwrites), and the HUD fades
        rgb = canvas.convert("RGB")
        self.draw_hud(rgb, t, shot, tt, info)
        P = self.P
        fade = min(C.smooth(C.window(t, 0.0, P["fade_in"])),
                   1 - C.smooth(C.window(t, self.duration - P["fade_out"], P["fade_out"])))
        if fade < 0.999:
            rgb = Image.eval(rgb, _fade_lut(fade))
        return rgb

    def draw_hud(self, canvas, t, shot, tt, info) -> None:
        hud = self.hud
        d = Pen(canvas)
        lo, hi = self.act_bounds[shot.act]
        dip = 0.22
        act_alpha = min(C.smooth(C.window(t, lo, dip)) if lo > 0 else 1.0,
                        1 - C.smooth(C.window(t, hi - dip, dip)) if hi < self.duration else 1.0)
        shot_alpha = min(C.smooth(C.window(tt, 0.0, dip)) if shot.start > 0 else 1.0,
                         1 - C.smooth(C.window(tt, shot.length - dip, dip))
                         if shot.end < self.duration - 1e-6 else 1.0)
        hud.title(d, self.title(shot.act), act_alpha)

        tags = []
        if self.label:
            tags.append((self.label, theme.INK_2, 1.0))
        ztag = info.get("ztag", 0.0)
        if ztag > 0.01:
            tags.append((f"z stretched ×{self.A.z_exag:g}", theme.INK_2, ztag))
        if tags:
            hud.tag(d, tags)

        if "counter" in info:
            label, value, total, sub = info["counter"]
            hud.counter(d, label, value, total, sub, shot_alpha)
        for x, y, name, colour, alpha in info.get("labels", []):
            hud.label(d, (x, y), name, colour, alpha, anchor="rm")
        above = hud.keys(canvas, d, info.get("keys", []), info.get("keys_alpha", 1.0) * act_alpha)
        if "scale" in info:
            ppu, alpha = info["scale"]
            hud.scale_bar(d, ppu, alpha * shot_alpha, y=above)
        panel = info.get("panel")
        if panel and panel["alpha"] > 0.01:
            self.draw_panel(canvas, d, panel)
        segments = [((lo_ / self.duration), (hi_ / self.duration))
                    for lo_, hi_ in self.act_bounds.values()]
        hud.progress(d, t / max(self.duration, 1e-6), segments)

    def draw_panel(self, canvas, d, panel) -> None:
        """The section being matched, its flanking ST sections, the convergence
        of the stored ICP iterations and the numbers that describe the match."""
        hud = self.hud
        a = panel["alpha"]
        x1 = self.W - hud.margin
        x0 = x1 - hud.px(400)
        y = hud.top + hud.px(150)
        hud.text(d, (x0, y), panel["name"], 34, "medium", theme.IF, a)
        x = x0
        yb = y + hud.px(34)
        parts = [("matched to ", theme.INK_2)]
        for i, nb in enumerate(panel["neighbours"]):
            if i:
                parts.append((" and ", theme.INK_2))
            parts.append((nb, theme.ST))
        for text, colour in parts:
            hud.text(d, (x, yb), text, 16, "medium" if colour == theme.ST else "regular", colour, a)
            x += hud.width(text, 16, "medium" if colour == theme.ST else "regular")
        top = y + hud.px(72)
        box = (x0, top, x1, top + hud.px(84))
        pad = hud.px(10)
        ov = Vectors(self.W, self.H, region=(x0 - pad, top - pad, x1 + pad, top + hud.px(330)))
        if panel["curve"]:
            self.hud_sparkline(ov, box, panel["curve"], panel["shown"], a)
        hud.text(d, (x0, top + hud.px(108)), "Mean distance between matched GCs, per ICP iteration",
                 12.5, "regular", theme.MUTED, a)
        end = hud.stats(d, x0, x1, top + hud.px(158), panel["rows"], a)
        hud.key_row(ov, d, (x0, end + hud.px(12)),
                    [("dot", theme.IF, "IF GC centre"), ("ring", theme.ST, "ST GC centre"),
                     ("line", theme.AGREE, "matched pair")], a, align="left", size=12.5)
        ov.composite(canvas)

    def hud_sparkline(self, ov: Vectors, box, values, shown, alpha) -> None:
        x0, y0, x1, y1 = box
        vals = [v for v in values if v == v]
        if not vals:
            return
        top = max(vals) * 1.08
        n = len(values)
        pts = [(x0 + (x1 - x0) * (i / max(n - 1, 1)), y1 - (y1 - y0) * (v / top))
               for i, v in enumerate(values)]
        k = self.H / 1080.0
        ov.line([(x0, y1), (x1, y1)], theme.LINE_2, alpha, 1.0 * k)
        shown = max(0, min(n, shown))
        if shown >= 2:
            ov.line(pts[:shown], theme.IF, alpha, 2.0 * k)
        if shown >= 1:
            ov.dot(pts[shown - 1], 4.0 * k, theme.INK, alpha)

    # ==================================================================== records
    def preview_times(self) -> dict:
        out = {}
        for s in self.shots:
            if s.kind == "spateo":
                h = s.params["hold_in"]
                run_len = s.params["frames"] / self.fps
                out["a1_start"] = s.start + h * 0.6
                out["a1_early"] = s.start + h + run_len * 0.12
                out["a1_middle"] = s.start + h + run_len * 0.45
                out["a1_late"] = s.start + h + run_len * 0.85
                out["a1_final"] = s.end - 0.2
            elif s.kind == "lift":
                o = s.params["opening"]
                if o:
                    out["a2_open"] = s.start + o * 0.9
                out["a2_tilt"] = s.start + o + (s.length - o) * 0.45
                out["a2_done"] = s.end - 0.05
            elif s.kind == "insert":
                ph = s.params["phases"]
                nm = s.params["name"]
                t_dive = ph["fly"] + ph["dive"]
                t_icp = t_dive + ph["icp"]
                if not any(k.startswith("a3_fly") for k in out):
                    out[f"a3_fly_{nm}"] = s.start + ph["fly"] * 0.6
                    out[f"a3_dive_{nm}"] = s.start + ph["fly"] + ph["dive"] * 0.6
                    out[f"a3_icp1_{nm}"] = s.start + t_dive + ph["icp"] / max(self.A.local[nm]["n_icp"], 1) * 0.5
                    out[f"a3_zoom_{nm}"] = s.start + min(ph["zoom_at"] + ph["zoom"] + 0.15, t_icp - 0.05)
                    out[f"a3_tps_{nm}"] = s.start + t_icp + ph["tps"] * 0.9
                    out[f"a3_rise_{nm}"] = s.start + t_icp + ph["tps"] + ph["rise"] * 0.75
                else:
                    out[f"a3_zoom_{nm}"] = s.start + min(ph["zoom_at"] + ph["zoom"] + 0.15, t_icp - 0.05)
            elif s.kind == "drop":
                out[f"a3_drop_{s.params['name']}"] = s.start + s.length * 0.5
            elif s.kind == "stack":
                out["a4_stack"] = s.start + s.length * 0.5
            elif s.kind == "turn":
                f, o = s.params["fade"], s.params["orbit"]
                out["a5_fade"] = s.start + f * 0.6
                out["a5_quarter"] = s.start + f + o * 0.30
                out["a5_half"] = s.start + f + o * 0.55
                out["a5_end"] = s.end - self.P["fade_out"] - 0.1
        return out

    def staging(self) -> list[str]:
        """Everything in this film that is staging and not measurement, one
        item per kind of staging, written for the run being rendered."""
        A = self.A
        items = [
            f"The z axis is stretched x{A.z_exag:g}. Each section sits at its true depth "
            f"(plane.z_um) times {A.z_exag:g}. Sections are {A.pitch_um:g} um thick. Each "
            f"section is drawn {A.slab / A.z_exag:.1f} um thick at true scale, so that every "
            "section stays visible.",
            "Each IF section enters from the right along an arc. This entry is choreography. "
            "The entry ends on the stored Stage-A placement (the placement that the align stage "
            "found before the ICP step), which is ICP iteration 0.",
            "Between two stored ICP iterations, the IF section moves along a straight-line "
            "blend of the two stored transforms for a fraction of a second. Every position at "
            "which the section comes to rest is a stored ICP iteration.",
            "The thin-plate-spline (TPS) result replaces the last ICP position by a cross-fade. "
            "The TPS warp itself is not interpolated.",
            "While an IF section is matched, its flanking ST sections (the ST sections directly "
            "before and after that IF section in cutting order) are drawn in the plane of that IF "
            "section, as GC outlines. The other sections fade out during the match.",
            "All camera moves are staging: the dive into the top view, the push-in on the window "
            "with the most matched GCs, the sway over the finished stack and the closing orbit.",
            "In the closing turn (act 5), each GC takes a shade of its section kind's viewer "
            f"colour (ST: {A.family_base['ST']}, IF: {A.family_base['IF']}). These shades "
            "separate neighbouring GCs and carry no other data.",
            "Transcriptomics bins are drawn as points with a fixed random offset of at most half "
            "a pixel, to avoid moire. Acts 1 and 2 draw every GC bin and every fifth other bin.",
        ]
        recut = [n for n in A.if_names if not A.tissue_stored[n]]
        if recut:
            items.append(
                "The outline of IF section(s) " + ", ".join(recut) + " is re-cut from the warped "
                "stain with the segmentation's tissue recipe. This run was made before gcalign "
                "0.2 and stored no warped tissue mask. GC identities, the stain and every number "
                "on screen do not depend on this outline.")
        if self.preset == "short":
            dropped = [n for n in A.if_names if n != self.featured]
            if dropped:
                items.append(
                    f"Short edit: only {self.featured} is matched on screen. "
                    + ", ".join(dropped) + " drop into their slots already registered. The drop "
                    "along z is staging.")
            if self.spateo is not None:
                items.append(
                    f"Short edit: act 1 shows {self._spateo_frames()} of the "
                    f"{self.spateo.n_iter} stored Spateo iterations, evenly spaced. Each frame "
                    "still shows one stored iteration.")
        return items

    def _spateo_frames(self) -> int:
        for s in self.shots:
            if s.kind == "spateo":
                return int(s.params["frames"])
        return 0

    ACT_CONTENT = {
        1: ("The ST sections converge in a top view, one stored Spateo iteration per frame "
            "(the short edit shows an evenly spaced sample of the stored iterations).",
            "uns/iter_<stage>/align_<stage>/<i> in the Spateo files, joined to "
            "00_scaffold/st_scaffold.npz by chip coordinate"),
        2: ("The registered ST sections tilt out of the top view and separate along z. Their "
            "bins become sections with a thickness.",
            "00_scaffold/st_scaffold.npz, 02_global_align/st_rasters.npz, "
            "03_gc_match/plane_gc_labels.npz"),
        3: ("Each IF section moves to its stored Stage-A placement, runs through its stored ICP "
            "iterations against its flanking ST sections, and is replaced by its "
            "thin-plate-spline result.",
            "01_if_gc/<IF>_gc.npz, the ICP steps (02_global_align/icp_trajectory.json, or the "
            "verified rebuild for older runs), 03_gc_match/plane_gc_labels.npz, "
            "03_gc_match/if_channels_warped.npz, 03_gc_match/match_report.json"),
        4: ("All sections stand in one coordinate frame.",
            "03_gc_match/plane_gc_labels.npz, 03_gc_match/if_channels_warped.npz, the tissue "
            "outlines"),
        5: ("The stack turns once. Each GC is coloured by its section kind.",
            "the same rasters as act 4; colours from viewer.st_hex and viewer.if_hex in the "
            "config"),
    }

    def record(self, out: Path | None, frames: int | None, seconds: float | None) -> dict:
        """The validation record written next to the film."""
        A, run = self.A, self.run
        acts = []
        for act, (lo, hi) in self.act_bounds.items():
            shows, data = self.ACT_CONTENT[act]
            acts.append({"act": act, "title_on_screen": self.title(act),
                         "start_s": round(lo, 3), "end_s": round(hi, 3),
                         "length_s": round(hi - lo, 3), "shows": shows, "driven_by": data})
        shots = []
        for s in self.shots:
            row = {"act": s.act, "kind": s.kind, "start_s": round(s.start, 3),
                   "end_s": round(s.end, 3)}
            if "name" in s.params:
                row["section"] = s.params["name"]
            if s.kind == "insert":
                ph = s.params["phases"]
                t_dive = s.start + ph["fly"] + ph["dive"]
                row.update(icp_start_s=round(t_dive, 3), icp_end_s=round(t_dive + ph["icp"], 3),
                           tps_end_s=round(t_dive + ph["icp"] + ph["tps"], 3))
            if s.kind == "spateo":
                row["stored_iterations_shown"] = int(s.params["frames"])
            shots.append(row)
        icp = {}
        for n in A.if_names:
            Lc = A.local[n]
            rebuilt = "rebuilt" in Lc["icp_origin"]
            icp[n] = {
                "iterations": Lc["n_icp"],
                "source": Lc["icp_origin"] + ("" if rebuilt else
                                              " (02_global_align/icp_trajectory.json)"),
                "stored_in_run": not rebuilt,
                "verified_against_stored_transform": True if rebuilt else None,
                **Lc["icp_errors"],
                "matched_gc_pairs_per_iteration": [p for p, _ in Lc["history"]],
                "mean_distance_between_matched_gcs_um": [round(d, 3) for _, d in Lc["history"]],
                "median_distance_after_tps_um": self.match.get(n, {}).get("tps_residual_median_um"),
            }
        files = ["03_gc_match/plane_gc_labels.npz", "03_gc_match/if_channels_warped.npz",
                 "03_gc_match/match_report.json", "03_gc_match/if_tissue_warped.npz",
                 "02_global_align/global_transforms.json", "02_global_align/st_rasters.npz",
                 "02_global_align/icp_trajectory.json", "00_scaffold/st_scaffold.npz",
                 "00_scaffold/st_scaffold_summary.json"]
        files += [f"01_if_gc/{n}_gc.npz" for n in A.if_names]
        hashes = {}
        for rel in files:
            path = run.root / rel
            if path.exists():
                hashes[rel] = _sha256(path)
        name = Path(out).name if out else "this storyboard"
        rec = {
            "about": (f"How the film {name} was made from the gcalign run '{self.cfg.name}' in "
                      f"{run.root}. Every moving thing in the film is a transform that the run "
                      "or its Spateo registration stored. The list 'staged_not_measured' names "
                      "everything in the film that is staging."),
            "abbreviations": {"IF": "immunofluorescence", "ST": "spatial transcriptomics",
                              "GC": "germinal centre", "ICP": "iterative closest point",
                              "TPS": "thin-plate spline"},
            "terms": {
                "Stage-A placement": "the rotation, mirror state, scale and position of an IF "
                                     "section that the align stage found before the ICP step; "
                                     "ICP iteration 0",
                "flanking ST sections": "the ST sections directly before and after an IF "
                                        "section in cutting order",
                "short edit featured section": "the IF section that the short edit matches in "
                                               "full: among the IF sections with an ST section "
                                               "on both sides, the one with the most matched "
                                               "GC pairs",
            },
            "film": {"path": str(out) if out else None, "preset": self.preset,
                     "quality": self.quality, "width_px": self.W, "height_px": self.H,
                     "fps": self.fps, "frames": frames,
                     "duration_s": round(frames / self.fps, 3) if frames else None,
                     "render_seconds": round(seconds, 1) if seconds else None},
            "run": {"name": self.cfg.name, "output_dir": str(run.root), "gcalign": __version__},
            "acts": acts,
            "shots": shots,
            "sections": [{"name": p.name, "kind": p.kind, "index": p.index, "z_um": p.z_um,
                          **({"flanking_st_sections": A.neighbours[p.name]}
                             if p.kind == "IF" else {})}
                         for p in A.planes],
            "z_stretch": {"factor": A.z_exag, "section_thickness_um": A.pitch_um,
                          "drawn_section_thickness_um_at_true_scale": round(A.slab / A.z_exag, 3)},
            "gc_count": len(A.ids),
            "colours": {"gc_identity_palette": self.cfg.viewer.palette, "IF": theme.IF,
                        "ST": theme.ST, "act5_families": A.family_base},
            "stain_display_rgb": dict(zip(("red", "green", "blue"), self.cfg.rgb_channels)),
            "icp": icp,
            "if_outline": {n: ("the stored warped tissue mask (03_gc_match/if_tissue_warped.npz)"
                               if A.tissue_stored[n] else
                               "re-cut from the warped stain (run made before gcalign 0.2)")
                           for n in A.if_names},
            "act1_spateo": (self.spateo.describe() if self.spateo is not None else
                            "not shown: no Spateo run was supplied (GCALIGN_SPATEO_DIR or "
                            "--spateo-dir), so the film opens on act 2"),
            "short_edit_featured_section": self.featured,
            "staged_not_measured": self.staging(),
            "input_sha256": hashes,
        }
        return rec


def _quiet(*args, **kwargs) -> None:
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_FADE_LUTS: dict = {}


def _fade_lut(f: float):
    key = int(round(f * 255))
    if key not in _FADE_LUTS:
        _FADE_LUTS[key] = lambda v, k=key: v * k // 255
    return _FADE_LUTS[key]


def _blur(radius: float):
    from PIL import ImageFilter
    return ImageFilter.GaussianBlur(radius)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
_WORKER_FILM: Film | None = None


def _render_bytes(index: int) -> bytes:
    film = _WORKER_FILM
    return film.render(index / film.fps).tobytes()


def _cores() -> int:
    """Cores this process may use (the allocation, not the whole node)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _default_workers() -> int:
    value = os.environ.get("GCALIGN_FILM_WORKERS", "").strip()
    if value.isdigit():
        return max(1, int(value))
    cores = _cores()
    return max(1, cores - 2) if cores > 3 else 1


def encode(film: Film, out: Path, workers: int | None = None, log=print,
           start: float = 0.0, end: float | None = None) -> tuple[int, float]:
    """Render every frame and pipe it through x264."""
    global _WORKER_FILM
    exe = find_ffmpeg()
    q = QUALITY[film.quality]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".render" + out.suffix)
    workers = workers or _default_workers()
    threads = max(2, min(8, _cores() - workers))
    cmd = [exe, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{film.W}x{film.H}", "-r", f"{film.fps:g}", "-i", "-", "-an",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", q["x264_preset"], "-crf", str(q["crf"]),
           "-x264-params", "aq-mode=3", "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-color_range", "tv", "-threads", str(threads),
           "-movflags", "+faststart", str(tmp)]
    n0 = int(round(start * film.fps))
    n1 = film.n_frames if end is None else min(film.n_frames, int(round(end * film.fps)))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        if workers > 1:
            import multiprocessing as mp
            _WORKER_FILM = film
            ctx = mp.get_context("fork")
            with ctx.Pool(workers) as pool:
                for i, frame in enumerate(pool.imap(_render_bytes, range(n0, n1), chunksize=4)):
                    proc.stdin.write(frame)
                    if i % 300 == 0:
                        el = time.time() - t0
                        log(f"  frame {n0 + i}/{n1}  {el:.0f} s  ({(i + 1) / max(el, 1e-6):.2f} fps)",
                            flush=True)
        else:
            for i in range(n0, n1):
                proc.stdin.write(film.render(i / film.fps).tobytes())
                if (i - n0) % 300 == 0:
                    el = time.time() - t0
                    log(f"  frame {i}/{n1}  {el:.0f} s  ({(i - n0 + 1) / max(el, 1e-6):.2f} fps)",
                        flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    except BaseException:
        proc.kill()
        raise
    finally:
        _WORKER_FILM = None
    tmp.replace(out)
    return n1 - n0, time.time() - t0


def _open_run(cfg, output_dir=None) -> Run:
    if isinstance(cfg, Run):
        return cfg
    if isinstance(cfg, Config):
        if output_dir is not None:
            cfg.output_dir = Path(output_dir)
        return Run(cfg)
    return Run.open(cfg, output_dir=output_dir)


def run(cfg, out=None, *, preset: str = "full", fps: float = 60, size=(1920, 1080),
        acts=None, preview=False, upstream=None, quality: str = "master",
        label: str | None = None, z_exaggeration: float | None = None,
        workers: int | None = None, output_dir=None, log=print) -> Path:
    """Render the reconstruction film of a finished run.

    cfg             a `Config`, a path to the experiment YAML, or an open `Run`
    out             the MP4; default ``<output_dir>/07_film/<name>.mp4``
                    (``<name>_short.mp4`` for the short edit)
    preset          "full" (about 86 s with act 1) or "short" (about 18 s)
    fps, size       frame rate and (width, height)
    acts            a subset of 1-5 to render, e.g. [3]
    preview         True renders key stills (PNG) into ``<stem>_previews/``
                    instead of the film; a list of names picks some of them
    upstream        the Spateo run for act 1: an `Upstream`, a directory, or
                    None to read GCALIGN_SPATEO_DIR; False skips act 1. Without
                    it the film opens on the registered ST sections.
    quality         "master" (crf 16, slow) or "draft" (fast, softer textures)
    label           a short tag shown top right, e.g. "rep1"
    workers         frame-rendering processes (default: SLURM allocation - 2)

    Writes ``<stem>_validation.json`` next to the film: act boundaries,
    iteration counts, where the ICP steps came from, input hashes and the list
    of what in the film is staging. Returns the film's path (or the preview
    folder).
    """
    started = time.time()
    log = log or _quiet
    run_ = _open_run(cfg, output_dir)
    if upstream is None:
        upstream = Upstream.from_env()
    elif upstream is False:
        upstream = None
    else:
        upstream = Upstream.coerce(upstream)
    if acts is not None and 1 in acts and upstream is None:
        log("  act 1 needs the Spateo run (upstream= / GCALIGN_SPATEO_DIR); skipping it", flush=True)
    film = Film(run_, size=size, fps=fps, preset=preset, upstream=upstream, acts=acts,
                quality=quality, label=label, z_exaggeration=z_exaggeration, log=log)
    stem_default = run_.cfg.name + ("" if preset == "full" else f"_{preset}")
    out = Path(out) if out is not None else run_.root / "07_film" / f"{stem_default}.mp4"
    log(f"assets ready in {film.build_seconds:.0f} s; {film.duration:.2f} s = "
        f"{film.n_frames} frames at {film.fps:g} fps; acts {sorted(film.act_bounds)}", flush=True)

    if preview:
        folder = out.with_name(out.stem + "_previews")
        folder.mkdir(parents=True, exist_ok=True)
        times = film.preview_times()
        if isinstance(preview, (list, tuple)):
            wanted = {}
            for item in preview:
                if isinstance(item, (int, float)):
                    wanted[f"t{item:07.3f}"] = float(item)
                elif item in times:
                    wanted[item] = times[item]
            times = wanted
        for key, t in times.items():
            s = time.time()
            film.render(t).save(folder / f"{key}.png")
            log(f"  {key:20s} t={t:7.2f}  {time.time() - s:.2f} s", flush=True)
        (folder / "storyboard.json").write_text(
            json.dumps(film.record(None, None, None) | {"preview_times": times}, indent=2,
                       ensure_ascii=False), encoding="utf-8")
        return folder

    frames, seconds = encode(film, out, workers=workers, log=log)
    record = film.record(out, frames, time.time() - started)
    record["preview_times"] = {k: round(v, 3) for k, v in film.preview_times().items()}
    out.with_name(out.stem + "_validation.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"DONE {out}  {out.stat().st_size / 1e6:.1f} MB  {frames} frames  "
        f"{time.time() - started:.0f} s", flush=True)
    return out
