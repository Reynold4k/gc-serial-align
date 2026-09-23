#!/usr/bin/env python3
"""Rendering primitives for the nine-section reconstruction film.

A real pinhole camera (look-at + perspective divide), not the fixed oblique
affine the previous animation used.  That is what makes the tilt out of the top
view, the dive back down onto one section, and the closing orbit read as one
continuous space instead of three unrelated diagrams.

Three primitives are enough for the whole film:

  ``Camera``       eye/target/fov, projects world micrometres to pixels.
  ``draw_slab``    a section: dark underside for thickness, then its texture
                   mapped onto the projected quad through a homography.
  ``splat``        the Stereo-seq point clouds, accumulated with bincount so a
                   1.4 M-point frame costs milliseconds instead of seconds.

World units are micrometres.  x and y are the common 10 um alignment grid; the
z axis is the cutting direction, section k at ``(k - 1) * 10 um * Z_EXAG``.
Screen-up is +Y, so every world y is negated on the way in (see ``world_y``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------
# palette - dark theatre so the fluorescence and the GC colours carry the image
# --------------------------------------------------------------------------
BG_TOP = (8, 12, 18)
BG_BOTTOM = (16, 22, 31)
INK = (232, 240, 248)
DIM = (128, 146, 165)
FAINT = (74, 90, 107)
RULE = (36, 48, 62)
ST_ACCENT = (255, 120, 170)          # Stereo-seq sections
IF_ACCENT = (86, 158, 255)           # immunofluorescence sections
ST_VIEWER_HEX = "#FF1493"            # the viewer's own two colours, reused
IF_VIEWER_HEX = "#0F52BA"
TISSUE_BODY = (26, 36, 48)
GOOD = (94, 222, 178)
WARN = (255, 196, 92)

from film_paths import FONT_DIR                                    # noqa: E402
_FONT_CACHE: dict = {}


def font(size: int, bold: bool = False, condensed: bool = False):
    key = (size, bold, condensed)
    if key not in _FONT_CACHE:
        name = "DejaVuSansCondensed" if condensed else "DejaVuSans"
        name += "-Bold" if bold else ""
        _FONT_CACHE[key] = ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), size)
    return _FONT_CACHE[key]


def text(d: ImageDraw.ImageDraw, xy, s, size=20, fill=INK, bold=False,
         anchor=None, condensed=False, spacing=4):
    d.text(xy, s, font=font(size, bold, condensed), fill=fill, anchor=anchor,
           spacing=spacing)


def tw(s, size=20, bold=False, condensed=False) -> float:
    """Width of a string, for right-aligned or centred layout maths."""
    f = font(size, bold, condensed)
    return f.getbbox(s)[2] - f.getbbox(s)[0]


# --------------------------------------------------------------------------
# easing
# --------------------------------------------------------------------------
def clamp01(t: float) -> float:
    return 0.0 if t < 0 else 1.0 if t > 1 else float(t)


def smooth(t: float) -> float:
    """smoothstep - zero velocity at both ends, for camera moves."""
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def smoother(t: float) -> float:
    """quintic - zero acceleration too, for the long moves that would
    otherwise show a visible kick at the start."""
    t = clamp01(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_out(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
class Camera:
    """Orbit camera: azimuth/elevation/distance around a target, plus fov.

    Elevation 90 degrees looks straight down.  ``right`` is defined from the
    azimuth alone so the basis never degenerates at the pole, which is what
    lets act 2 tilt continuously out of the top view.
    """

    def __init__(self, W: int, H: int, target, azimuth=270.0, elevation=90.0,
                 distance=20000.0, fov=18.0, shift=(0.0, 0.0)):
        self.W, self.H = W, H
        self.shift = np.asarray(shift, float)   # pixels, applied after projection
        self.target = np.asarray(target, float)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.distance = float(distance)
        self.fov = float(fov)

    def copy(self) -> "Camera":
        return Camera(self.W, self.H, self.target.copy(), self.azimuth,
                      self.elevation, self.distance, self.fov, tuple(self.shift))

    @property
    def focal(self) -> float:
        return (self.H / 2) / math.tan(math.radians(self.fov) / 2)

    def basis(self):
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        d = np.array([math.cos(el) * math.cos(az),
                      math.cos(el) * math.sin(az),
                      math.sin(el)])
        eye = self.target + self.distance * d
        fwd = -d
        right = np.array([-math.sin(az), math.cos(az), 0.0])
        up = np.cross(right, fwd)
        return eye, right, up, fwd

    def project(self, P):
        """(N,3) world -> (N,2) pixels and (N,) depth along the view axis."""
        P = np.atleast_2d(np.asarray(P, float))
        eye, right, up, fwd = self.basis()
        v = P - eye
        zc = v @ fwd
        safe = np.maximum(zc, 1.0)
        f = self.focal
        x = self.W / 2 + f * (v @ right) / safe + self.shift[0]
        y = self.H / 2 - f * (v @ up) / safe + self.shift[1]
        return np.stack([x, y], axis=1), zc

    def distance_for_width(self, half_width_um: float, screen_fraction: float) -> float:
        """Distance that makes ``half_width_um`` span ``screen_fraction`` of W.

        Used to hold the framing constant while the field of view opens up, so
        act 2 reads as the room turning rather than the slide zooming."""
        return self.focal * half_width_um / (self.W * screen_fraction / 2)


def interp_camera(a: Camera, b: Camera, t: float) -> Camera:
    """Blend two camera states.  Azimuth takes the short way round."""
    daz = (b.azimuth - a.azimuth + 180) % 360 - 180
    c = a.copy()
    c.target = lerp(a.target, b.target, t)
    c.azimuth = a.azimuth + daz * t
    c.elevation = lerp(a.elevation, b.elevation, t)
    c.fov = lerp(a.fov, b.fov, t)
    # distance in log space: a linear ramp between very different distances
    # spends most of the move close to the far end and looks like a stall.
    c.distance = math.exp(lerp(math.log(a.distance), math.log(b.distance), t))
    c.shift = lerp(a.shift, b.shift, t)
    return c


# --------------------------------------------------------------------------
# background
# --------------------------------------------------------------------------
def backdrop(W: int, H: int) -> Image.Image:
    """Vertical gradient plus a soft centre lift, rendered once and reused."""
    y = np.linspace(0, 1, H)[:, None]
    grad = (np.array(BG_TOP) * (1 - y) + np.array(BG_BOTTOM) * y)
    img = np.repeat(grad[:, None, :], W, axis=1)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - W * 0.5) / (W * 0.62), (yy - H * 0.46) / (H * 0.72))
    img += np.clip(1 - r, 0, 1)[:, :, None] ** 2 * np.array([13.0, 17.0, 23.0])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


# --------------------------------------------------------------------------
# textured section planes
# --------------------------------------------------------------------------
def _homography(dst_quad, src_quad):
    """PIL PERSPECTIVE coefficients mapping output pixels back to the texture."""
    A, B = [], []
    for (ox, oy), (sx, sy) in zip(dst_quad, src_quad):
        A.append([ox, oy, 1, 0, 0, 0, -sx * ox, -sx * oy])
        A.append([0, 0, 0, ox, oy, 1, -sy * ox, -sy * oy])
        B += [sx, sy]
    return np.linalg.solve(np.asarray(A, float), np.asarray(B, float))


def grid_corners(bbox, z):
    """World corners of a grid-aligned section plane, texture (0,0) first.

    bbox is (x_lo, x_hi, Y_lo, Y_hi) in world micrometres with Y already
    negated, so texture row 0 belongs to the Y_hi edge."""
    x0, x1, y0, y1 = bbox
    return np.array([[x0, y1, z], [x1, y1, z], [x1, y0, z], [x0, y0, z]], float)


def local_corners(shape, um_per_px, linear, offset, z):
    """Corners of an IF section still in its own frame, placed by one
    Stage-A similarity transform.

    Composing this with the camera homography means a mid-ICP section costs the
    same single texture transform as a finished one - the iteration is shown at
    the transform the optimiser actually held, not an interpolation of it."""
    h, w = shape
    px = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    local_um = px * np.asarray(um_per_px, float)[None, :]
    world = local_um @ np.asarray(linear, float).T + np.asarray(offset, float)
    return np.column_stack([world[:, 0], -world[:, 1], np.full(4, float(z))])


def draw_slab(canvas: Image.Image, cam: Camera, corners, texture: Image.Image,
              thickness: float = 0.0, shadow: Image.Image | None = None,
              alpha: float = 1.0, resample=Image.BILINEAR) -> bool:
    """Map one section onto the canvas.  Returns False if it is off camera.

    ``corners`` are the world positions of texture pixels (0,0), (W,0), (W,H),
    (0,H).  ``shadow`` is the same silhouette in flat dark pixels drawn one slab
    thickness lower; where it sticks out past the top face it becomes the cut
    edge of the section, which is most of what sells the depth.
    """
    if alpha <= 0.004:
        return False
    corners = np.asarray(corners, float)
    pts, depth = cam.project(corners)
    if (depth < 1.0).any():
        return False

    def blit(img, quad_pts, a):
        lo = np.floor(quad_pts.min(axis=0)).astype(int) - 1
        hi = np.ceil(quad_pts.max(axis=0)).astype(int) + 1
        lo = np.maximum(lo, [-4, -4])
        hi = np.minimum(hi, [canvas.width + 4, canvas.height + 4])
        w, h = int(hi[0] - lo[0]), int(hi[1] - lo[1])
        if w < 2 or h < 2:
            return
        if hi[0] <= 0 or hi[1] <= 0 or lo[0] >= canvas.width or lo[1] >= canvas.height:
            return
        src = [(0, 0), (img.width, 0), (img.width, img.height), (0, img.height)]
        try:
            coeffs = _homography([tuple(p - lo) for p in quad_pts], src)
        except np.linalg.LinAlgError:
            return
        warped = img.transform((w, h), Image.PERSPECTIVE, coeffs, resample)
        if a < 1:
            warped.putalpha(warped.getchannel("A").point(lambda v: int(v * a)))
        canvas.alpha_composite(warped, (int(lo[0]), int(lo[1])))

    if shadow is not None and thickness > 0:
        sunk = corners.copy()
        sunk[:, 2] -= thickness
        low, dlow = cam.project(sunk)
        if (dlow >= 1.0).all():
            blit(shadow, low, alpha)
    blit(texture, pts, alpha)
    return True


def depth_shade(img: Image.Image, t: float) -> Image.Image:
    """Push a texture toward the backdrop - aerial perspective for far planes."""
    if t <= 0.002:
        return img
    haze = Image.new("RGBA", img.size, (*BG_BOTTOM, 255))
    out = Image.blend(img.convert("RGBA"), haze, min(t, 0.85))
    out.putalpha(img.getchannel("A"))
    return out


# --------------------------------------------------------------------------
# point clouds
# --------------------------------------------------------------------------
def splat(W: int, H: int, layers, gain: float = 0.62, radius: float = 0.0) -> Image.Image:
    """Accumulate coloured points into an RGBA image.

    ``layers`` is a list of (screen_xy, rgb_0_255, weight) where rgb is either
    one triple for the whole layer or one row per point.

    Binning happens straight at output resolution: supersampling and averaging
    back down costs four times the memory traffic and buys nothing here, since
    a bin20 point is about one pixel wide anyway.  ``gain`` sets how opaque a
    lone point is, so a sparse cloud stays visible while dense tissue
    saturates.  A flat-coloured layer skips the three colour accumulators.
    """
    n = H * W
    acc = None
    cnt = np.zeros(n, np.float32)
    flat = None
    for xy, rgb, weight in layers:
        if len(xy) == 0:
            continue
        xi = np.rint(xy[:, 0]).astype(np.int64)
        yi = np.rint(xy[:, 1]).astype(np.int64)
        keep = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if not keep.any():
            continue
        idx = yi[keep] * W + xi[keep]
        rgb = np.asarray(rgb, np.float32)
        w = float(weight)
        cnt += np.bincount(idx, weights=np.full(len(idx), w, np.float32), minlength=n)
        if rgb.ndim == 1:
            flat = rgb if flat is None else flat        # one colour for the layer
            continue
        if acc is None:
            acc = np.zeros((n, 3), np.float32)
        c = rgb[keep]
        for k in range(3):
            acc[:, k] += np.bincount(idx, weights=c[:, k] * w, minlength=n)
    a = np.clip(cnt * gain, 0, 1).reshape(H, W)
    if acc is None:
        col = np.broadcast_to(np.asarray(flat if flat is not None else (255, 255, 255),
                                         np.float32), (H, W, 3))
    else:
        col = np.where(cnt[:, None] > 1e-6, acc / np.maximum(cnt, 1e-6)[:, None],
                       0.0).reshape(H, W, 3)
    rgba = np.concatenate([np.clip(col, 0, 255), (a * 255)[..., None]], axis=2)
    img = Image.fromarray(rgba.astype(np.uint8), "RGBA")
    if radius:
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return img


# --------------------------------------------------------------------------
# small HUD pieces
# --------------------------------------------------------------------------
def glow_line(d: ImageDraw.ImageDraw, p0, p1, color, width=2, alpha=255):
    d.line([tuple(p0), tuple(p1)], fill=(*color, alpha), width=width)


def chip(d: ImageDraw.ImageDraw, xy, label, color, size=17, pad=(10, 5)):
    x, y = xy
    w = tw(label, size, True, True) + pad[0] * 2
    h = size + pad[1] * 2
    d.rounded_rectangle((x, y, x + w, y + h), radius=(h // 2), fill=(*color, 34),
                        outline=(*color, 150), width=1)
    text(d, (x + pad[0], y + pad[1] - 1), label, size, (*color, 255), True, condensed=True)
    return w


def progress(d: ImageDraw.ImageDraw, box, t, marks=(), color=(120, 190, 255)):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=(y1 - y0) / 2, fill=(*RULE, 210))
    if t > 0:
        d.rounded_rectangle((x0, y0, x0 + (x1 - x0) * clamp01(t), y1),
                            radius=(y1 - y0) / 2, fill=(*color, 255))
    for m in marks:
        mx = x0 + (x1 - x0) * m
        d.line((mx, y0 - 3, mx, y1 + 3), fill=(*DIM, 160), width=1)


def scrim(W: int, H: int) -> Image.Image:
    """Darken the two text bands.  Act 3 zooms into bright fluorescence right
    where the readout sits, and a HUD that is only legible over a dark plate is
    not a HUD."""
    a = np.zeros(H, np.float32)
    top = np.linspace(1.0, 0.0, 165) ** 1.6 * 0.72
    bot = np.linspace(0.0, 1.0, 245) ** 1.5 * 0.80
    a[:165] = top
    a[H - 245:] = np.maximum(a[H - 245:], bot)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = BG_TOP
    rgba[..., 3] = np.repeat((a * 255)[:, None], W, axis=1).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def vignette(W: int, H: int, strength=0.34) -> Image.Image:
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    a = np.clip((r - 0.62) / 0.75, 0, 1) ** 2 * strength
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")
