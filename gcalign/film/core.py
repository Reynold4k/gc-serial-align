"""Camera, geometry and drawing primitives for the film.

A real pinhole camera (look-at plus perspective divide), not a fixed oblique
projection: that is what lets one continuous move go from the top view, out
into the stack, down onto a single section and round it.

Three primitives draw the whole film:

  ``Camera``      eye, target and field of view; projects world micrometres
                  to pixels
  ``draw_slab``   a section: its cut edge, then its texture mapped onto the
                  projected quad through a homography
  ``splat``       the transcriptomics point clouds, accumulated with bincount
                  so a frame of several hundred thousand bins costs
                  milliseconds

World units are micrometres. x and y are the run's common grid; z is the
cutting axis, exaggerated so the sections can be told apart. Image rows grow
downwards, as in every figure of a run, so world Y is the negated grid y and
screen-up is +Y (see ``to_world``).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------
# easing
# --------------------------------------------------------------------------
def clamp01(t: float) -> float:
    return 0.0 if t < 0 else 1.0 if t > 1 else float(t)


def window(t: float, start: float, length: float) -> float:
    """Progress 0..1 of ``t`` through [start, start + length]."""
    if length <= 0:
        return 1.0 if t >= start else 0.0
    return clamp01((t - start) / length)


def smooth(t: float) -> float:
    """Smoothstep: zero velocity at both ends."""
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def smoother(t: float) -> float:
    """Quintic smootherstep: zero acceleration at both ends too, for the long
    camera moves that would otherwise show a kick as they start."""
    t = clamp01(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def smooth_array(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
class Camera:
    """Orbit camera: azimuth, elevation and distance around a target.

    Elevation 90 degrees looks straight down. ``right`` is defined from the
    azimuth alone, so the basis never degenerates at the pole and the frame can
    tilt continuously out of the top view.
    """

    def __init__(self, W: int, H: int, target, azimuth: float = 270.0,
                 elevation: float = 90.0, distance: float = 20000.0,
                 fov: float = 18.0, shift=(0.0, 0.0)):
        self.W, self.H = int(W), int(H)
        self.shift = np.asarray(shift, float)       # pixels, after projection
        self.target = np.asarray(target, float)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.distance = float(distance)
        self.fov = float(fov)

    def copy(self) -> "Camera":
        return Camera(self.W, self.H, self.target.copy(), self.azimuth, self.elevation,
                      self.distance, self.fov, tuple(self.shift))

    @property
    def focal(self) -> float:
        return (self.H / 2) / math.tan(math.radians(self.fov) / 2)

    def basis(self):
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        eye = self.target + self.distance * d
        fwd = -d
        right = np.array([-math.sin(az), math.cos(az), 0.0])
        up = np.cross(right, fwd)
        return eye, right, up, fwd

    def project(self, P):
        """(N, 3) world -> (N, 2) pixels and (N,) depth along the view axis."""
        P = np.atleast_2d(np.asarray(P, float))
        eye, right, up, fwd = self.basis()
        v = P - eye
        zc = v @ fwd
        safe = np.maximum(zc, 1.0)
        f = self.focal
        x = self.W / 2 + f * (v @ right) / safe + self.shift[0]
        y = self.H / 2 - f * (v @ up) / safe + self.shift[1]
        return np.stack([x, y], axis=1), zc

    def px_per_um(self) -> float:
        """Screen pixels per micrometre at the target, for a scale bar."""
        return self.focal / self.distance

    def distance_for(self, half_um: float, screen_fraction: float, axis: str = "w") -> float:
        """The distance that makes ``half_um`` span ``screen_fraction`` of the
        frame's width (or height): holds the framing while the lens changes."""
        span = self.W if axis == "w" else self.H
        return self.focal * half_um / (span * screen_fraction / 2)


def interp_camera(a: Camera, b: Camera, t: float) -> Camera:
    """Blend two camera states. Azimuth takes the short way round; distance
    moves in log space, or a long dolly spends most of its time at the far end
    and reads as a stall."""
    daz = (b.azimuth - a.azimuth + 180) % 360 - 180
    c = a.copy()
    c.target = lerp(a.target, b.target, t)
    c.azimuth = a.azimuth + daz * t
    c.elevation = lerp(a.elevation, b.elevation, t)
    c.fov = lerp(a.fov, b.fov, t)
    c.distance = math.exp(lerp(math.log(a.distance), math.log(b.distance), t))
    c.shift = lerp(a.shift, b.shift, t)
    return c


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def to_world(xy_um, z) -> np.ndarray:
    """Grid micrometres (x right, y down) -> render world (x, -y, z)."""
    xy = np.asarray(xy_um, float).reshape(-1, 2)
    zz = np.broadcast_to(np.asarray(z, float), (len(xy),))
    return np.column_stack([xy[:, 0], -xy[:, 1], zz])


def grid_corners(grid, z: float) -> np.ndarray:
    """World corners of a plane on the run's grid, texture pixel (0, 0) first.

    Pixel (r, c) is centred on (x0 + c um, y0 + r um), so the raster's outer
    edge sits half a pixel outside the first and last centres."""
    u = grid.um
    x_lo, x_hi = grid.x0 - u / 2, grid.x0 + (grid.width - 0.5) * u
    y_lo, y_hi = grid.y0 - u / 2, grid.y0 + (grid.height - 0.5) * u
    return to_world([[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]], z)


def local_corners(shape, um: float, linear, offset, z: float) -> np.ndarray:
    """Corners of an IF section still in its own frame, placed by one
    similarity transform (local micrometres -> grid micrometres).

    Composed with the camera's homography this costs one texture transform per
    frame, so a section mid-ICP is drawn at exactly the transform the optimiser
    held at that iteration."""
    h, w = shape
    px = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
    local = px * float(um)
    placed = local @ np.asarray(linear, float).T + np.asarray(offset, float)
    return to_world(placed, z)


def place(points_um, linear, offset) -> np.ndarray:
    """Apply a similarity transform to local micrometre points."""
    return np.asarray(points_um, float) @ np.asarray(linear, float).T + np.asarray(offset, float)


# --------------------------------------------------------------------------
# textured quads
# --------------------------------------------------------------------------
def homography(dst_quad, src_quad) -> np.ndarray:
    """PIL PERSPECTIVE coefficients mapping output pixels back to the texture."""
    A, B = [], []
    for (ox, oy), (sx, sy) in zip(dst_quad, src_quad):
        A.append([ox, oy, 1, 0, 0, 0, -sx * ox, -sx * oy])
        A.append([0, 0, 0, ox, oy, 1, -sy * ox, -sy * oy])
        B += [sx, sy]
    return np.linalg.solve(np.asarray(A, float), np.asarray(B, float))


class Texture:
    """A section texture with pre-filtered smaller copies (a rip-map).

    A slab seen at 20 degrees elevation is squeezed about seven times along
    one texture axis and two to three times along the other. Bilinear sampling
    of the full texture at that ratio skips texels, so outlines break up and
    the stain crawls as the camera moves. Each copy here is box-filtered by
    1, 2 or 4 independently along x and y, and every frame samples the copy
    whose texel spacing is closest to one screen pixel along each axis."""

    FACTORS = (1, 2, 4)

    def __init__(self, image: Image.Image):
        self.size = image.size
        self.levels = {(1, 1): image}
        premultiplied = image.convert("RGBa")
        for fx in self.FACTORS:
            for fy in self.FACTORS:
                if (fx, fy) != (1, 1):
                    self.levels[(fx, fy)] = premultiplied.reduce((fx, fy)).convert("RGBA")

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def pick(self, ratio_x: float, ratio_y: float) -> Image.Image:
        def factor(r):
            return 4 if r >= 5.0 else 2 if r >= 2.5 else 1
        return self.levels[(factor(ratio_x), factor(ratio_y))]


_ALPHA_LUTS: dict = {}


def scale_alpha(img: Image.Image, a: float) -> Image.Image:
    """Multiply an RGBA image's alpha by ``a`` (in place on a copy)."""
    if a >= 0.999:
        return img
    key = int(round(a * 1000))
    lut = _ALPHA_LUTS.get(key)
    if lut is None:
        lut = _ALPHA_LUTS[key] = [int(v * key / 1000 + 0.5) for v in range(256)]
    out = img.copy()
    out.putalpha(img.getchannel("A").point(lut))
    return out


def blit_quad(canvas: Image.Image, quad_px: np.ndarray, texture,
              alpha: float = 1.0, resample=Image.BILINEAR) -> None:
    """Map ``texture`` (an Image or a `Texture`) onto the screen quad
    ``quad_px`` and composite it."""
    lo = np.floor(quad_px.min(axis=0)).astype(int) - 1
    hi = np.ceil(quad_px.max(axis=0)).astype(int) + 1
    if hi[0] <= 0 or hi[1] <= 0 or lo[0] >= canvas.width or lo[1] >= canvas.height:
        return
    lo = np.maximum(lo, [-2, -2])
    hi = np.minimum(hi, [canvas.width + 2, canvas.height + 2])
    w, h = int(hi[0] - lo[0]), int(hi[1] - lo[1])
    if w < 2 or h < 2:
        return
    if isinstance(texture, Texture):
        q = quad_px
        along_x = max(np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[3])), 1e-3)
        along_y = max(np.hypot(*(q[3] - q[0])), np.hypot(*(q[2] - q[1])), 1e-3)
        texture = texture.pick(texture.width / along_x, texture.height / along_y)
    src = [(0, 0), (texture.width, 0), (texture.width, texture.height), (0, texture.height)]
    try:
        coeffs = homography([tuple(p - lo) for p in quad_px], src)
    except np.linalg.LinAlgError:
        return
    warped = texture.transform((w, h), Image.PERSPECTIVE, tuple(coeffs), resample)
    if alpha < 0.999:
        warped = scale_alpha(warped, alpha)
    canvas.alpha_composite(warped, (int(lo[0]), int(lo[1])))


def draw_slab(canvas: Image.Image, cam: Camera, corners, texture: Image.Image,
              edge: Image.Image | None = None, thickness: float = 0.0,
              alpha: float = 1.0) -> bool:
    """Draw one section. Returns False when it is behind the camera.

    ``corners`` are the world positions of texture pixels (0,0), (W,0), (W,H),
    (0,H). ``edge`` is the same silhouette in the cut-edge colour, drawn one
    slab thickness lower; where it shows past the top face it is the section's
    thickness, which is most of what sells the depth.
    """
    if alpha <= 0.004:
        return False
    corners = np.asarray(corners, float)
    pts, depth = cam.project(corners)
    if (depth < 1.0).any():
        return False
    if edge is not None and thickness > 0:
        low = corners.copy()
        low[:, 2] -= thickness
        lpts, ldepth = cam.project(low)
        if (ldepth >= 1.0).all():
            blit_quad(canvas, lpts, edge, alpha)
    blit_quad(canvas, pts, texture, alpha)
    return True


# --------------------------------------------------------------------------
# point clouds
# --------------------------------------------------------------------------
def splat(W: int, H: int, layers, gain: float = 0.6) -> Image.Image:
    """Accumulate coloured points into an RGBA image.

    ``layers`` is a list of (screen_xy, rgb, weight): rgb is one triple for the
    whole layer or one row per point. Binning happens straight at output
    resolution - a bin is about one pixel at the framing the film uses - and
    ``gain`` sets how opaque a lone point is, so a sparse cloud stays visible
    while dense tissue saturates.
    """
    n = H * W
    cnt = np.zeros(n, np.float32)
    per_point = any(np.asarray(rgb).ndim == 2 for _, rgb, _ in layers)
    acc = np.zeros((n, 3), np.float32) if per_point else None
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
        cnt += np.bincount(idx, minlength=n).astype(np.float32) * w
        if acc is None:
            flat = rgb
            continue
        colours = rgb[keep] if rgb.ndim == 2 else np.broadcast_to(rgb, (int(keep.sum()), 3))
        for k in range(3):
            acc[:, k] += np.bincount(idx, weights=colours[:, k] * w, minlength=n).astype(np.float32)
    a = np.clip(cnt * gain, 0, 1).reshape(H, W)
    if acc is None:
        col = np.broadcast_to(np.asarray(flat if flat is not None else (255, 255, 255),
                                         np.float32), (H, W, 3))
    else:
        col = (acc / np.maximum(cnt, 1e-6)[:, None]).reshape(H, W, 3)
    rgba = np.empty((H, W, 4), np.uint8)
    rgba[..., :3] = np.clip(col, 0, 255)
    rgba[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


# --------------------------------------------------------------------------
# an anti-aliased vector layer
# --------------------------------------------------------------------------
class Vectors:
    """Lines and markers drawn at twice the frame size and box-filtered down:
    PIL draws hard-edged lines, and a hairline over fluorescence needs edges."""

    SS = 2

    def __init__(self, W: int, H: int, region=None):
        if region is None:
            region = (0, 0, W, H)
        x0, y0, x1, y1 = (int(math.floor(region[0])), int(math.floor(region[1])),
                          int(math.ceil(region[2])), int(math.ceil(region[3])))
        self.x0, self.y0 = x0, y0
        self.size = (max(x1 - x0, 1), max(y1 - y0, 1))
        self.img = None

    def _draw(self):
        if self.img is None:
            self.img = Image.new("RGBA", (self.size[0] * self.SS, self.size[1] * self.SS),
                                 (0, 0, 0, 0))
            self.d = ImageDraw.Draw(self.img, "RGBA")
        return self.d

    def _xy(self, x, y):
        return (float(x) - self.x0) * self.SS, (float(y) - self.y0) * self.SS

    def line(self, pts, colour: str, alpha: float, width: float) -> None:
        if alpha <= 0.004 or len(pts) < 2:
            return
        self._draw().line([self._xy(x, y) for x, y in pts], fill=_rgba(colour, alpha),
                          width=max(1, int(round(width * self.SS))), joint="curve")

    def dot(self, xy, r: float, colour: str, alpha: float) -> None:
        if alpha <= 0.004:
            return
        x, y = self._xy(*xy)
        r = r * self.SS
        self._draw().ellipse((x - r, y - r, x + r, y + r), fill=_rgba(colour, alpha))

    def ring(self, xy, r: float, colour: str, alpha: float, width: float) -> None:
        if alpha <= 0.004:
            return
        x, y = self._xy(*xy)
        r = r * self.SS
        self._draw().ellipse((x - r, y - r, x + r, y + r), outline=_rgba(colour, alpha),
                             width=max(1, int(round(width * self.SS))))

    def composite(self, canvas: Image.Image) -> None:
        """Filter the layer down and lay it over ``canvas`` (RGBA or RGB)."""
        if self.img is not None:
            small = self.img.convert("RGBa").reduce(self.SS).convert("RGBA")
            if canvas.mode == "RGBA":
                canvas.alpha_composite(small, (self.x0, self.y0))
            else:
                canvas.paste(small, (self.x0, self.y0), small)


def _rgba(colour, alpha: float):
    """A theme hex colour, or an RGB tuple, with alpha, for PIL."""
    from .. import theme
    if isinstance(colour, str):
        return theme.rgba(colour, alpha)
    return (int(colour[0]), int(colour[1]), int(colour[2]), int(round(max(0.0, min(alpha, 1.0)) * 255)))
