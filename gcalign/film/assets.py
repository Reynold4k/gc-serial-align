"""Turn a finished run into the film's textures.

Nothing here recomputes a registration. Every texture is cut from what the run
stored, read through `gcalign.runio.Run`:

  run.labels(name)      the germinal-centre id raster of every plane
  run.rgb(name)         the warped stain, red/green/blue per ``cfg.rgb_channels``
                        with the run's shared display limits - the same picture
                        the figures and the viewer show
  run.tissue(name)      the section outline (stored, or re-cut for runs older
                        than 0.2 - see ``tissue_stored``)
  run.local(name)       each IF section still in its own frame, for the ICP replay
  run.icp(name)         the ICP iterations as transforms (stored, or rebuilt and
                        verified against the stored transform)
  run.identity          one colour per germinal centre, shared with every view

Textures are built once at ``scale`` times the grid resolution, with outlines
anti-aliased at that resolution, so a section still has clean edges when the
camera pushes in to several times its full-frame size.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .. import theme
from ..runio import Run
from .core import Texture

SLAB_FRACTION = 0.17        # drawn slab thickness as a fraction of the drawn pitch
STACK_ASPECT = 0.44         # drawn stack height / tissue width, for the automatic z factor


def _rgb(hexcolour: str) -> np.ndarray:
    return np.asarray(theme.hex_to_rgb(hexcolour), np.float32)


# --------------------------------------------------------------------------
# raster helpers (all at texture resolution)
# --------------------------------------------------------------------------
def _resize(arr: np.ndarray, shape, resample=Image.BILINEAR) -> np.ndarray:
    """Resample a float plane to ``shape`` (rows, cols), pixel centres aligned."""
    img = Image.fromarray(np.ascontiguousarray(arr, np.float32))     # float32, 2D: Pillow mode "F"
    return np.asarray(img.resize((shape[1], shape[0]), resample), np.float32)


def soft_mask(mask: np.ndarray, scale: int, sigma: float = 0.7) -> np.ndarray:
    """An anti-aliased alpha for a binary mask, upsampled ``scale`` times.

    The mask is blurred slightly at its own resolution, upsampled, and pushed
    back through a steep ramp: the outline comes out smooth instead of a
    staircase of 10 um pixels, and its last texture pixel is fractional."""
    m = ndi.gaussian_filter(np.asarray(mask, np.float32), sigma)
    up = _resize(m, (mask.shape[0] * scale, mask.shape[1] * scale))
    return np.clip((up - 0.5) * 2.6 + 0.5, 0.0, 1.0)


def inner_band(alpha: np.ndarray, width_px: int) -> np.ndarray:
    """A line of ``width_px`` texture pixels just inside an alpha's edge."""
    hard = alpha > 0.5
    inner = ndi.binary_erosion(hard, structure=np.ones((3, 3), bool), iterations=width_px)
    inner = ndi.gaussian_filter(inner.astype(np.float32), 0.6)
    return np.clip(alpha - inner, 0.0, 1.0)


def upsample_labels(labels: np.ndarray, scale: int, alpha: np.ndarray) -> np.ndarray:
    """Labels at texture resolution, filled out to the smoothed outline."""
    lab = np.asarray(labels).repeat(scale, 0).repeat(scale, 1)
    want = alpha > 0.02
    missing = want & (lab == 0)
    if missing.any():
        rows, cols = np.nonzero(want)
        r0, r1 = max(rows.min() - 4, 0), rows.max() + 5
        c0, c1 = max(cols.min() - 4, 0), cols.max() + 5
        crop = lab[r0:r1, c0:c1]
        _, (ri, ci) = ndi.distance_transform_edt(crop == 0, return_indices=True)
        filled = crop[ri, ci]
        crop[missing[r0:r1, c0:c1]] = filled[missing[r0:r1, c0:c1]]
    lab[~want] = 0
    return lab


def colour_field(labels: np.ndarray, lut: dict, default) -> np.ndarray:
    """(H, W, 3) float colours from an integer label raster and {id: rgb}."""
    lo, hi = int(labels.min()), int(labels.max())
    table = np.tile(np.asarray(default, np.float32), (hi - lo + 1, 1))
    for gid, rgb in lut.items():
        if lo <= gid <= hi:
            table[gid - lo] = rgb
    return table[labels - lo]


def glow(alpha_grid: np.ndarray, colour_grid: np.ndarray, scale: int, sigma: float):
    """A soft halo of each structure's own colour, computed at grid resolution
    and upsampled (it is a blur, so the extra resolution would buy nothing)."""
    a = ndi.gaussian_filter(alpha_grid.astype(np.float32), sigma)
    weighted = np.stack([ndi.gaussian_filter(alpha_grid * colour_grid[..., k], sigma)
                         for k in range(3)], -1)
    col = weighted / np.maximum(a, 1e-4)[..., None]
    shape = (alpha_grid.shape[0] * scale, alpha_grid.shape[1] * scale)
    a_up = _resize(a, shape)
    col_up = np.stack([_resize(col[..., k], shape) for k in range(3)], -1)
    return a_up, col_up


def compose(shape, layers) -> Image.Image:
    """Stack (alpha, colour) layers, source-over, into one RGBA image."""
    rgb = np.zeros(tuple(shape) + (3,), np.float32)      # premultiplied
    acc = np.zeros(tuple(shape), np.float32)
    for alpha, colour in layers:
        a = np.clip(np.asarray(alpha, np.float32), 0, 1)
        col = np.asarray(colour, np.float32)
        if col.ndim == 1:
            col = col[None, None, :]
        rgb = col * a[..., None] + rgb * (1 - a[..., None])
        acc = a + acc * (1 - a)
    out = np.empty(tuple(shape) + (4,), np.uint8)
    out[..., :3] = np.clip(rgb / np.maximum(acc, 1e-6)[..., None], 0, 255)
    out[..., 3] = np.clip(acc * 255 + 0.5, 0, 255)
    return Image.fromarray(out)                           # uint8, four channels: Pillow mode "RGBA"


def screen(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Screen blend, 0..255: light adds without clipping to white."""
    return 255.0 - (255.0 - base) * (255.0 - layer) / 255.0


# --------------------------------------------------------------------------
class Assets:
    """Every texture and number the film draws, from one finished run."""

    def __init__(self, run: Run, scale: int = 3, z_exaggeration: float | None = None,
                 log=print):
        self.run = run
        self.cfg = cfg = run.cfg
        self.scale = int(scale)
        say = log or (lambda *args, **kwargs: None)

        self.planes = list(run.planes)
        self.names = [p.name for p in self.planes]
        self.kind = {p.name: p.kind for p in self.planes}
        self.if_names = [p.name for p in self.planes if p.kind == "IF"]
        self.st_names = [p.name for p in self.planes if p.kind == "ST"]
        index = {p.name: p.index for p in self.planes}
        self.neighbours = {n: list(cfg.neighbours.get(index[n], [])) for n in self.if_names}
        self.grid = g = run.grid
        self.tex_shape = (g.height * self.scale, g.width * self.scale)

        # ---- outlines and framing --------------------------------------
        self.tissue = {n: np.asarray(run.tissue(n), bool) for n in self.names}
        self.tissue_stored = {n: bool(run.tissue_is_stored(n)) for n in self.names}
        union = np.logical_or.reduce([self.tissue[n] for n in self.names])
        rows, cols = np.nonzero(union)
        if len(rows) == 0:
            rows, cols = np.array([0, g.height - 1]), np.array([0, g.width - 1])
        x_lo, x_hi = g.x0 + (cols.min() - 0.5) * g.um, g.x0 + (cols.max() + 0.5) * g.um
        y_lo, y_hi = g.y0 + (rows.min() - 0.5) * g.um, g.y0 + (rows.max() + 0.5) * g.um
        self.centre = np.array([(x_lo + x_hi) / 2, -(y_lo + y_hi) / 2])   # world X, Y
        self.half_w = (x_hi - x_lo) / 2
        self.half_h = (y_hi - y_lo) / 2
        self.half = max(self.half_w, self.half_h)

        # ---- the cutting axis -------------------------------------------
        self.pitch_um = float(cfg.section_thickness_um)
        z_all = [p.z_um for p in self.planes]
        span = max(z_all) - min(z_all)
        if z_exaggeration is None:
            raw = STACK_ASPECT * 2 * self.half / max(span, self.pitch_um)
            z_exaggeration = max(5.0, round(raw / 5.0) * 5.0)
        self.z_exag = float(z_exaggeration)
        self.z0 = min(z_all)
        self.z = {p.name: (p.z_um - self.z0) * self.z_exag for p in self.planes}
        self.z_mid = (max(self.z.values()) + min(self.z.values())) / 2
        self.slab = SLAB_FRACTION * self.pitch_um * self.z_exag

        # ---- colour --------------------------------------------------------
        self.ids = list(run.gc_ids)
        self.identity_hex = dict(run.identity)
        self.identity = {int(k): _rgb(v) for k, v in self.identity_hex.items()}
        self.other = _rgb(theme.MUTED)                       # IF-only objects
        self.family_hex = {
            "ST": theme.kind_colours(self.ids, "ST", cfg.viewer.st_hex),
            "IF": theme.kind_colours(self.ids, "IF", cfg.viewer.if_hex),
        }
        self.family = {k: {int(i): _rgb(h) for i, h in v.items()} for k, v in self.family_hex.items()}
        self.family_base = {"ST": cfg.viewer.st_hex, "IF": cfg.viewer.if_hex}
        self.limits = run.rgb_limits() if self.if_names else {}

        # ---- textures -------------------------------------------------------
        self.tex, self.edge, self.overlay, self.view = {}, {}, {}, {}
        for name in self.names:
            self._plane(name)
            say(f"  texture {name}", flush=True)
        self.local = {}
        for name in self.if_names:
            self.local[name] = self._local(name)
            say(f"  ICP {name}: {self.local[name]['n_icp']} iterations "
                f"({self.local[name]['icp_origin']})", flush=True)

    # ---- geometry --------------------------------------------------------
    def corners(self, name: str | None = None, z: float | None = None) -> np.ndarray:
        from .core import grid_corners
        return grid_corners(self.grid, self.z[name] if z is None else z)

    # ---- per-plane textures ------------------------------------------------
    def _gc_layers(self, labels: np.ndarray):
        s = self.scale
        gc_grid = labels != 0
        gc_a = soft_mask(gc_grid, s)
        lab_up = upsample_labels(labels, s, gc_a)
        return gc_grid, gc_a, lab_up

    def _plane(self, name: str) -> None:
        s, shape = self.scale, self.tex_shape
        kind = self.kind[name]
        labels = np.asarray(self.run.labels(name))
        tissue = self.tissue[name]
        t_a = soft_mask(tissue, s)
        outline = inner_band(t_a, 2)
        gc_grid, gc_a, lab_up = self._gc_layers(labels)
        col = colour_field(lab_up, self.identity, self.other)
        accent = _rgb(theme.KIND[kind])
        edge_col = _rgb(theme.mix(theme.TISSUE_EDGE, theme.KIND[kind], 0.22))
        ring = inner_band(gc_a, 2)

        if kind == "ST":
            body = np.broadcast_to(_rgb(theme.TISSUE), shape + (3,))
            col_grid = colour_field(labels, self.identity, self.other)
            halo_a, halo_col = glow(gc_grid.astype(np.float32), col_grid, s, 2.0)
            self.tex[name] = Texture(compose(shape, [
                (t_a, body),
                (outline * 0.55, accent),
                (np.clip(halo_a * 0.55, 0, 1) * (1 - gc_a) * t_a, halo_col),
                (gc_a * 0.97, col),
                (ring * 0.55, np.clip(col * 1.18 + 18, 0, 255)),
            ]))
            # in the matching close-up the transcriptomics GCs are drawn as
            # outlines over the stain: a fill would hide what it is checked against
            self.overlay[name] = Texture(compose(shape, [
                (gc_a * 0.07, col),
                (ring * 0.98, np.clip(col * 1.15 + 20, 0, 255)),
            ]))
        else:
            stain = np.clip(self.run.rgb(name, limits=self.limits), 0, 1) * 255.0
            stain_up = np.stack([_resize(stain[..., k], shape) for k in range(3)], -1)
            body = screen(np.broadcast_to(_rgb(theme.TISSUE), shape + (3,)), stain_up)
            self.tex[name] = Texture(compose(shape, [
                (t_a, body),
                (outline * 0.70, accent),
                (gc_a * 0.10, col),
                (ring * 0.95, np.clip(col * 1.1 + 10, 0, 255)),
            ]))

        self.edge[name] = Texture(compose(shape, [(t_a, edge_col)]))
        # the viewer's look for the closing turn: glass sheets, GCs in the
        # modality's colour family, so the columns read through the stack
        fam = self.family[kind]
        fcol = colour_field(lab_up, fam, self.other)
        self.view[name] = Texture(compose(shape, [
            (t_a * 0.20, _rgb(theme.mix(theme.BG, theme.KIND[kind], 0.10))),
            (outline * 0.50, accent),
            (gc_a * 0.95, fcol),
        ]))

    # ---- the IF section in its own frame, for the ICP replay ----------------
    def local_rgb(self, name: str, local=None) -> np.ndarray:
        """The stain of an IF section before registration, on the display scale of
        ``run.rgb`` (`Run.local_rgb`), so the section looks the same before and
        after it is warped and only its geometry changes."""
        return self.run.local_rgb(name, limits=self.limits)

    def _local(self, name: str) -> dict:
        s = self.scale
        local = self.run.local(name)
        tissue = np.asarray(local["tissue"], bool)
        objects = np.asarray(local["gc_labels"], np.int32)
        shape = (tissue.shape[0] * s, tissue.shape[1] * s)
        # match_report's `assigned` is keyed by regionprops label (1-based)
        assigned = {int(k): int(v) for k, v in self.run.match[name].get("assigned", {}).items()}
        lut = np.zeros(max(int(objects.max()), max(assigned, default=0)) + 1, np.int32)
        for label, gid in assigned.items():
            if label < len(lut):
                lut[label] = gid
        unmatched = (objects > 0) & (lut[objects] == 0)
        ids = lut[objects]
        ids[unmatched] = -1                                   # IF-only: neutral grey
        t_a = soft_mask(tissue, s)
        stain = self.local_rgb(name, local) * 255.0
        stain_up = np.stack([_resize(stain[..., k], shape) for k in range(3)], -1)
        body = screen(np.broadcast_to(_rgb(theme.TISSUE), shape + (3,)), stain_up)
        # before the match the section carries no identities: the stain shows
        # its germinal centres, and the centroids are drawn as points on top
        tex = Texture(compose(shape, [
            (t_a, body),
            (inner_band(t_a, 2) * 0.70, _rgb(theme.IF)),
        ]))
        edge = Texture(compose(shape, [(t_a, _rgb(theme.mix(theme.TISSUE_EDGE, theme.IF, 0.22)))]))

        icp = self.run.icp(name)
        steps = list(icp.get("steps", []))
        source = np.asarray(icp.get("source_centroids_um",
                                    self.run.transforms[name]["if_centroids_um"]), float)
        history = []
        for step in steps[1:]:
            pairs = len(step.get("matched_if_index", []))
            dist = step.get("mean_distance_before_update_um")
            history.append((pairs, float(dist) if dist is not None else float("nan")))
        matched = np.zeros(len(source), bool)
        for label in assigned:
            if 1 <= label <= len(source):
                matched[label - 1] = True
        return {
            "tex": tex, "edge": edge, "shape": tissue.shape,
            "centroids_um": source, "matched": matched,
            "steps": steps, "n_icp": max(len(steps) - 1, 0), "history": history,
            "icp_origin": str(icp.get("origin", "stored")),
            "icp_errors": {k: icp[k] for k in ("max_linear_error", "max_offset_error_um",
                                               "max_history_error") if k in icp},
        }

    # ---- act 3 close-up window ------------------------------------------------
    def zoom_window(self, name: str, half_um: float):
        """The window holding the most paired GCs at the end of the ICP, so the
        refinement is watched where there is something to watch."""
        steps = self.local[name]["steps"]
        targets = np.asarray(steps[-1].get("target_points_um", []), float) if steps else np.empty((0, 2))
        if len(targets) == 0:
            return None, 0
        best, count = targets.mean(axis=0), -1
        for p in targets:
            n = int((np.abs(targets - p).max(axis=1) < half_um).sum())
            if n > count:
                best, count = p, n
        inside = targets[(np.abs(targets - best).max(axis=1) < half_um)]
        return inside.mean(axis=0), count
