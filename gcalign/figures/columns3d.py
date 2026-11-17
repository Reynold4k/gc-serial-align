"""3D figures of GC columns, rendered off-screen with PyVista on a black background.

A column is the closed 3D surface that the `columns` stage draws through one
GC's cross-sections. The meshes are drawn exactly as that stage built them,
with no subdivision and no smoothing, using smooth shading from per-vertex
normals, three lights (key, fill and rim) and a low specular highlight. Depth
is exaggerated by the viewer's default factor so that a short stack reads as a
column. Every tile uses the same camera and the same scale, so the sizes of
different GCs can be compared.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import theme
from . import _base as B

COLUMNS_FILE = "04_columns/gc_columns.npz"
PATCHES_FILE = "04_columns/gc_plane_patches.npz"
ELEVATION_DEG = 20.0
AZIMUTH_DEG = -60.0
TILE_ASPECT = 1.18                      # height / width of one column tile
MATERIAL = dict(ambient=0.12, diffuse=0.85, specular=0.20, specular_power=24.0)
# (position in camera space, intensity): key upper-left-front, soft fill from
# the right, rim from behind and below to lift the silhouette off the black
LIGHTS = (((-1.0, 1.3, 1.2), 0.95), ((1.4, -0.1, 0.5), 0.40), ((0.2, 0.9, -1.6), 0.60))
SLICE_OPACITY = 0.45

_RENDERER: dict[str, bool] = {}


def _px(points: float, dpi: float) -> float:
    return points * dpi / 72.0


def _note(message: str) -> None:
    print(f"[gcalign.figures] {message}", flush=True)


def renderer_available() -> bool:
    """True when this machine can render 3D off-screen; tested once per process.

    A missing OpenGL or EGL library can abort the Python interpreter instead of
    raising an error, so the test renders a sphere in a child process and only
    that process's exit code is used.
    """
    if "ok" not in _RENDERER:
        probe = ("import pyvista as pv; p = pv.Plotter(off_screen=True, window_size=[48, 48]); "
                 "p.add_mesh(pv.Sphere()); p.screenshot(return_img=True); p.close()")
        try:
            result = subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=180)
            _RENDERER["ok"] = result.returncode == 0
        except Exception:
            _RENDERER["ok"] = False
    return _RENDERER["ok"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_meshes(run) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The column meshes of a run as {GC id: (vertices in um, triangle indices)}; {} before the `columns` stage."""
    if not run.has(COLUMNS_FILE):
        return {}
    store = np.load(run.root / COLUMNS_FILE, allow_pickle=False)
    ids = [int(g) for g in store["gc_ids"]] if "gc_ids" in store.files else sorted(
        int(k[1:]) for k in store.files if k.startswith("v") and k[1:].isdigit())
    return {g: (np.asarray(store[f"v{g}"], float), np.asarray(store[f"f{g}"], np.int64))
            for g in ids if f"v{g}" in store.files and f"f{g}" in store.files}


def load_slices(run, gid: int) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """The measured cross-sections of one GC as (section name, outline xy in um, triangles, z in um)."""
    if not run.has(PATCHES_FILE):
        return []
    store = np.load(run.root / PATCHES_FILE, allow_pickle=False)
    out = []
    for name in run.names:
        index = 0
        while f"{gid}|{name}|{index}|xy" in store.files:
            out.append((name, np.asarray(store[f"{gid}|{name}|{index}|xy"], float),
                        np.asarray(store[f"{gid}|{name}|{index}|f"], np.int64),
                        float(np.asarray(store[f"{gid}|{name}|{index}|z"]).ravel()[0])))
            index += 1
    return out


def _cells(faces: np.ndarray) -> np.ndarray:
    return np.hstack([np.full((len(faces), 1), 3, np.int64), faces.astype(np.int64)]).ravel()


def _polydata(vertices: np.ndarray, faces: np.ndarray, z_scale: float):
    import pyvista as pv
    points = vertices.astype(float).copy()
    points[:, 2] *= z_scale
    mesh = pv.PolyData(points, _cells(faces))
    return mesh.compute_normals(auto_orient_normals=True, feature_angle=180.0,
                                split_vertices=False, consistent_normals=True)


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
def _view_axes(elevation: float = ELEVATION_DEG, azimuth: float = AZIMUTH_DEG):
    el, az = np.deg2rad(elevation), np.deg2rad(azimuth)
    towards_camera = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(np.array([0.0, 0.0, 1.0]), towards_camera)
    right /= np.linalg.norm(right)
    up = np.cross(towards_camera, right)
    return towards_camera, right, up


def _screen_box(points: np.ndarray):
    """Projected extent of a point set: (3D centre, half-width, half-height) on screen."""
    towards, right, up = _view_axes()
    u, v, w = points @ right, points @ up, points @ towards
    centre = ((u.min() + u.max()) / 2 * right + (v.min() + v.max()) / 2 * up
              + (w.min() + w.max()) / 2 * towards)
    return centre, (u.max() - u.min()) / 2, (v.max() - v.min()) / 2


def _render(add_actors, *, size: tuple[int, int], half_height: float, centre: np.ndarray,
            lit: bool = True, translucent: bool = False) -> np.ndarray:
    """Set up a black off-screen plotter, let `add_actors` fill it, return RGB uint8."""
    import pyvista as pv
    towards, _, _ = _view_axes()
    plotter = pv.Plotter(off_screen=True, window_size=[int(size[0]), int(size[1])], lighting="none")
    try:
        plotter.set_background(theme.BG)
        if translucent:
            plotter.enable_depth_peeling(number_of_peels=16)
        add_actors(plotter)
        distance = 20.0 * max(half_height, 1.0)
        plotter.camera_position = [tuple(centre + towards * distance), tuple(centre), (0.0, 0.0, 1.0)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = half_height
        if lit:
            for position, intensity in LIGHTS:
                light = pv.Light(light_type="camera light", intensity=intensity)
                light.position = position
                light.focal_point = (0.0, 0.0, 0.0)
                plotter.add_light(light)
        plotter.enable_anti_aliasing("ssaa")
        image = plotter.screenshot(return_img=True)
    finally:
        plotter.close()
    return np.asarray(image)[..., :3].astype(np.uint8)


def render_surface(mesh, colour: str, **kwargs) -> np.ndarray:
    return _render(lambda p: p.add_mesh(mesh, color=colour, smooth_shading=True, **MATERIAL), **kwargs)


def render_slices(run, slices, z_scale: float, **kwargs) -> np.ndarray:
    """Render the measured cross-sections of one GC as translucent discs, blue for IF and pink for ST."""
    import pyvista as pv

    def add(plotter):
        for name, xy, faces, z in slices:
            colour = B.kind_colour(run.plane(name).kind)
            points = np.column_stack([xy, np.full(len(xy), z * z_scale)])
            plotter.add_mesh(pv.PolyData(points, _cells(faces)), color=colour, opacity=SLICE_OPACITY,
                             ambient=1.0, diffuse=0.0, specular=0.0)
            ring = pv.lines_from_points(np.vstack([points, points[:1]]))
            plotter.add_mesh(ring, color=colour, line_width=2.0, ambient=1.0, diffuse=0.0)
    return _render(add, lit=False, translucent=True, **kwargs)


def _ranked_ids(run, meshes: dict) -> list[int]:
    """Columns by measured voxel volume, largest first."""
    if run.metrics is not None and "volume_um3" in run.metrics:
        table = run.metrics.set_index("merged_gc_id")["volume_um3"]
        known = [int(g) for g in table.sort_values(ascending=False).index if int(g) in meshes]
        return known + [g for g in sorted(meshes) if g not in known]
    return sorted(meshes, key=lambda g: -_polydata(*meshes[g], 1.0).volume)


# --------------------------------------------------------------------------
# 6. columns
# --------------------------------------------------------------------------
def columns(run, out=None, *, dpi: float = 220, fmt: str = "png", n: int = 8,
            ids=None, ncols: int = 4) -> Path | None:
    """Draw the largest GC columns by measured volume, one tile each, all at one scale."""
    run = B.as_run(run)
    out = Path(out) if out is not None else run.root / "06_figures" / f"columns.{fmt}"
    meshes = load_meshes(run)
    if not meshes:
        _note(f"skipped the columns figure: {COLUMNS_FILE} is missing (run the `columns` stage first)")
        return None
    chosen = [int(g) for g in ids if int(g) in meshes] if ids is not None else _ranked_ids(run, meshes)[:n]
    if not chosen:
        _note("skipped the columns figure: none of the requested GCs has a 3D mesh")
        return None
    if not renderer_available():
        _note("skipped the columns figure: this machine has no off-screen OpenGL renderer")
        return None
    z_scale = float(run.cfg.viewer.default_z_exaggeration)
    polys = {g: _polydata(*meshes[g], z_scale) for g in chosen}
    boxes = {g: _screen_box(np.asarray(polys[g].points)) for g in chosen}

    with B.styled():
        ncols = max(1, min(ncols, len(chosen)))
        nrows = int(np.ceil(len(chosen) / ncols))
        margin, gap = _px(B.MARGIN_PT, dpi), _px(B.GAP_PT, dpi)
        tile_w = int(round((B.FIGURE_WIDTH_IN * dpi - 2 * margin - (ncols - 1) * gap) / ncols))
        tile_h = int(round(tile_w * TILE_ASPECT))
        label_h = _px(B.BAND_PT, dpi)
        half = max(max(b[2], b[1] * TILE_ASPECT) for b in boxes.values()) * 1.06
        width = 2 * margin + ncols * tile_w + (ncols - 1) * gap
        height = margin * 0.6 + nrows * (tile_h + label_h) + (nrows - 1) * gap + margin * 0.4
        canvas = B.Canvas(width, height, dpi)
        for index, gid in enumerate(chosen):
            col, row = index % ncols, index // ncols
            x = margin + col * (tile_w + gap)
            y = margin * 0.6 + row * (tile_h + label_h + gap)
            image = render_surface(polys[gid], run.identity.get(gid, theme.INK), size=(tile_w, tile_h),
                                   half_height=half, centre=boxes[gid][0])
            canvas.raster(image, x, y)
            canvas.text(x + tile_w / 2, y + tile_h + label_h * 0.62, f"GC{gid}",
                        colour=theme.INK_2, ha="center")
        return canvas.save(out, fmt)


def column(run, gid: int, out=None, *, dpi: float = 220, fmt: str = "png") -> Path | None:
    """Draw one GC's column: its measured cross-sections beside the surface drawn through them.

    Left: the cross-sections as translucent discs at their depths, blue for IF
    sections and pink for ST sections. Right: the column surface in the GC's
    identity colour. Both use the same camera and scale.
    """
    run = B.as_run(run)
    gid = int(gid)
    out = Path(out) if out is not None else run.root / "06_figures" / "columns" / f"GC{gid}.{fmt}"
    meshes = load_meshes(run)
    if gid not in meshes:
        _note(f"skipped the column figure of GC{gid}: {COLUMNS_FILE} has no mesh for this GC")
        return None
    if not renderer_available():
        _note("skipped the column figure: this machine has no off-screen OpenGL renderer")
        return None
    z_scale = float(run.cfg.viewer.default_z_exaggeration)
    poly = _polydata(*meshes[gid], z_scale)
    slices = load_slices(run, gid)
    points = [np.asarray(poly.points)]
    for _, xy, _, z in slices:
        points.append(np.column_stack([xy, np.full(len(xy), z * z_scale)]))
    centre, half_w, half_h = _screen_box(np.vstack(points))
    with B.styled():
        margin, gap = _px(B.MARGIN_PT, dpi), _px(B.GAP_PT * 1.4, dpi)
        n_tiles = 2 if slices else 1
        tile_w = int(round((B.FIGURE_WIDTH_IN * 0.62 * dpi - 2 * margin - (n_tiles - 1) * gap) / n_tiles))
        tile_h = int(round(tile_w * TILE_ASPECT))
        foot = _px(B.FOOT_PT, dpi)
        width = 2 * margin + n_tiles * tile_w + (n_tiles - 1) * gap
        height = margin * 0.6 + tile_h + foot
        canvas = B.Canvas(width, height, dpi)
        half = max(half_h, half_w * TILE_ASPECT) * 1.08
        view = dict(size=(tile_w, tile_h), half_height=half, centre=centre)
        x = margin
        if slices:
            canvas.raster(render_slices(run, slices, z_scale, **view), x, margin * 0.6)
            x += tile_w + gap
        canvas.raster(render_surface(poly, run.identity.get(gid, theme.INK), **view), x, margin * 0.6)
        canvas.text(margin, height - _px(B.FOOT_BAR_PT, dpi) + _px(1.0, dpi), f"GC{gid}",
                    colour=theme.INK, weight="semibold")
        return canvas.save(out, fmt)
