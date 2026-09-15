"""Stage 5 - stack each GC's cross-sections into one smooth closed surface.

The smoothing happens in the voxel volume, not on the mesh: the iso-surface of a
blurred stack is already smooth, whereas a mesh smoother collapses columns this
thin. The iso-level is solved so the enclosed volume reproduces the measured
voxel count, which keeps the rounding cosmetic and the volume honest. Padding is
at least four sigma on every axis, otherwise marching cubes clips at the array
border and the surface never closes.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyvista as pv
from scipy import ndimage as ndi
from skimage import measure

from .config import Config, stage_dir

def outline_patches(cfg, mask: np.ndarray, x0: float, y0: float, z_um: float):
    """Per connected component: simplified outer contour + a flat triangulation."""
    from matplotlib.path import Path as MplPath
    from scipy.spatial import Delaunay

    patches = []
    labels, count = ndi.label(mask)
    for index in range(1, count + 1):
        piece = ndi.binary_fill_holes(labels == index)
        if piece.sum() < cfg.columns.min_px_per_plane:
            continue
        padded = np.pad(piece.astype(float), 1)
        contours = measure.find_contours(padded, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len) - 1.0
        tolerance = 0.45
        simple = measure.approximate_polygon(contour, tolerance)
        while len(simple) > 48 and tolerance < 6:
            tolerance *= 1.35
            simple = measure.approximate_polygon(contour, tolerance)
        if len(simple) < 4:
            continue
        if np.allclose(simple[0], simple[-1]):
            simple = simple[:-1]
        if len(simple) < 3:
            continue
        pts = np.column_stack([x0 + simple[:, 1] * cfg.grid_um, y0 + simple[:, 0] * cfg.grid_um])
        try:
            tri = Delaunay(pts)
        except Exception:
            continue
        centres = pts[tri.simplices].mean(axis=1)
        inside = MplPath(pts).contains_points(centres)
        faces = tri.simplices[inside]
        if len(faces) == 0:
            continue
        patches.append({
            "z_um": z_um,
            "xy_um": pts.astype(np.float32),
            "faces": faces.astype(np.int32),
            "area_um2": float(piece.sum() * cfg.grid_um ** 2),
        })
    return patches


def build_column(cfg, volume: np.ndarray, x0: float, y0: float, z0: float):
    """Oversample in z, Gaussian-smooth, march, decimate.

    The smoothing happens in the volume, not on the mesh: the iso-surface of a
    Gaussian-blurred stack is already smooth, and a mesh smoother on top of it
    collapses thin columns. Padding is >= 4 sigma on every axis so the surface
    closes instead of being clipped at the array border, which is what makes the
    meshes watertight and keeps the enclosed volume equal to the measured one.
    """
    padded = np.pad(volume.astype(np.float32), ((cfg.columns.pad_z, cfg.columns.pad_z), (cfg.columns.pad_xy, cfg.columns.pad_xy),
                                                (cfg.columns.pad_xy, cfg.columns.pad_xy)))
    n_z = padded.shape[0]
    zoomed = ndi.zoom(padded, (cfg.columns.z_oversample, 1, 1), order=1)
    z_gain = (n_z - 1) / (zoomed.shape[0] - 1)     # exact ndi.zoom index mapping
    smoothed = ndi.gaussian_filter(zoomed, (cfg.columns.sigma_z_sub, cfg.columns.sigma_xy_px, cfg.columns.sigma_xy_px))
    peak = float(smoothed.max())
    if peak < 0.30:
        return None

    # iso-level that reproduces the measured voxel count: the rounding is
    # cosmetic, the enclosed volume stays what the sections actually say
    target_voxels = float(padded.sum()) / z_gain
    order = np.sort(smoothed.ravel())[::-1]
    index = int(min(max(round(target_voxels), 1), order.size - 1))
    level = float(order[index])
    if not (0 < level < peak):
        level = 0.5 * peak
    try:
        verts, faces, _, _ = measure.marching_cubes(smoothed, level=level)
    except Exception:
        return None
    if len(faces) < 8:
        return None

    xyz = np.column_stack([
        x0 + (verts[:, 2] - cfg.columns.pad_xy) * cfg.grid_um,
        y0 + (verts[:, 1] - cfg.columns.pad_xy) * cfg.grid_um,
        z0 + (verts[:, 0] * z_gain - cfg.columns.pad_z) * cfg.section_thickness_um,
    ])
    mesh = pv.PolyData(xyz, np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64))
    if mesh.n_faces_strict > cfg.columns.target_faces:
        mesh = mesh.decimate(1 - cfg.columns.target_faces / mesh.n_faces_strict, volume_preservation=True)
    cells = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    if len(cells) < 8:
        return None
    shell = mesh
    if mesh.n_faces_strict > cfg.columns.shell_faces and mesh.is_all_triangles:
        shell = mesh.decimate(1 - cfg.columns.shell_faces / mesh.n_faces_strict, volume_preservation=True)
    shell_cells = np.asarray(shell.faces).reshape(-1, 4)[:, 1:]
    return (np.asarray(mesh.points, np.float32), cells.astype(np.int32),
            float(mesh.volume), int(mesh.n_open_edges),
            np.asarray(shell.points, np.float32), shell_cells.astype(np.int32))


def run(cfg: Config) -> None:
    OUT = stage_dir(cfg, "04_columns")
    PLANES = cfg.planes
    store = np.load(cfg.output_dir / "03_gc_match/plane_gc_labels.npz")
    x0, y0, height, width = store["grid"]
    x0, y0 = float(x0), float(y0)
    labels = {name: store[name] for name, _, _ in PLANES}

    gc_ids = sorted({int(v) for lab in labels.values() for v in np.unique(lab) if v > 0})
    print(f"{len(gc_ids)} merged GCs on the stack")

    columns, patches_out, rows = {}, {}, []
    for gid in gc_ids:
        masks = np.stack([labels[name] == gid for name, _, _ in PLANES])
        present = [i for i in range(9) if masks[i].sum() >= cfg.columns.min_px_per_plane]
        if len(present) < cfg.columns.min_planes:
            continue
        ys, xs = np.nonzero(masks.any(axis=0))
        pad = 4
        r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, masks.shape[1])
        c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, masks.shape[2])
        sub = masks[:, r0:r1, c0:c1]

        built = build_column(cfg, sub, x0 + c0 * cfg.grid_um, y0 + r0 * cfg.grid_um, 0.0)
        if built is None:
            continue
        verts, faces, mesh_volume, open_edges, shell_v, shell_f = built
        columns[gid] = {"verts": verts, "faces": faces,
                        "shell_verts": shell_v, "shell_faces": shell_f}

        per_plane = []
        for index, (name, kind, section) in enumerate(PLANES):
            z_um = (section - 1) * cfg.section_thickness_um
            mask = masks[index]
            area_px = int(mask.sum())
            patch = outline_patches(cfg, mask[r0:r1, c0:c1], x0 + c0 * cfg.grid_um,
                                    y0 + r0 * cfg.grid_um, z_um) if area_px else []
            if patch:
                patches_out[f"{gid}|{name}"] = patch
            cy, cx = ndi.center_of_mass(mask) if area_px else (np.nan, np.nan)
            per_plane.append({
                "plane": name, "kind": kind, "z_um": z_um,
                "area_um2": area_px * cfg.grid_um ** 2,
                "equiv_radius_um": float(np.sqrt(area_px * cfg.grid_um ** 2 / np.pi)) if area_px else 0.0,
                "centroid_x_um": float(x0 + cx * cfg.grid_um) if area_px else None,
                "centroid_y_um": float(y0 + cy * cfg.grid_um) if area_px else None,
            })

        areas = np.array([p["area_um2"] for p in per_plane])
        radii = np.array([p["equiv_radius_um"] for p in per_plane if p["area_um2"] > 0])
        centres = np.array([[p["centroid_x_um"], p["centroid_y_um"]]
                            for p in per_plane if p["area_um2"] > 0])
        rows.append({
            "merged_gc_id": gid,
            "n_planes": len(present),
            "n_st_planes": sum(1 for i in present if PLANES[i][1] == "ST"),
            "n_if_planes": sum(1 for i in present if PLANES[i][1] == "IF"),
            "volume_um3": float(areas.sum() * cfg.section_thickness_um),
            "mesh_volume_um3": float(mesh_volume),
            "mesh_open_edges": int(open_edges),
            "max_area_um2": float(areas.max()),
            "max_equiv_radius_um": float(radii.max()),
            "median_equiv_radius_um": float(np.median(radii)),
            "radius_cv": float(radii.std() / radii.mean()),
            "centroid_drift_um": float(np.linalg.norm(centres - centres.mean(0), axis=1).max()),
            "z_span_um": float((present[-1] - present[0]) * cfg.section_thickness_um),
            "n_vertices": int(len(verts)),
            "n_faces": int(len(faces)),
        })

    table = pd.DataFrame(rows).sort_values("volume_um3", ascending=False)
    table.to_csv(OUT / "gc_column_metrics.csv", index=False)

    np.savez_compressed(
        OUT / "gc_columns.npz",
        **{f"v{gid}": columns[gid]["verts"] for gid in columns},
        **{f"f{gid}": columns[gid]["faces"] for gid in columns},
        **{f"sv{gid}": columns[gid]["shell_verts"] for gid in columns},
        **{f"sf{gid}": columns[gid]["shell_faces"] for gid in columns},
        gc_ids=np.array(sorted(columns), np.int32),
    )
    flat = {}
    for key, patch_list in patches_out.items():
        for index, patch in enumerate(patch_list):
            flat[f"{key}|{index}|xy"] = patch["xy_um"]
            flat[f"{key}|{index}|f"] = patch["faces"]
            flat[f"{key}|{index}|z"] = np.array([patch["z_um"]], np.float32)
    np.savez_compressed(OUT / "gc_plane_patches.npz", **flat)

    (OUT / "column_build_report.json").write_text(json.dumps({
        "n_columns": len(columns),
        "total_faces": int(sum(len(c["faces"]) for c in columns.values())),
        "total_shell_faces": int(sum(len(c["shell_faces"]) for c in columns.values())),
        "total_vertices": int(sum(len(c["verts"]) for c in columns.values())),
        "n_plane_patches": len(patches_out),
        "z_step_um": cfg.section_thickness_um, "grid_um": cfg.grid_um,
        "z_oversample": cfg.columns.z_oversample,
        "sigma_xy_px": cfg.columns.sigma_xy_px, "sigma_z_sub": cfg.columns.sigma_z_sub,
        "mesh_smoothing": "volume-domain Gaussian only; no mesh smoother",
        "pad_z": cfg.columns.pad_z, "pad_xy": cfg.columns.pad_xy,
        "iso_level": "volume-preserving (matches the measured voxel count)",
        "target_faces": cfg.columns.target_faces,
        "planes": [{"name": n, "kind": k, "z_um": (i - 1) * cfg.section_thickness_um} for n, k, i in PLANES],
    }, indent=2))

    print(f"built {len(columns)} columns, {sum(len(c['faces']) for c in columns.values()):,} faces, "
          f"{len(patches_out)} plane patches")
    print(table.head(10)[["merged_gc_id", "n_planes", "volume_um3", "max_equiv_radius_um",
                          "radius_cv", "centroid_drift_um", "n_faces"]].to_string(index=False))

