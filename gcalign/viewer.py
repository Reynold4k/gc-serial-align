"""Stage 6 - write the standalone viewer.

Everything is embedded: coordinates are quantised to 16 bits and base64'd, and
plotly.js is inlined, so the page needs no network and no server. The 53 GC
meshes are merged into a single trace because one mesh per GC costs about a
second per dragged frame on a machine without hardware acceleration.
"""
from __future__ import annotations

import base64
import colorsys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.offline as pyo

from .config import Config, stage_dir

def b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.dtype(dtype)).tobytes()).decode()


def v4_palette(ids: list[int]) -> dict[int, str]:
    """Reuse the four-slice viewer's HSV merged-GC palette so IDs keep colours."""
    out = {}
    for index, gid in enumerate(ids):
        hue = index / len(ids)
        sat = 0.6 + (index % 3) * 0.15
        val = 0.7 + (index % 2) * 0.2
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        out[gid] = f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
    return out


def hue_family(ids: list[int], base_hex: str, hue_spread: float,
               sat_range: tuple[float, float], val_range: tuple[float, float]) -> dict[int, str]:
    base = tuple(int(base_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h0, _, _ = colorsys.rgb_to_hsv(*base)
    out = {}
    for index, gid in enumerate(ids):
        t = index / max(len(ids) - 1, 1)
        hue = (h0 + hue_spread * (t - 0.5)) % 1.0
        sat = sat_range[0] + (sat_range[1] - sat_range[0]) * ((index * 7) % 5) / 4
        val = val_range[0] + (val_range[1] - val_range[0]) * ((index * 3) % 4) / 3
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        out[gid] = f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"
    return out


def run(cfg: Config) -> None:
    OUT = stage_dir(cfg, "05_viewer")
    Z_STEP_UM = cfg.section_thickness_um
    SLAB_HALF_UM = cfg.viewer.slab_half_um
    PLANES = cfg.planes
    LEIDEN_GC_HEX = cfg.viewer.st_hex
    SAPPHIRE_HEX = cfg.viewer.if_hex
    columns = np.load(cfg.output_dir / "04_columns/gc_columns.npz")
    patches = np.load(cfg.output_dir / "04_columns/gc_plane_patches.npz")
    metrics = pd.read_csv(cfg.output_dir / "04_columns/gc_column_metrics.csv")
    match_report = json.loads((cfg.output_dir / "03_gc_match/match_report.json").read_text())
    build_report = json.loads((cfg.output_dir / "04_columns/column_build_report.json").read_text())
    global_report = json.loads((cfg.output_dir / "02_global_align/global_transforms.json").read_text())

    gc_ids = [int(v) for v in columns["gc_ids"]]
    metrics = metrics.set_index("merged_gc_id").loc[gc_ids].reset_index()

    # ---- bounds shared by every buffer -------------------------------------
    stacked = np.concatenate([columns[f"v{gid}"] for gid in gc_ids])
    lo = stacked.min(axis=0).astype(np.float64)
    hi = stacked.max(axis=0).astype(np.float64)
    span = np.maximum(hi - lo, 1e-6)

    def quantise(points: np.ndarray) -> np.ndarray:
        return np.rint(np.clip((points - lo) / span, 0, 1) * 65535).astype("<u2")

    # ---- columns, merged into ONE mesh -------------------------------------
    # 53 separate mesh3d traces cost ~1.5 s per frame while dragging; one merged
    # trace with a per-vertex GC id renders the same thing at interactive rates.
    verts_all, faces_all, owner_all = [], [], []
    cursor = 0
    for gid in gc_ids:
        verts = columns[f"v{gid}"].astype(np.float64)
        faces = columns[f"f{gid}"]
        verts_all.append(verts)
        faces_all.append(faces + cursor)
        owner_all.append(np.full(len(verts), gid, np.int32))
        cursor += len(verts)
    cverts = np.concatenate(verts_all)
    cfaces = np.concatenate(faces_all)
    cowner = np.concatenate(owner_all)
    def merge(vkey: str, fkey: str) -> dict:
        verts, faces, owner, cursor = [], [], [], 0
        for gid in gc_ids:
            v = columns[f"{vkey}{gid}"].astype(np.float64)
            verts.append(v)
            faces.append(columns[f"{fkey}{gid}"] + cursor)
            owner.append(np.full(len(v), gid, np.int32))
            cursor += len(v)
        v = np.concatenate(verts); f = np.concatenate(faces); o = np.concatenate(owner)
        return {"n": int(len(v)), "nf": int(len(f)),
                "xyz": b64(quantise(v), "<u2"),
                "f": b64(f, "<u4" if len(v) > 65535 else "<u2"),
                "fBytes": 4 if len(v) > 65535 else 2,
                "owner": b64(o, "<u2")}

    col_payload = merge("v", "f")
    shell_payload = merge("sv", "sf")

    # ---- plane patches, merged per plane with a per-vertex GC id ------------
    def extrude(xy: np.ndarray, cap: np.ndarray, z: float):
        """Thin prism: bottom cap, top cap, and a wall around the ordered ring."""
        n = len(xy)
        verts = np.vstack([
            np.column_stack([xy, np.full(n, z - SLAB_HALF_UM)]),
            np.column_stack([xy, np.full(n, z + SLAB_HALF_UM)]),
        ])
        nxt = (np.arange(n) + 1) % n
        wall = np.concatenate([
            np.column_stack([np.arange(n), nxt, np.arange(n) + n]),
            np.column_stack([nxt, nxt + n, np.arange(n) + n]),
        ])
        faces = np.vstack([cap[:, ::-1], cap + n, wall])
        return verts, faces.astype(np.int32)

    plane_payload = []
    for name, kind, section in PLANES:
        verts_all, faces_all, owner_all = [], [], []
        cursor = 0
        for gid in gc_ids:
            index = 0
            while f"{gid}|{name}|{index}|xy" in patches:
                xy = patches[f"{gid}|{name}|{index}|xy"].astype(np.float64)
                cap = patches[f"{gid}|{name}|{index}|f"]
                z = float(patches[f"{gid}|{name}|{index}|z"][0])
                verts, faces = extrude(xy, cap, z)
                verts_all.append(verts)
                faces_all.append(faces + cursor)
                owner_all.append(np.full(len(verts), gid, np.int32))
                cursor += len(verts)
                index += 1
        if not verts_all:
            continue
        verts = np.concatenate(verts_all)
        faces = np.concatenate(faces_all)
        owner = np.concatenate(owner_all)
        plane_payload.append({
            "name": name, "kind": kind, "section": section,
            "z": (section - 1) * Z_STEP_UM,
            "n": int(len(verts)), "nf": int(len(faces)),
            "xyz": b64(quantise(verts), "<u2"),
            "f": b64(faces, "<u4" if len(verts) > 65535 else "<u2"),
            "fBytes": 4 if len(verts) > 65535 else 2,
            "owner": b64(owner, "<u2"),
        })

    identity = v4_palette(sorted(gc_ids))
    st_colours = hue_family(sorted(gc_ids), LEIDEN_GC_HEX, 0.085, (0.62, 1.0), (0.74, 1.0))
    if_colours = hue_family(sorted(gc_ids), SAPPHIRE_HEX, 0.075, (0.58, 0.98), (0.62, 1.0))

    info = {}
    for row in metrics.itertuples():
        info[int(row.merged_gc_id)] = {
            "planes": int(row.n_planes), "st": int(row.n_st_planes), "if": int(row.n_if_planes),
            "vol": float(row.volume_um3), "rmax": float(row.max_equiv_radius_um),
            "rmed": float(row.median_equiv_radius_um), "cv": float(row.radius_cv),
            "drift": float(row.centroid_drift_um),
        }

    payload = {
        "lo": lo.tolist(), "span": span.tolist(),
        "columns": col_payload, "shell": shell_payload, "planes": plane_payload,
        "ids": sorted(gc_ids),
        "identity": identity, "stColours": st_colours, "ifColours": if_colours,
        "info": info,
        "meta": {
            "sectionThicknessUm": Z_STEP_UM,
            "slabHalfUm": SLAB_HALF_UM,
            "nColumns": len(gc_ids),
            "totalFaces": int(col_payload["nf"]),
            "perGcBounds": {int(g): [float(columns[f"v{g}"][:, 0].min()), float(columns[f"v{g}"][:, 0].max()),
                                     float(columns[f"v{g}"][:, 1].min()), float(columns[f"v{g}"][:, 1].max())]
                            for g in gc_ids},
            "leidenGcHex": LEIDEN_GC_HEX,
            "alignment": {k: {"angle": v["angle_deg"], "flip": v["flip"],
                              "scale": v["total_scale"],
                              "matched": v["n_matched"], "close": v["n_matched_under_150um"],
                              "median_um": v["match_median_um"]}
                          for k, v in global_report["sections"].items()},
            "warp": {k: {"anchors": v["n_anchor_pairs"],
                         "pre_um": v["centroid_error_pre_um"],
                         "post_um": v["tps_residual_median_um"],
                         "loo_um": v["tps_loo_median_um"],
                         "twoSided": (v["two_sided_disagreement"] or {}).get("median_um")}
                     for k, v in match_report.items() if k.startswith("IF")},
            "mesh": {"pad_z": build_report["pad_z"], "sigma_xy_px": build_report["sigma_xy_px"],
                     "sigma_z_sub": build_report["sigma_z_sub"]},
        },
    }

    template = (Path(__file__).parent / "templates/viewer.html").read_text(encoding="utf-8")
    html = template.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__PLOTLYJS__", pyo.get_plotlyjs())
    html = html.replace("__TITLE__", cfg.viewer.title)
    html = html.replace("__SUBTITLE__", cfg.viewer.subtitle or cfg.name)
    html = html.replace("__ZDEFAULT__", str(cfg.viewer.default_z_exaggeration))
    target = OUT / cfg.viewer.filename
    target.write_text(html, encoding="utf-8")
    print(f"wrote {target}  ({target.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(gc_ids)} columns merged into 1 mesh, {col_payload['nf']:,} faces "
          f"(+ {shell_payload['nf']:,}-face envelope for the plane views)")
    print(f"  {len(plane_payload)} plane meshes, "
          f"{sum(p['nf'] for p in plane_payload):,} faces")

