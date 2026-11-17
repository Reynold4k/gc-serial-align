"""Stage 6 - write the standalone viewer.

Everything is embedded: coordinates are quantised to 16 bits and base64'd, and
plotly.js and a small subset of the theme typeface are inlined, so the page needs
no network and no server. The GC meshes are merged into a single trace because
one mesh per GC costs about a second per dragged frame on a machine without
hardware acceleration.

The look comes from `gcalign.theme`: the page's CSS custom properties are
`theme.css_vars()`, and the Plotly side reads the same colours from the payload
(`theme.as_dict()`), so nothing in the template restates a colour.

Payload (schema 2), as read by ``templates/viewer.html``::

    lo, span            quantisation frame (um) shared by every mesh buffer
    columns, shell      merged GC meshes: n, nf, xyz, f, fBytes, owner, ownerBytes
    planes[]            per-plane cross-section slabs: name, kind, index, z, + mesh
    follicle[], dzlz[]  optional point layers;  zoneBlocks: optional DZ/LZ solids
    ifImages            optional IF photographs (`if_images.json`)
    ids                 GC ids that have a column, ascending
    identity            one colour per GC id (theme.identity_colours)
    stColours, ifColours  per-GC shades of the two modality families
    info[gid]           planes, st, if, vol, rmax, rmed, cv, drift,
                        areas: um^2 on every plane in cutting order (null = absent)
    theme, fonts        the theme's named colours and font stacks
    kind                the colour that stands for each modality in this run
    scene               derived scene colours (follicle backdrop, envelopes)
    meta                planes (every plane: name, kind, index, z), counts,
                        alignment / warp QC per IF section, rgb channels, ...
"""
from __future__ import annotations

import base64
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.offline as pyo

from . import theme
from .config import Config, stage_dir

TEMPLATE = Path(__file__).parent / "templates" / "viewer.html"
SCHEMA = 2
Z_MAX = 24                      # top of the z-exaggeration slider


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
def b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.dtype(dtype)).tobytes()).decode()


def _num(value, digits: int | None = None):
    """A JSON-safe number: NaN / inf / missing become None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return round(value, digits) if digits is not None else value


def _empty_mesh() -> dict:
    return {"n": 0, "nf": 0, "xyz": "", "f": "", "fBytes": 2, "owner": "", "ownerBytes": 2}


class _Frame:
    """The quantisation box every mesh buffer shares."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo = np.asarray(lo, np.float64)
        self.span = np.maximum(np.asarray(hi, np.float64) - self.lo, 1e-6)

    def quantise(self, points: np.ndarray) -> np.ndarray:
        return np.rint(np.clip((points - self.lo) / self.span, 0, 1) * 65535).astype("<u2")

    def outside(self, points: np.ndarray) -> int:
        hi = self.lo + self.span
        return int(((points < self.lo - 1e-3) | (points > hi + 1e-3)).any(axis=1).sum())


def _merge(items: list[tuple[int, np.ndarray, np.ndarray]], frame: _Frame) -> dict:
    """Merge (gc id, vertices, faces) pieces into one mesh with a per-vertex owner id."""
    if not items:
        return _empty_mesh()
    verts, faces, owner, cursor = [], [], [], 0
    for gid, v, f in items:
        v = np.asarray(v, np.float64)
        verts.append(v)
        faces.append(np.asarray(f, np.int64) + cursor)
        owner.append(np.full(len(v), int(gid), np.int64))
        cursor += len(v)
    v = np.concatenate(verts)
    f = np.concatenate(faces)
    o = np.concatenate(owner)
    wide_faces = len(v) > 65535
    wide_owner = int(o.max()) > 65535
    return {"n": int(len(v)), "nf": int(len(f)),
            "xyz": b64(frame.quantise(v), "<u2"),
            "f": b64(f, "<u4" if wide_faces else "<u2"), "fBytes": 4 if wide_faces else 2,
            "owner": b64(o, "<u4" if wide_owner else "<u2"), "ownerBytes": 4 if wide_owner else 2,
            "clipped": frame.outside(v)}


def _extrude(xy: np.ndarray, cap: np.ndarray, z: float, half: float):
    """Thin prism: bottom cap, top cap, and a wall around the ordered ring."""
    n = len(xy)
    verts = np.vstack([np.column_stack([xy, np.full(n, z - half)]),
                       np.column_stack([xy, np.full(n, z + half)])])
    nxt = (np.arange(n) + 1) % n
    wall = np.concatenate([np.column_stack([np.arange(n), nxt, np.arange(n) + n]),
                           np.column_stack([nxt, nxt + n, np.arange(n) + n])])
    faces = np.vstack([cap[:, ::-1], cap + n, wall])
    return verts, faces.astype(np.int64)


# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------
def _luminance(value: str) -> float:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in theme.rgb01(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two hex colours."""
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _legible(value: str, ratio: float = 4.5) -> str:
    """`value` if it reads on the page as text, else lightened towards INK until it does."""
    for step in range(21):
        candidate = theme.mix(value, theme.INK, step / 20)
        if contrast(candidate, theme.BG) >= ratio:
            return candidate
    return theme.INK


def ui_kind_colours(cfg: Config) -> dict[str, str]:
    """The colour that stands for each modality in this run's interface.

    The theme's blue and pink, unless the config gives a modality its own family
    colour (an atlas whose GC cluster is orange, say) - then the interface
    follows the scene, so a chip, a legend dot and the planes it names agree.
    """
    out = {}
    for kind, base, default, ui in (("ST", cfg.viewer.st_hex, theme.ST_FAMILY_HEX, theme.ST),
                                    ("IF", cfg.viewer.if_hex, theme.IF_FAMILY_HEX, theme.IF)):
        out[kind] = ui if str(base).upper() == default.upper() else _legible(str(base))
    return out


# --------------------------------------------------------------------------
# type: a subset of the theme face, so the page looks the same everywhere
# --------------------------------------------------------------------------
def font_face_css(extra_text: str = "") -> str:
    """The theme font embedded for this page (`theme.font_face_css`)."""
    return theme.font_face_css(extra_text)


# --------------------------------------------------------------------------
# reading the run
# --------------------------------------------------------------------------
def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _presence(cfg: Config):
    """The presence table (which GC is on which plane, and how large), or None."""
    try:
        from .runio import Run
        return Run(cfg).presence
    except Exception as error:
        for relative in ("03_gc_match/gc_presence.csv", "03_gc_match/gc_presence_9plane.csv"):
            path = cfg.output_dir / relative
            if path.exists():
                return pd.read_csv(path)
        print(f"  no presence table ({error}); the per-GC profile is left out")
        return None


def _identity(cfg: Config, ids: list[int]) -> dict[int, str]:
    """One colour per GC, keyed on every GC of the run so it matches the figures."""
    try:
        from .runio import Run
        colours = dict(Run(cfg).identity)
        if all(i in colours for i in ids):
            return colours
    except Exception:
        pass
    return theme.identity_colours(ids, cfg.viewer.palette)


def _grid_frame(cfg: Config, global_report: dict, match_report: dict) -> _Frame:
    """The section grid as a frame, for a run that has no geometry to measure."""
    grid = global_report.get("grid") or match_report.get("grid") or {}
    try:
        um = float(grid.get("um_per_px", cfg.grid_um))
        x0, y0 = float(grid["x0_um"]), float(grid["y0_um"])
        lo = [x0, y0, 0.0]
        hi = [x0 + float(grid["width"]) * um, y0 + float(grid["height"]) * um, cfg.z_span_um]
    except (KeyError, TypeError, ValueError):
        lo, hi = [0.0, 0.0, 0.0], [1000.0, 1000.0, max(cfg.z_span_um, 1.0)]
    return _Frame(np.array(lo), np.array(hi))


# --------------------------------------------------------------------------
# the payload
# --------------------------------------------------------------------------
def build_payload(cfg: Config) -> dict:
    root = Path(cfg.output_dir)
    slab_half = float(cfg.viewer.slab_half_um)
    planes = cfg.planes                                     # (name, kind, index), cutting order
    if_names = [name for name, kind, _ in planes if kind == "IF"]

    global_report = _json(root / "02_global_align/global_transforms.json")
    match_report = _json(root / "03_gc_match/match_report.json")
    build_report = _json(root / "04_columns/column_build_report.json")

    column_file = root / "04_columns/gc_columns.npz"
    columns = np.load(column_file) if column_file.exists() else None
    gc_ids = sorted(int(v) for v in columns["gc_ids"]) if columns is not None and "gc_ids" in columns.files else []
    metrics_file = root / "04_columns/gc_column_metrics.csv"
    metrics = pd.read_csv(metrics_file) if metrics_file.exists() and metrics_file.stat().st_size else None
    if metrics is None or "merged_gc_id" not in getattr(metrics, "columns", []):
        gc_ids = []
    else:
        known = set(int(v) for v in metrics["merged_gc_id"])
        dropped = [g for g in gc_ids if g not in known]
        if dropped:
            print(f"  {len(dropped)} column(s) have no metrics row and are left out: {dropped[:8]}")
        gc_ids = [g for g in gc_ids if g in known]

    patch_file = root / "04_columns/gc_plane_patches.npz"
    patches = np.load(patch_file) if patch_file.exists() else None
    patch_keys = set(patches.files) if patches is not None else set()

    zb_file = root / "04_columns/zone_blocks.npz"
    zone_store = np.load(zb_file) if zb_file.exists() and gc_ids else None
    zb_ids = [int(v) for v in zone_store["gc_ids"]] if zone_store is not None and "gc_ids" in zone_store.files else []
    zb_ids = [g for g in zb_ids if f"DZ_v{g}" in zone_store.files and f"LZ_v{g}" in zone_store.files] if zb_ids else []

    # ---- the frame: every piece of geometry that is quantised -----------------
    plane_pieces: dict[str, list] = {name: [] for name, _, _ in planes}
    for name, _, _ in planes:
        for gid in gc_ids:
            index = 0
            while f"{gid}|{name}|{index}|xy" in patch_keys:
                key = f"{gid}|{name}|{index}"
                xy = patches[f"{key}|xy"].astype(np.float64)
                cap = patches[f"{key}|f"]
                z = float(patches[f"{key}|z"][0]) if f"{key}|z" in patch_keys else 0.0
                verts, faces = _extrude(xy, cap, z, slab_half)
                plane_pieces[name].append((gid, verts, faces))
                index += 1

    if gc_ids:
        stacks = [columns[f"v{g}"].astype(np.float64) for g in gc_ids]
        stacks += [v for pieces in plane_pieces.values() for _, v, _ in pieces]
        stacks += [zone_store[f"{c}_v{g}"].astype(np.float64) for g in zb_ids for c in ("DZ", "LZ")]
        everything = np.concatenate(stacks)
        frame = _Frame(everything.min(axis=0), everything.max(axis=0))
    else:
        frame = _grid_frame(cfg, global_report, match_report)

    # ---- columns, merged into ONE mesh ----------------------------------------
    # one mesh3d trace per GC costs ~1.5 s per frame while dragging; one merged
    # trace with a per-vertex GC id renders the same thing at interactive rates
    col_payload = _merge([(g, columns[f"v{g}"], columns[f"f{g}"]) for g in gc_ids], frame)
    # the translucent envelope is blurry by design, so it uses a low-poly copy
    shell_items = [(g, columns[f"sv{g}"], columns[f"sf{g}"]) for g in gc_ids
                   if columns is not None and f"sv{g}" in columns.files and f"sf{g}" in columns.files]
    shell_payload = _merge(shell_items, frame) if len(shell_items) == len(gc_ids) else col_payload

    # ---- plane slabs, merged per plane with a per-vertex GC id -----------------
    plane_payload = []
    for name, kind, section in planes:
        pieces = plane_pieces[name]
        if not pieces:
            continue
        mesh = _merge(pieces, frame)
        plane_payload.append({"name": name, "kind": kind, "index": int(section),
                              "z": float(cfg.z_um(section)), **mesh})

    # ---- optional DZ / LZ solids ----------------------------------------------
    # Built by viewer_layers/build_zone_blocks.py, which partitions each GC rather
    # than thresholding the two zones independently, so the pair tiles it.
    zone_blocks = None
    if zb_ids:
        zone_blocks = {cls: _merge([(g, zone_store[f"{cls}_v{g}"], zone_store[f"{cls}_f{g}"])
                                    for g in zb_ids], frame) for cls in ("DZ", "LZ")}
        zone_blocks["ids"] = zb_ids
        print(f"  DZ/LZ blocks: {len(zb_ids)} GCs, "
              f"{zone_blocks['DZ']['nf'] + zone_blocks['LZ']['nf']:,} faces")

    # ---- optional B-follicle backdrop ------------------------------------------
    follicle_payload = []
    foll_file = root / "05_viewer/follicle_points.npz"
    if foll_file.exists() and gc_ids:
        store = np.load(foll_file)
        for name, kind, section in planes:
            if name not in store.files:
                continue
            pts = np.asarray(store[name], np.float32).reshape(-1, 2)
            if not len(pts):
                continue
            follicle_payload.append({"name": name, "kind": kind, "index": int(section),
                                     "z": float(cfg.z_um(section)), "n": int(len(pts)),
                                     "x": b64(pts[:, 0], "<f4"), "y": b64(pts[:, 1], "<f4")})
        print(f"  follicle backdrop: {len(follicle_payload)} planes, "
              f"{sum(p['n'] for p in follicle_payload):,} points")

    # ---- optional dark-zone / light-zone points ---------------------------------
    dzlz_payload = []
    dz_file = root / "05_viewer/dzlz_points.npz"
    if dz_file.exists() and gc_ids:
        store = np.load(dz_file)
        for name, kind, section in planes:
            entry = {"name": name, "kind": kind, "index": int(section), "z": float(cfg.z_um(section))}
            for cls in ("DZ", "LZ"):
                key = f"{name}|{cls}"
                if key not in store.files:
                    continue
                pts = np.asarray(store[key], np.float32).reshape(-1, 2)
                block = {"n": int(len(pts)), "x": b64(pts[:, 0], "<f4"), "y": b64(pts[:, 1], "<f4")}
                if f"{key}|gc" in store.files:
                    gc = np.asarray(store[f"{key}|gc"])
                    wide = len(gc) and int(gc.max()) > 65535
                    block["gc"] = b64(gc, "<u4" if wide else "<u2")
                    block["gcBytes"] = 4 if wide else 2
                entry[cls] = block
            if "DZ" in entry or "LZ" in entry:
                dzlz_payload.append(entry)
        print(f"  dark/light zone: {len(dzlz_payload)} planes, "
              f"{sum(e.get(c, {}).get('n', 0) for e in dzlz_payload for c in ('DZ', 'LZ')):,} points")

    # ---- optional full-resolution IF photographs --------------------------------
    if_images = None
    img_file = root / "05_viewer/if_images.json"
    if img_file.exists():
        try:
            if_images = json.loads(img_file.read_text(encoding="utf-8"))
            if not isinstance(if_images, dict) or not if_images.get("planes"):
                if_images = None
            else:
                print(f"  IF photographs: {len(if_images['planes'])} planes, "
                      f"{img_file.stat().st_size / 1e6:.2f} MB")
        except ValueError as error:
            print(f"  IF photographs skipped: {img_file} is not valid JSON ({error})")
            if_images = None

    # ---- per-GC numbers ----------------------------------------------------------
    presence = _presence(cfg)
    area_rows: dict[int, list] = {}
    if presence is not None and "merged_gc_id" in presence.columns:
        # by column name, not attribute access: a plane can be called anything
        gids = [int(v) for v in presence["merged_gc_id"]]
        per_plane = []
        for name, _, _ in planes:
            col = f"{name}_area_um2"
            per_plane.append(list(presence[col]) if col in presence.columns else [None] * len(gids))
        for row, gid in enumerate(gids):
            values = []
            for column in per_plane:
                area = _num(column[row])
                values.append(int(round(area)) if area and area > 0 else None)
            area_rows[gid] = values

    all_ids = sorted(set(gc_ids) | set(area_rows))
    identity = _identity(cfg, all_ids) if all_ids else {}
    st_colours = theme.kind_colours(all_ids, "ST", cfg.viewer.st_hex)
    if_colours = theme.kind_colours(all_ids, "IF", cfg.viewer.if_hex)

    info = {}
    if gc_ids:
        table = metrics.set_index("merged_gc_id")
        for gid in gc_ids:
            row = table.loc[gid]
            if isinstance(row, pd.DataFrame):            # a duplicated id: take the first
                row = row.iloc[0]
            info[gid] = {
                "planes": int(row["n_planes"]), "st": int(row["n_st_planes"]),
                "if": int(row["n_if_planes"]),
                "vol": _num(row["volume_um3"]), "rmax": _num(row["max_equiv_radius_um"]),
                "rmed": _num(row["median_equiv_radius_um"]), "cv": _num(row["radius_cv"]),
                "drift": _num(row["centroid_drift_um"]),
            }
            if gid in area_rows:
                info[gid]["areas"] = area_rows[gid]

    per_gc_bounds = {}
    for gid in gc_ids:
        v = columns[f"v{gid}"]
        per_gc_bounds[gid] = [float(v[:, 0].min()), float(v[:, 0].max()),
                              float(v[:, 1].min()), float(v[:, 1].max())]

    # ---- alignment QC, keyed on the IF planes by kind, never by name -------------
    sections = global_report.get("sections", {})
    alignment = {}
    for name in if_names:
        entry = sections.get(name)
        if not entry:
            continue
        index = cfg.section(name).index
        alignment[name] = {
            "angle": _num(entry.get("angle_deg"), 2), "flip": bool(entry.get("flip", False)),
            "scale": _num(entry.get("total_scale"), 4),
            "matched": entry.get("n_matched"), "nIf": entry.get("n_if_gc"),
            "close": entry.get("n_matched_under_150um"),
            "median_um": _num(entry.get("match_median_um"), 2),
            "neighbours": list(cfg.neighbours.get(index, [])),
            "oneSided": bool(cfg.one_sided(index)),
        }
    warp = {}
    for name in if_names:
        entry = match_report.get(name)
        if not isinstance(entry, dict):
            continue
        two_sided = entry.get("two_sided_disagreement") or {}
        warp[name] = {"anchors": entry.get("n_anchor_pairs"),
                      "pre_um": _num(entry.get("centroid_error_pre_um"), 2),
                      "post_um": _num(entry.get("tps_residual_median_um"), 2),
                      "loo_um": _num(entry.get("tps_loo_median_um"), 2),
                      "twoSided": _num(two_sided.get("median_um"), 2) if two_sided else None}

    from . import __version__
    kinds = ui_kind_colours(cfg)
    stem = Path(cfg.viewer.filename).stem or "gcalign"
    payload = {
        "schema": SCHEMA,
        "lo": frame.lo.tolist(), "span": frame.span.tolist(),
        "columns": col_payload, "shell": shell_payload, "planes": plane_payload,
        "follicle": follicle_payload, "dzlz": dzlz_payload, "ifImages": if_images,
        "zoneBlocks": zone_blocks,
        "ids": gc_ids,
        "identity": {int(k): v for k, v in identity.items()},
        "stColours": st_colours, "ifColours": if_colours,
        "info": info,
        "theme": theme.as_dict(),
        "fonts": {"sans": theme.CSS_FONT_STACK, "mono": theme.CSS_MONO_STACK},
        "kind": kinds,
        "scene": {
            "follicle": theme.FOLLICLE,
            "envelope": {"st": theme.mix(kinds["ST"], theme.INK, 0.45),
                         "if": theme.mix(kinds["IF"], theme.INK, 0.45),
                         "both": theme.mix(theme.mix(kinds["ST"], kinds["IF"], 0.5), theme.INK, 0.45)},
            "scaleBar": theme.INK_2,
        },
        "meta": {
            "version": __version__,
            "name": cfg.name,
            "title": cfg.viewer.title,
            "subtitle": cfg.viewer.subtitle or cfg.name,
            "fileStem": stem,
            "sectionThicknessUm": float(cfg.section_thickness_um),
            "gridUm": float(cfg.grid_um),
            "zSpanUm": float(cfg.z_span_um),
            "zDefault": int(min(max(int(cfg.viewer.default_z_exaggeration), 1), Z_MAX)),
            "zMax": Z_MAX,
            "planes": [{"name": n, "kind": k, "index": int(i), "z": float(cfg.z_um(i))}
                       for n, k, i in planes],
            "nPlanes": len(planes),
            "nColumns": len(gc_ids),
            "nGcs": len(area_rows) if area_rows else len(gc_ids),
            "minPlanes": int(cfg.columns.min_planes),
            "gcIdSource": cfg.st_scaffold.gc_id_col,
            "slabHalfUm": slab_half,
            "totalFaces": int(col_payload["nf"]),
            "perGcBounds": per_gc_bounds,
            "rgb": list(cfg.rgb_channels),
            "closeUm": float(cfg.align.close_match_um),
            "stHex": cfg.viewer.st_hex, "ifHex": cfg.viewer.if_hex,
            "alignment": alignment,
            "warp": warp,
            "mesh": {k: build_report.get(k) for k in ("pad_z", "sigma_xy_px", "sigma_z_sub")},
        },
    }
    clipped = col_payload.get("clipped", 0) + sum(p.get("clipped", 0) for p in plane_payload)
    if zone_blocks:
        clipped += zone_blocks["DZ"].get("clipped", 0) + zone_blocks["LZ"].get("clipped", 0)
    if clipped:
        print(f"  warning: {clipped} vertices fell outside the shared frame and were clamped")
    return payload


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
def _script_json(payload: dict) -> str:
    """JSON that can sit inside a <script> element and still parse."""
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    return (text.replace("</", "<\\/").replace("<!--", "<\\u0021--")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def render_html(cfg: Config, payload: dict) -> str:
    kinds = payload.get("kind") or ui_kind_colours(cfg)
    extra_vars = {"kind-st": kinds["ST"], "kind-if": kinds["IF"],
                  "follicle": payload.get("scene", {}).get("follicle", theme.IF)}
    css_vars = theme.css_vars() + "".join(f"--{k}:{v};" for k, v in extra_vars.items())
    named = " ".join([cfg.viewer.title, cfg.viewer.subtitle or cfg.name]
                     + [name for name, _, _ in cfg.planes]
                     + [c for c in cfg.rgb_channels if c])
    title = cfg.viewer.title
    subtitle = cfg.viewer.subtitle or cfg.name
    fills = {
        "__THEME_VARS__": css_vars,
        "__FONT_FACE__": font_face_css(named),
        "__TITLE__": html.escape(title),
        "__SUBTITLE__": html.escape(subtitle),
        "__PLOTLYJS__": pyo.get_plotlyjs(),
        "__PAYLOAD__": _script_json(payload),
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    # one pass, so nothing inside an inserted block is ever substituted again
    return re.sub(r"__[A-Z][A-Z0-9_]*__", lambda m: fills.get(m.group(0), m.group(0)), template)


def run(cfg: Config) -> None:
    OUT = stage_dir(cfg, "05_viewer")
    payload = build_payload(cfg)
    page = render_html(cfg, payload)
    target = OUT / cfg.viewer.filename
    target.write_text(page, encoding="utf-8")
    columns = payload["columns"]
    print(f"wrote {target}  ({target.stat().st_size / 1e6:.1f} MB)")
    if not payload["ids"]:
        print("  no GC columns in this run - the page opens on an empty state")
        return
    print(f"  {len(payload['ids'])} columns merged into 1 mesh, {columns['nf']:,} faces "
          f"(+ {payload['shell']['nf']:,}-face envelope for the plane views)")
    print(f"  {len(payload['planes'])} plane meshes, "
          f"{sum(p['nf'] for p in payload['planes']):,} faces")
