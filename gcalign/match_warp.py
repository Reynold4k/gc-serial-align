"""Stage 4 - match germinal centres, warp, and give every GC one identity.

Each IF section is warped towards each flanking Stereo-seq section
independently. The two answers are averaged by true z distance, and how much
they disagree is kept as that section's error bar - an honest uncertainty that
costs nothing extra to compute. The Stereo-seq scaffold itself never moves.

The warp is a thin-plate spline anchored only on matched GC centroids, and its
stiffness is picked by leave-one-out cross-validation, so the field is never
given more freedom than the anchors actually support.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
from skimage.segmentation import watershed

from .config import Config, stage_dir
from .global_align import match_pairs
from .parallel import run_sections

PRESENCE_FILE = "gc_presence.csv"
TRANSFORMS_FILE = "section_transforms.json"
MIN_ANCHORS = 6


def st_on_if_grid(cfg, st_map, linear, offset, shape, grid):
    """Look up a world-space Stereo-seq label map at every IF pixel."""
    x0, y0, height, width = grid
    rows, cols = np.indices(shape)
    local = np.stack([cols * cfg.grid_um, rows * cfg.grid_um], axis=-1)
    world = local @ np.asarray(linear).T + np.asarray(offset)
    c = np.rint((world[..., 0] - x0) / cfg.grid_um).astype(int)
    r = np.rint((world[..., 1] - y0) / cfg.grid_um).astype(int)
    ok = (r >= 0) & (r < height) & (c >= 0) & (c < width)
    out = np.zeros(shape, np.int32)
    out[ok] = st_map[r[ok], c[ok]]
    return out


def split_touching(cfg, gc_mask, st_views):
    """Split IF objects that two or more Stereo-seq GCs sit under.

    The Stereo-seq side is the evidence for *how many* structures are present;
    a watershed seeded on those regions decides where the boundary goes. Only
    objects with two well-supported claimants are touched, so a genuinely
    single, elongated GC is left alone.
    """
    labels, count = ndi.label(gc_mask)
    if not cfg.align.split_touching_gcs or count == 0 or not st_views:
        return labels, count, []

    objects = ndi.find_objects(labels)
    out = np.zeros_like(labels)
    next_label = 0
    report = []
    for index in range(1, count + 1):
        box = objects[index - 1]
        obj = labels[box] == index
        size = int(obj.sum())

        claims = {}
        for view in st_views:                      # one per flanking ST section
            sub = view[box]
            for gid, n in zip(*np.unique(sub[obj], return_counts=True)):
                if gid > 0:
                    claims[int(gid)] = max(claims.get(int(gid), 0), int(n))
        strong = sorted((g for g, n in claims.items()
                         if n >= cfg.align.split_min_px
                         and n >= cfg.align.split_min_frac * size),
                        key=lambda g: -claims[g])

        if len(strong) < 2:
            next_label += 1
            out[box][obj] = next_label
            continue

        markers = np.zeros(obj.shape, np.int32)
        for rank, gid in enumerate(strong, start=1):
            seed = np.zeros(obj.shape, bool)
            for view in st_views:
                seed |= (view[box] == gid)
            seed &= obj
            if seed.any():
                markers[seed] = rank
        if markers.max() < 2:
            next_label += 1
            out[box][obj] = next_label
            continue

        distance = ndi.distance_transform_edt(obj)
        pieces = watershed(-distance, markers, mask=obj)
        sizes = []
        for rank, gid in enumerate(strong, start=1):
            piece = pieces == rank
            if not piece.any():
                continue
            next_label += 1
            out[box][piece] = next_label
            sizes.append((gid, int(piece.sum())))
        report.append({"object_px": size,
                       "claimants": {f"GC{g}": claims[g] for g in strong},
                       "split_into": {f"GC{g}": n for g, n in sizes}})
    return out, next_label, report


def fit_tps(src, dst, smoothing):
    return RBFInterpolator(src, dst, kernel="thin_plate_spline",
                           degree=1, smoothing=smoothing)


def choose_smoothing(cfg, src, dst):
    """Leave-one-out CV: how well does the field predict an anchor it never saw?"""
    best, curve = None, []
    for smoothing in cfg.align.tps_smoothing_grid:
        errors = []
        for i in range(len(src)):
            keep = np.ones(len(src), bool); keep[i] = False
            model = fit_tps(src[keep], dst[keep], smoothing)
            errors.append(float(np.linalg.norm(model(src[i:i + 1])[0] - dst[i])))
        score = float(np.median(errors))
        curve.append((smoothing, score))
        if best is None or score < best[1]:
            best = (smoothing, score)
    return best[0], best[1], curve


def source_pixels(cfg, shape, linear, offset, inverse_tps, grid):
    """For every world-grid pixel, the IF pixel it came from (through A then B).

    Computed once and reused for the labels and for the fluorescence channels, so
    the warped image and the warped mask can never drift apart.
    """
    x0, y0, height, width = grid
    GRID_UM = cfg.grid_um
    rows, cols = np.indices((height, width))
    world = np.stack([x0 + cols * GRID_UM, y0 + rows * GRID_UM], axis=-1).reshape(-1, 2)
    pre = inverse_tps(world) if inverse_tps is not None else world
    local = (pre - offset) @ np.linalg.inv(linear).T
    src_c = np.rint(local[:, 0] / GRID_UM).astype(int)
    src_r = np.rint(local[:, 1] / GRID_UM).astype(int)
    ok = (src_r >= 0) & (src_r < shape[0]) & (src_c >= 0) & (src_c < shape[1])
    return src_r, src_c, ok, (height, width)


def sample(plane, src_r, src_c, ok, out_shape, dtype):
    out = np.zeros(out_shape[0] * out_shape[1], dtype)
    out[ok] = plane[src_r[ok], src_c[ok]]
    return out.reshape(out_shape)


def warp_labels(cfg, mask, labels_by_object, linear, offset, inverse_tps, grid):
    """Pull-sample the IF label map onto the world grid through A then B."""
    src_r, src_c, ok, out_shape = source_pixels(
        cfg, mask.shape, linear, offset, inverse_tps, grid)
    return sample(labels_by_object, src_r, src_c, ok, out_shape, np.int32)


def channel_top(plane: np.ndarray) -> float:
    """The intensity shown as white: the 99.9th percentile of the positive pixels."""
    plane = np.asarray(plane)
    return float(np.percentile(plane[plane > 0], 99.9)) if (plane > 0).any() else 1.0


def to_display_units(plane: np.ndarray) -> np.ndarray:
    """A channel rescaled to 0..65535, with `channel_top` at 65535.

    The registered channels are stored on this scale, one scale per section and
    channel, so every section uses the full 16-bit range.
    """
    plane = np.asarray(plane, np.float32)
    return np.clip(plane / max(channel_top(plane), 1e-6), 0, 1) * 65535.0


def st_plane_labels(cfg: Config, grid) -> dict[str, np.ndarray]:
    """Rasterise the frozen Stereo-seq GC regions onto the grid. Nothing moves."""
    GRID_UM = cfg.grid_um
    shape = (int(grid[2]), int(grid[3]))
    scaffold = np.load(cfg.output_dir / "00_scaffold/st_scaffold.npz")
    slice_short = scaffold["slice_short"]
    gc_id = scaffold["gc_id"]
    world = scaffold["world_um"].astype(np.float64)
    out = {}
    for name in cfg.st_names:
        keep = (slice_short == name) & (gc_id > 0)
        label = np.zeros(shape, np.int32)
        col = np.rint((world[keep, 0] - grid[0]) / GRID_UM).astype(int)
        row = np.rint((world[keep, 1] - grid[1]) / GRID_UM).astype(int)
        inside = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
        label[row[inside], col[inside]] = gc_id[keep][inside]
        # close the 1-px gaps the bin -> grid rounding leaves behind
        filled = ndi.grey_closing(label, size=(3, 3))
        label = np.where(label > 0, label, np.where(filled > 0, filled, 0))
        out[name] = label
    return out


def per_slice_centroids(cfg: Config) -> dict[str, dict[int, np.ndarray]]:
    summary = json.loads((cfg.output_dir / "00_scaffold/st_scaffold_summary.json").read_text())
    return {
        s: {r["merged_gc_id"]: np.array([r["per_slice"][s]["x_um"], r["per_slice"][s]["y_um"]])
            for r in summary["merged_gc"] if s in r["per_slice"]}
        for s in cfg.st_names
    }


def section_anchors(cfg: Config, name: str, entry: dict, st_labels: dict, centroids: dict,
                    grid, verbose: bool = True) -> dict:
    """Pair one IF section's germinal centres with the flanking Stereo-seq ones.

    Returns everything the warp is fitted on: the (possibly split) IF objects,
    their centroids after the Stage-A placement, the per-neighbour matches, and
    the anchors - `src` where each matched GC is, `dst` where its Stereo-seq
    counterpart says it should be. Deterministic given a finished stage `align`,
    which is how a transform can be rebuilt for a run that never stored it.
    """
    GRID_UM = cfg.grid_um
    say = print if verbose else (lambda *a, **k: None)
    neighbours = cfg.neighbours[cfg.section(name).index]
    linear = np.array(entry["linear"])
    offset = np.array(entry["offset"])
    data = np.load(cfg.output_dir / f"01_if_gc/{name}_gc.npz")
    gc = data["gc"]
    st_views = [st_on_if_grid(cfg, st_labels[n], linear, offset, gc.shape, grid)
                for n in neighbours]
    labels, n_obj, split_report = split_touching(cfg, gc, st_views)
    for item in split_report:
        say(f"{name}: split a {item['object_px']} px object claimed by "
            + " and ".join(f"{g} ({n} px)" for g, n in item["claimants"].items())
            + " -> " + ", ".join(f"{g} {n} px" for g, n in item["split_into"].items()),
            flush=True)
    centroids_local = np.array([[c * GRID_UM, r * GRID_UM] for r, c in
                                ndi.center_of_mass(gc, labels, np.arange(1, n_obj + 1))]).reshape(-1, 2)
    centroids_world = centroids_local @ linear.T + offset

    # match against each flanking slice independently
    per_target = {}
    for neighbour in neighbours:
        table = centroids[neighbour]
        ids = np.array(sorted(table))
        pts = np.array([table[i] for i in ids])
        rows, cols, dists = match_pairs(centroids_world, pts, cfg.align.warp_match_radius_um)
        per_target[neighbour] = {"rows": rows, "ids": ids[cols],
                                 "targets": pts[cols], "dists": dists}
        say(f"{name} vs {neighbour}: {len(rows)}/{n_obj} matched, "
            f"median {np.median(dists):.0f} um" if len(rows) else
            f"{name} vs {neighbour}: 0/{n_obj} matched", flush=True)

    # blended target: equidistant flanks, so a plain mean where both agree
    blended: dict[int, list] = {}
    votes: dict[int, list] = {}
    for neighbour, info in per_target.items():
        for row, gid, target in zip(info["rows"], info["ids"], info["targets"]):
            blended.setdefault(int(row), []).append(target)
            votes.setdefault(int(row), []).append(int(gid))
    src, dst, assigned = [], [], {}
    for row, targets in blended.items():
        ids = votes[row]
        if len(set(ids)) > 1:        # flanks disagree on identity -> drop as anchor
            continue
        src.append(centroids_world[row])
        dst.append(np.mean(targets, axis=0))
        assigned[row] = ids[0]
    return {"gc": gc, "data": data, "labels": labels, "n_obj": n_obj,
            "split_report": split_report, "centroids_world": centroids_world,
            "per_target": per_target, "linear": linear, "offset": offset,
            "src": np.asarray(src, float).reshape(-1, 2),
            "dst": np.asarray(dst, float).reshape(-1, 2), "assigned": assigned}


def warp_section(cfg: Config, name: str, entry: dict, st_labels: dict, centroids: dict, grid):
    """Stage B for one IF section. Independent of the other IF sections."""
    found = section_anchors(cfg, name, entry, st_labels, centroids, grid)
    gc, data, labels, n_obj = found["gc"], found["data"], found["labels"], found["n_obj"]
    linear, offset = found["linear"], found["offset"]
    src, dst, assigned = found["src"], found["dst"], found["assigned"]
    per_target, centroids_world = found["per_target"], found["centroids_world"]
    if len(src) < MIN_ANCHORS:
        raise RuntimeError(
            f"{name}: only {len(src)} germinal centres are matched to the flanking sections "
            f"with one agreed identity; a warp needs at least {MIN_ANCHORS}. Look at "
            "02_global_align/global_transforms.json for this section first - if it matched "
            "few structures there too, the placement itself is the problem.")

    # forward for QC, inverse for rasterising
    smoothing, loo_median, loo_curve = choose_smoothing(cfg, src, dst)
    forward = fit_tps(src, dst, smoothing)
    inverse = fit_tps(dst, src, smoothing)
    residual = np.linalg.norm(forward(src) - dst, axis=1)
    pre = np.linalg.norm(src - dst, axis=1)
    print(f"{name}: smoothing {smoothing:.0e} (LOO median {loo_median:.0f} um), "
          f"centroid error {np.median(pre):.0f} -> {np.median(residual):.0f} um", flush=True)

    # two-sided disagreement: warp each flank separately, compare fields
    disagreement = None
    if len(per_target) == 2:
        fields = []
        for neighbour, info in per_target.items():
            keep = [i for i, row in enumerate(info["rows"]) if row in assigned]
            if len(keep) < 6:
                continue
            rows_k = info["rows"][keep]
            fields.append(fit_tps(centroids_world[rows_k], info["targets"][keep], smoothing))
        if len(fields) == 2:
            probe = src
            delta = np.linalg.norm(fields[0](probe) - fields[1](probe), axis=1)
            disagreement = {"median_um": float(np.median(delta)),
                            "p90_um": float(np.percentile(delta, 90)),
                            "max_um": float(delta.max())}

    # object label map -> merged GC id, then warp
    object_to_gc = np.zeros(n_obj + 1, np.int32)
    for row, gid in assigned.items():
        object_to_gc[row + 1] = gid
    unmatched = [i for i in range(n_obj) if i not in assigned]
    for offset_id, row in enumerate(unmatched, start=1):
        object_to_gc[row + 1] = -offset_id          # IF-only object
    id_map = object_to_gc[labels]
    src_r, src_c, ok, out_shape = source_pixels(
        cfg, gc.shape, linear, offset, inverse, grid)
    plane_label = sample(id_map, src_r, src_c, ok, out_shape, np.int32)

    # the fluorescence itself, warped by exactly the same map, so figures can
    # show the real stain under the segmented outline rather than a flat fill
    warped_channels = {}
    for channel in cfg.if_images.channels.values():
        scaled = to_display_units(data[channel])
        warped_channels[f"{name}|{channel}"] = sample(
            scaled.astype(np.uint16), src_r, src_c, ok, out_shape, np.uint16)
    # and the section outline, through the same map: it is what gives the plane
    # a body in every picture, and it is measured, not re-cut afterwards
    tissue_warped = sample(data["tissue"].astype(np.uint8), src_r, src_c, ok,
                           out_shape, np.uint8).astype(bool)

    report = {
        "n_if_objects": int(n_obj),
        "objects_split_on_st_evidence": found["split_report"],
        "n_anchor_pairs": int(len(src)),
        "n_if_only": len(unmatched),
        "tps_smoothing": float(smoothing),
        "tps_loo_median_um": float(loo_median),
        "tps_loo_curve": [[float(a), float(b)] for a, b in loo_curve],
        "centroid_error_pre_um": float(np.median(np.linalg.norm(src - dst, axis=1))),
        "tps_residual_median_um": float(np.median(residual)),
        "tps_residual_p90_um": float(np.percentile(residual, 90)),
        "pre_warp_median_um": {k: float(np.median(v["dists"])) for k, v in per_target.items()},
        "two_sided_disagreement": disagreement,
        "assigned": {str(k + 1): int(v) for k, v in assigned.items()},
    }
    transform = {
        "linear": linear.tolist(), "offset": offset.tolist(),
        "tps": {"smoothing": float(smoothing),
                "source_um": src.tolist(), "target_um": dst.tolist()},
    }
    print(f"{name}: {len(src)} anchors, TPS residual median "
          f"{np.median(residual):.0f} um, {len(unmatched)} IF-only objects"
          + (f", two-sided disagreement median {disagreement['median_um']:.0f} um"
             if disagreement else "") + "\n", flush=True)
    return name, plane_label, warped_channels, tissue_warped, report, transform


def transforms_document(cfg: Config, grid_meta: dict, sections: dict) -> dict:
    """Everything needed to map a point between an IF section and the frame."""
    manifest = {}
    manifest_file = cfg.output_dir / "01_if_gc/if_gc_manifest.json"
    if manifest_file.exists():
        by_index = {s.index: s.name for s in cfg.if_sections}
        for item in json.loads(manifest_file.read_text()):
            manifest[item.get("name") or by_index.get(item["section"])] = item
    out = {}
    for name, body in sections.items():
        index = cfg.section(name).index
        entry = dict(body)
        entry["z_um"] = cfg.z_um(index)
        entry["neighbours"] = cfg.neighbours[index]
        entry["one_sided"] = cfg.one_sided(index)
        if name in manifest:
            m = manifest[name]
            entry["image"] = {"um_per_px": cfg.if_images.um_per_px,
                              "raster_shape": m["shape"],
                              "raster_um_per_px": [m["um_per_px_x"], m["um_per_px_y"]]}
        out[name] = entry
    return {
        "note": ("Stage A is the similarity `linear`, `offset` (local um -> frame um); stage B is "
                 "a thin-plate spline fitted from `target_um` back to `source_um`, i.e. the pull "
                 "map the rasters were warped with. Local um = raster pixel index x "
                 "`local_um_per_px`. See gcalign.transforms."),
        "frame": grid_meta,
        "local_um_per_px": cfg.grid_um,
        "tps_kernel": "thin_plate_spline", "tps_degree": 1,
        "sections": out,
    }


def run(cfg: Config, jobs: int = 1) -> None:
    OUT = stage_dir(cfg, "03_gc_match")
    GRID_UM = cfg.grid_um
    PLANES = cfg.planes
    globals_json = json.loads((cfg.output_dir / "02_global_align/global_transforms.json").read_text())
    grid_meta = globals_json["grid"]
    grid = (grid_meta["x0_um"], grid_meta["y0_um"], grid_meta["height"], grid_meta["width"])

    # --- Stereo-seq planes: rasterise the frozen scaffold, no movement -----
    plane_labels: dict[str, np.ndarray] = dict(st_plane_labels(cfg, grid))
    for name in cfg.st_names:
        label = plane_labels[name]
        print(f"{name}: {int((label > 0).sum()):,} GC px, "
              f"{len(np.unique(label[label > 0]))} merged GCs", flush=True)
    st_labels = {name: plane_labels[name] for name in cfg.st_names}
    centroids = per_slice_centroids(cfg)

    # --- IF planes: Stage A similarity, then GC-anchored TPS --------------
    report, warped_channels, tissue_warped, transforms = {}, {}, {}, {}
    for name, plane_label, channels, tissue, entry, transform in run_sections(
            warp_section,
            [(cfg, name, globals_json["sections"][name], st_labels, centroids, grid)
             for name in cfg.if_names], jobs):
        plane_labels[name] = plane_label
        warped_channels.update(channels)
        tissue_warped[name] = tissue
        report[name] = entry
        transforms[name] = transform

    np.savez_compressed(OUT / "plane_gc_labels.npz",
                        **plane_labels,
                        grid=np.array([grid[0], grid[1], grid[2], grid[3]], np.float64))
    np.savez_compressed(OUT / "if_channels_warped.npz",
                        **warped_channels,
                        grid=np.array([grid[0], grid[1], grid[2], grid[3]], np.float64),
                        channels=np.array(sorted(cfg.if_images.channels.values()), dtype="U32"))
    np.savez_compressed(OUT / "if_tissue_warped.npz", **tissue_warped)
    (OUT / TRANSFORMS_FILE).write_text(
        json.dumps(transforms_document(cfg, grid_meta, transforms), indent=1))
    print(f"warped {len(warped_channels)} IF channel planes into the aligned frame")

    # --- presence table over every plane -----------------------------------
    rows = []
    all_ids = sorted({int(v) for label in plane_labels.values() for v in np.unique(label) if v > 0})
    for gid in all_ids:
        row = {"merged_gc_id": gid}
        for name, kind, index in PLANES:
            area = int((plane_labels[name] == gid).sum())
            row[f"{name}_px"] = area
            row[f"{name}_area_um2"] = area * GRID_UM ** 2
        present = [n for n, _, _ in PLANES if row[f"{n}_px"] > 0]
        row["n_planes"] = len(present)
        row["planes"] = "|".join(present)
        row["n_st_planes"] = sum(1 for n, k, _ in PLANES if k == "ST" and row[f"{n}_px"] > 0)
        row["n_if_planes"] = sum(1 for n, k, _ in PLANES if k == "IF" and row[f"{n}_px"] > 0)
        row["total_px"] = sum(row[f"{n}_px"] for n, _, _ in PLANES)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("total_px", ascending=False)
    table.to_csv(OUT / PRESENCE_FILE, index=False)

    report["planes"] = [{"name": n, "kind": k, "section_index": i,
                         "z_um": cfg.z_um(i)} for n, k, i in PLANES]
    report["grid"] = grid_meta
    (OUT / "match_report.json").write_text(json.dumps(report, indent=2))
    n_planes = len(PLANES)
    print(f"{len(table)} GCs on the {n_planes}-plane stack; "
          f"{int((table.n_planes == n_planes).sum())} present on all {n_planes}")
    print(table.head(12)[["merged_gc_id", "n_planes", "n_st_planes", "n_if_planes",
                          "total_px"]].to_string(index=False))
