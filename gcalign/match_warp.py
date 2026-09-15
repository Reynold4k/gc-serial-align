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
from scipy.optimize import linear_sum_assignment

from .config import Config, stage_dir

def match_pairs(src, dst, radius):
    cost = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
    blocked = cost > radius
    rows, cols = linear_sum_assignment(np.where(blocked, radius * 10, cost))
    keep = ~blocked[rows, cols]
    return rows[keep], cols[keep], cost[rows[keep], cols[keep]]


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


def warp_labels(cfg, mask, labels_by_object, linear, offset, inverse_tps, grid):
    """Pull-sample the IF label map onto the world grid through A then B."""
    x0, y0, height, width = grid
    GRID_UM = cfg.grid_um
    rows, cols = np.indices((height, width))
    world = np.stack([x0 + cols * GRID_UM, y0 + rows * GRID_UM], axis=-1).reshape(-1, 2)
    pre = inverse_tps(world) if inverse_tps is not None else world
    local = (pre - offset) @ np.linalg.inv(linear).T
    src_c = np.rint(local[:, 0] / GRID_UM).astype(int)
    src_r = np.rint(local[:, 1] / GRID_UM).astype(int)
    ok = (src_r >= 0) & (src_r < mask.shape[0]) & (src_c >= 0) & (src_c < mask.shape[1])
    out = np.zeros(height * width, np.int32)
    out[ok] = labels_by_object[src_r[ok], src_c[ok]]
    return out.reshape(height, width)


def run(cfg: Config) -> None:
    OUT = stage_dir(cfg, "03_gc_match")
    GRID_UM = cfg.grid_um
    SECTION_THICKNESS_UM = cfg.section_thickness_um
    PLANES = cfg.planes
    NEIGHBOURS = cfg.neighbours
    globals_json = json.loads((cfg.output_dir / "02_global_align/global_transforms.json").read_text())
    grid_meta = globals_json["grid"]
    grid = (grid_meta["x0_um"], grid_meta["y0_um"], grid_meta["height"], grid_meta["width"])
    shape = (grid_meta["height"], grid_meta["width"])

    scaffold = np.load(cfg.output_dir / "00_scaffold/st_scaffold.npz")
    slice_short = scaffold["slice_short"]
    gc_id = scaffold["gc_id"]
    world = scaffold["world_um"].astype(np.float64)

    summary = json.loads((cfg.output_dir / "00_scaffold/st_scaffold_summary.json").read_text())
    per_slice_centroid = {
        s: {r["merged_gc_id"]: np.array([r["per_slice"][s]["x_um"], r["per_slice"][s]["y_um"]])
            for r in summary["merged_gc"] if s in r["per_slice"]}
        for s in cfg.st_names
    }

    plane_labels: dict[str, np.ndarray] = {}

    # --- Stereo-seq planes: rasterise the frozen scaffold, no movement -----
    for name in cfg.st_names:
        keep = (slice_short == name) & (gc_id > 0)
        label = np.zeros(shape, np.int32)
        col = np.rint((world[keep, 0] - grid[0]) / GRID_UM).astype(int)
        row = np.rint((world[keep, 1] - grid[1]) / GRID_UM).astype(int)
        inside = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
        label[row[inside], col[inside]] = gc_id[keep][inside]
        # close the 1-px gaps the bin20 -> 10 um rounding leaves behind
        filled = ndi.grey_closing(label, size=(3, 3))
        label = np.where(label > 0, label, np.where(filled > 0, filled, 0))
        plane_labels[name] = label
        print(f"{name}: {int((label > 0).sum()):,} GC px, "
              f"{len(np.unique(label[label > 0]))} merged GCs", flush=True)

    # --- IF planes: Stage A similarity, then GC-anchored TPS --------------
    report = {}
    for entry in cfg.if_sections:
        section = entry.index
        key = f"IF{section}"
        entry = globals_json["sections"][key]
        linear = np.array(entry["linear"])
        offset = np.array(entry["offset"])
        data = np.load(cfg.output_dir / f"01_if_gc/IF{section}_gc.npz")
        gc = data["gc"]
        labels, n_obj = ndi.label(gc)
        centroids_local = np.array([[c * GRID_UM, r * GRID_UM] for r, c in
                                    ndi.center_of_mass(gc, labels, np.arange(1, n_obj + 1))])
        centroids_world = centroids_local @ linear.T + offset

        # match against each flanking slice independently
        per_target = {}
        for neighbour in NEIGHBOURS[section]:
            table = per_slice_centroid[neighbour]
            ids = np.array(sorted(table))
            pts = np.array([table[i] for i in ids])
            rows, cols, dists = match_pairs(centroids_world, pts, cfg.align.warp_match_radius_um)
            per_target[neighbour] = {"rows": rows, "ids": ids[cols],
                                     "targets": pts[cols], "dists": dists}
            print(f"{key} vs {neighbour}: {len(rows)}/{n_obj} matched, "
                  f"median {np.median(dists):.0f} um", flush=True)

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
        src = np.asarray(src); dst = np.asarray(dst)

        # forward for QC, inverse for rasterising
        smoothing, loo_median, loo_curve = choose_smoothing(cfg, src, dst)
        forward = fit_tps(src, dst, smoothing)
        inverse = fit_tps(dst, src, smoothing)
        residual = np.linalg.norm(forward(src) - dst, axis=1)
        pre = np.linalg.norm(src - dst, axis=1)
        print(f"{key}: smoothing {smoothing:.0e} (LOO median {loo_median:.0f} um), "
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
        plane_labels[key] = warp_labels(cfg, gc, id_map, linear, offset, inverse, grid)

        report[key] = {
            "n_if_objects": int(n_obj),
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
        print(f"{key}: {len(src)} anchors, TPS residual median "
              f"{np.median(residual):.0f} um, {len(unmatched)} IF-only objects"
              + (f", two-sided disagreement median {disagreement['median_um']:.0f} um"
                 if disagreement else "") + "\n", flush=True)

    np.savez_compressed(OUT / "plane_gc_labels.npz",
                        **plane_labels,
                        grid=np.array([grid[0], grid[1], grid[2], grid[3]], np.float64))

    # --- presence table over all nine planes ------------------------------
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
    table.to_csv(OUT / "gc_presence_9plane.csv", index=False)

    report["planes"] = [{"name": n, "kind": k, "section_index": i,
                         "z_um": (i - 1) * SECTION_THICKNESS_UM} for n, k, i in PLANES]
    report["grid"] = grid_meta
    (OUT / "match_report.json").write_text(json.dumps(report, indent=2))
    print(f"{len(table)} GCs on the 9-plane stack; "
          f"{int((table.n_planes == 9).sum())} present on all nine")
    print(table.head(12)[["merged_gc_id", "n_planes", "n_st_planes", "n_if_planes",
                          "total_px"]].to_string(index=False))

