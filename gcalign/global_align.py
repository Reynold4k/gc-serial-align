"""Stage 3 - place every IF section into the Stereo-seq coordinate frame.

Three deterministic steps, no optimiser that can wander off:
  A1  brute force over {mirror} x {rotation} x FFT translation on tissue masks
  A2  local refinement of rotation and scale
  A3  ICP on GC centroids against the transcriptomics consensus

Mirror state and large rotations are discrete and non-convex, which is exactly
what a gradient optimiser gets wrong, so they are enumerated instead. The
winning candidate is chosen by how well the germinal centres agree, not by how
well the tissue outlines overlap: a near-convex section overlaps almost as well
under the wrong mirror, and that is how you silently flip a section.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from .config import Config, stage_dir
from .parallel import run_sections

def load_scaffold(cfg: Config):
    data = np.load(cfg.output_dir / "00_scaffold/st_scaffold.npz", allow_pickle=False)
    return data["slice_short"], data["gc_id"], data["world_um"].astype(np.float64)


def world_grid(cfg: Config, world: np.ndarray):
    GRID_UM = cfg.grid_um
    x0 = world[:, 0].min() - cfg.align.margin_px * GRID_UM
    y0 = world[:, 1].min() - cfg.align.margin_px * GRID_UM
    width = int(np.ceil((world[:, 0].max() - x0) / GRID_UM)) + cfg.align.margin_px
    height = int(np.ceil((world[:, 1].max() - y0) / GRID_UM)) + cfg.align.margin_px
    return x0, y0, height, width


def rasterise(cfg: Config, points: np.ndarray, x0: float, y0: float, shape) -> np.ndarray:
    GRID_UM = cfg.grid_um
    grid = np.zeros(shape, bool)
    col = np.rint((points[:, 0] - x0) / GRID_UM).astype(int)
    row = np.rint((points[:, 1] - y0) / GRID_UM).astype(int)
    keep = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    grid[row[keep], col[keep]] = True
    return ndi.binary_closing(grid, np.ones((3, 3)))


def shrink(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return mask
    h = mask.shape[0] // factor * factor
    w = mask.shape[1] // factor * factor
    return mask[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)) > 0.4


class Correlator:
    """Circular FFT cross-correlation against one fixed mask, FFT cached."""

    def __init__(self, fixed: np.ndarray, factor: int = 1):
        self.factor = factor
        small = shrink(fixed, factor)
        self.pad = tuple(int(2 ** np.ceil(np.log2(2 * n))) for n in small.shape)
        board = np.zeros(self.pad, np.float32)
        board[:small.shape[0], :small.shape[1]] = small
        self.fixed_fft = np.fft.rfft2(board)

    def best_shift(self, moving: np.ndarray):
        small = shrink(moving, self.factor)
        board = np.zeros(self.pad, np.float32)
        board[:small.shape[0], :small.shape[1]] = small
        corr = np.fft.irfft2(self.fixed_fft * np.conj(np.fft.rfft2(board)), s=self.pad)
        dy, dx = np.unravel_index(int(np.argmax(corr)), self.pad)
        dy = dy - self.pad[0] if dy > self.pad[0] // 2 else dy
        dx = dx - self.pad[1] if dx > self.pad[1] // 2 else dx
        return int(dy) * self.factor, int(dx) * self.factor, float(corr.max())


def place(mask: np.ndarray, centre_rc, shape, angle_deg: float, scale: float, flip: bool):
    """Centre `mask` on the world grid centre, then mirror / scale / rotate."""
    work = mask[:, ::-1] if flip else mask
    cr, cc = centre_rc
    if flip:
        cc = mask.shape[1] - 1 - cc
    canvas = np.zeros(shape, np.float32)
    tr = int(round(shape[0] / 2 - cr))
    tc = int(round(shape[1] / 2 - cc))
    rs, re = max(tr, 0), min(tr + work.shape[0], shape[0])
    cs, ce = max(tc, 0), min(tc + work.shape[1], shape[1])
    canvas[rs:re, cs:ce] = work[rs - tr:re - tr, cs - tc:ce - tc]
    if abs(scale - 1.0) > 1e-6:
        canvas = ndi.affine_transform(
            canvas,
            np.diag([1 / scale, 1 / scale]),
            offset=[shape[0] / 2 * (1 - 1 / scale), shape[1] / 2 * (1 - 1 / scale)],
            order=1, output_shape=shape,
        )
    if abs(angle_deg) > 1e-6:
        canvas = ndi.rotate(canvas, angle_deg, reshape=False, order=1, prefilter=False)
    return canvas > 0.5


def warp_mask(cfg: Config, mask: np.ndarray, linear: np.ndarray, offset: np.ndarray,
              x0: float, y0: float, shape) -> np.ndarray:
    """Pull-sample `mask` (IF pixel grid) onto the world grid under linear/offset."""
    GRID_UM = cfg.grid_um
    inv = np.linalg.inv(linear)
    rows, cols = np.indices(shape)
    wx = x0 + cols * GRID_UM
    wy = y0 + rows * GRID_UM
    local = np.stack([wx - offset[0], wy - offset[1]], axis=-1) @ inv.T
    src_c = local[..., 0] / GRID_UM
    src_r = local[..., 1] / GRID_UM
    valid = (src_r >= 0) & (src_r < mask.shape[0] - 1) & (src_c >= 0) & (src_c < mask.shape[1] - 1)
    out = np.zeros(shape, bool)
    out[valid] = mask[np.rint(src_r[valid]).astype(int), np.rint(src_c[valid]).astype(int)]
    return out


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else 0.0


def similarity_from_pairs(src: np.ndarray, dst: np.ndarray):
    """Umeyama similarity fit (rotation + isotropic scale + translation)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    correction = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[1, 1] = -1
    rot = u @ correction @ vt
    scale = float(np.trace(np.diag(sig) @ correction) / max((s ** 2).sum() / len(src), 1e-12))
    return rot, scale, mu_d - scale * rot @ mu_s


def match_pairs(src: np.ndarray, dst: np.ndarray, radius: float):
    cost = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
    blocked = cost > radius
    padded = np.where(blocked, radius * 10, cost)
    rows, cols = linear_sum_assignment(padded)
    keep = ~blocked[rows, cols]
    return rows[keep], cols[keep], cost[rows[keep], cols[keep]]


MAX_ICP_ITERATIONS = 12


def icp_targets(summary: dict, neighbours: list[str]):
    """The Stereo-seq centroids an IF section is matched against, with their ids.

    Two flanking sections: the consensus centroid of every merged GC. One
    flanking section (the ends of the stack): that section's own centroids.
    """
    points, ids = [], []
    for row in summary["merged_gc"]:
        if len(neighbours) == 1:
            if neighbours[0] not in row["per_slice"]:
                continue
            here = row["per_slice"][neighbours[0]]
            points.append([here["x_um"], here["y_um"]])
        else:
            points.append([row["consensus_x_um"], row["consensus_y_um"]])
        ids.append(row["merged_gc_id"])
    return np.asarray(points, float).reshape(-1, 2), np.asarray(ids, int)


def run_icp(source: np.ndarray, target: np.ndarray, target_ids: np.ndarray,
            linear: np.ndarray, offset: np.ndarray, radius_um: float):
    """ICP on GC centroids, keeping every step it takes.

    Returns the final (linear, offset), the per-iteration history - (pairs,
    mean paired distance) - and the steps as transforms, so the optimisation
    can be shown or audited afterwards and not only its end point.
    """
    history = []
    steps = [{"iteration": 0, "linear": linear.tolist(), "offset": offset.tolist()}]
    for index in range(MAX_ICP_ITERATIONS):
        moved = source @ linear.T + offset
        rows, cols, dists = match_pairs(moved, target, radius_um)
        if len(rows) < 6:
            break
        rot_i, scale_i, trans_i = similarity_from_pairs(moved[rows], target[cols])
        linear = scale_i * rot_i @ linear
        offset = scale_i * rot_i @ offset + trans_i
        history.append((int(len(rows)), float(dists.mean())))
        steps.append({
            "iteration": index + 1,
            "linear": linear.tolist(), "offset": offset.tolist(),
            "matched_if_index": rows.tolist(),
            "matched_gc_id": target_ids[cols].astype(int).tolist(),
            "target_points_um": target[cols].tolist(),
            "mean_distance_before_update_um": float(dists.mean()),
        })
    return linear, offset, history, steps


def stage_a_start(centre_rc, angle_deg: float, scale: float, flip: bool,
                  dy: int, dx: int, grid_um: float, x0: float, y0: float, shape):
    """The similarity transform a (mirror, angle, scale, shift) candidate stands for."""
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    mirror = np.diag([-1.0, 1.0]) if flip else np.eye(2)
    linear = scale * rot @ mirror
    centre_um = np.array([centre_rc[1] * grid_um, centre_rc[0] * grid_um])
    grid_centre = np.array([x0 + shape[1] / 2 * grid_um, y0 + shape[0] / 2 * grid_um])
    offset = grid_centre + np.array([dx * grid_um, dy * grid_um]) - linear @ centre_um
    return linear, offset


def align_section(cfg: Config, name: str, shared: dict):
    """Place one IF section. Independent of every other section, so it can run
    in its own process."""
    GRID_UM = cfg.grid_um
    x0, y0, shape = shared["x0"], shared["y0"], shared["shape"]
    neighbours = cfg.neighbours[cfg.section(name).index]
    data = np.load(cfg.output_dir / f"01_if_gc/{name}_gc.npz")
    tissue, gc = data["tissue"], data["gc"]
    centre = ndi.center_of_mass(tissue)

    target_tissue = np.logical_or.reduce([shared["st_tissue"][s] for s in neighbours])
    target_gc = np.logical_or.reduce([shared["st_gc"][s] for s in neighbours])

    # --- A1 exhaustive orientation search on tissue, 40 um pyramid ----
    # Keep several candidates: a near-convex section overlaps well under the
    # wrong mirror too, so tissue Dice alone picks the wrong branch.
    coarse = Correlator(target_tissue, factor=4)
    coarse_target = shrink(target_tissue, 4)
    scored = []
    for flip in (False, True):
        for angle in range(0, 360, cfg.align.coarse_angle_step_deg):
            moved = place(tissue, centre, shape, angle, 1.0, flip)
            dy, dx, _ = coarse.best_shift(moved)
            score = dice(np.roll(shrink(moved, 4), (dy // 4, dx // 4), axis=(0, 1)),
                         coarse_target)
            scored.append((score, flip, float(angle)))
    scored.sort(key=lambda item: -item[0])
    candidates, seen = [], []
    for score, flip, angle in scored:
        if any(f == flip and min(abs(angle - a), 360 - abs(angle - a)) < 25 for _, f, a in seen):
            continue
        seen.append((score, flip, angle))
        candidates.append((score, flip, angle))
        if len(candidates) == cfg.align.n_orientation_candidates:
            break
    print(f"{name}  A1 candidates: "
          + ", ".join(f"{'M' if f else 'N'}{a:.0f}({d:.3f})" for d, f, a in candidates),
          flush=True)

    fine = Correlator(target_tissue, factor=2)
    fine_target = shrink(target_tissue, 2)
    labels, _ = ndi.label(gc)
    props = ndi.center_of_mass(gc, labels, np.arange(1, labels.max() + 1))
    if_centroids = np.array([[c * GRID_UM, r * GRID_UM] for r, c in props]).reshape(-1, 2)
    areas = np.bincount(labels.ravel())[1:] * GRID_UM ** 2

    target_pts, target_ids = icp_targets(shared["summary"], neighbours)

    winner = None
    searched = []
    for coarse_dice, flip, angle0 in candidates:
        # --- A2 local rotation / scale refinement, 20 um pyramid ------
        best = (-1.0, flip, angle0, 1.0, 0, 0)
        for angle in np.arange(angle0 - 6, angle0 + 6.01, 1.0):
            for scale in np.arange(*cfg.align.scale_search):
                moved = place(tissue, centre, shape, float(angle), float(scale), flip)
                dy, dx, _ = fine.best_shift(moved)
                score = dice(np.roll(shrink(moved, 2), (dy // 2, dx // 2), axis=(0, 1)),
                             fine_target)
                if score > best[0]:
                    best = (score, flip, float(angle), float(scale), dy, dx)
        tissue_dice, flip_b, angle, scale, dy, dx = best
        linear, offset = stage_a_start(centre, angle, scale, flip_b, dy, dx,
                                       GRID_UM, x0, y0, shape)

        # --- A3 ICP on GC centroids -----------------------------------
        linear, offset, history, steps = run_icp(
            if_centroids, target_pts, target_ids, linear, offset,
            cfg.align.icp_match_radius_um)
        moved_pts = if_centroids @ linear.T + offset
        rows, cols, dists = match_pairs(moved_pts, target_pts, cfg.align.icp_match_radius_um)
        # the decision metric is GC agreement, not tissue overlap
        close = int((dists < cfg.align.close_match_um).sum()) if len(rows) else 0
        searched.append({
            "flip": bool(flip_b), "coarse_angle_deg": float(angle0),
            "coarse_tissue_dice": float(coarse_dice),
            "angle_deg": float(angle), "scale": float(scale),
            "tissue_dice": float(tissue_dice),
            "n_matched": int(len(rows)), "n_close": int(close),
            "median_um": float(np.median(dists)) if len(rows) else None,
        })
        if winner is None or close > winner[0]:
            winner = (close, tissue_dice, bool(flip_b), float(angle), float(scale),
                      linear, offset, rows, cols, dists, history, steps, len(searched) - 1)
        print(f"{name}    cand {'M' if flip_b else 'N'}{angle0:.0f}: "
              f"tissue {tissue_dice:.3f}, matched {len(rows)}, "
              f"<{cfg.align.close_match_um:.0f}um {close}, "
              f"median {np.median(dists):.0f} um" if len(rows) else
              f"{name}    cand {'M' if flip_b else 'N'}{angle0:.0f}: no match", flush=True)

    (close, tissue_dice, flip, angle, scale, linear, offset,
     rows, cols, dists, history, steps, winner_index) = winner

    warped_gc = warp_mask(cfg, gc, linear, offset, x0, y0, shape)
    gc_dice = dice(warped_gc, target_gc)
    warped_tissue = warp_mask(cfg, tissue, linear, offset, x0, y0, shape)
    tissue_dice_final = dice(warped_tissue, target_tissue)

    result = {
        "section": cfg.section(name).index,
        "neighbours": neighbours,
        "flip": bool(flip), "angle_deg": float(angle), "scale_a2": float(scale),
        "linear": linear.tolist(), "offset": offset.tolist(),
        "total_scale": float(np.sqrt(abs(np.linalg.det(linear)))),
        "tissue_dice_a2": float(tissue_dice),
        "tissue_dice_final": float(tissue_dice_final),
        "gc_dice_final": float(gc_dice),
        "n_matched_under_150um": int(close),
        "n_if_gc": int(len(if_centroids)),
        "n_matched": int(len(rows)),
        "match_rms_um": float(np.sqrt((dists ** 2).mean())) if len(rows) else None,
        "match_median_um": float(np.median(dists)) if len(rows) else None,
        "icp_history": history,
        "if_centroids_um": if_centroids.tolist(),
        "if_areas_um2": areas.tolist(),
        "matched_if_index": rows.tolist(),
        "matched_merged_gc_id": target_ids[cols].astype(int).tolist(),
        "matched_distance_um": dists.tolist(),
    }
    trajectory = {
        "source_centroids_um": if_centroids.tolist(),
        "steps": steps,
        "n_iterations": len(history),
        "origin": "stored by stage align",
    }
    search = {"winner": int(winner_index), "candidates": searched}
    if len(rows):
        print(f"{name}  WON {'mirror' if flip else 'direct'} {angle:.1f} deg scale {scale:.3f} | "
              f"matched {len(rows)}/{len(if_centroids)} ({close} under "
              f"{cfg.align.close_match_um:.0f} um), "
              f"median {np.median(dists):.0f} um | tissue dice {tissue_dice_final:.3f}, "
              f"GC dice {gc_dice:.3f}\n", flush=True)
    else:
        print(f"{name}  no orientation matched any germinal centre\n", flush=True)
    return name, result, trajectory, search


def run(cfg: Config, jobs: int = 1) -> None:
    OUT = stage_dir(cfg, "02_global_align")
    GRID_UM = cfg.grid_um
    ST_SLICES = cfg.st_names
    slice_short, gc_id, world = load_scaffold(cfg)
    x0, y0, height, width = world_grid(cfg, world)
    shape = (height, width)
    print(f"world grid {shape} at {GRID_UM} um/px, origin ({x0:.0f}, {y0:.0f}) um")

    st_tissue = {s: rasterise(cfg, world[slice_short == s], x0, y0, shape) for s in ST_SLICES}
    st_gc = {s: rasterise(cfg, world[(slice_short == s) & (gc_id > 0)], x0, y0, shape) for s in ST_SLICES}
    st_tissue_all = np.logical_or.reduce(list(st_tissue.values()))

    summary = json.loads((cfg.output_dir / "00_scaffold/st_scaffold_summary.json").read_text())
    shared = {"x0": x0, "y0": y0, "shape": shape, "st_tissue": st_tissue, "st_gc": st_gc,
              "summary": summary}

    results, trajectories, searches = {}, {}, {}
    for name, result, trajectory, search in run_sections(
            align_section, [(cfg, name, shared) for name in cfg.if_names], jobs):
        results[name] = result
        trajectories[name] = trajectory
        searches[name] = search

    unmatched = [name for name, r in results.items() if r["n_matched"] < 6]
    if unmatched:
        raise RuntimeError(
            f"fewer than six germinal centres could be matched on {unmatched}. The registration "
            "is driven by matched structures, so it cannot continue. Check that `gc_channel` "
            "is the marker that shows the structure, that the sections are listed in cutting "
            "order, and look at 01_if_gc/*_gc_mask.tif - a tissue outline that lost part of "
            "the section (see `tissue_threshold_scale`) takes its structures with it.")

    (OUT / "global_transforms.json").write_text(json.dumps({
        "grid": {"um_per_px": GRID_UM, "x0_um": x0, "y0_um": y0,
                 "height": height, "width": width},
        "sections": results,
    }, indent=2))
    # how the answer was reached: the ICP as steps, and every orientation
    # candidate with both of its scores
    (OUT / "icp_trajectory.json").write_text(json.dumps(trajectories))
    (OUT / "orientation_search.json").write_text(json.dumps({
        "close_match_um": cfg.align.close_match_um, "sections": searches}, indent=2))
    np.savez_compressed(OUT / "st_rasters.npz",
                        **{f"tissue_{s}": st_tissue[s] for s in ST_SLICES},
                        **{f"gc_{s}": st_gc[s] for s in ST_SLICES},
                        tissue_all=st_tissue_all)
    print("wrote", OUT / "global_transforms.json")
