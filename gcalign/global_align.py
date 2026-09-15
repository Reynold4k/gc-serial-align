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


def run(cfg: Config) -> None:
    OUT = stage_dir(cfg, "02_global_align")
    GRID_UM = cfg.grid_um
    IF_SECTIONS = [s.index for s in cfg.if_sections]
    ST_SLICES = cfg.st_names
    NEIGHBOURS = cfg.neighbours
    slice_short, gc_id, world = load_scaffold(cfg)
    x0, y0, height, width = world_grid(cfg, world)
    shape = (height, width)
    print(f"world grid {shape} at {GRID_UM} um/px, origin ({x0:.0f}, {y0:.0f}) um")

    st_tissue = {s: rasterise(cfg, world[slice_short == s], x0, y0, shape) for s in ST_SLICES}
    st_gc = {s: rasterise(cfg, world[(slice_short == s) & (gc_id > 0)], x0, y0, shape) for s in ST_SLICES}
    st_tissue_all = np.logical_or.reduce(list(st_tissue.values()))

    summary = json.loads((cfg.output_dir / "00_scaffold/st_scaffold_summary.json").read_text())
    consensus = np.array([[r["consensus_x_um"], r["consensus_y_um"]] for r in summary["merged_gc"]])
    consensus_id = np.array([r["merged_gc_id"] for r in summary["merged_gc"]])
    per_slice_centroid = {
        s: {r["merged_gc_id"]: (r["per_slice"][s]["x_um"], r["per_slice"][s]["y_um"])
            for r in summary["merged_gc"] if s in r["per_slice"]}
        for s in ST_SLICES
    }

    results = {}
    for section in IF_SECTIONS:
        data = np.load(cfg.output_dir / f"01_if_gc/IF{section}_gc.npz")
        tissue, gc = data["tissue"], data["gc"]
        centre = ndi.center_of_mass(tissue)

        target_tissue = np.logical_or.reduce([st_tissue[s] for s in NEIGHBOURS[section]])
        target_gc = np.logical_or.reduce([st_gc[s] for s in NEIGHBOURS[section]])

        # --- A1 exhaustive orientation search on tissue, 40 um pyramid ----
        # Keep several candidates: a near-convex section overlaps well under the
        # wrong mirror too, so tissue Dice alone picks the wrong branch (IF9).
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
        print(f"IF{section}  A1 candidates: "
              + ", ".join(f"{'M' if f else 'N'}{a:.0f}({d:.3f})" for d, f, a in candidates),
              flush=True)

        fine = Correlator(target_tissue, factor=2)
        fine_target = shrink(target_tissue, 2)
        labels, _ = ndi.label(gc)
        props = ndi.center_of_mass(gc, labels, np.arange(1, labels.max() + 1))
        if_centroids = np.array([[c * GRID_UM, r * GRID_UM] for r, c in props])
        areas = np.bincount(labels.ravel())[1:] * GRID_UM ** 2

        target_pts = consensus
        target_ids = consensus_id
        if len(NEIGHBOURS[section]) == 1:
            table = per_slice_centroid[NEIGHBOURS[section][0]]
            target_ids = np.array([i for i in consensus_id if i in table])
            target_pts = np.array([table[i] for i in target_ids])

        winner = None
        for _, flip, angle0 in candidates:
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

            theta = np.deg2rad(angle)
            rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            mirror = np.diag([-1.0, 1.0]) if flip_b else np.eye(2)
            linear = scale * rot @ mirror
            centre_um = np.array([centre[1] * GRID_UM, centre[0] * GRID_UM])
            grid_centre = np.array([x0 + shape[1] / 2 * GRID_UM, y0 + shape[0] / 2 * GRID_UM])
            offset = grid_centre + np.array([dx * GRID_UM, dy * GRID_UM]) - linear @ centre_um

            # --- A3 ICP on GC centroids -----------------------------------
            history = []
            for _ in range(12):
                moved_pts = if_centroids @ linear.T + offset
                rows, cols, dists = match_pairs(moved_pts, target_pts, cfg.align.icp_match_radius_um)
                if len(rows) < 6:
                    break
                rot_i, scale_i, trans_i = similarity_from_pairs(moved_pts[rows], target_pts[cols])
                linear = scale_i * rot_i @ linear
                offset = scale_i * rot_i @ offset + trans_i
                history.append((int(len(rows)), float(dists.mean())))
            moved_pts = if_centroids @ linear.T + offset
            rows, cols, dists = match_pairs(moved_pts, target_pts, cfg.align.icp_match_radius_um)
            # the decision metric is GC agreement, not tissue overlap
            close = int((dists < cfg.align.close_match_um).sum()) if len(rows) else 0
            if winner is None or close > winner[0]:
                winner = (close, tissue_dice, bool(flip_b), float(angle), float(scale),
                          linear, offset, rows, cols, dists, history)
            print(f"IF{section}    cand {'M' if flip_b else 'N'}{angle0:.0f}: "
                  f"tissue {tissue_dice:.3f}, matched {len(rows)}, <150um {close}, "
                  f"median {np.median(dists):.0f} um" if len(rows) else
                  f"IF{section}    cand {'M' if flip_b else 'N'}{angle0:.0f}: no match", flush=True)

        (close, tissue_dice, flip, angle, scale, linear, offset,
         rows, cols, dists, history) = winner

        warped_gc = warp_mask(cfg, gc, linear, offset, x0, y0, shape)
        gc_dice = dice(warped_gc, target_gc)
        warped_tissue = warp_mask(cfg, tissue, linear, offset, x0, y0, shape)
        tissue_dice_final = dice(warped_tissue, target_tissue)

        results[f"IF{section}"] = {
            "section": section,
            "neighbours": NEIGHBOURS[section],
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
        print(f"IF{section}  WON {'mirror' if flip else 'direct'} {angle:.1f} deg scale {scale:.3f} | "
              f"matched {len(rows)}/{len(if_centroids)} ({close} under 150 um), "
              f"median {np.median(dists):.0f} um | tissue dice {tissue_dice_final:.3f}, "
              f"GC dice {gc_dice:.3f}\n", flush=True)

    (OUT / "global_transforms.json").write_text(json.dumps({
        "grid": {"um_per_px": GRID_UM, "x0_um": x0, "y0_um": y0,
                 "height": height, "width": width},
        "sections": results,
    }, indent=2))
    np.savez_compressed(OUT / "st_rasters.npz",
                        **{f"tissue_{s}": st_tissue[s] for s in ST_SLICES},
                        **{f"gc_{s}": st_gc[s] for s in ST_SLICES},
                        tissue_all=st_tissue_all)
    print("wrote", OUT / "global_transforms.json")

