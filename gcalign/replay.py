"""The registration as a trajectory, not just an answer.

Stage `align` ends in one transform per IF section, but it got there in steps:
a placement from the orientation search, then up to twelve ICP iterations on the
germinal-centre centroids. Runs made with gcalign 0.2 or later store those steps
in ``02_global_align/icp_trajectory.json``. For an older run they are rebuilt
here from what the run did store - the winning orientation, the masks and the
centroids - with the pipeline's own functions, and the rebuild is only accepted
if it lands on the stored transform and the stored per-iteration history.

That check is the point: an animation driven from these steps shows the
optimisation that actually ran, not a tween between two endpoints.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import ndimage as ndi

from .config import Config
from .global_align import Correlator, icp_targets, place, run_icp, stage_a_start

TRAJECTORY_FILE = "02_global_align/icp_trajectory.json"


def rebuild(cfg: Config, name: str) -> dict:
    """Reconstruct one section's ICP steps from a finished run and verify them."""
    root = cfg.output_dir
    transforms = json.loads((root / "02_global_align/global_transforms.json").read_text())
    entry = transforms["sections"][name]
    grid = transforms["grid"]
    shape = (int(grid["height"]), int(grid["width"]))
    um = float(grid["um_per_px"])
    summary = json.loads((root / "00_scaffold/st_scaffold_summary.json").read_text())
    rasters = np.load(root / "02_global_align/st_rasters.npz")
    local = np.load(root / f"01_if_gc/{name}_gc.npz")

    # the winning candidate's placement, exactly as the orientation search made it
    tissue = local["tissue"]
    centre = ndi.center_of_mass(tissue)
    target_mask = np.logical_or.reduce([rasters[f"tissue_{n}"] for n in entry["neighbours"]])
    moved = place(tissue, centre, shape, entry["angle_deg"], entry["scale_a2"], entry["flip"])
    dy, dx, _ = Correlator(target_mask, factor=2).best_shift(moved)
    linear, offset = stage_a_start(centre, entry["angle_deg"], entry["scale_a2"], entry["flip"],
                                   dy, dx, um, grid["x0_um"], grid["y0_um"], shape)

    source = np.asarray(entry["if_centroids_um"], float).reshape(-1, 2)
    target, target_ids = icp_targets(summary, entry["neighbours"])
    linear, offset, history, steps = run_icp(
        source, target, target_ids, linear, offset, cfg.align.icp_match_radius_um)

    error_linear = float(np.max(np.abs(linear - np.asarray(entry["linear"]))))
    error_offset = float(np.max(np.abs(offset - np.asarray(entry["offset"]))))
    saved = np.asarray(entry["icp_history"], float).reshape(-1, 2)
    mine = np.asarray(history, float).reshape(-1, 2)
    if len(saved) != len(mine):
        error_history = float("inf")
    else:
        error_history = float(np.max(np.abs(mine - saved))) if len(mine) else 0.0
    if not (error_linear < 1e-9 and error_offset < 1e-6 and error_history < 1e-6):
        raise RuntimeError(
            f"{name}: the rebuilt ICP does not land on the stored transform "
            f"(linear {error_linear:.2e}, offset {error_offset:.2e} um, history {error_history:.2e}); "
            "the run's intermediate files do not belong to its global_transforms.json")
    return {
        "source_centroids_um": source.tolist(),
        "steps": steps,
        "n_iterations": len(history),
        "origin": "rebuilt from the stored orientation and verified against the stored transform",
        "max_linear_error": error_linear,
        "max_offset_error_um": error_offset,
        "max_history_error": error_history,
    }


def icp_trajectory(cfg: Config, name: str) -> dict:
    """The ICP steps of one IF section: stored if the run has them, rebuilt if not."""
    stored = cfg.output_dir / TRAJECTORY_FILE
    if stored.exists():
        data = json.loads(stored.read_text())
        if name in data:
            return data[name]
    return rebuild(cfg, name)
