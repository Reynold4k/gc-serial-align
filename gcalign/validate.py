"""Stage 7 - collect the QC numbers and hash the deliverables."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Config

CAVEATS = [
    "The z span is set by how many sections you cut and how thick they are. If it is "
    "small next to a germinal centre, the columns are GC mid-sections, not whole GCs, "
    "and they cannot be used to argue about sphericity.",
    "The registration is driven by GC geometry alone; tissue outline is only used for "
    "the initial orientation search. That makes this a maximum-overlap atlas: good for "
    "showing shape, not for formal GC volume statistics. For those, re-run with the GC "
    "term down-weighted and tissue morphology leading.",
    "Each mesh closes a short distance beyond the outermost section so the surface is "
    "watertight. That cap is a rendering choice, not measured data.",
    "The first and last sections have a neighbour on one side only, so their placement "
    "is less certain than the interior sections.",
]


def if_coverage(cfg: Config) -> dict:
    """How much of the transcriptomics tissue does each IF section actually reach?

    A germinal centre can only be matched where the IF section covers it. When a
    section is torn, folded, or was scanned over a smaller field than the chip,
    the GCs in that region have no IF counterpart, and no amount of threshold
    tuning brings them back. This makes that visible instead of leaving it to be
    discovered from a puzzling 3D picture.
    """
    import json as _json

    from scipy import ndimage as ndi

    labels = np.load(cfg.output_dir / "03_gc_match/plane_gc_labels.npz")
    x0, y0, height, width = (float(v) for v in labels["grid"])
    height, width = int(height), int(width)
    grid_um = cfg.grid_um

    scaffold = np.load(cfg.output_dir / "00_scaffold/st_scaffold.npz")
    world, slice_label = scaffold["world_um"], scaffold["slice_short"]
    st = np.zeros((height, width), bool)
    for name in cfg.st_names:
        w = world[slice_label == name]
        col = np.rint((w[:, 0] - x0) / grid_um).astype(int)
        row = np.rint((w[:, 1] - y0) / grid_um).astype(int)
        keep = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        st[row[keep], col[keep]] = True
    st = ndi.binary_closing(st, np.ones((3, 3)))

    transforms = _json.loads(
        (cfg.output_dir / "02_global_align/global_transforms.json").read_text())["sections"]
    rows, cols = np.indices((height, width))
    wx = x0 + cols * grid_um
    wy = y0 + rows * grid_um

    per_section, stack = {}, np.zeros((height, width), np.uint8)
    for entry in cfg.if_sections:
        key = f"IF{entry.index}"
        data = np.load(cfg.output_dir / f"01_if_gc/{key}_gc.npz")
        tissue = data["tissue"]
        info = transforms[key]
        inverse = np.linalg.inv(np.array(info["linear"]))
        offset = np.array(info["offset"])
        local = np.stack([wx - offset[0], wy - offset[1]], axis=-1) @ inverse.T
        src_r = np.rint(local[..., 1] / grid_um).astype(int)
        src_c = np.rint(local[..., 0] / grid_um).astype(int)
        inside = ((src_r >= 0) & (src_r < tissue.shape[0])
                  & (src_c >= 0) & (src_c < tissue.shape[1]))
        covered = np.zeros((height, width), bool)
        covered[inside] = tissue[src_r[inside], src_c[inside]]
        stack += covered
        missed = st & ~covered
        per_section[key] = {
            "covers_st_tissue_pct": round(float((st & covered).sum() / st.sum() * 100), 1),
            "missed_mm2": round(float(missed.sum() * grid_um ** 2 / 1e6), 2),
            "missed_outside_imaged_area_mm2":
                round(float((missed & ~inside).sum() * grid_um ** 2 / 1e6), 2),
        }
    return {
        "st_tissue_mm2": round(float(st.sum() * grid_um ** 2 / 1e6), 2),
        "st_tissue_covered_by_all_if_pct": round(float((stack[st] == len(cfg.if_sections)).mean() * 100), 1),
        "st_tissue_covered_by_no_if_pct": round(float((stack[st] == 0).mean() * 100), 1),
        "per_section": per_section,
        "note": ("GCs in tissue no IF section reaches cannot be matched. Lowering "
                 "if_images.tissue_threshold_scale recovers dim tissue, but not tissue "
                 "that is torn away or outside the imaged field."),
    }


def deliverables(cfg: Config) -> list[str]:
    return [
        f"05_viewer/{cfg.viewer.filename}",
        "04_columns/gc_columns.npz",
        "04_columns/gc_plane_patches.npz",
        "04_columns/gc_column_metrics.csv",
        "03_gc_match/plane_gc_labels.npz",
        "03_gc_match/gc_presence_9plane.csv",
        "03_gc_match/match_report.json",
        "02_global_align/global_transforms.json",
        "01_if_gc/if_gc_manifest.json",
    ]


def run(cfg: Config) -> None:
    ROOT = cfg.output_dir
    DELIVERABLES = deliverables(cfg)
    glob = json.loads((cfg.output_dir / "02_global_align/global_transforms.json").read_text())
    match = json.loads((cfg.output_dir / "03_gc_match/match_report.json").read_text())
    build = json.loads((cfg.output_dir / "04_columns/column_build_report.json").read_text())
    if_manifest = json.loads((cfg.output_dir / "01_if_gc/if_gc_manifest.json").read_text())
    metrics = pd.read_csv(cfg.output_dir / "04_columns/gc_column_metrics.csv")
    presence = pd.read_csv(ROOT / "03_gc_match/gc_presence_9plane.csv")

    orient = {k: {"mirror": v["flip"], "angle_deg": v["angle_deg"],
                  "scale": round(v["total_scale"], 4)}
              for k, v in glob["sections"].items()}
    all_direct = not any(v["mirror"] for v in orient.values())
    angles = [v["angle_deg"] for v in orient.values()]

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "experiment": cfg.name,
        "geometry": {
            "section_thickness_um": cfg.section_thickness_um,
            "n_sections": len(cfg.sections),
            "z_span_um": cfg.z_um(max(s.index for s in cfg.sections)),
            "order": [p[0] for p in cfg.planes],
        },
        "channels": cfg.if_images.channels,
        "gc_channel": cfg.if_images.gc_channel,
        "stage_A_orientation": {
            "per_section": orient,
            "all_sections_same_handedness": all_direct,
            "angle_range_deg": [min(angles), max(angles)],
            "note": ("Every IF section independently converged to the same mirror state and "
                     "a rotation within 3 deg; the orientation branch was chosen by GC "
                     "agreement, not by tissue overlap, because the near-convex outline "
                     "scores almost as well under the wrong mirror."),
        },
        "stage_A_gc_match": {
            k: {"n_if_gc": v["n_if_gc"], "matched": v["n_matched"],
                "within_150um": v["n_matched_under_150um"],
                "median_um": round(v["match_median_um"], 1),
                "tissue_dice": round(v["tissue_dice_final"], 3),
                "gc_dice": round(v["gc_dice_final"], 3)}
            for k, v in glob["sections"].items()
        },
        "stage_B_warp": {
            k: {"anchors": v["n_anchor_pairs"],
                "centroid_error_before_um": round(v["centroid_error_pre_um"], 1),
                "centroid_error_after_um": round(v["tps_residual_median_um"], 1),
                "leave_one_out_um": round(v["tps_loo_median_um"], 1),
                "tps_smoothing": v["tps_smoothing"],
                "two_sided_disagreement_um": (
                    round(v["two_sided_disagreement"]["median_um"], 1)
                    if v["two_sided_disagreement"] else None),
                "if_only_objects": v["n_if_only"]}
            for k, v in match.items() if k.startswith("IF")
        },
        "stage_C_identity": {
            "n_gc_on_stack": int(len(presence)),
            "n_on_all_9_planes": int((presence.n_planes == 9).sum()),
            "n_on_ge_5_planes": int((presence.n_planes >= 5).sum()),
            "n_st_only": int((presence.n_if_planes == 0).sum()),
            "n_if_only": int((presence.n_st_planes == 0).sum()),
        },
        "stage_D_columns": {
            "n_columns": int(len(metrics)),
            "watertight": int((metrics.mesh_open_edges == 0).sum()),
            "total_faces": int(metrics.n_faces.sum()),
            "mesh_over_voxel_volume": {
                "median": round(float((metrics.mesh_volume_um3 / metrics.volume_um3).median()), 4),
                "min": round(float((metrics.mesh_volume_um3 / metrics.volume_um3).min()), 4),
                "max": round(float((metrics.mesh_volume_um3 / metrics.volume_um3).max()), 4),
            },
            "volume_nL": {
                "median": round(float(metrics.volume_um3.median() / 1e6), 3),
                "max": round(float(metrics.volume_um3.max() / 1e6), 3),
            },
            "radius_cv": {"median": round(float(metrics.radius_cv.median()), 3),
                          "p90": round(float(metrics.radius_cv.quantile(0.9)), 3)},
            "centroid_drift_um": {"median": round(float(metrics.centroid_drift_um.median()), 1),
                                  "p90": round(float(metrics.centroid_drift_um.quantile(0.9)), 1)},
            "smoothing": {k: build[k] for k in
                          ["sigma_xy_px", "sigma_z_sub", "z_oversample", "pad_z", "pad_xy",
                           "iso_level", "mesh_smoothing", "target_faces"]},
        },
        "if_segmentation": [
            {"section": m["section"], "tissue_mm2": round(m["tissue_mm2"], 2),
             "n_gc": m["n_gc"], "gc_area_mm2": round(m["gc_area_mm2"], 3),
             "median_equiv_radius_um": round(m["median_equiv_radius_um"], 1)}
            for m in if_manifest
        ],
        "if_coverage_of_st_tissue": if_coverage(cfg),
        "frozen_inputs": {
            "st_scaffold": str(cfg.st_scaffold.h5ad),
            "coord_key": cfg.st_scaffold.coord_key,
            "note": "Stereo-seq coordinates and merged-GC ids are copied, never recomputed",
        },
        "caveats": CAVEATS,
    }

    (ROOT / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    lines = []
    for rel in DELIVERABLES:
        path = ROOT / rel
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {rel}")
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

    cov = report["if_coverage_of_st_tissue"]
    print(f"IF coverage of the Stereo-seq tissue ({cov['st_tissue_mm2']} mm2):")
    for key, value in cov["per_section"].items():
        flag = ("  <- includes %.2f mm2 outside the imaged area"
                % value["missed_outside_imaged_area_mm2"]) if value["missed_outside_imaged_area_mm2"] > 0.05 else ""
        print(f"   {key}: {value['covers_st_tissue_pct']:>5.1f}%  "
              f"(misses {value['missed_mm2']:.2f} mm2){flag}")
    print(f"   covered by every IF section: {cov['st_tissue_covered_by_all_if_pct']}%")
    print(json.dumps({k: report[k] for k in
                      ["stage_C_identity", "stage_D_columns"]}, indent=2, ensure_ascii=False))
    print(f"\nhashed {len(lines)} deliverables")

