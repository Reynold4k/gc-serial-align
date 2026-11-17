"""Stage 7 - collect the QC numbers and hash the deliverables."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import __version__
from .config import Config

def _number(value, digits: int):
    """A rounded float, or None where there is nothing to summarise (an empty table)."""
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def limitations(cfg: Config, metrics: pd.DataFrame) -> list[str]:
    """The limits that apply to this run, stated with this run's own numbers."""
    depth = cfg.z_span_um + cfg.section_thickness_um
    out = []
    if len(metrics):
        diameter = 2 * float(metrics.median_equiv_radius_um.median())
        if depth < diameter:
            out.append(
                f"Depth. The stack is {depth:.0f} um of tissue ({cfg.n_planes} sections, "
                f"{cfg.section_thickness_um:g} um each, first to last). The median diameter of "
                f"a germinal centre (GC) cross-section in this run is {diameter:.0f} um. Each 3D "
                "column therefore covers only part of the depth of a GC. The columns show how a "
                "GC continues from section to section; they do not show the whole shape of a GC "
                "and cannot be used to judge whether GCs are spherical.")
        else:
            out.append(
                f"Depth. The stack is {depth:.0f} um of tissue ({cfg.n_planes} sections, "
                f"{cfg.section_thickness_um:g} um each, first to last); the median diameter of "
                f"a germinal centre (GC) cross-section in this run is {diameter:.0f} um, so the "
                "stack is deep enough to contain whole GCs.")
    out.append(
        "Registration by GCs. The registration uses the GCs as its only landmarks; the tissue "
        "outline is used only to find the orientation. The registration therefore maximises "
        "the overlap of each GC between sections. The 3D columns are suitable for showing GC "
        "shape. They are not suitable for statistics on GC volume, because the overlap "
        "between sections is partly produced by the registration.")
    out.append(
        "Closed surfaces. Each 3D surface ends in a rounded cap that extends slightly beyond "
        "the first and the last section on which the GC appears. The caps close the surface; "
        "they are not measured.")
    one_sided = [s.name for s in cfg.if_sections if cfg.one_sided(s.index)]
    if one_sided:
        out.append(
            f"One-sided IF sections. {', '.join(one_sided)} "
            f"{'has' if len(one_sided) == 1 else 'have'} a spatial-transcriptomics section on one "
            "side only. For these sections no two-sided disagreement can be computed, and their "
            "registration is less certain than that of IF sections inside the stack.")
    return out


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
        key = entry.name
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
        "note": ("A GC can be matched on an IF section only where that section has tissue. "
                 "Lowering if_images.tissue_threshold_scale recovers dimly stained tissue. It "
                 "cannot recover tissue that is torn off or lies outside the imaged area."),
    }


def deliverables(cfg: Config) -> list[str]:
    return [
        f"05_viewer/{cfg.viewer.filename}",
        "04_columns/gc_columns.npz",
        "04_columns/gc_plane_patches.npz",
        "04_columns/gc_column_metrics.csv",
        "03_gc_match/plane_gc_labels.npz",
        "03_gc_match/if_channels_warped.npz",
        "03_gc_match/if_tissue_warped.npz",
        "03_gc_match/gc_presence.csv",
        "03_gc_match/match_report.json",
        "03_gc_match/section_transforms.json",
        "02_global_align/global_transforms.json",
        "02_global_align/icp_trajectory.json",
        "02_global_align/orientation_search.json",
        "01_if_gc/if_gc_manifest.json",
    ]


def angular_spread_deg(angles: list[float]) -> float:
    """The smallest arc that holds every angle, so 355 and 2 are 7 apart, not 353."""
    if len(angles) < 2:
        return 0.0
    ordered = sorted(a % 360.0 for a in angles)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])] + [ordered[0] + 360.0 - ordered[-1]]
    return float(360.0 - max(gaps))


def orientation_summary(orient: dict) -> dict:
    """What the independent per-section orientation decisions add up to.

    Each IF section is oriented on its own. Sections cut from one block and
    mounted the same way round should come out with the same mirror state, so a
    section that disagrees with the rest is worth a look - it is either mounted
    flipped or placed wrongly. Their rotations only agree if they were also laid
    down at the same angle, so the spread is reported, not judged.
    """
    mirrors = {k: bool(v["mirror"]) for k, v in orient.items()}
    states = sorted(set(mirrors.values()))
    odd_ones = []
    if len(states) > 1:
        minority = min(states, key=lambda state: sum(1 for m in mirrors.values() if m == state))
        odd_ones = [k for k, m in mirrors.items() if m == minority]
    angles = [v["angle_deg"] for v in orient.values()]
    spread = angular_spread_deg(angles)
    if len(states) == 1:
        handed = ("all IF sections were placed "
                  + ("mirrored" if states[0] else "without mirroring"))
    else:
        handed = (f"the mirror state of {', '.join(odd_ones)} differs from that of the other IF "
                  "sections; check how these sections were mounted")
    return {
        "all_sections_same_handedness": len(states) == 1,
        "sections_with_minority_handedness": odd_ones,
        "angle_range_deg": [min(angles), max(angles)],
        "angle_spread_deg": round(spread, 1),
        "note": (f"{handed[0].upper()}{handed[1:]}. The rotation angles of the IF sections span "
                 f"{spread:.0f} degrees. The orientation of each IF section was determined "
                 "separately. It was chosen by the number of matched germinal centres, not by "
                 "tissue overlap, because a nearly convex tissue outline overlaps almost as "
                 "well when it is mirrored."),
    }


def run(cfg: Config) -> None:
    ROOT = cfg.output_dir
    DELIVERABLES = deliverables(cfg)
    glob = json.loads((cfg.output_dir / "02_global_align/global_transforms.json").read_text())
    match = json.loads((cfg.output_dir / "03_gc_match/match_report.json").read_text())
    build = json.loads((cfg.output_dir / "04_columns/column_build_report.json").read_text())
    if_manifest = json.loads((cfg.output_dir / "01_if_gc/if_gc_manifest.json").read_text())
    metrics = pd.read_csv(cfg.output_dir / "04_columns/gc_column_metrics.csv")
    presence_file = ROOT / "03_gc_match/gc_presence.csv"
    if not presence_file.exists():                      # a run made before 0.2
        presence_file = ROOT / "03_gc_match/gc_presence_9plane.csv"
    presence = pd.read_csv(presence_file)
    n_planes = cfg.n_planes
    majority = n_planes // 2 + 1

    orient = {k: {"mirror": v["flip"], "angle_deg": v["angle_deg"],
                  "scale": round(v["total_scale"], 4)}
              for k, v in glob["sections"].items()}

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "gcalign_version": __version__,
        "experiment": cfg.name,
        "geometry": {
            "section_thickness_um": cfg.section_thickness_um,
            "n_sections": len(cfg.sections),
            "z_span_um": cfg.z_um(max(s.index for s in cfg.sections)),
            "order": [p[0] for p in cfg.planes],
        },
        "channels": cfg.if_images.channels,
        "gc_channel": cfg.if_images.gc_channel,
        "stage_A_orientation": {"per_section": orient, **orientation_summary(orient)},
        "close_match_um": cfg.align.close_match_um,
        "stage_A_gc_match": {
            k: {"n_if_gc": v["n_if_gc"], "matched": v["n_matched"],
                "within_150um": v["n_matched_under_150um"],
                "median_um": (round(v["match_median_um"], 1)
                              if v["match_median_um"] is not None else None),
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
                "if_only_objects": v["n_if_only"],
                "objects_split_on_st_evidence": v.get("objects_split_on_st_evidence", [])}
            for k, v in match.items() if k.startswith("IF")
        },
        "stage_C_identity": {
            "n_planes": n_planes,
            "n_gc_on_stack": int(len(presence)),
            "n_on_all_planes": int((presence.n_planes == n_planes).sum()),
            "majority_of_planes": majority,
            "n_on_majority_of_planes": int((presence.n_planes >= majority).sum()),
            "n_st_only": int((presence.n_if_planes == 0).sum()),
            "n_if_only": int((presence.n_st_planes == 0).sum()),
        },
        "stage_D_columns": {
            "n_columns": int(len(metrics)),
            "watertight": int((metrics.mesh_open_edges == 0).sum()),
            "total_faces": int(metrics.n_faces.sum()),
            "mesh_over_voxel_volume": {
                "median": _number((metrics.mesh_volume_um3 / metrics.volume_um3).median(), 4),
                "min": _number((metrics.mesh_volume_um3 / metrics.volume_um3).min(), 4),
                "max": _number((metrics.mesh_volume_um3 / metrics.volume_um3).max(), 4),
            },
            "volume_nL": {
                "median": _number(metrics.volume_um3.median() / 1e6, 3),
                "max": _number(metrics.volume_um3.max() / 1e6, 3),
            },
            "radius_cv": {"median": _number(metrics.radius_cv.median(), 3),
                          "p90": _number(metrics.radius_cv.quantile(0.9), 3)},
            "centroid_drift_um": {"median": _number(metrics.centroid_drift_um.median(), 1),
                                  "p90": _number(metrics.centroid_drift_um.quantile(0.9), 1)},
            "smoothing": {k: build[k] for k in
                          ["sigma_xy_px", "sigma_z_sub", "z_oversample", "pad_z", "pad_xy",
                           "iso_level", "mesh_smoothing", "target_faces"]},
        },
        "if_segmentation": [
            {"section": m["section"], "tissue_mm2": round(m["tissue_mm2"], 2),
             "n_gc": m["n_gc"], "gc_area_mm2": round(m["gc_area_mm2"], 3),
             "median_equiv_radius_um": (round(m["median_equiv_radius_um"], 1)
                                        if m["median_equiv_radius_um"] is not None else None)}
            for m in if_manifest
        ],
        "if_coverage_of_st_tissue": if_coverage(cfg),
        "frozen_inputs": {
            "st_scaffold": str(cfg.st_scaffold.h5ad),
            "coord_key": cfg.st_scaffold.coord_key,
            "note": "Stereo-seq coordinates and merged-GC ids are copied, never recomputed",
        },
        "caveats": limitations(cfg, metrics),
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
    splits = [(k, item) for k, v in report["stage_B_warp"].items()
              for item in v["objects_split_on_st_evidence"]]
    if splits:
        print(f"IF objects split because two Stereo-seq GCs claimed them ({len(splits)}):")
        for section, item in splits:
            print(f"   {section}: {item['object_px']} px -> "
                  + ", ".join(f"{g} {n} px" for g, n in item["split_into"].items()))
    print(json.dumps({k: report[k] for k in
                      ["stage_C_identity", "stage_D_columns"]}, indent=2, ensure_ascii=False))
    print(f"\nhashed {len(lines)} deliverables")

