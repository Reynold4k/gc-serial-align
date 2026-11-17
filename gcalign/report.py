"""Write one self-contained HTML quality report for a finished gcalign run.

    from gcalign import config, report
    report.run(config.load("my_experiment.yaml"))     # -> <output_dir>/06_figures/report.html

The report opens without a network connection: every figure is embedded in the
HTML file (as PNG, or as JPEG when JPEG is less than half the size), and the
Inter typeface is embedded as a subset. The report reads runs made with gcalign
0.1 and 0.2. Counts whose names changed between the two versions are
recomputed from the run's own presence table.
"""
from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__, theme
from . import figures as figures_module

TEMPLATE = Path(__file__).parent / "templates" / "report.html"
MAX_IMAGE_PX = 2000            # embedded width: sharp on a 2x screen at the page's 1000 px
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
DASH = '<span class="dim">—</span>'


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _num(value, digits: int = 1, suffix: str = "") -> str:
    """A number for a table cell; a dim dash when there is no value."""
    if value is None or isinstance(value, bool):
        return DASH
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}{suffix}"
    value = float(value)
    if not np.isfinite(value):
        return DASH
    return f"{value:,.{digits}f}{suffix}"


def _power(value) -> str:
    """3e+05 -> 3 x 10^5 with a real superscript."""
    if not value:
        return DASH
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    head = "" if abs(mantissa - 1) < 1e-9 else f"{mantissa:g} × "
    return f"{head}10{str(exponent).translate(_SUPERSCRIPT)}"


def _plane(run, name: str) -> str:
    kind = run.plane(name).kind if name in run.names else ""
    return f'<span class="kind-{_e(kind)}">{_e(name)}</span>'


def _table(headers, rows, text_columns=(0,), compact: bool = False) -> str:
    head = "".join(f'<th class="text">{h}</th>' if i in text_columns else f"<th>{h}</th>"
                   for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="text">{c}</td>' if i in text_columns else f"<td>{c}</td>"
                                    for i, c in enumerate(row)) + "</tr>" for row in rows)
    cls = ' class="compact"' if compact else ""
    return f'<div class="scroll"><table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<p class="sub">{sub}</p>' if sub else ""
    return f'<div class="tile"><p class="label">{label}</p><p class="value">{value}</p>{sub_html}</div>'


def _spread(values, unit: str = " µm") -> str:
    values = [v for v in values if v is not None]
    if not values:
        return ""
    return f"{min(values):.0f}–{max(values):.0f}{unit} across IF sections; lower is better"


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------
def embed_image(path: Path, max_width: int = MAX_IMAGE_PX) -> str:
    """Return a figure as a data URI: PNG, or JPEG when JPEG is under half the PNG size."""
    from PIL import Image
    image = Image.open(path).convert("RGB")
    if image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)), Image.LANCZOS)
    png, jpeg = io.BytesIO(), io.BytesIO()
    image.save(png, format="PNG", optimize=True)
    image.save(jpeg, format="JPEG", quality=90, subsampling=0, optimize=True)
    if jpeg.tell() < 0.5 * png.tell():
        return "data:image/jpeg;base64," + base64.b64encode(jpeg.getvalue()).decode("ascii")
    return "data:image/png;base64," + base64.b64encode(png.getvalue()).decode("ascii")


def font_faces() -> str:
    """Return the @font-face rules that embed the theme font (`theme.font_face_css`)."""
    return theme.font_face_css()


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------
def identity_counts(run) -> dict:
    """Count GCs by the sections they appear on, from the run's presence table.

    The same columns exist in runs made with gcalign 0.1 and 0.2, so the counts
    are comparable across versions.
    """
    table = run.presence
    present = lambda names: (table[[f"{n}_px" for n in names]] > 0).sum(axis=1)  # noqa: E731
    n_sections = len(run.names)
    on = table["n_planes"] if "n_planes" in table else present(run.names)
    on_st = table["n_st_planes"] if "n_st_planes" in table else present(run.st_names)
    on_if = table["n_if_planes"] if "n_if_planes" in table else present(run.if_names)
    majority = n_sections // 2 + 1
    return {"n_sections": n_sections, "on_stack": int(len(table)),
            "on_all": int((on == n_sections).sum()),
            "majority": majority, "on_majority": int((on >= majority).sum()),
            "st_only": int((on_if == 0).sum()), "if_only": int((on_st == 0).sum())}


def warp_displacement(run) -> dict:
    """How far the elastic warp moves each IF section's tissue, in um: (median, 95th percentile, max).

    Only runs that store their transforms (gcalign 0.2 and later) have them;
    other runs get an empty result.
    """
    if not run.has("03_gc_match/section_transforms.json"):
        return {}
    try:
        from .transforms import Transforms
        transforms = Transforms.load(run.cfg)
    except Exception:
        return {}
    out = {}
    for name in run.if_names:
        try:
            rows, cols = np.nonzero(run.tissue(name)[::3, ::3])
            shift = transforms[name].displacement_um(run.grid.to_xy(rows * 3, cols * 3))
            out[name] = (float(np.median(shift)), float(np.percentile(shift, 95)), float(shift.max()))
        except Exception:
            continue
    return out


def mesh_summary(run) -> dict | None:
    """Summary numbers of the 3D columns, from 04_columns/gc_column_metrics.csv."""
    metrics = run.metrics
    if metrics is None or not len(metrics):
        return None
    ratio = metrics["mesh_volume_um3"] / metrics["volume_um3"]
    return {"n": int(len(metrics)), "watertight": int((metrics["mesh_open_edges"] == 0).sum()),
            "ratio_median": float(ratio.median()), "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "volume_median_nl": float(metrics["volume_um3"].median() / 1e6),
            "volume_max_nl": float(metrics["volume_um3"].max() / 1e6),
            "cv_median": float(metrics["radius_cv"].median()),
            "cv_p90": float(metrics["radius_cv"].quantile(0.9)),
            "drift_median": float(metrics["centroid_drift_um"].median()),
            "drift_p90": float(metrics["centroid_drift_um"].quantile(0.9))}


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
def _figure(paths: dict, name: str, alt: str, narrow: bool = False) -> str:
    path = paths.get(name)
    if path is None or not Path(path).exists():
        return ""
    cls = ' class="narrow"' if narrow else ""
    return f'<figure{cls}><img alt="{_e(alt)}" src="{embed_image(Path(path))}"></figure>'


def _section(heading: str, *blocks: str) -> str:
    return f"<section><h2>{heading}</h2>{''.join(b for b in blocks if b)}</section>"


def _p(*sentences: str) -> str:
    return "<p>" + " ".join(s for s in sentences if s) + "</p>"


def build(run, figure_paths: dict) -> str:
    """The report body as HTML: what the run is, then one section per check."""
    cfg = run.cfg
    validation = run.validation or {}
    run_version = validation.get("gcalign_version") or ("0.1" if validation else "unknown")
    counts = identity_counts(run)
    align_rows = figures_module.alignment_table(run)
    meshes = mesh_summary(run)
    coverage = validation.get("if_coverage_of_st_tissue") or {}
    search = run.orientation_search
    close_um = float(validation.get("close_match_um") or (search or {}).get("close_match_um")
                     or cfg.align.close_match_um)
    n_if, n_st, n_all = len(run.if_names), len(run.st_names), len(run.names)
    depth = max(p.z_um for p in run.planes) - min(p.z_um for p in run.planes)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = []

    # what the run is ------------------------------------------------------
    parts.append(
        '<header><p class="eyebrow">gcalign quality report</p>'
        f"<h1>{_e(cfg.name)}</h1>"
        '<p class="lead">This report checks one finished gcalign run. '
        f"The run aligned {n_if} immunofluorescence (IF) sections into the coordinate frame of "
        f"{n_st} spatial-transcriptomics (ST) sections cut from the same tissue block, using "
        "germinal centres (GCs) as landmarks. "
        f"Each section is {cfg.section_thickness_um:g} µm thick. From the first to the last "
        f"section the stack is {depth + cfg.section_thickness_um:g} µm of tissue. "
        f"{counts['on_stack']} GCs are followed through the stack. "
        f"The run was made with gcalign {_e(run_version)}; this report was written by gcalign "
        f"{_e(__version__)} on {generated}.</p></header>")

    # key numbers ----------------------------------------------------------
    tiles = [_tile("GCs followed through the stack", _num(counts["on_stack"]),
                   f"{counts['st_only']} found on ST sections only; {counts['if_only']} on IF sections only"),
             _tile(f"GCs present on all {counts['n_sections']} sections",
                   f"{counts['on_all']}<small>of {counts['on_stack']}</small>",
                   f"{counts['on_majority']} present on {counts['majority']} or more sections")]
    if align_rows:
        after = [r["after_um"] for r in align_rows]
        held = [r["held_out_um"] for r in align_rows]
        tiles.append(_tile("GC centre error after the elastic warp",
                           f"{np.median(after):.0f}<small>µm</small>", _spread(after)))
        tiles.append(_tile("Leave-one-out (LOO) GC centre error",
                           f"{np.median(held):.0f}<small>µm</small>", _spread(held)))
    if coverage.get("st_tissue_covered_by_all_if_pct") is not None:
        tiles.append(_tile("ST tissue covered by every IF section",
                           f"{coverage['st_tissue_covered_by_all_if_pct']:.0f}<small>%</small>",
                           f"{coverage.get('st_tissue_covered_by_no_if_pct', 0):.1f}% covered by no IF section"))
    if meshes:
        tiles.append(_tile("3D columns with a closed surface",
                           f"{meshes['watertight']}<small>of {meshes['n']}</small>",
                           f"median of surface volume ÷ measured volume: {meshes['ratio_median']:.3f}"))
    parts.append(f'<div class="tiles">{"".join(tiles)}</div>'
                 '<p class="note">The two error tiles show the middle value over the IF sections of '
                 "each section's median error.</p>")

    # 1. the stack ----------------------------------------------------------
    rgb = cfg.rgb_channels
    shown = ", ".join(f"{c} as {band}" for c, band in zip(rgb, ("red", "green", "blue")) if c)
    parts.append(_section(
        "Every section of the stack, after alignment",
        _p(f"The figure shows all {n_all} sections in cutting order, left to right and top to "
           "bottom, on the common grid after alignment.",
           f"Each IF section shows its own stain: {_e(shown)}. All IF sections share one set of "
           "display limits, so brightness is comparable between them.",
           "Each ST section shows the section outline in dark grey and every GC region in the "
           "identity colour of that GC. A GC keeps its identity colour in every figure gcalign "
           "draws, including the 3D columns and the interactive viewer.",
           "The scale bar is at the bottom right."),
        _figure(figure_paths, "sections", "all sections in cutting order")))

    # 2. agreement ----------------------------------------------------------
    parts.append(_section(
        "Agreement of the GC masks between IF and ST",
        _p("The left panel overlays the GC masks of all sections on the common grid.",
           "A pixel is pink when only ST sections call it GC, blue when only IF sections call it "
           "GC, and white when every section of both kinds calls it GC.",
           "Pink and blue get brighter with the fraction of sections of that kind that call the "
           "pixel GC.",
           "The right panel enlarges the square marked by the corner brackets in the left panel: "
           "the square that holds the most GC area summed over all sections."),
        _figure(figure_paths, "overlay", "GC masks of IF and ST sections overlaid")))

    # 3. orientation --------------------------------------------------------
    rows = []
    for name in run.if_names:
        t = run.transforms.get(name, {})
        neighbours = cfg.neighbours.get(run.plane(name).index, [])
        flank = " + ".join(_plane(run, n) for n in neighbours)
        if len(neighbours) < 2:
            flank += ' <span class="flag">(one side only)</span>'
        rejected = DASH
        if search and name in search.get("sections", {}):
            entry = search["sections"][name]
            winner = int(entry.get("winner", 0))
            others = [c for i, c in enumerate(entry["candidates"]) if i != winner]
            if others:
                best = max(others, key=lambda c: c["n_close"])
                rejected = f"{best['n_close']} ({'mirrored' if best.get('flip') else 'not mirrored'})"
        rows.append([_plane(run, name), flank, "yes" if t.get("flip") else "no",
                     _num(t.get("angle_deg"), 0), _num(t.get("total_scale"), 3),
                     _num(t.get("n_if_gc")), _num(t.get("n_matched")), _num(t.get("n_matched_under_150um")),
                     rejected, _num(t.get("match_median_um"), 0), _num(t.get("tissue_dice_final"), 3),
                     _num(t.get("gc_dice_final"), 3)])
    if search is not None:
        explain = _p(
            "Stage A of the alignment places each IF section with a similarity transform: a "
            "mirror state, a rotation, a scale and a shift.",
            "Stage A tries both mirror states and every rotation, refines the best candidates, "
            f"and keeps the candidate that brings the most IF GC centres within {close_um:g} µm "
            "of an ST GC centre.",
            "In the figure, each panel is one IF section and each mark is one candidate orientation.",
            "The x axis is the Dice overlap of the IF and ST tissue outlines (0 to 1, higher means "
            f"more overlap). The y axis is the number of GC centres within {close_um:g} µm.",
            "The large blue dot is the orientation kept, labelled with its rotation in degrees. Grey "
            "dots are rejected candidates without mirroring; grey rings are rejected mirrored candidates.",
            "All candidates overlap the tissue about equally well; the GC count separates the kept "
            "orientation from the rest.")
    else:
        explain = _p(
            "Stage A of the alignment places each IF section with a similarity transform: a "
            "mirror state, a rotation, a scale and a shift.",
            "This run was made before gcalign 0.2 and did not store the candidate orientations that "
            "stage A compared, so the table lists only the orientation kept and there is no figure.")
    parts.append(_section(
        "Orientation of each IF section (stage A)", explain,
        _figure(figure_paths, "orientation", "orientation candidates of each IF section"),
        _table(["IF section", "ST sections either side", "mirrored", "rotation (°)", "scale (×)",
                "IF GCs (count)", "matched to an ST GC (count)",
                f"matched within {close_um:g} µm (count)",
                f"best rejected candidate: GCs within {close_um:g} µm",
                "median distance between matched GC centres (µm)", "tissue overlap, Dice (0–1)",
                "GC Dice (0–1)"], rows, text_columns=(0, 1, 2)),
        '<p class="note">Every number in this table describes stage A, before the elastic warp. '
        "Higher is better for the counts within the radius and for both Dice overlaps; lower is "
        "better for the median distance. An IF section at an end of the stack has only one ST "
        "section next to it.</p>"))

    # 4. warp -----------------------------------------------------------------
    shift = warp_displacement(run)
    rows = []
    for r in align_rows:
        match = run.match.get(r["plane"], {})
        two = match.get("two_sided_disagreement") or {}
        splits = match.get("objects_split_on_st_evidence") or []
        moved = shift.get(r["plane"])
        none = DASH if len(r["neighbours"]) >= 2 else '<span class="dim">one ST neighbour</span>'
        rows.append([_plane(run, r["plane"]), _num(r["anchors"]), _num(r["before_um"], 1),
                     _num(r["after_um"], 1), _num(r["held_out_um"], 1), _power(match.get("tps_smoothing")),
                     _num(two.get("median_um"), 1) if two else none,
                     _num(two.get("p90_um"), 1) if two else DASH,
                     _num(match.get("n_if_only")), _num(len(splits)),
                     _num(moved[0], 0) if moved else DASH, _num(moved[2], 0) if moved else DASH])
    parts.append(_section(
        "Error of the elastic warp (stage B)",
        _p("Stage B corrects the remaining stretch of each IF section with a thin-plate spline "
           "(TPS): an elastic warp anchored on the centres of matched GC pairs.",
           "The stiffness of the TPS is chosen by leave-one-out cross-validation (LOO): each anchor "
           "is left out in turn, the warp is fitted without it, and the distance between the "
           "predicted and the true position of the left-out GC is measured.",
           "An IF section that lies between two ST sections is warped to each of them separately. "
           "The two-sided disagreement is the distance between where those two warps put the same "
           "points; it is an error bar for the section that costs no extra experiment."),
        _p("In the figure, each row is one IF section in cutting order.",
           "Panel 1 (centroid error): distance between the centres of matched IF and ST GCs, in "
           "micrometres, before the warp (ring) and after the warp (arrow head).",
           "Panel 2 (leave-one-out error): the LOO error, in micrometres.",
           "Panel 3 (two-sided disagreement): in micrometres; a section at an end of the stack has "
           "only one ST neighbour and therefore no value.",
           "Panel 4 (matched GCs): IF GCs used as TPS anchors (blue) out of all GC objects found on "
           "the IF section (grey).",
           "Lower is better in panels 1 to 3."),
        _figure(figure_paths, "alignment", "warp error of each IF section"),
        _table(["IF section", "TPS anchors (count)", "GC centre error before warp (µm)",
                "GC centre error after warp (µm)", "leave-one-out (LOO) error (µm)", "TPS stiffness (chosen by LOO)",
                "two-sided disagreement, median (µm)", "two-sided disagreement, 90th percentile (µm)",
                "IF GCs with no ST partner (count)", "IF objects split between two ST GCs (count)",
                "distance the warp moves the tissue, median (µm)",
                "distance the warp moves the tissue, maximum (µm)"], rows),
        '<p class="note">Errors are medians over the matched GC pairs of each section; lower is '
        "better. TPS stiffness is the smoothing parameter of the spline, without a unit: a larger "
        "value gives a stiffer, smoother warp. The distance the warp moves the tissue is listed "
        "only for runs that store their transforms (gcalign 0.2 and later).</p>"))

    # 5. presence -----------------------------------------------------------
    rows = [["GCs followed through the stack", _num(counts["on_stack"])],
            [f"GCs present on all {counts['n_sections']} sections", _num(counts["on_all"])],
            [f"GCs present on {counts['majority']} or more sections", _num(counts["on_majority"])],
            ["GCs present on ST sections only", _num(counts["st_only"])],
            ["GCs present on IF sections only", _num(counts["if_only"])]]
    parts.append(_section(
        "Which germinal centre is on which section",
        _p("In the figure, each row is one GC, labelled with its id; the rows are ordered by the "
           "GC's cross-section area summed over all sections, largest first.",
           "Each column is one section in cutting order: blue columns are IF sections and pink "
           "columns are ST sections.",
           "The brightness of a cell is the cross-section area of that GC on that section; the "
           "scale under the matrix is in thousands of square micrometres. A black cell means the "
           "GC is absent from that section."),
        _figure(figure_paths, "presence", "GC presence on every section", narrow=True),
        _table(["", "count"], rows, compact=True)))

    # 6. columns ------------------------------------------------------------
    if meshes:
        rows = [["3D columns built (count)", _num(meshes["n"])],
                ["columns with a closed surface, no open edges (count)", _num(meshes["watertight"])],
                ["surface volume ÷ measured volume, median (minimum–maximum); 1 is exact",
                 f"{meshes['ratio_median']:.3f} ({meshes['ratio_min']:.3f}–{meshes['ratio_max']:.3f})"],
                ["GC volume within the stack, median / maximum (nL)",
                 f"{meshes['volume_median_nl']:.2f} / {meshes['volume_max_nl']:.2f}"],
                ["variation of the GC radius along the stack (coefficient of variation), median / 90th percentile",
                 f"{meshes['cv_median']:.3f} / {meshes['cv_p90']:.3f}"],
                ["largest distance of a GC's centre from its mean position, median / 90th percentile (µm)",
                 f"{meshes['drift_median']:.0f} / {meshes['drift_p90']:.0f}"]]
        parts.append(_section(
            "3D columns of the largest germinal centres",
            _p("A column is the closed 3D surface that the columns stage of the pipeline draws "
               "through one GC's cross-sections on all sections.",
               f"The figure shows the {min(8, meshes['n'])} largest columns by measured volume, all at "
               f"the same scale, with depth exaggerated {cfg.viewer.default_z_exaggeration:g} times. Each column has "
               "its GC's identity colour and is labelled with the GC id.",
               "The surface is smoothed in the image volume, and its level is chosen so that the "
               "enclosed volume equals the volume measured on the sections."),
            _figure(figure_paths, "columns", "largest GC columns in 3D"),
            _table(["", "value"], rows, compact=True)))

    # 7. coverage -----------------------------------------------------------
    blocks = [_p("A GC can be matched only where an IF section covers the ST tissue.",
                 "A torn or folded IF section, or one imaged over a smaller field than the ST chip, "
                 "leaves the GCs in the uncovered tissue without an IF partner.")]
    cov_rows = [[_plane(run, name), _num(v.get("covers_st_tissue_pct"), 1), _num(v.get("missed_mm2"), 2),
                 _num(v.get("missed_outside_imaged_area_mm2"), 2)]
                for name, v in (coverage.get("per_section") or {}).items()]
    if cov_rows:
        blocks.append("<h3>ST tissue covered by each IF section</h3>"
                      + _table(["IF section", "ST tissue covered (%)", "ST tissue not covered (mm²)",
                                "part of the uncovered tissue outside the imaged field (mm²)"], cov_rows)
                      + f'<p class="note">ST tissue area: {coverage.get("st_tissue_mm2", 0):.2f} mm². '
                        f"Covered by every IF section: {coverage.get('st_tissue_covered_by_all_if_pct', 0):.1f}%. "
                        f"Covered by no IF section: {coverage.get('st_tissue_covered_by_no_if_pct', 0):.1f}%. "
                        "Higher coverage is better.</p>")
    transfer = None
    if run.has("06_transfer/st_bins_if.json"):
        try:
            transfer = json.loads((run.root / "06_transfer/st_bins_if.json").read_text(encoding="utf-8"))
        except Exception:
            transfer = None
    if transfer and transfer.get("sections"):
        rows = [[_plane(run, name), " + ".join(_plane(run, n) for n in v.get("if_sections", [])),
                 _num(v.get("bins")), _num(v.get("pct_under_if_tissue"), 1)]
                for name, v in transfer["sections"].items()]
        blocks.append("<h3>ST bins that receive IF signal</h3>"
                      + _table(["ST section", "IF sections either side", "ST bins (count)",
                                "bins under IF tissue (%)"], rows, text_columns=(0, 1))
                      + '<p class="note">gcalign transfers the IF signal of the IF sections either side '
                        "onto each ST bin that lies under IF tissue (06_transfer/st_bins_if.npz).</p>")
    seg_rows = []
    if run.has("01_if_gc/if_gc_manifest.json"):
        for name in run.if_names:
            item = run.manifest.get(name, {})
            seg_rows.append([_plane(run, name), _num(item.get("tissue_mm2"), 2), _num(item.get("n_gc")),
                             _num(item.get("gc_area_mm2"), 2), _num(item.get("median_equiv_radius_um"), 0)])
    if seg_rows:
        blocks.append("<h3>GC segmentation on each IF section</h3>"
                      + _table(["IF section", "tissue area (mm²)", "GC objects (count)",
                                "GC area (mm²)", "median GC radius (µm)"], seg_rows)
                      + '<p class="note">The GC radius is the radius of a circle with the same area '
                        "as the GC object.</p>")
    parts.append(_section("IF coverage of the ST tissue", *blocks))

    # 8. caveats --------------------------------------------------------------
    caveats = validation.get("caveats") or []
    if caveats:
        items = "".join(f"<li>{_e(c)}</li>" for c in caveats)
        parts.append(_section("Limitations recorded by the run",
                              _p("The validation stage of the run writes these limitations into "
                                 "validation_report.json. They apply to every result above."),
                              f'<ul class="caveats">{items}</ul>'))
    else:
        parts.append(_section("Limitations recorded by the run",
                              _p("This run has no validation_report.json, so its recorded limitations "
                                 "and IF coverage numbers are missing. Run "
                                 "<code>gcalign run --only validate</code> to write the file.")))

    parts.append(f"<footer>Run folder: <code>{_e(run.root)}</code><br>Report written by gcalign "
                 f"{_e(__version__)} on {generated}.</footer>")
    return "\n".join(parts)


def run(cfg, out_dir=None, *, render: bool = True, dpi: float = 220) -> Path:
    """Write ``report.html`` for a finished run and return its path.

    `out_dir` defaults to ``cfg.output_dir / "06_figures"``. The whole-run
    figures are drawn into the same folder first; with `render=False` the PNGs
    already in the folder are reused and only missing ones are drawn.
    """
    data = figures_module.as_run(cfg)
    out = Path(out_dir) if out_dir is not None else Path(data.cfg.output_dir) / "06_figures"
    out.mkdir(parents=True, exist_ok=True)
    paths: dict = {}
    if not render:
        paths = {n: out / f"{n}.png" for n in figures_module.WHOLE_RUN if (out / f"{n}.png").exists()}
    missing = [n for n in figures_module.WHOLE_RUN if n not in paths]
    if missing:
        paths.update(figures_module.run(data, out_dir=out, which=missing, dpi=dpi))
    content = build(data, paths)
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in (("{{TITLE}}", _e(f"{data.cfg.name} · gcalign quality report")),
                       ("{{GENERATOR}}", _e(f"gcalign {__version__}")),
                       ("{{FONTS}}", font_faces()), ("{{CSS_VARS}}", theme.css_vars()),
                       ("{{CONTENT}}", content)):
        page = page.replace(key, value)
    target = out / "report.html"
    target.write_text(page, encoding="utf-8")
    print(f"[gcalign.report] wrote the quality report: {target} ({target.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    return target
