"""Look at the inputs before spending a run on them.

`gcalign check --config my_experiment.yaml` opens every file the config points
at and verifies the things a run would otherwise discover one stage at a time:
that the images exist and have the channels the config names, that the pixel
size written in the config agrees with the image's own metadata, that the
AnnData has the coordinate key and columns asked for, that every listed section
is in it, and whether the depth of the stack can support a claim about whole
structures.

Nothing is written. The exit status is 0 when a run can start.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .if_sections import reader_for


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def ok(self, what: str, detail: str = "") -> None:
        self.rows.append(("ok", what, detail))

    def warn(self, what: str, detail: str = "") -> None:
        self.rows.append(("warn", what, detail))

    def fail(self, what: str, detail: str = "") -> None:
        self.rows.append(("FAIL", what, detail))

    @property
    def failures(self) -> int:
        return sum(1 for level, _, _ in self.rows if level == "FAIL")

    @property
    def warnings(self) -> int:
        return sum(1 for level, _, _ in self.rows if level == "warn")

    def show(self) -> None:
        for level, what, detail in self.rows:
            print(f"  {level:<4}  {what}" + (f"\n        {detail}" if detail else ""))


def _categories(node) -> list[str]:
    import h5py
    if isinstance(node, h5py.Group):
        values = node["categories"][:]
    else:
        values = np.unique(node[:])
    return sorted(v.decode() if isinstance(v, bytes) else str(v) for v in values)


def check_layout(cfg: Config, report: Report) -> None:
    n_if, n_st = len(cfg.if_sections), len(cfg.st_sections)
    report.ok(f"{cfg.n_planes} sections: {n_if} IF + {n_st} ST; "
              f"{cfg.z_span_um + cfg.section_thickness_um:.0f} um of tissue from the first to the last")
    orphans = [s.name for s in cfg.if_sections if not cfg.neighbours[s.index]]
    if orphans:
        report.fail(f"IF section(s) {orphans} have no ST section to be registered to")
    one_sided = [s.name for s in cfg.if_sections if len(cfg.neighbours[s.index]) == 1]
    if one_sided:
        report.warn(f"{', '.join(one_sided)}: only one neighbouring ST section",
                    "normal for the first and last section of a stack; for these IF sections no "
                    "two-sided error estimate can be computed")
    indices = [i for _, _, i in cfg.planes]
    steps = sorted(set(b - a for a, b in zip(indices, indices[1:])))
    if steps and steps != [1]:
        report.warn(f"the section numbers are not consecutive (steps {steps})",
                    "missing numbers are treated as sections that were cut and not kept; the "
                    "depth of each section follows its number")
    if n_st == 1:
        report.warn("the config lists only one ST section",
                    "every IF section then has one ST neighbour, and no two-sided error "
                    "estimate can be computed")


def check_images(cfg: Config, report: Report) -> None:
    needed = max(cfg.if_images.channels) + 1
    for section in cfg.if_sections:
        path = cfg.if_path(section)
        if not path.exists():
            report.fail(f"{section.name}: image not found", str(path))
            continue
        try:
            kind = reader_for(cfg, path)
            if kind == "czi":
                from aicspylibczi import CziFile
                czi = CziFile(path)
                dims = czi.get_dims_shape()[0]
                n_channels = dims["C"][1] - dims["C"][0] if "C" in dims else 1
                box = czi.get_mosaic_bounding_box()
                size = (box.h, box.w)
                found = czi.meta.find(".//Scaling/Items/Distance[@Id='X']/Value")
                pixel = float(found.text) * 1e6 if found is not None and found.text else None
            else:
                import tifffile
                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    axes, shape = series.axes, series.shape
                    size = (shape[axes.index("Y")], shape[axes.index("X")])
                    rest = [n for a, n in zip(axes, shape) if a not in "YX" and n > 1]
                    n_channels = int(np.prod(rest)) if rest else 1
                    pixel = None
                    if tif.is_ome and tif.ome_metadata:
                        import re
                        hit = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tif.ome_metadata)
                        pixel = float(hit.group(1)) if hit else None
        except Exception as error:                               # an unreadable file is a finding
            report.fail(f"{section.name}: cannot open {path.name}", f"{type(error).__name__}: {error}")
            continue
        if n_channels < needed:
            report.fail(f"{section.name}: {path.name} has {n_channels} channel(s)",
                        f"the config names channel index {needed - 1}")
            continue
        detail = f"{size[1]} x {size[0]} px, {n_channels} channels"
        if pixel is None:
            report.ok(f"{section.name}: {path.name}", detail + "; no pixel size in the file to compare")
        elif abs(pixel - cfg.if_images.um_per_px) / cfg.if_images.um_per_px > 0.01:
            report.fail(f"{section.name}: pixel size disagrees",
                        f"the file says {pixel:.4f} um, the config says {cfg.if_images.um_per_px:.4f} um; "
                        "the scale between the two modalities comes from this number")
        else:
            report.ok(f"{section.name}: {path.name}", detail + f", {pixel:.4f} um/px")


def check_scaffold(cfg: Config, report: Report) -> dict:
    """Returns {section name: median GC equivalent diameter in um} where it can."""
    import h5py

    path = Path(cfg.st_scaffold.h5ad)
    if not path.exists():
        report.fail("the ST AnnData file was not found", str(path))
        return {}
    diameters = {}
    with h5py.File(path, "r") as handle:
        obs, obsm = handle["obs"], handle["obsm"]
        good = True
        for key in (cfg.st_scaffold.coord_key, "spatial"):
            if key not in obsm:
                report.fail(f"obsm['{key}'] is missing", f"obsm holds {sorted(obsm)}")
                good = False
        columns = [cfg.st_scaffold.slice_col, cfg.st_scaffold.gc_id_col]
        if cfg.st_scaffold.cluster_col:
            columns.append(cfg.st_scaffold.cluster_col)
        for column in columns:
            if column not in obs:
                report.fail(f"obs['{column}'] is missing")
                good = False
        if not good:
            return {}
        labels = _categories(obs[cfg.st_scaffold.slice_col])
        wanted = {s.source: s.name for s in cfg.st_sections}
        missing = sorted(set(wanted) - set(labels))
        if missing:
            report.fail(f"section(s) {missing} are not in obs['{cfg.st_scaffold.slice_col}']",
                        f"it holds {labels}")
            return {}
        extra = sorted(set(labels) - set(wanted))
        if extra:
            report.warn(f"{len(extra)} section(s) in the AnnData are not in the config and will be ignored",
                        ", ".join(extra))
        node = obs[cfg.st_scaffold.slice_col]
        if isinstance(node, h5py.Group):
            cats = np.array([c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]])
            section = cats[node["codes"][:]]
        else:
            section = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in node[:]])
        gc_node = obs[cfg.st_scaffold.gc_id_col]
        if isinstance(gc_node, h5py.Group):
            report.fail(f"obs['{cfg.st_scaffold.gc_id_col}'] is categorical",
                        "the structure id has to be an integer column, 0 where there is no structure")
            return {}
        gc_id = gc_node[:].astype(np.int64)
        # the unregistered coordinates still sit on the capture lattice, which
        # is what gives the bin pitch; the registered ones no longer do
        lattice_x = obsm["spatial"][:, 0].astype(np.float64) * cfg.st_scaffold.dnb_um
        report.ok(f"{path.name}: {len(gc_id):,} bins, coordinates in obsm['{cfg.st_scaffold.coord_key}']")
        for source, name in wanted.items():
            here = section == source
            ids = gc_id[here]
            n_structures = len(np.unique(ids[ids > 0]))
            if n_structures < 6:
                report.fail(f"{name}: only {n_structures} labelled structure(s)",
                            "the registration needs at least six matched structures per section, "
                            "and works best with a few dozen")
                continue
            xs = np.unique(np.round(lattice_x[here], 3))
            pitch = float(np.median(np.diff(xs))) if len(xs) > 1 else cfg.grid_um
            counts = np.bincount(ids[ids > 0])
            counts = counts[counts > 0]
            diameters[name] = float(np.median(2 * np.sqrt(counts * pitch ** 2 / np.pi)))
            detail = (f"{int(here.sum()):,} bins at ~{pitch:.0f} um, {n_structures} structures, "
                      f"median diameter ~{diameters[name]:.0f} um")
            if abs(pitch - cfg.grid_um) / cfg.grid_um > 0.25:
                report.warn(f"{name}: bins are ~{pitch:.0f} um but align.grid_um is {cfg.grid_um:g}",
                            detail + "; set grid_um to the bin size")
            else:
                report.ok(f"{name}", detail)
    return diameters


def check_depth(cfg: Config, diameters: dict, report: Report) -> None:
    if not diameters:
        return
    typical = float(np.median(list(diameters.values())))
    span = cfg.z_span_um + cfg.section_thickness_um
    if span < typical:
        report.warn(f"the stack is {span:.0f} um of tissue; the median structure is "
                    f"~{typical:.0f} um in diameter",
                    "each 3D column will cover only part of the depth of a structure; the columns "
                    "show how a structure continues between sections, not its whole shape")
    else:
        report.ok(f"the stack ({span:.0f} um of tissue) is deeper than the median structure "
                  f"diameter (~{typical:.0f} um)")


def run(cfg: Config) -> int:
    """Print the findings; return the number of problems that would stop a run."""
    report = Report()
    check_layout(cfg, report)
    check_images(cfg, report)
    diameters = check_scaffold(cfg, report)
    check_depth(cfg, diameters, report)
    print(f"check  {cfg.name}")
    report.show()
    if report.failures:
        print(f"\n{report.failures} problem(s) would stop a run.")
    elif report.warnings:
        print(f"\nA run can start. Read the {report.warnings} line(s) marked `warn` above first.")
    else:
        print("\nA run can start.")
    return report.failures
