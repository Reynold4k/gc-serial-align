"""Smoke tests for the static figures and the HTML report, run on the synthetic experiment.

The synthetic experiment (tests/synthetic.py) has five sections named IF1, S2,
IF3, S4 and IF5, and channels named RING, BODY, BLOB and DAPI. Passing here
shows that the figures depend on none of the section names, channel names or
section count of the project they were designed on. The synthetic data is test
data only and never appears in a deliverable figure.
"""
from __future__ import annotations

import copy
import csv
import re

import numpy as np
import pytest
from PIL import Image

from gcalign import cli, config, figures, report, theme
from gcalign.runio import Run

FLAT = [name for name in figures.WHOLE_RUN if name != "columns"]     # no 3D renderer needed
DPI = 96


def check_png(path, min_side: int = 120) -> np.ndarray:
    """A real figure: big enough, pure black in every corner, and not blank."""
    assert path.exists() and path.suffix == ".png", path
    image = np.asarray(Image.open(path).convert("RGB"))
    height, width = image.shape[:2]
    assert min(height, width) >= min_side
    for row, col in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
        assert tuple(int(v) for v in image[row, col]) == (0, 0, 0), f"corner {row},{col} of {path.name}"
    assert (image.max(axis=2) > 40).mean() > 0.002, f"{path.name} is blank"
    return image


def identity_pixels(image: np.ndarray, colours) -> int:
    """How many pixels carry exactly one of the given colours."""
    flat = image.reshape(-1, 3).astype(np.int32)
    keys = flat[:, 0] * 65536 + flat[:, 1] * 256 + flat[:, 2]
    wanted = [r * 65536 + g * 256 + b for r, g, b in (theme.hex_to_rgb(c) for c in colours)]
    return int(np.isin(keys, wanted).sum())


@pytest.fixture(scope="module")
def drawn(finished, tmp_path_factory):
    cfg = config.load(finished.config)
    out = tmp_path_factory.mktemp("figures")
    return cfg, out, figures.run(cfg, out_dir=out, which=FLAT, dpi=DPI)


@pytest.mark.parametrize("name", FLAT)
def test_whole_run_figure_is_written_on_black(drawn, name):
    _, _, written = drawn
    assert name in written, f"{name} was not written"
    check_png(written[name])


def test_sections_use_the_identity_colours_of_the_config(drawn, tmp_path):
    cfg, _, written = drawn
    run = Run(cfg)
    image = check_png(written["sections"])
    assert identity_pixels(image, run.identity.values()) > 50
    other_cfg = copy.deepcopy(cfg)                   # another palette, same run
    other_cfg.viewer.palette = "hsv"
    other = Run(other_cfg)
    assert other.identity != run.identity
    path = figures.sections(other, tmp_path / "sections_hsv.png", dpi=DPI)
    image = check_png(path)
    assert identity_pixels(image, other.identity.values()) > 50
    assert identity_pixels(image, run.identity.values()) < identity_pixels(image, other.identity.values())


def test_presence_has_one_row_per_gc(drawn):
    cfg, _, _ = drawn
    run = Run(cfg)
    ids, area = figures.presence_matrix(run)
    assert sorted(ids) == run.gc_ids
    assert area.shape == (len(run.gc_ids), len(run.names))
    totals = area.sum(axis=1)
    assert np.all(np.diff(totals) <= 0)                 # largest total area first


def test_orientation_is_skipped_without_a_stored_search(drawn, tmp_path, capsys):
    cfg, _, _ = drawn
    run = Run(cfg)
    run.__dict__["orientation_search"] = None          # what a gcalign 0.1 run looks like
    assert figures.orientation(run, tmp_path / "orientation.png") is None
    assert "skipped the orientation figure" in capsys.readouterr().out
    assert not (tmp_path / "orientation.png").exists()


def test_gc_closeup_and_its_table(drawn):
    cfg, out, _ = drawn
    run = Run(cfg)
    gid = run.gc_ids[0]
    written = figures.run(cfg, out_dir=out, which=["gc"], gcs=[gid], dpi=DPI)
    check_png(written[f"gc{gid}"])
    with written["gc_closeups_csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["plane"] for r in rows] == run.names
    presence = run.presence.set_index("merged_gc_id").loc[gid]
    for row in rows:
        assert int(row["area_px"]) == int(presence[f"{row['plane']}_px"])


def test_3d_columns(drawn):
    if not figures.renderer_available():
        pytest.skip("no off-screen OpenGL renderer on this machine")
    cfg, out, _ = drawn
    run = Run(cfg)
    written = figures.run(cfg, out_dir=out, which=["columns", "column"], gcs=[run.gc_ids[0]], dpi=DPI)
    check_png(written["columns"])
    check_png(written[f"column{run.gc_ids[0]}"])


def test_report_is_self_contained(finished, tmp_path):
    assert cli.main(["run", "--config", str(finished.config), "--only", "validate"]) == 0
    cfg = config.load(finished.config)
    path = report.run(cfg, out_dir=tmp_path, dpi=DPI)
    page = path.read_text(encoding="utf-8")
    assert re.search(r"\{\{\s*[A-Z_]+\s*\}\}", page) is None             # every placeholder filled
    assert page.count("data:image/") >= len(FLAT)
    assert "src=\"http" not in page and "href=\"http" not in page        # nothing to fetch
    assert cfg.name in page
    for name in Run(cfg).names:
        assert f">{name}<" in page
