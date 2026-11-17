"""The viewer stage on the synthetic run: five planes, made-up names, no optional layers.

Nothing here knows the mouse-spleen experiment, which is the point: the page has
to be built from what the run says - its planes, their kinds and heights, its
GC ids - and not from names or a plane count baked into the template.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import pytest

from gcalign import cli, config
from gcalign.runio import Run

TEMPLATE = Path(__file__).resolve().parents[1] / "gcalign" / "templates" / "viewer.html"
PAYLOAD = re.compile(r'<script id="gc-data" type="application/json">(.*?)</script>', re.S)
PLACEHOLDERS = ("__THEME_VARS__", "__FONT_FACE__", "__TITLE__", "__SUBTITLE__",
                "__PLOTLYJS__", "__PAYLOAD__")
GL_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
           "--ignore-gpu-blocklist"]


@pytest.fixture(scope="module")
def built(finished):
    assert cli.main(["run", "--config", str(finished.config), "--only", "viewer"]) == 0
    cfg = config.load(finished.config)
    path = cfg.output_dir / "05_viewer" / cfg.viewer.filename
    text = path.read_text(encoding="utf-8")
    match = PAYLOAD.search(text)
    assert match, "no embedded payload"
    return finished, cfg, path, text, json.loads(match.group(1))


def test_the_page_is_written_and_every_placeholder_filled(built):
    _, _, path, text, _ = built
    assert path.stat().st_size > 1_000_000              # plotly.js is inlined
    for token in PLACEHOLDERS:
        assert token not in text
    # anything left that looks like a placeholder, outside plotly.js and the data
    outside = PAYLOAD.sub("", text)
    outside = re.sub(r"<script>.*?</script>", "", outside, flags=re.S)
    assert not re.findall(r"__[A-Z][A-Z0-9_]*__", outside)
    # nothing is fetched: the only URL outside the scripts is the SVG namespace name
    cleaned = outside.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in cleaned and "https://" not in cleaned


def test_the_payload_describes_this_run(built):
    finished, cfg, _, _, data = built
    run = Run.open(finished.config)
    planes = data["meta"]["planes"]
    assert [p["name"] for p in planes] == run.names == [name for name, _, _ in finished.planes]
    assert [p["kind"] for p in planes] == [p.kind for p in run.planes]
    assert [p["z"] for p in planes] == pytest.approx([p.z_um for p in run.planes])
    assert data["meta"]["nPlanes"] == len(run.planes) == 5
    for slab in data["planes"]:                          # the drawn cross-sections
        plane = run.plane(slab["name"])
        assert slab["kind"] == plane.kind
        assert slab["z"] == pytest.approx(plane.z_um)
    assert data["ids"] == sorted(int(v) for v in run.metrics["merged_gc_id"])
    assert set(data["ids"]) <= set(run.gc_ids)
    assert data["meta"]["nGcs"] == len(run.presence)


def test_per_gc_numbers_come_from_the_tables(built):
    finished, _, _, _, data = built
    run = Run.open(finished.config)
    metrics = run.metrics.set_index("merged_gc_id")
    presence = run.presence.set_index("merged_gc_id")
    for gid in data["ids"]:
        entry = data["info"][str(gid)]
        assert entry["planes"] == metrics.loc[gid, "n_planes"]
        assert entry["vol"] == pytest.approx(metrics.loc[gid, "volume_um3"])
        assert len(entry["areas"]) == len(run.names)
        for name, area in zip(run.names, entry["areas"]):
            assert (area or 0) == pytest.approx(presence.loc[gid, f"{name}_area_um2"])


def test_qc_and_colours_follow_the_config(built):
    _, cfg, _, _, data = built
    alignment = data["meta"]["alignment"]
    assert sorted(alignment) == sorted(cfg.if_names)
    for name, entry in alignment.items():
        assert entry["oneSided"] == cfg.one_sided(cfg.section(name).index)
    assert sorted(data["meta"]["warp"]) == sorted(cfg.if_names)
    assert data["meta"]["rgb"] == list(cfg.rgb_channels)
    assert data["theme"]["bg"] == "#000000"
    assert set(data["identity"]) >= {str(g) for g in data["ids"]}
    # no optional layers in this run, and the page must not need them
    assert data["follicle"] == [] and data["dzlz"] == []
    assert data["ifImages"] is None and data["zoneBlocks"] is None


def test_the_template_names_no_experiment():
    text = TEMPLATE.read_text(encoding="utf-8")
    for word in ("four-slice", "CD4", "GL7", "IgD", "AF594", "AF647", "nine", "Nine",
                 "startsWith('IF')", 'startsWith("IF")'):
        assert word not in text, word
    # colours come from the theme: no hex literal anywhere in the template
    assert not re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b(?![0-9a-zA-Z])", text.replace("&#39;", ""))


def _chromium():
    explicit = os.environ.get("GCALIGN_CHROMIUM")
    if explicit:
        return explicit
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if root:
        for pattern in ("chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell",
                        "chromium-*/chrome-linux64/chrome"):
            found = sorted(glob.glob(os.path.join(root, pattern)))
            if found:
                return found[-1]
    return None


def test_the_page_opens_without_errors(built):
    sync_api = pytest.importorskip("playwright.sync_api")
    _, _, path, _, data = built
    errors: list[str] = []
    with sync_api.sync_playwright() as pw:
        options = {"args": GL_ARGS}
        if _chromium():
            options["executable_path"] = _chromium()
        try:
            browser = pw.chromium.launch(**options)
        except Exception as error:                      # no browser on this machine
            pytest.skip(f"no Chromium to drive: {str(error).splitlines()[0]}")
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
            page.route("**/*", lambda route: route.continue_()
                       if route.request.url.startswith(("file:", "data:", "blob:")) else route.abort())
            page.goto(path.as_uri())
            page.wait_for_function("window.__gcalign && window.__gcalign.ready === true", timeout=120_000)
            rows = page.evaluate("document.querySelectorAll('#list .row').length")
            assert rows == len(data["ids"])
            for key, view in (("2", "st"), ("3", "if"), ("4", "both"), ("1", "columns")):
                page.keyboard.press(key)
                page.evaluate("window.__gcalign.idle()")
                assert page.evaluate("window.__gcalign.state().view") == view
            page.keyboard.press("4")
            chips = page.evaluate("Array.from(document.querySelectorAll('#chips .chip')).map(c => c.dataset.plane)")
            assert chips == [p["name"] for p in data["planes"]]
            page.keyboard.press("ArrowDown")
            page.evaluate("window.__gcalign.idle()")
            assert page.evaluate("window.__gcalign.state().gc") == data["ids"][0]
            assert page.evaluate("document.querySelectorAll('#card .prof rect.bar').length") > 0
            assert not page.evaluate("document.getElementById('fatal') !== null")
        finally:
            browser.close()
    assert not errors, errors
