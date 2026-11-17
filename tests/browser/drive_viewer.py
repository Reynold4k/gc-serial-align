#!/usr/bin/env python3
"""Drive a built gcalign viewer in headless Chromium and check every control.

    python tests/browser/drive_viewer.py VIEWER.html OUTDIR [--chromium PATH]
                                          [--size 1600x1000] [--small 1100x700]
                                          [--expect-empty] [--keep-going]

Opens the page with the network blocked (it must be fully self-contained), then
works through every view, control and keyboard shortcut, checks the state the
page reports after each step, measures text contrast and layout at two window
sizes, and saves screenshots to OUTDIR. Exits 1 if any check failed or the page
logged an error.

Needs the `playwright` Python package and a Chromium build. WebGL runs on
SwiftShader, so it works on a server with no GPU - slowly: allow a few minutes.
Point --chromium (or $GCALIGN_CHROMIUM) at a Chromium / headless-shell binary if
Playwright's own download is not installed.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path

GL_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
           "--ignore-gpu-blocklist"]
READY = "window.__gcalign && window.__gcalign.ready === true"
# Chromium's own advice about canvas read-back, raised inside plotly.js; not an error
BENIGN = ("willReadFrequently",)


class Driver:
    def __init__(self, page, out: Path, keep_going: bool):
        self.page = page
        self.out = out
        self.keep_going = keep_going
        self.failures: list[str] = []
        self.passed = 0
        self.console: list[str] = []
        self.blocked: list[str] = []
        self.shots: list[str] = []
        self.step = "loading"           # the last check or screenshot, to place a page error

    # ---- bookkeeping -------------------------------------------------------
    def check(self, name: str, ok: bool, detail="") -> bool:
        self.step = name
        if ok:
            self.passed += 1
            print(f"  ok    {name}")
        else:
            message = f"{name}: {detail}" if detail != "" else name
            self.failures.append(message)
            print(f"  FAIL  {message}")
            if not self.keep_going and len(self.failures) > 25:
                raise SystemExit("too many failures")
        return ok

    def js(self, expr: str, arg=None):
        return self.page.evaluate(expr, arg) if arg is not None else self.page.evaluate(expr)

    def settle(self, pause: float = 0.6):
        """Wait until every Plotly call has finished and two frames have been drawn."""
        self.page.evaluate("window.__gcalign.idle()")
        time.sleep(pause)
        self.page.evaluate("window.__gcalign.idle()")

    def state(self) -> dict:
        return self.js("window.__gcalign.state()")

    def shot(self, name: str, pause: float = 1.0):
        self.settle(pause)
        path = self.out / f"{name}.png"
        self.step = f"screenshot {name}"
        self.page.screenshot(path=str(path))
        self.shots.append(str(path))
        print(f"  shot  {path.name}")

    def key(self, key: str):
        self.page.keyboard.press(key)
        self.settle(0.3)

    def click(self, selector: str):
        self.page.click(selector)
        self.settle(0.3)

    # ---- page facts --------------------------------------------------------
    def payload(self) -> dict:
        return self.js("JSON.parse(document.getElementById('gc-data').textContent)")

    def visible(self, selector: str) -> bool:
        return bool(self.js("""s => { const e = document.querySelector(s);
            if (!e) return false; const r = e.getBoundingClientRect();
            return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden'; }""", selector))

    def trace(self, index: int) -> dict:
        return self.js("""i => { const t = document.getElementById('plot').data[i];
            return {visible: t.visible, opacity: t.opacity, type: t.type, name: t.name,
                    nf: t.i ? t.i.length : null, n: t.x ? t.x.length : null,
                    marker: t.marker ? {opacity: t.marker.opacity} : null}; }""", index)


# --------------------------------------------------------------------------
# measurements run inside the page
# --------------------------------------------------------------------------
CONTRAST_JS = r"""
() => {
  function parse(c){
    const m = c && c.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(',').map(s => parseFloat(s));
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1};
  }
  const lin = v => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const lum = c => 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
  const over = (fg, bg, a) => ({r: fg.r * a + bg.r * (1 - a), g: fg.g * a + bg.g * (1 - a), b: fg.b * a + bg.b * (1 - a)});
  function background(el){
    const stack = [];
    for (let e = el; e; e = e.parentElement){
      const c = parse(getComputedStyle(e).backgroundColor);
      if (c && c.a > 0) stack.push(c);
      if (c && c.a >= 0.999) break;
    }
    let bg = parse(getComputedStyle(document.body).backgroundColor) || {r: 0, g: 0, b: 0, a: 1};
    for (let i = stack.length - 1; i >= 0; i--) bg = over(stack[i], bg, stack[i].a);
    return bg;
  }
  function opacity(el){ let o = 1; for (let e = el; e; e = e.parentElement) o *= parseFloat(getComputedStyle(e).opacity || 1); return o; }
  const out = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())){
    const text = node.nodeValue.trim();
    if (!text) continue;
    const el = node.parentElement;
    if (!el || el.closest('script,style,svg,#ifStage,[data-off="1"],#plot')) continue;
    const rects = el.getClientRects();
    if (!rects.length) continue;
    const r = rects[0];
    if (r.width === 0 || r.height === 0) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden') continue;
    const fg = parse(st.color);
    if (!fg) continue;
    const bg = background(el);
    const a = fg.a * opacity(el);
    const shown = over(fg, bg, a);
    const L1 = lum(shown), L2 = lum(bg);
    const ratio = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
    out.push({text: text.slice(0, 40), ratio: Math.round(ratio * 100) / 100, size: st.fontSize,
              where: (el.id || el.className || el.tagName).toString().slice(0, 40)});
  }
  return out;
}
"""

LAYOUT_JS = r"""
() => {
  const vw = innerWidth, vh = innerHeight, problems = [];
  const doc = document.scrollingElement || document.documentElement;
  if (doc.scrollWidth > vw + 1) problems.push('page scrolls sideways: ' + doc.scrollWidth + ' > ' + vw);
  const side = document.getElementById('side');
  if (side && side.offsetParent !== null && side.scrollWidth > side.clientWidth + 1)
    problems.push('sidebar overflows sideways: ' + side.scrollWidth + ' > ' + side.clientWidth);
  const shown = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  document.querySelectorAll('.btn, .chip, .seg button, .link, .row .meta, .facts span, .lab, .card-sub, #badge, #legend span, .stats dd')
    .forEach(e => {
      if (!shown(e)) return;
      if (e.scrollWidth > e.clientWidth + 1)
        problems.push('text clipped in ' + (e.id || e.className) + ': "' + e.textContent.trim().slice(0, 30) + '"');
    });
  const qc = document.getElementById('qc');
  if (qc && shown(qc) && qc.scrollWidth > qc.clientWidth + 1)
    problems.push('the QC table needs sideways scrolling: ' + qc.scrollWidth + ' > ' + qc.clientWidth);
  if (/[\u039C]/.test(document.body.innerText)) problems.push('a capital mu (an upper-cased µ) is on screen');
  ['tools', 'card', 'qc', 'legend', 'badge'].forEach(id => {
    const e = document.getElementById(id);
    if (!e || !shown(e)) return;
    const r = e.getBoundingClientRect();
    if (r.left < -1 || r.top < -1 || r.right > vw + 1 || r.bottom > vh + 1)
      problems.push(id + ' leaves the window: ' + JSON.stringify([r.left, r.top, r.right, r.bottom].map(Math.round)));
  });
  const boxes = ['tools', 'card', 'qc', 'legend', 'badge'].map(id => document.getElementById(id))
    .filter(e => e && shown(e)).map(e => [e.id, e.getBoundingClientRect()]);
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++){
    const a = boxes[i][1], b = boxes[j][1];
    if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1)
      problems.push(boxes[i][0] + ' overlaps ' + boxes[j][0]);
  }
  return problems;
}
"""


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as handle:
        head = handle.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", head[16:24])


def png_corners_black(path: Path) -> bool:
    """Decode the PNG (no Pillow needed) and look at the four corner pixels."""
    data = path.read_bytes()
    width, height = png_size(path)
    pos, chunks, bit_depth, colour = 8, [], 8, 6
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            bit_depth, colour = body[8], body[9]
        elif kind == b"IDAT":
            chunks.append(body)
        pos += 12 + length
    if bit_depth != 8 or colour not in (2, 6):
        return True                                     # unusual encoding: do not guess
    bpp = 4 if colour == 6 else 3
    raw = zlib.decompress(b"".join(chunks))
    stride = width * bpp
    rows, prev, offset = [], bytearray(stride), 0
    for _ in range(height):
        filt = raw[offset]
        line = bytearray(raw[offset + 1:offset + 1 + stride])
        offset += 1 + stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if filt == 1:
                line[i] = (line[i] + a) & 255
            elif filt == 2:
                line[i] = (line[i] + b) & 255
            elif filt == 3:
                line[i] = (line[i] + ((a + b) >> 1)) & 255
            elif filt == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    corners = [rows[0][:3], rows[0][stride - bpp:stride - bpp + 3],
               rows[-1][:3], rows[-1][stride - bpp:stride - bpp + 3]]
    return all(max(px) <= 8 for px in corners)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def open_page(browser, url: str, size: tuple[int, int], drv_holder: dict, out: Path, keep_going: bool):
    context = browser.new_context(viewport={"width": size[0], "height": size[1]}, accept_downloads=True,
                                  device_scale_factor=1)
    page = context.new_page()
    drv = Driver(page, out, keep_going)
    drv_holder["drv"] = drv

    def on_route(route):
        url = route.request.url
        if url.startswith(("file:", "data:", "blob:", "about:")):
            route.continue_()
        else:
            drv.blocked.append(url)
            route.abort()

    context.route("**/*", on_route)
    page.on("console", lambda m: drv.console.append(f"console.{m.type} (after: {drv.step}): {m.text}")
            if m.type in ("error",) or (m.type == "warning" and not any(b in m.text for b in BENIGN)) else None)
    page.on("pageerror", lambda e: drv.console.append(
        f"pageerror (after: {drv.step}): {e} | {str(getattr(e, 'stack', '') or '').strip().splitlines()[1:4]}"))
    started = time.time()
    page.goto(url)
    page.wait_for_function(READY, timeout=240_000)
    drv.settle(0.5)
    print(f"  ready in {time.time() - started:.1f} s at {size[0]}x{size[1]}")
    return context, page, drv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("html", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--chromium", default=os.environ.get("GCALIGN_CHROMIUM"),
                    help="Chromium or chrome-headless-shell binary (default: Playwright's own)")
    ap.add_argument("--size", default="1600x1000")
    ap.add_argument("--small", default="1100x700")
    ap.add_argument("--expect-empty", action="store_true", help="the run has no columns: check the empty state")
    ap.add_argument("--keep-going", action="store_true", help="do not stop after 25 failures")
    args = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    html = args.html.resolve()
    if not html.exists():
        print(f"no such file: {html}")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    size = tuple(int(v) for v in args.size.lower().split("x"))
    small = tuple(int(v) for v in args.small.lower().split("x"))
    url = html.as_uri()
    print(f"viewer  {html}  ({html.stat().st_size / 1e6:.2f} MB)")

    launch = {"args": GL_ARGS}
    if args.chromium:
        launch["executable_path"] = args.chromium
    holder: dict = {}
    all_failures: list[str] = []
    console: list[str] = []
    blocked: list[str] = []
    passed = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch)
        try:
            context, page, drv = open_page(browser, url, size, holder, args.out, args.keep_going)
            if args.expect_empty:
                run_empty(drv)
            else:
                run_full(drv, url, small)
            all_failures += drv.failures
            console += drv.console
            blocked += drv.blocked
            passed += drv.passed
            context.close()
        except SystemExit as stop:
            drv = holder.get("drv")
            all_failures += (drv.failures if drv else []) + [str(stop)]
            console += drv.console if drv else []
        except Exception as error:                      # a crash in the driver is a failure too
            drv = holder.get("drv")
            all_failures += (drv.failures if drv else []) + [f"driver crashed: {type(error).__name__}: {error}"]
            console += drv.console if drv else []
            if drv:
                try:
                    drv.page.screenshot(path=str(args.out / "crash.png"))
                except Exception:
                    pass
        finally:
            browser.close()

    if blocked:
        all_failures.append(f"the page tried to reach the network: {blocked[:5]}")
    if console:
        all_failures += [f"page logged: {line}" for line in console]
    print()
    print(f"{passed} checks passed, {len(all_failures)} failed")
    for line in all_failures:
        print(f"  - {line}")
    summary = {"html": str(html), "bytes": html.stat().st_size, "passed": passed,
               "failed": all_failures, "network": blocked}
    (args.out / "drive_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 1 if all_failures else 0


def run_empty(d: Driver):
    d.check("empty state shown", d.visible("#empty"))
    d.check("no list in the empty state", not d.visible("#list"))
    d.check("no fatal error", not d.visible("#fatal"))
    d.shot("00_empty")
    if d.visible("#tQC"):
        d.key("q")
        d.check("QC drawer opens in the empty state", d.visible("#qc"))
        d.shot("00_empty_qc")
        d.key("Escape")
        d.check("Esc closes the QC drawer", not d.visible("#qc"))


def run_full(d: Driver, url: str, small: tuple[int, int]):
    page = d.page
    data = d.payload()
    meta = data["meta"]
    ids = data["ids"]
    planes = [p["name"] for p in data["planes"]]
    kinds = {p["name"]: p["kind"] for p in meta["planes"]}
    info = {int(k): v for k, v in data["info"].items()}
    traces = d.js("window.__gcalign.traces()")
    has_blocks = bool(data.get("zoneBlocks")) or bool(data.get("dzlz"))
    has_points = bool(data.get("dzlz"))
    has_solid = bool(data.get("zoneBlocks"))
    has_foll = bool(data.get("follicle"))
    has_photo = bool(data.get("ifImages"))
    print(f"  payload: {len(ids)} columns, {len(meta['planes'])} planes, blocks={has_blocks}, "
          f"follicle={has_foll}, photo={has_photo}")

    # ---- first impression ---------------------------------------------------
    d.check("no fatal error box", not d.visible("#fatal"))
    d.check("opens on the columns view", d.state()["view"] == "columns")
    rows = d.js("document.querySelectorAll('#list .row').length")
    d.check("one list row per column", rows == len(ids), f"{rows} rows, {len(ids)} ids")
    d.check("columns trace visible", d.trace(traces["columns"])["visible"] is True)
    bg = d.js("getComputedStyle(document.body).backgroundColor")
    d.check("page background is the theme black", bg == "rgb(0, 0, 0)", bg)
    scene_bg = d.js("document.getElementById('plot')._fullLayout.scene.bgcolor")
    d.check("scene background is the theme black", scene_bg.lower() in ("#000000", "rgb(0, 0, 0)", "#000"), scene_bg)
    fonts = d.js("Array.from(document.fonts).filter(f => f.family.replace(/\"/g,'') === 'Inter' && f.status === 'loaded').length")
    d.check("embedded Inter faces loaded", fonts >= 1, fonts)
    d.shot("01_columns")

    # ---- every view, by button ------------------------------------------------
    for view, name in (("st", "02_st"), ("if", "03_if"), ("both", "04_both")):
        d.click(f'#views [data-view="{view}"]')
        st = d.state()
        d.check(f"view button {view}", st["view"] == view, st["view"])
        pressed = d.js(f"document.querySelector('#views [data-view=\"{view}\"]').getAttribute('aria-pressed')")
        d.check(f"view button {view} pressed", pressed == "true")
        want = {"st": ["ST"], "if": ["IF"], "both": ["ST", "IF"]}[view]
        chips = d.js("Array.from(document.querySelectorAll('#chips .chip')).map(c => c.dataset.plane)")
        d.check(f"{view}: one chip per visible plane", chips == [p for p in planes if kinds[p] in want], chips)
        vis = [d.trace(traces["plane0"] + k)["visible"] for k in range(len(planes))]
        d.check(f"{view}: plane traces follow the view",
                vis == [kinds[p] in want for p in planes], vis)
        d.check(f"{view}: columns hidden", d.trace(traces["columns"])["visible"] is False)
        legend = d.js("document.getElementById('legend').textContent")
        d.check(f"{view}: legend names the modalities", all(k in legend for k in want), legend)
        d.shot(name)
    if has_blocks:
        d.click('#views [data-view="blocks"]')
        d.check("view button blocks", d.state()["view"] == "blocks")
        d.shot("05_blocks")

    # ---- x-ray by chip ----------------------------------------------------------
    d.click('#views [data-view="both"]')
    for name in planes:
        d.click(f'#chips .chip[data-plane="{name}"]')
        st = d.state()
        k = planes.index(name)
        focus = d.trace(traces["plane0"] + k)["opacity"]
        other = d.trace(traces["plane0"] + (k + 1) % len(planes))["opacity"] if len(planes) > 1 else 0.14
        d.check(f"x-ray chip {name}", st["slice"] == name and focus == 1 and other < 0.5,
                f"slice={st['slice']} focus={focus} other={other}")
    mid = planes[len(planes) // 2]
    d.click(f'#chips .chip[data-plane="{mid}"]')
    d.shot("06_xray")
    d.click(f'#chips .chip[data-plane="{mid}"]')
    d.check("clicking the active chip clears the x-ray", d.state()["slice"] is None)

    # ---- GC selection: list, search ----------------------------------------------
    pick = ids[min(4, len(ids) - 1)]
    d.click(f'#list .row[data-gc="{pick}"]')
    st = d.state()
    d.check("select a GC from the list", st["gc"] == pick, st["gc"])
    d.check("GC card shown", d.visible("#card"))
    title = d.js("document.querySelector('#card .t').textContent")
    d.check("card names the GC", title == f"GC{pick}", title)
    if info[pick].get("areas"):
        bars = d.js("document.querySelectorAll('#card .prof rect.bar').length")
        present = sum(1 for a in info[pick]["areas"] if a)
        d.check("profile: one bar per plane the GC is on", bars == present, f"{bars} bars, {present} planes")
        slots = d.js("document.querySelectorAll('#card .prof svg > rect, #card .prof svg > g').length")
        d.check("profile: one slot per plane", slots == len(meta["planes"]), f"{slots} slots")
    faces = d.trace(traces["plane0"])["nf"]
    total = data["planes"][0]["nf"]
    d.check("selecting a GC isolates it in the planes", faces is not None and faces < total, f"{faces} < {total}")
    d.shot("07_gc_selected")
    d.click("#cardClose")
    d.check("card close clears the GC", d.state()["gc"] is None and not d.visible("#card"))

    probe = str(ids[-1])[:2] if len(str(ids[-1])) > 1 else str(ids[-1])
    page.fill("#search", probe)
    d.settle(0.2)
    shown = d.js("window.__gcalign.list()")
    d.check("search filters the list", shown and all(str(g).startswith(probe) for g in shown), shown[:10])
    page.press("#search", "Enter")
    d.settle(0.3)
    d.check("Enter selects the first match", d.state()["gc"] == shown[0], d.state()["gc"])
    page.fill("#search", "zzz")
    d.settle(0.2)
    d.check("a search with no match says so", d.js("document.querySelector('#list .none') !== null"))
    page.press("#search", "Escape")
    d.settle(0.2)
    d.check("Esc in the search clears it", d.js("document.getElementById('search').value") == ""
            and len(d.js("window.__gcalign.list()")) == len(ids))

    # ---- sliders -------------------------------------------------------------------
    d.js("() => { const e = document.getElementById('zs'); e.value = 12; e.dispatchEvent(new Event('input')); }")
    d.settle(0.3)
    aspect = d.js("document.getElementById('plot').layout.scene.aspectratio")
    d.check("z slider sets the exaggeration", d.state()["z"] == 12, d.state()["z"])
    # the scale bar must stay true under exaggeration: flat in z, equal x/y scale, exact length
    bar = d.js("""(() => { const g = document.getElementById('plot'), t = g.data[window.__gcalign.traces().scale];
        const s = g.layout.scene, a = (s.annotations || [])[0] || {};
        return {x: Array.from(t.x), y: Array.from(t.y), z: Array.from(t.z), L: window.__gcalign.scaleBar().L,
                label: a.text, xr: s.xaxis.range, yr: s.yaxis.range, asp: s.aspectratio}; })()""")
    length = ((bar["x"][1] - bar["x"][0]) ** 2 + (bar["y"][1] - bar["y"][0]) ** 2) ** 0.5
    d.check("scale bar: exact length", abs(length - bar["L"]) < 1e-6 * bar["L"], f"{length} vs {bar['L']}")
    d.check("scale bar: lies flat in one section", max(bar["z"]) - min(bar["z"]) < 1e-9, bar["z"])
    unit_x = bar["asp"]["x"] / (bar["xr"][1] - bar["xr"][0])
    unit_y = bar["asp"]["y"] / (bar["yr"][1] - bar["yr"][0])
    d.check("scale bar: x and y share one scale", abs(unit_x / unit_y - 1) < 1e-6, f"{unit_x} vs {unit_y}")
    expect = f"{bar['L'] // 1000:g} mm" if bar["L"] >= 1000 else f"{bar['L']:g} µm"
    d.check("scale bar: labelled with its length", bar["label"] == expect, f"{bar['label']!r} vs {expect!r}")
    d.js("() => { const e = document.getElementById('zs'); e.value = 5; e.dispatchEvent(new Event('input')); }")
    d.settle(0.3)
    aspect5 = d.js("document.getElementById('plot').layout.scene.aspectratio")
    d.check("z exaggeration changes the scene aspect",
            aspect["z"] / aspect["x"] > aspect5["z"] / aspect5["x"] * 2, f"{aspect} vs {aspect5}")
    d.click('#views [data-view="st"]')
    d.js("() => { const e = document.getElementById('op'); e.value = 30; e.dispatchEvent(new Event('input')); }")
    d.settle(0.3)
    shell = d.trace(traces["shell"])
    want = 0.30 if d.state()["gc"] is None else min(0.30 + 0.13, 0.45)    # a selected GC is drawn more solid
    d.check("surface opacity slider", abs(d.state()["env"] - 0.30) < 1e-9 and abs(shell["opacity"] - want) < 1e-6
            and shell["visible"] is True, shell)
    d.js("() => { const e = document.getElementById('op'); e.value = 11; e.dispatchEvent(new Event('input')); }")
    d.settle(0.3)

    # ---- follicle --------------------------------------------------------------------
    if has_foll:
        d.click("#foll")
        st = d.state()
        foll = [d.trace(traces["follicle0"] + k) for k in range(len(data["follicle"]))]
        want = [p["kind"] == "ST" for p in data["follicle"]]
        d.check("follicle switch", st["follicle"] is True and [t["visible"] for t in foll] == want,
                [t["visible"] for t in foll])
        d.check("follicle switch reports its state", d.js("document.getElementById('foll').getAttribute('aria-checked')") == "true")
        d.click('#views [data-view="both"]')
        d.shot("08_follicle")
        d.click("#foll")
        d.check("follicle switch off", d.state()["follicle"] is False)
    else:
        d.check("no follicle switch without the layer", not d.visible("#follRow"))

    # ---- DZ / LZ ------------------------------------------------------------------------
    if has_blocks:
        d.key("5")
        d.check("key 5 opens DZ / LZ", d.state()["view"] == "blocks")
        if has_solid and has_points:
            d.click('#zmode [data-zmode="points"]')
            zt = [d.trace(traces["zone0"] + k)["visible"] for k in range(sum(1 for e in data["dzlz"] for c in ("DZ", "LZ") if c in e))]
            d.check("DZ / LZ points", d.state()["zone"] == "points" and any(zt), zt)
            d.shot("09_dzlz_points")
            d.click('#zmode [data-zmode="solid"]')
            bt = [d.trace(traces["blocks0"])["visible"], d.trace(traces["blocks0"] + 1)["visible"]]
            d.check("DZ / LZ solids", d.state()["zone"] == "solid" and bt == [True, True], bt)
            d.shot("10_dzlz_solid")
            gid = data["zoneBlocks"]["ids"][0]
            d.click(f'#list .row[data-gc="{gid}"]')
            d.check("a GC can be isolated in the DZ / LZ view", d.state()["gc"] == gid)
            d.shot("11_dzlz_gc")
            d.key("Escape")
        else:
            d.check("no zone switch with only one representation", not d.visible("#zoneRow"))

    # ---- IF photograph --------------------------------------------------------------------
    if has_photo:
        d.click('#views [data-view="both"]')
        gid = ids[0]
        d.click(f'#list .row[data-gc="{gid}"]')
        d.click("#ifOpen")
        d.check("photo panel opens", d.visible("#ifPanel"))
        loaded = d.js("(() => { const i = document.getElementById('ifImg'); return i.complete && i.naturalWidth > 0; })()")
        d.check("photo decoded", loaded)
        polys = d.js("document.querySelectorAll('#ifSvg polygon').length")
        d.check("photo outlines drawn", polys > 0, polys)
        foot = d.js("document.getElementById('ifChannels').textContent")
        d.check("photo footer names the channels", all(c.replace("_", " ") in foot for c in meta["rgb"] if c), foot)
        d.shot("12_photo_gc")
        before = d.js("document.getElementById('ifWrap').style.transform")
        box = page.locator("#ifStage").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.wheel(0, -400)
        d.settle(0.3)
        after = d.js("document.getElementById('ifWrap').style.transform")
        d.check("photo zooms with the wheel", before != after, f"{before} -> {after}")
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2 + 60, box["y"] + box["height"] / 2 + 40, steps=4)
        page.mouse.up()
        d.check("photo pans with a drag", d.js("document.getElementById('ifWrap').style.transform") != after)
        d.click("#ifFit")
        names = [p["plane"] for p in data["ifImages"]["planes"]]
        if len(names) > 1:
            d.click(f'#ifChips button[data-plane="{names[1]}"]')
            d.check("photo plane chips", d.js("document.getElementById('ifTitle').textContent") == names[1])
            d.key("ArrowRight")
            d.check("→ steps the photo plane", d.js("document.getElementById('ifTitle').textContent") == names[2 % len(names)])
        d.click("#ifOut")
        d.check("outlines toggle", d.js("document.querySelectorAll('#ifSvg polygon').length") == 0)
        d.click("#ifOut")
        d.key("Escape")
        d.check("Esc closes the photo panel", not d.visible("#ifPanel"))
        d.key("p")
        d.check("P opens the photo panel", d.visible("#ifPanel"))
        d.shot("13_photo")
        d.click("#ifClose")
        d.check("× closes the photo panel", not d.visible("#ifPanel"))
        d.key("Escape")
    else:
        d.check("no photo row without photographs", not d.visible("#photoRow"))

    # ---- QC drawer ---------------------------------------------------------------------------
    al, wp = meta.get("alignment", {}), meta.get("warp", {})
    qc_names = [p["name"] for p in meta["planes"] if p["kind"] == "IF" and (p["name"] in al or p["name"] in wp)]
    if qc_names:
        d.click("#tQC")
        d.check("QC drawer opens from the toolbar", d.visible("#qc") and d.state()["qc"] is True)
        rows = d.js("Array.from(document.querySelectorAll('#qc tbody tr')).map(r => r.dataset.plane)")
        d.check("QC: one row per IF section", rows == qc_names, rows)
        flags = d.js("Array.from(document.querySelectorAll('#qc tbody tr')).filter(r => r.querySelector('.flag')).map(r => r.dataset.plane)")
        one_sided = [n for n in qc_names if al.get(n, {}).get("oneSided")]
        d.check("QC: one-sided sections flagged", flags == one_sided, f"{flags} vs {one_sided}")
        cells = d.js("Array.from(document.querySelectorAll('#qc tbody tr:first-child td')).map(c => c.textContent)")
        first = qc_names[0]
        anchors = wp.get(first, {}).get("anchors")
        d.check("QC: numbers come from the payload", anchors is None or str(anchors) in cells, cells)
        d.shot("14_qc")
        d.key("q")
        d.check("Q closes the QC drawer", not d.visible("#qc"))
        d.key("Q")
        d.check("Q opens the QC drawer", d.visible("#qc"))
        d.key("Escape")
        d.check("Esc closes the QC drawer", not d.visible("#qc"))

    # ---- sort ----------------------------------------------------------------------------------
    vol = lambda g: info[g]["vol"] or 0
    for mode, expect in (("vol", sorted(ids, key=lambda g: (-vol(g), g))),
                         ("planes", sorted(ids, key=lambda g: (-info[g]["planes"], -vol(g), g))),
                         ("id", sorted(ids))):
        d.click(f'#sort [data-sort="{mode}"]')
        order = d.js("Array.from(document.querySelectorAll('#list .row')).map(r => +r.dataset.gc)")
        d.check(f"sort by {mode}", order == expect and d.state()["sort"] == mode, order[:6])

    # ---- keyboard --------------------------------------------------------------------------------
    d.key("Escape")
    d.key("Escape")
    for number, view in enumerate(["columns", "st", "if", "both"] + (["blocks"] if has_blocks else []), start=1):
        d.key(str(number))
        d.check(f"key {number} -> {view}", d.state()["view"] == view, d.state()["view"])
    d.key("4")
    for step in range(len(planes) + 1):
        d.key("ArrowRight")
    d.check("→ steps through every section and back to none", d.state()["slice"] is None, d.state()["slice"])
    d.key("ArrowRight")
    d.check("→ x-rays the first section", d.state()["slice"] == planes[0], d.state()["slice"])
    d.key("ArrowLeft")
    d.key("ArrowLeft")
    d.check("← steps backwards and wraps", d.state()["slice"] == planes[-1], d.state()["slice"])
    order = d.js("window.__gcalign.list()")
    d.key("ArrowDown")
    d.check("↓ selects the first GC", d.state()["gc"] == order[0], d.state()["gc"])
    d.key("ArrowDown")
    d.check("↓ steps to the next GC", d.state()["gc"] == order[1 % len(order)], d.state()["gc"])
    d.key("ArrowUp")
    d.check("↑ steps back", d.state()["gc"] == order[0], d.state()["gc"])
    d.key("Escape")
    st = d.state()
    d.check("Esc clears section and GC", st["slice"] is None and st["gc"] is None, st)
    d.key("?")
    d.check("? opens the shortcut sheet", d.visible("#help"))
    d.shot("15_help")
    d.key("?")
    d.check("? closes the shortcut sheet", not d.visible("#help"))
    d.key("?")
    d.key("Escape")
    d.check("Esc closes the shortcut sheet", not d.visible("#help"))
    d.key("/")
    d.check("/ focuses the search", d.js("document.activeElement && document.activeElement.id") == "search")
    page.press("#search", "Escape")
    if has_foll:
        d.key("f")
        d.check("F toggles the follicle backdrop", d.state()["follicle"] is True)
        d.key("f")
    eye_before = d.js("window.__gcalign.camera().eye")
    box = page.locator("#plot").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.62, box["y"] + box["height"] * 0.5, steps=6)
    page.mouse.up()
    d.settle(0.5)
    eye_dragged = d.js("window.__gcalign.camera().eye")
    d.check("dragging the scene turns the camera", eye_dragged != eye_before, f"{eye_before} -> {eye_dragged}")
    d.key("0")
    eye = d.js("window.__gcalign.camera().eye")
    n = (eye["x"] ** 2 + eye["y"] ** 2 + eye["z"] ** 2) ** 0.5
    home = (0.57, -0.82, 0.53)                      # EYE0 in the template
    m = sum(v * v for v in home) ** 0.5
    d.check("0 resets the camera direction",
            all(abs(eye[a] / n - home[k] / m) < 0.02 for k, a in enumerate("xyz")), eye)

    # ---- a click selects a GC in the scene; a drag that starts on a GC only turns the scene ------
    d.key("1")
    d.key("Escape")
    d.js("""() => { window.__hov = null; document.getElementById('plot').on('plotly_hover', e => {
            const p = e.points && e.points[0]; if (p) window.__hov = {c: p.curveNumber, v: p.pointNumber}; }); }""")
    spot = None
    for fy in (0.5, 0.42, 0.58, 0.35, 0.65):
        for fx in (0.5, 0.42, 0.58, 0.35, 0.65, 0.28, 0.72):
            page.mouse.move(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
            d.settle(0.25)
            hov = d.js("window.__hov")
            if hov and hov["c"] == traces["columns"]:
                spot = (box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
                break
            d.js("() => { window.__hov = null; }")
        if spot:
            break
    if d.check("found a GC column under the pointer", spot is not None):
        cam0 = d.js("window.__gcalign.camera().eye")
        page.mouse.move(*spot)
        page.mouse.down()
        page.mouse.move(spot[0] + 90, spot[1] + 10, steps=6)
        page.mouse.up()
        d.settle(0.6)
        d.check("a drag that starts on a GC turns the scene without selecting the GC",
                d.state()["gc"] is None and d.js("window.__gcalign.camera().eye") != cam0, d.state()["gc"])
        d.key("0")
        page.mouse.move(*spot)
        d.settle(0.3)
        page.mouse.down()
        d.settle(0.3)
        page.mouse.up()
        d.settle(0.6)
        d.check("a click on a GC column selects it", d.state()["gc"] is not None, d.state()["gc"])
        d.key("Escape")

    # ---- auto-rotate -------------------------------------------------------------------------------
    d.key("r")
    d.check("R starts the turntable", d.state()["rot"] is True)
    start = d.js("window.__gcalign.camera().eye")
    time.sleep(3.0)
    moved = d.js("window.__gcalign.camera().eye")
    d.check("the turntable turns the camera", moved != start, f"{start} -> {moved}")
    d.check("the turntable keeps the height", abs(moved["z"] - start["z"]) < 1e-4, f"{start} -> {moved}")
    page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.5, steps=3)
    page.mouse.up()
    d.settle(0.5)
    d.check("a drag stops the turntable", d.state()["rot"] is False)
    d.click("#tRot")
    d.check("the toolbar starts the turntable", d.state()["rot"] is True)
    d.click("#tRot")
    d.check("the toolbar stops the turntable", d.state()["rot"] is False)
    d.key("0")

    # ---- clean capture -------------------------------------------------------------------------------
    d.key("1")
    d.key("h")
    d.check("H hides the interface", not d.visible("#side") and not d.visible("#tools"))
    width = d.js("document.getElementById('plot').clientWidth")
    d.check("the scene takes the whole window", width == page.viewport_size["width"], width)
    d.shot("16_bare_columns", pause=1.5)
    d.key("h")
    d.check("H shows the interface again", d.visible("#side") and d.visible("#tools"))
    d.key("h")
    d.key("Escape")
    d.check("Esc also brings the interface back", d.visible("#side"))

    for trigger in ("button", "key"):
        with page.expect_download(timeout=180_000) as info_dl:
            if trigger == "button":
                page.click("#tPng")
            else:
                page.keyboard.press("s")
        download = info_dl.value
        target = d.out / f"export_{trigger}_{download.suggested_filename}"
        download.save_as(str(target))
        w, h = png_size(target)
        pw_, ph_ = d.js("[document.getElementById('plot').clientWidth, document.getElementById('plot').clientHeight]")
        d.check(f"PNG ({trigger}) is twice the scene size", (w, h) == (2 * pw_, 2 * ph_), f"{w}x{h} for {pw_}x{ph_}")
        d.check(f"PNG ({trigger}) has a black background", png_corners_black(target))
        st = d.state()
        expect = (f"{meta['fileStem']}_{st['view']}" + (f"_{st['slice']}" if st["slice"] else "")
                  + (f"_GC{st['gc']}" if st["gc"] is not None else "") + ".png")
        d.check(f"PNG ({trigger}) is named after the view", download.suggested_filename == expect,
                f"{download.suggested_filename} vs {expect}")
        d.settle(0.5)

    # ---- the URL hash --------------------------------------------------------------------------------
    gid = ids[min(2, len(ids) - 1)]
    d.key("4")
    sec = next((p for p in planes if kinds[p] == "IF"), planes[0])
    d.click(f'#chips .chip[data-plane="{sec}"]')
    d.click(f'#list .row[data-gc="{gid}"]')
    d.js("() => { const e = document.getElementById('zs'); e.value = 8; e.dispatchEvent(new Event('input')); }")
    if has_foll:
        d.click("#foll")
    d.click('#sort [data-sort="vol"]')
    if qc_names:
        d.key("q")
    time.sleep(0.8)
    wanted = d.state()
    cam_before = d.js("window.__gcalign.camera().eye")
    hash_text = d.js("location.hash")
    d.check("state is written to the hash", f"gc={gid}" in hash_text and "view=both" in hash_text, hash_text)
    page.reload()
    page.wait_for_function(READY, timeout=240_000)
    d.settle(0.8)
    got = d.state()
    same = {k: (wanted[k], got[k]) for k in ("view", "slice", "gc", "z", "follicle", "sort", "qc", "env", "zone")
            if wanted[k] != got[k]}
    d.check("reload restores the state from the hash", not same, same)
    cam_after = d.js("window.__gcalign.camera().eye")
    d.check("reload restores the camera", all(abs(cam_after[a] - cam_before[a]) < 0.01 for a in "xyz"),
            f"{cam_before} vs {cam_after}")
    d.shot("17_restored")
    d.js("location.hash = 'view=if&z=3'")
    time.sleep(0.6)
    d.settle(0.5)
    st = d.state()
    d.check("editing the hash by hand applies it", st["view"] == "if" and st["z"] == 3 and st["gc"] is None, st)

    page.goto("about:blank")
    page.goto(url + "#view=bogus&gc=9999999&z=abc&sec=nope&env=-5&foll=2&zone=x&sort=%3Cb%3E&cam=1,2&ui=maybe")
    page.wait_for_function(READY, timeout=240_000)
    d.settle(0.5)
    st = d.state()
    d.check("an invalid hash falls back to the defaults",
            st["view"] == "columns" and st["gc"] is None and st["slice"] is None and st["sort"] == "id"
            and st["ui"] is True, st)
    d.check("no fatal error after an invalid hash", not d.visible("#fatal"))

    # ---- contrast and layout at full size ----------------------------------------------------------
    d.key("4")
    d.click(f'#list .row[data-gc="{ids[0]}"]')
    if qc_names:
        d.key("q")
    d.settle(0.5)
    texts = d.js(CONTRAST_JS)
    low = [t for t in texts if t["ratio"] < 4.5]
    d.check(f"text contrast >= 4.5:1 ({len(texts)} text runs, min "
            f"{min((t['ratio'] for t in texts), default=0):.2f})", not low, low[:8])
    problems = d.js(LAYOUT_JS)
    d.check("layout at full size: nothing clipped, overlapping or off-screen", not problems, problems)
    d.shot("18_full_gc_qc")
    if qc_names:
        d.key("q")

    # ---- small window ---------------------------------------------------------------------------------
    page.set_viewport_size({"width": small[0], "height": small[1]})
    d.settle(1.0)
    d.key("Escape")
    d.key("1")
    d.shot("19_small_columns")
    problems = d.js(LAYOUT_JS)
    d.check(f"layout at {small[0]}x{small[1]}: columns view", not problems, problems)
    d.key("4")
    d.click(f'#list .row[data-gc="{ids[0]}"]')
    if qc_names:
        d.key("q")
    problems = d.js(LAYOUT_JS)
    d.check(f"layout at {small[0]}x{small[1]}: GC card + QC drawer", not problems, problems)
    list_h = d.js("document.getElementById('list').getBoundingClientRect().height")
    d.check("the GC list keeps a usable height in a small window", list_h >= 100, list_h)
    d.shot("20_small_gc_qc")
    if has_photo:
        d.key("p")
        d.shot("21_small_photo")
        d.key("Escape")
    texts = d.js(CONTRAST_JS)
    low = [t for t in texts if t["ratio"] < 4.5]
    d.check("text contrast in the small window", not low, low[:8])


if __name__ == "__main__":
    sys.exit(main())
