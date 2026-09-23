#!/usr/bin/env python3
"""Load the real pipeline outputs and turn them into the film's textures.

Nothing here recomputes a registration.  Every coordinate comes from a file the
nine-section pipeline already wrote:

  02_global_align/global_transforms.json   Stage-A similarity per IF section
  rep{n}_verified_icp_replay.json          the 12 saved ICP iterations, replayed
                                           with gcalign's own functions and
                                           asserted identical to the saved result
  03_gc_match/plane_gc_labels.npz          final GC id raster, all nine planes
  03_gc_match/if_channels_warped.npz       the real stain, warped by the same map
  03_gc_match/match_report.json            anchors, TPS residual, LOO error
  02_global_align/st_rasters.npz           frozen Stereo-seq tissue and GC masks
  01_if_gc/IF*_gc.npz                      each IF section still in its own frame

Act 1 additionally needs the upstream Stereo-seq run, because the per-iteration
coordinates live there and not in gcalign's output.  All of it is located
through ``film_paths``; nothing below hard-codes an installation.

The one recomputation is cosmetic and flagged as such: the warped IF silhouette
is re-cut from the warped channels with gcalign's own tissue recipe, because the
pipeline never stored a warped tissue mask.
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

import film_paths as P
import story_core as S

NINE = P.NINE_SLICE_DIR

DNB_UM = 0.5                 # one Stereo-seq spatial unit, from the joint yaml
SECTION_UM = 10.0            # true section pitch
Z_EXAG = 45.0                # z stretch, stated on screen
SLAB_FRACTION = 0.17         # drawn slab thickness as a fraction of the pitch
TEX_SCALE = 2

# The three Spateo stages, in the order they ran, and the obsm key each one
# stored its per-iteration coordinates under.
STAGES = [("iter_rigid", "align_rigid"),
          ("iter_nonrigid", "align_nonrigid"),
          ("iter_deformable", "align_deformable")]

CHANNEL_TINT = {                     # the three channels the GC figures use
    "GL7_AF488": (84, 255, 158),     # germinal-centre marker
    "IgD_AF647": (255, 92, 198),     # mantle / follicle
    "DAPI": (86, 132, 255),
}
CHANNEL_GAIN = {"GL7_AF488": 1.55, "IgD_AF647": 1.00, "DAPI": 0.42}


# --------------------------------------------------------------------------
# The GC identity palette.  Reproduced here rather than imported so a GC keeps
# the same colour in the film as in every other figure of the study: hues walk
# the replicate's band by the golden ratio, so neighbouring ids never land on
# neighbouring hues, and four saturation/value tiers separate ids that do.
HUE_BAND = {"rep1": (0.00, 0.42), "rep2": (0.52, 0.94)}
TIERS = [(0.95, 0.90), (0.72, 0.62), (0.98, 0.58), (0.60, 0.97)]
GOLDEN = 0.6180339887498949


def gc_palette(rep, ids):
    lo, hi = HUE_BAND[rep]
    out = {}
    for k, g in enumerate(sorted(int(i) for i in ids)):
        rgb = colorsys.hsv_to_rgb(lo + ((0.13 + k * GOLDEN) % 1.0) * (hi - lo),
                                  *TIERS[k % len(TIERS)])
        out[int(g)] = tuple(int(round(c * 255)) for c in rgb)
    return out


def hue_family(ids, base_hex, hue_spread, sat_range, val_range):
    """gcalign's viewer palette, copied so act 5 matches the HTML exactly."""
    base = tuple(int(base_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h0, _, _ = colorsys.rgb_to_hsv(*base)
    out = {}
    for index, gid in enumerate(ids):
        t = index / max(len(ids) - 1, 1)
        hue = (h0 + hue_spread * (t - 0.5)) % 1.0
        sat = sat_range[0] + (sat_range[1] - sat_range[0]) * ((index * 7) % 5) / 4
        val = val_range[0] + (val_range[1] - val_range[0]) * ((index * 3) % 4) / 3
        out[int(gid)] = tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, sat, val))
    return out


# --------------------------------------------------------------------------
def _over(base_rgb, base_a, col, al):
    """Source-over one layer onto a float RGBA accumulator."""
    col = np.asarray(col, np.float32)
    if col.ndim == 1:
        col = np.broadcast_to(col, base_rgb.shape)
    na = al + base_a * (1 - al)
    num = col * al[..., None] + base_rgb * (base_a * (1 - al))[..., None]
    rgb = np.where(na[..., None] > 1e-6, num / np.maximum(na, 1e-6)[..., None], 0.0)
    return rgb.astype(np.float32), na.astype(np.float32)


def compose(shape, layers, scale=TEX_SCALE, blur=0.0) -> Image.Image:
    """Stack (alpha, colour) layers and upsample once, softening the 10 um grid."""
    h, w = shape
    rgb = np.zeros((h, w, 3), np.float32)
    a = np.zeros((h, w), np.float32)
    for al, col in layers:
        rgb, a = _over(rgb, a, col, np.clip(np.asarray(al, np.float32), 0, 1))
    arr = np.concatenate([np.clip(rgb, 0, 255), np.clip(a * 255, 0, 255)[..., None]], 2)
    img = Image.fromarray(arr.astype(np.uint8), "RGBA")
    if scale != 1:
        img = img.resize((w * scale, h * scale), Image.BICUBIC)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


def body_shade(mask, reach=14.0):
    """Darken the last few pixels inside the outline - a cut edge, not paper."""
    d = ndi.distance_transform_edt(np.asarray(mask, bool))
    return (0.58 + 0.42 * np.clip(d / reach, 0, 1)).astype(np.float32)


def outer_rim(mask):
    """A halo just outside a mask, so even a 2-px GC keeps its own fill."""
    m = np.asarray(mask, bool)
    return (ndi.binary_dilation(m, iterations=1) & ~m).astype(np.float32)


def rim(mask, outer=1, inner=1):
    """A one-pixel-ish band hugging the outside of a mask."""
    m = np.asarray(mask, bool)
    return (ndi.binary_dilation(m, iterations=outer) &
            ~ndi.binary_erosion(m, iterations=inner)).astype(np.float32)


def colour_by_label(labels, lut, default=(120, 130, 142)):
    """Per-pixel RGB from an integer label raster."""
    h, w = labels.shape
    out = np.zeros((h, w, 3), np.float32)
    out[...] = default
    for gid in np.unique(labels):
        if gid == 0:
            continue
        out[labels == gid] = lut.get(int(gid), default)
    return out


def fluorescence(channels: dict, mask=None, gain=1.0) -> np.ndarray:
    """Additive three-channel composite, each channel on its own 99.5 % ceiling."""
    first = next(iter(channels.values()))
    out = np.zeros(first.shape + (3,), np.float32)
    for name, tint in CHANNEL_TINT.items():
        if name not in channels:
            continue
        plane = np.asarray(channels[name], np.float32)
        inside = plane[mask] if mask is not None else plane[plane > 0]
        if inside.size:
            floor, top = np.percentile(inside, [55.0, 99.5])
        else:
            floor, top = 0.0, 1.0
        norm = np.clip((plane - floor) / max(float(top - floor), 1e-6), 0, 1) ** 0.85
        out += norm[..., None] * np.array(tint, np.float32) * CHANNEL_GAIN[name] * gain
    return np.clip(out, 0, 255)


def tissue_from_warped(channels: dict) -> np.ndarray:
    """Re-cut the section outline with gcalign's own recipe.

    Cosmetic only - the pipeline stores no warped tissue mask, and the film
    needs a silhouette to give each plane a body and a cut edge."""
    weights = {"CD4_AF594": 1.0, "IgD_AF647": 0.6, "GL7_AF488": 0.3}
    combined = np.zeros(next(iter(channels.values())).shape, np.float32)
    for name, weight in weights.items():
        sm = ndi.gaussian_filter(np.asarray(channels[name], np.float32), 6.0)
        low, high = np.percentile(sm, [1, 99.5])
        combined += weight * np.clip((sm - low) / max(high - low, 1.0), 0, 1)
    mask = combined > filters.threshold_otsu(combined)
    mask = morphology.binary_closing(mask, morphology.disk(6))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, 3000)
    lab = measure.label(mask)
    if lab.max() > 1:
        areas = np.bincount(lab.ravel())
        areas[0] = 0
        mask = lab == int(areas.argmax())
    return ndi.binary_fill_holes(mask)


# --------------------------------------------------------------------------
class Assets:
    def __init__(self, rep: str, verbose=True):
        self.rep = rep
        root = NINE / rep
        self.root = root
        say = print if verbose else (lambda *a, **k: None)

        self.labels = {k: v for k, v in np.load(root / "03_gc_match/plane_gc_labels.npz").items()
                       if k != "grid"}
        self.chan = np.load(root / "03_gc_match/if_channels_warped.npz")
        self.st_ras = np.load(root / "02_global_align/st_rasters.npz")
        self.transforms = json.loads((root / "02_global_align/global_transforms.json").read_text())
        self.match = json.loads((root / "03_gc_match/match_report.json").read_text())
        self.validation = json.loads((root / "validation_report.json").read_text())
        self.icp = json.loads(
            (P.ICP_REPLAY_DIR / f"{rep}_verified_icp_replay.json").read_text())
        self.manifest = {f"IF{m['section']}": m
                         for m in json.loads((root / "01_if_gc/if_gc_manifest.json").read_text())}

        self.order = list(self.validation["geometry"]["order"])
        self.kind = {k: ("IF" if k.startswith("IF") else "ST") for k in self.order}
        self.section = {k: i + 1 for i, k in enumerate(self.order)}
        self.st_names = [k for k in self.order if self.kind[k] == "ST"]
        self.if_names = [k for k in self.order if self.kind[k] == "IF"]
        self.neighbours = {k: self.transforms["sections"][k]["neighbours"] for k in self.if_names}

        g = self.transforms["grid"]
        self.grid = g
        self.shape = (int(g["height"]), int(g["width"]))
        self.um = float(g["um_per_px"])
        x0, y0 = float(g["x0_um"]), float(g["y0_um"])
        # world Y is negated once, here, so screen-up is +Y everywhere downstream
        self.bbox = (x0, x0 + self.shape[1] * self.um,
                     -(y0 + self.shape[0] * self.um), -y0)
        self.centre = np.array([(self.bbox[0] + self.bbox[1]) / 2,
                                (self.bbox[2] + self.bbox[3]) / 2])
        self.half_width = (self.bbox[1] - self.bbox[0]) / 2

        self.ids = sorted({int(v) for lab in self.labels.values() for v in np.unique(lab) if v > 0})
        self.pal = gc_palette(rep, self.ids)
        self.st_view = hue_family(self.ids, S.ST_VIEWER_HEX, 0.085, (0.62, 1.0), (0.74, 1.0))
        self.if_view = hue_family(self.ids, S.IF_VIEWER_HEX, 0.075, (0.58, 0.98), (0.62, 1.0))

        self.slab = SECTION_UM * Z_EXAG * SLAB_FRACTION
        self.tissue = {}
        self.tex, self.shadow, self.view_tex = {}, {}, {}
        self.overlay_tex = {}
        for key in self.order:
            self._build_plane(key)
            say(f"  texture {key}", flush=True)
        self.if_local = {k: self._build_local(k) for k in self.if_names}
        say(f"  {rep}: {len(self.ids)} GC ids, {len(self.order)} planes ready", flush=True)

    # -- geometry ----------------------------------------------------------
    def z(self, key) -> float:
        return (self.section[key] - 1) * SECTION_UM * Z_EXAG

    def corners(self, key, z=None):
        return S.grid_corners(self.bbox, self.z(key) if z is None else z)

    def world_xy(self, dnb_xy):
        """Stereo-seq spatial units -> render world (x, -y)."""
        p = np.asarray(dnb_xy, np.float32) * DNB_UM
        return np.column_stack([p[:, 0], -p[:, 1]])

    # -- textures ----------------------------------------------------------
    def _build_plane(self, key):
        lab = self.labels[key]
        gc = lab > 0
        if self.kind[key] == "ST":
            tissue = np.asarray(self.st_ras[f"tissue_{key}"], bool)
            accent = S.ST_ACCENT
            body = np.array(S.TISSUE_BODY, np.float32)
            body_layer = np.broadcast_to(body, tissue.shape + (3,))
            gc_alpha = 0.97
        else:
            chans = {c: self.chan[f"{key}|{c}"] for c in
                     ["IgD_AF647", "CD4_AF594", "GL7_AF488", "DAPI"]}
            tissue = tissue_from_warped(chans)
            accent = S.IF_ACCENT
            stain = fluorescence(chans, mask=tissue)
            body_layer = np.clip(np.array([14.0, 20.0, 30.0]) + stain * 0.92, 0, 255)
            gc_alpha = 0.50
        self.tissue[key] = tissue

        tb = tissue.astype(np.float32)
        body_layer = np.asarray(body_layer, np.float32) * body_shade(tissue)[..., None]
        edge = rim(tissue, 2, 1)
        gcc = colour_by_label(lab, self.pal)
        gc_rim = outer_rim(gc)
        layers = [
            (tb * 0.93, body_layer),
            (edge * 0.80, np.array(accent, np.float32) * 0.75),
            (gc.astype(np.float32) * gc_alpha, gcc),
            (gc_rim * 0.72, np.clip(gcc * 1.18 + 14, 0, 255)),
        ]
        self.tex[key] = compose(self.shape, layers)
        self.shadow[key] = compose(self.shape, [(tb * 0.92, (5, 8, 13))], blur=1.2)

        # act 3 lays the Stereo-seq GC over the IF stain as a contour, the way
        # an overlay figure does: a thick outline reads at any zoom, a fill
        # would hide the stain it is supposed to be checked against
        thick = (ndi.binary_dilation(gc, iterations=2) &
                 ~ndi.binary_erosion(gc, iterations=1)).astype(np.float32)
        self.overlay_tex[key] = compose(self.shape, [
            (gc.astype(np.float32) * 0.22, gcc),
            (thick * 0.98, np.clip(gcc * 1.25 + 26, 0, 255)),
        ])

        lut = self.st_view if self.kind[key] == "ST" else self.if_view
        vcol = colour_by_label(lab, lut)
        self.view_tex[key] = compose(self.shape, [
            (tb * 0.09, np.array(lut[self.ids[0]], np.float32) * 0.26 + 18),
            (edge * 0.55, np.array(accent, np.float32) * 0.8),
            (gc.astype(np.float32) * 0.95, vcol),
            (gc_rim * 0.80, np.clip(vcol * 1.25 + 26, 0, 255)),
        ])

    def _build_local(self, key):
        """The IF section still in its own frame, for the ICP replay."""
        z = np.load(self.root / f"01_if_gc/{key}_gc.npz")
        tissue = np.asarray(z["tissue"], bool)
        obj = np.asarray(z["gc_labels"], np.int32)
        # assigned is keyed by regionprops label.  When match_warp splits one IF
        # object between two GCs it appends a row past labels.max(); that
        # fragment has no pixels of its own here, so the local raster keeps the
        # object's dominant identity and the extra key is simply unused.
        assigned = {int(k): int(v) for k, v in self.match[key]["assigned"].items()}
        merged = np.zeros(max(int(obj.max()), max(assigned)) + 1, np.int32)
        for label, gid in assigned.items():
            merged[label] = gid
        idmap = merged[obj]
        chans = {c: np.asarray(z[c], np.float32) for c in CHANNEL_TINT}
        stain = fluorescence(chans, mask=tissue)
        gc = obj > 0
        matched = idmap > 0
        gcc = colour_by_label(idmap, self.pal, default=(118, 130, 145))
        tb = tissue.astype(np.float32)
        layers = [
            (tb * 0.95, np.clip((np.array([14.0, 20.0, 30.0]) + stain * 0.95)
                                * body_shade(tissue)[..., None], 0, 255)),
            (rim(tissue, 2, 1) * 0.85, np.array(S.IF_ACCENT, np.float32) * 0.8),
            (gc.astype(np.float32) * 0.46, gcc),
            (outer_rim(gc) * 0.80, np.clip(gcc * 1.22 + 16, 0, 255)),
        ]
        entry = self.transforms["sections"][key]
        cents = np.asarray(entry["if_centroids_um"], float)
        # the object ids match the manifest order, i.e. regionprops label - 1
        return {
            "tex": compose(tissue.shape, layers),
            "shadow": compose(tissue.shape, [(tb * 0.92, (5, 8, 13))], blur=1.2),
            "shape": tissue.shape,
            # source_pixels() divides local micrometres by the 10 um grid, so the
            # raster's effective pitch in the alignment is exactly grid_um
            "um_per_px": (self.um, self.um),
            "centroids_um": cents,
            "matched": np.array([(i + 1) in assigned for i in range(len(cents))]),
            "steps": self.icp[key]["steps"],
            "history": self.transforms["sections"][key]["icp_history"],
        }

    # -- act 1 -------------------------------------------------------------
    def load_trajectory(self, tissue_stride=10, gc_stride=1):
        """Open the Stereo-seq runs and hand back a streaming handle per slice.

        The h5ad files stay open: act 1 draws 1,100 iterations, and holding all
        of them in memory is ~3 GB per moving slice, so each frame reads the one
        iteration it needs.  Nothing is interpolated - every frame is a stored
        optimiser state.

        The GC identity per bin is joined on the *unaligned* integer coordinate,
        which is the only key shared by the curation table and the h5ad.
        """
        import h5py
        import pandas as pd

        fam = pd.read_csv(P.GC_FAMILY_CSV)
        fam = fam[fam.rep == self.rep]
        slices, seq = [], {}
        for name in P.SLICES[self.rep]:
            reg = pd.read_parquet(P.GC_REGIONS_DIR / name / f"{name}_gc_regions.parquet",
                                  columns=["x", "y", "gc_region_new"])
            lut = fam[fam["slice"] == name].set_index("new_gc_id")["merged_gc_id"]
            merged = np.where(reg.gc_region_new.to_numpy() > 0,
                              reg.gc_region_new.map(lut).fillna(-1).astype(int).to_numpy(), 0)
            rx = reg.x.to_numpy(np.int64)
            ry = reg.y.to_numpy(np.int64)
            key = pd.Series(merged, index=(rx << 32) | ry)
            key = key[~key.index.duplicated()]

            h = h5py.File(P.SPATEO_ITER_DIR[self.rep] /
                          f"{name}_gc_bin20_aligned_all_stages.h5ad", "r")
            start = np.asarray(h["obsm"]["spatial"]).astype(np.int64)
            final = np.asarray(h["obsm"]["align_deformable"]).astype(np.float32)
            keys = {}
            for group, coord in STAGES:
                path = f"uns/{group}/{coord}"
                if path in h:
                    ks = sorted(h[path], key=int)
                    keys[group] = ks
                    seq.setdefault(group, ks)
                    assert seq[group] == ks, "stored iteration keys differ between slices"

            gid = key.reindex((start[:, 0] << 32) | start[:, 1]).fillna(0).astype(int).to_numpy()
            gmask = gid != 0
            # the reference slice has no stored iterations; it never moves
            match = [n for n in self.st_names if name.endswith(n)]
            assert len(match) == 1, f"cannot map {name} onto {self.st_names}"
            slices.append(dict(name=name, short=match[0], h=h, keys=keys,
                               moving=bool(keys), final=final,
                               start=start.astype(np.float32), gid=gid,
                               tsel=np.flatnonzero(~gmask)[::tissue_stride],
                               gsel=np.flatnonzero(gmask)[::gc_stride]))
            print(f"  {name}: {len(start):,} bins, GC {int(gmask.sum()):,}"
                  + (f", {sum(len(v) for v in keys.values())} stored iterations"
                     if keys else ", reference"), flush=True)
        return slices, seq, [g for g, _ in STAGES]


def coords_at(s, stage, key):
    if not s["moving"] or stage == "start":
        return s["start"]
    if stage == "final":
        return s["final"]
    coord = dict(STAGES)[stage]
    return s["h"][f"uns/{stage}/{coord}/{key}"][...]
