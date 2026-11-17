"""Read a finished run.

Everything that draws or exports - the figures, the report, the film, the
transfer of IF signal onto transcriptomics bins - goes through `Run`, so none of
them has to know which stage wrote which file, and all of them keep working on
a run made with an older version of the package.

    run = Run.open("my_experiment.yaml")
    run.names                    # ['IF1', 'D1', 'IF3', ...] in cutting order
    run.labels("IF3")            # GC id raster on the common grid
    run.rgb("IF3")               # the stain as a displayable composite
    run.icp("IF3")["steps"]      # the registration as it actually proceeded

All rasters share one grid: `run.grid` gives its origin, shape and pixel size in
micrometres, and `run.grid.extent()` is what `imshow` wants.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from . import replay, theme
from .config import Config
from .config import load as load_config

PRESENCE_FILES = ("03_gc_match/gc_presence.csv", "03_gc_match/gc_presence_9plane.csv")


@dataclass(frozen=True)
class Plane:
    name: str           # "IF3", "D1"
    kind: str           # "IF" or "ST"
    index: int          # cryostat section number
    z_um: float         # height above the first section


@dataclass(frozen=True)
class Grid:
    """The common raster every plane lives on. Row 0 is the smallest y."""
    x0: float
    y0: float
    height: int
    width: int
    um: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def full(self) -> tuple[int, int, int, int]:
        """The whole grid as a window (r0, r1, c0, c1)."""
        return (0, self.height, 0, self.width)

    def extent(self, window: tuple[int, int, int, int] | None = None) -> list[float]:
        """[left, right, bottom, top] in micrometres for imshow(origin="upper")."""
        r0, r1, c0, c1 = window or self.full
        return [self.x0 + c0 * self.um, self.x0 + c1 * self.um,
                self.y0 + r1 * self.um, self.y0 + r0 * self.um]

    def to_rc(self, xy_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World micrometres -> nearest (row, col)."""
        xy = np.asarray(xy_um, float)
        return (np.rint((xy[..., 1] - self.y0) / self.um).astype(int),
                np.rint((xy[..., 0] - self.x0) / self.um).astype(int))

    def to_xy(self, rows, cols) -> np.ndarray:
        """(row, col) -> world micrometres."""
        return np.stack([self.x0 + np.asarray(cols, float) * self.um,
                         self.y0 + np.asarray(rows, float) * self.um], axis=-1)

    def inside(self, rows, cols) -> np.ndarray:
        return (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)


class Run:
    """A finished pipeline run, opened for reading."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = Path(cfg.output_dir)
        if not (self.root / "03_gc_match/plane_gc_labels.npz").exists():
            raise FileNotFoundError(
                f"{self.root} holds no finished run (03_gc_match/plane_gc_labels.npz is missing). "
                "Run `gcalign run` first, or point --output-dir at the run.")
        self._npz: dict[str, object] = {}

    @classmethod
    def open(cls, config: str | Path, output_dir: str | Path | None = None) -> "Run":
        return cls(load_config(config, output_dir=output_dir))

    # ---- files ------------------------------------------------------------
    def has(self, relative: str) -> bool:
        return (self.root / relative).exists()

    def _load(self, relative: str):
        if relative not in self._npz:
            self._npz[relative] = np.load(self.root / relative, allow_pickle=False)
        return self._npz[relative]

    def _json(self, relative: str):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    # ---- layout -----------------------------------------------------------
    @cached_property
    def planes(self) -> list[Plane]:
        return [Plane(name, kind, index, self.cfg.z_um(index))
                for name, kind, index in self.cfg.planes]

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.planes]

    @property
    def if_names(self) -> list[str]:
        return [p.name for p in self.planes if p.kind == "IF"]

    @property
    def st_names(self) -> list[str]:
        return [p.name for p in self.planes if p.kind == "ST"]

    def plane(self, name: str) -> Plane:
        for item in self.planes:
            if item.name == name:
                return item
        raise KeyError(f"no plane named {name!r}; this run has {self.names}")

    @cached_property
    def grid(self) -> Grid:
        x0, y0, height, width = (float(v) for v in self._load("03_gc_match/plane_gc_labels.npz")["grid"])
        # the pixel size the run was made with, not the one in a config edited since
        um = float(self.cfg.grid_um)
        if self.has("02_global_align/global_transforms.json"):
            um = float(self._json("02_global_align/global_transforms.json")["grid"]["um_per_px"])
        return Grid(x0, y0, int(height), int(width), um)

    # ---- rasters ----------------------------------------------------------
    def labels(self, name: str) -> np.ndarray:
        """Germinal-centre id per pixel: > 0 a curated id, < 0 an IF-only object."""
        return self._load("03_gc_match/plane_gc_labels.npz")[name]

    @cached_property
    def gc_ids(self) -> list[int]:
        found: set[int] = set()
        for name in self.names:
            values = np.unique(self.labels(name))
            found.update(int(v) for v in values if v > 0)
        return sorted(found)

    @cached_property
    def channel_names(self) -> list[str]:
        return list(self.cfg.if_images.channels.values())

    def channel(self, name: str, channel: str) -> np.ndarray:
        """One fluorescence channel of an IF plane, warped onto the grid (uint16)."""
        return self._load("03_gc_match/if_channels_warped.npz")[f"{name}|{channel}"]

    def channels(self, name: str) -> dict[str, np.ndarray]:
        store = self._load("03_gc_match/if_channels_warped.npz")
        return {c: store[f"{name}|{c}"] for c in self.channel_names if f"{name}|{c}" in store}

    def tissue(self, name: str) -> np.ndarray:
        """The section outline on the grid.

        Spatial-transcriptomics planes: the rasterised bins. IF planes: the
        tissue mask carried through the same warp as the image; a run made
        before gcalign 0.2 did not store one, so it is cut again from the warped
        channels with the segmentation's own recipe.
        """
        if self.plane(name).kind == "ST":
            return np.asarray(self._load("02_global_align/st_rasters.npz")[f"tissue_{name}"], bool)
        stored = "03_gc_match/if_tissue_warped.npz"
        if self.has(stored) and name in self._load(stored):
            return np.asarray(self._load(stored)[name], bool)
        key = f"_tissue_{name}"
        if key not in self._npz:
            self._npz[key] = _recut_tissue(self.cfg, self.channels(name))
        return self._npz[key]

    def tissue_is_stored(self, name: str) -> bool:
        """False when `tissue` had to re-cut an IF outline (run older than 0.2)."""
        if self.plane(name).kind == "ST":
            return True
        stored = "03_gc_match/if_tissue_warped.npz"
        return self.has(stored) and name in self._load(stored)

    # ---- the stain as a picture ---------------------------------------------
    def rgb_limits(self, window: tuple[int, int, int, int] | None = None,
                   low: float = 2.0, high: float = 99.6) -> dict[str, tuple[float, float]]:
        """Display limits per channel, shared by every IF plane.

        Sharing them is what makes brightness comparable from one section to the
        next. Give a window to set the limits for a close-up from the close-up.
        """
        r0, r1, c0, c1 = window or self.grid.full
        limits = {}
        for channel in self.cfg.rgb_channels:
            if channel is None:
                continue
            pool = []
            for name in self.if_names:
                piece = self.channel(name, channel)[r0:r1, c0:c1]
                pool.append(piece[piece > 0] if window is None else piece.ravel())
            pool = np.concatenate(pool) if pool else np.zeros(1)
            if pool.size == 0:
                pool = np.zeros(1)
            lo, hi = np.percentile(pool, [low, high])
            limits[channel] = (float(lo), float(max(hi, lo + 1e-6)))
        return limits

    def rgb(self, name: str, window: tuple[int, int, int, int] | None = None,
            limits: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
        """An IF plane as a float RGB image in 0..1 (red, green, blue per the config)."""
        r0, r1, c0, c1 = window or self.grid.full
        limits = limits or self.rgb_limits(window)
        out = np.zeros((r1 - r0, c1 - c0, 3), np.float32)
        for band, channel in enumerate(self.cfg.rgb_channels):
            if channel is None:
                continue
            piece = self.channel(name, channel)[r0:r1, c0:c1].astype(np.float32)
            lo, hi = limits[channel]
            out[..., band] = np.clip((piece - lo) / max(hi - lo, 1e-6), 0, 1)
        return out

    def local_rgb(self, name: str,
                  limits: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
        """An IF section before registration, as a float RGB image in 0..1.

        The image is in the section's own orientation (`run.local(name)`). Each
        channel is rescaled exactly as the registered channels are, and shown
        with the same display limits as `run.rgb` (all IF sections share them),
        so the section looks the same before and after registration.
        """
        from .match_warp import to_display_units

        local = self.local(name)
        limits = limits or self.rgb_limits()
        height, width = local["tissue"].shape
        out = np.zeros((height, width, 3), np.float32)
        for band, channel in enumerate(self.cfg.rgb_channels):
            if channel is None or channel not in local.files:
                continue
            scaled = to_display_units(local[channel])
            lo, hi = limits[channel]
            out[..., band] = np.clip((scaled - lo) / max(hi - lo, 1e-6), 0, 1)
        return out

    # ---- reports and tables -----------------------------------------------
    @cached_property
    def transforms(self) -> dict:
        """Stage-A placement and match quality per IF plane."""
        return self._json("02_global_align/global_transforms.json")["sections"]

    @cached_property
    def match(self) -> dict:
        """Stage-B warp report per IF plane."""
        report = self._json("03_gc_match/match_report.json")
        return {k: v for k, v in report.items() if k in self.if_names}

    @cached_property
    def manifest(self) -> dict:
        """IF segmentation summary per IF plane."""
        entries = self._json("01_if_gc/if_gc_manifest.json")
        by_index = {s.index: s.name for s in self.cfg.if_sections}
        return {item.get("name") or by_index[item["section"]]: item for item in entries}

    @cached_property
    def presence(self):
        """Which germinal centre appears on which plane, and how large."""
        import pandas as pd
        for relative in PRESENCE_FILES:
            if self.has(relative):
                return pd.read_csv(self.root / relative)
        raise FileNotFoundError(f"no presence table under {self.root / '03_gc_match'}")

    @cached_property
    def metrics(self):
        """Per-GC column metrics (volume, radius, drift), or None before stage `columns`."""
        import pandas as pd
        path = self.root / "04_columns/gc_column_metrics.csv"
        return pd.read_csv(path) if path.exists() else None

    @cached_property
    def validation(self) -> dict | None:
        path = self.root / "validation_report.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @cached_property
    def orientation_search(self) -> dict | None:
        """Every orientation candidate with its tissue and GC score (0.2+ runs)."""
        relative = "02_global_align/orientation_search.json"
        return self._json(relative) if self.has(relative) else None

    def icp(self, name: str) -> dict:
        """The ICP iterations of one IF plane, as transforms (see `gcalign.replay`)."""
        key = f"_icp_{name}"
        if key not in self._npz:
            self._npz[key] = replay.icp_trajectory(self.cfg, name)
        return self._npz[key]

    def local(self, name: str):
        """An IF section still in its own frame: tissue, gc, gc_labels and channels."""
        return self._load(f"01_if_gc/{name}_gc.npz")

    @cached_property
    def scaffold(self):
        """The frozen spatial-transcriptomics bins: slice_short, gc_id, world_um, chip_um."""
        return self._load("00_scaffold/st_scaffold.npz")

    # ---- 3D surfaces ------------------------------------------------------
    @cached_property
    def column_ids(self) -> list[int]:
        """Ids of the germinal centres that have a 3D surface (stage `columns`)."""
        if not self.has("04_columns/gc_columns.npz"):
            return []
        return [int(v) for v in self._load("04_columns/gc_columns.npz")["gc_ids"]]

    def column(self, gc_id: int, shell: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """The 3D surface of one germinal centre: vertices (x, y, z in um) and triangles.

        `shell=True` returns the same surface with fewer triangles.
        """
        store = self._load("04_columns/gc_columns.npz")
        prefix = ("sv", "sf") if shell else ("v", "f")
        if f"{prefix[0]}{gc_id}" not in store:
            raise KeyError(f"GC{gc_id} has no 3D surface; ids with one: {self.column_ids}")
        return store[f"{prefix[0]}{gc_id}"], store[f"{prefix[1]}{gc_id}"]

    def outline_polygons(self, gc_id: int, name: str) -> list[dict]:
        """The flat outline(s) of one germinal centre on one plane.

        Each entry has `xy_um` (polygon vertices), `faces` (triangles of the
        filled polygon) and `z_um` (depth of the plane). Empty if the germinal
        centre is absent from the plane.
        """
        if not self.has("04_columns/gc_plane_patches.npz"):
            return []
        store = self._load("04_columns/gc_plane_patches.npz")
        out, index = [], 0
        while f"{gc_id}|{name}|{index}|xy" in store:
            key = f"{gc_id}|{name}|{index}"
            out.append({"xy_um": store[f"{key}|xy"], "faces": store[f"{key}|f"],
                        "z_um": float(store[f"{key}|z"][0])})
            index += 1
        return out

    # ---- colour -----------------------------------------------------------
    @cached_property
    def identity(self) -> dict[int, str]:
        """One colour per germinal centre, shared by every view of this run."""
        return theme.identity_colours(self.gc_ids, self.cfg.viewer.palette)

    # ---- geometry helpers ---------------------------------------------------
    def footprint(self, gc_id: int) -> np.ndarray:
        """Union of one germinal centre's masks over every plane."""
        return np.logical_or.reduce([self.labels(n) == gc_id for n in self.names])

    def square_window(self, mask: np.ndarray, pad_fraction: float = 0.35,
                      min_pad_px: int = 14) -> tuple[int, int, int, int]:
        """A square window (r0, r1, c0, c1) around a mask, clipped to the grid."""
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            raise ValueError("empty mask")
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        spread = max(np.ptp(ys), np.ptp(xs))
        half = spread / 2 + max(min_pad_px, pad_fraction * spread)
        r0, r1 = int(max(cy - half, 0)), int(min(cy + half, self.grid.height))
        c0, c1 = int(max(cx - half, 0)), int(min(cx + half, self.grid.width))
        side = min(r1 - r0, c1 - c0)
        return r0, r0 + side, c0, c0 + side


def _recut_tissue(cfg: Config, channels: dict[str, np.ndarray]) -> np.ndarray:
    """Cut a section outline out of registered channels (runs older than 0.2 only).

    Uses the tissue recipe of stage `if` (`if_sections.tissue_mask`), including
    `if_images.tissue_threshold_scale`, on the registered channels.
    """
    from .if_sections import tissue_mask

    planes = {name: np.asarray(plane, np.float32) for name, plane in channels.items()}
    usable = {name: weight for name, weight in cfg.if_images.tissue_channels.items()
              if name in planes}
    if not usable:
        raise ValueError("none of the tissue_channels is among the registered channels")
    return np.asarray(tissue_mask(cfg, planes, sigma=6.0), bool)
