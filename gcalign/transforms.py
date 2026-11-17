"""Move a point between an IF section and the common frame.

The pipeline registers every IF section in two steps - a similarity transform
(stage A: mirror, rotation, scale, shift) and a thin-plate spline anchored on
matched germinal centres (stage B) - and writes its rasters through them. This
module gives the same mapping for arbitrary points, so anything measured on the
IF image (a cell segmentation, an annotation, a region of interest) can be put
into the spatial-transcriptomics frame, and any transcriptomics bin can be
looked up in the IF image.

    transforms = Transforms.load(cfg)
    t = transforms["IF3"]
    frame_um = t.image_px_to_frame(cells_xy_px)     # IF pixels -> frame micrometres
    image_px = t.frame_to_image_px(bins_xy_um)      # frame micrometres -> IF pixels

Three coordinate systems are involved:

  image pixels   (x, y) in the original IF image, pixel centres at integers
  local um       the section on the common grid before registration: raster
                 pixel index x the grid pitch (what stage A takes as input)
  frame um       the registered frame the transcriptomics coordinates live in
                 (divide by `dnb_um` for the units of the AnnData coordinates)

`from_frame` is exactly the map the rasters were warped with; `to_frame` is its
numerical inverse, so the two round-trip and a point lands where the warped
image shows it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RBFInterpolator

from .config import Config

TRANSFORMS_FILE = "03_gc_match/section_transforms.json"


def _tps(src: np.ndarray, dst: np.ndarray, smoothing: float) -> RBFInterpolator:
    return RBFInterpolator(src, dst, kernel="thin_plate_spline", degree=1, smoothing=smoothing)


@dataclass
class SectionTransform:
    name: str
    linear: np.ndarray                    # stage A, local um -> frame um
    offset: np.ndarray
    smoothing: float                      # stage B stiffness chosen by cross-validation
    source_um: np.ndarray                 # anchors after stage A
    target_um: np.ndarray                 # where the transcriptomics says they belong
    local_um_per_px: float
    z_um: float = 0.0
    neighbours: list[str] = field(default_factory=list)
    one_sided: bool = False
    image: dict | None = None             # um_per_px, raster_shape, raster_um_per_px
    _pull: RBFInterpolator | None = field(default=None, repr=False)
    _push: RBFInterpolator | None = field(default=None, repr=False)

    # ---- the two spline fields, built on first use ------------------------
    @property
    def pull(self) -> RBFInterpolator:
        """Frame -> stage-A position: the field the rasters were sampled through."""
        if self._pull is None:
            self._pull = _tps(self.target_um, self.source_um, self.smoothing)
        return self._pull

    @property
    def push(self) -> RBFInterpolator:
        """Stage-A position -> frame, as fitted for the QC residuals."""
        if self._push is None:
            self._push = _tps(self.source_um, self.target_um, self.smoothing)
        return self._push

    # ---- um <-> um ---------------------------------------------------------
    def from_frame(self, xy_frame_um, warp: bool = True) -> np.ndarray:
        """Frame micrometres -> local micrometres of this section."""
        xy = np.atleast_2d(np.asarray(xy_frame_um, float))
        pre = _chunked(self.pull, xy) if warp else xy
        return (pre - self.offset) @ np.linalg.inv(self.linear).T

    def to_frame(self, xy_local_um, warp: bool = True, tolerance_um: float = 1e-6,
                 max_iterations: int = 30) -> np.ndarray:
        """Local micrometres of this section -> frame micrometres.

        Stage A is applied directly. Stage B is the inverse of the pull field,
        found by Newton iteration from the fitted push field; it converges in a
        few steps because the field is smooth and close to the identity.
        """
        xy = np.atleast_2d(np.asarray(xy_local_um, float))
        placed = xy @ self.linear.T + self.offset
        if not warp:
            return placed
        frame = _chunked(self.push, placed)
        step = 1.0                                            # um, for the Jacobian
        for _ in range(max_iterations):
            here = _chunked(self.pull, frame)
            residual = placed - here
            if np.abs(residual).max() < tolerance_um:
                break
            dx = (_chunked(self.pull, frame + [step, 0.0]) - here) / step
            dy = (_chunked(self.pull, frame + [0.0, step]) - here) / step
            jacobian = np.stack([dx, dy], axis=2)             # (n, 2, 2), columns d/dx, d/dy
            frame = frame + np.linalg.solve(jacobian, residual[..., None])[..., 0]
        else:
            worst = float(np.abs(placed - _chunked(self.pull, frame)).max())
            if worst > 1e-3:
                raise RuntimeError(f"{self.name}: inverting the warp did not converge "
                                   f"(worst residual {worst:.3g} um)")
        return frame

    # ---- pixels ------------------------------------------------------------
    def raster_px_to_local(self, xy_px) -> np.ndarray:
        """(col, row) on the section's grid raster -> local micrometres."""
        return np.atleast_2d(np.asarray(xy_px, float)) * self.local_um_per_px

    def local_to_raster_px(self, xy_local_um) -> np.ndarray:
        return np.atleast_2d(np.asarray(xy_local_um, float)) / self.local_um_per_px

    def _image_ratio(self) -> np.ndarray:
        if not self.image:
            raise ValueError(f"{self.name}: this run did not record the image geometry "
                             "(01_if_gc/if_gc_manifest.json is missing)")
        return float(self.image["um_per_px"]) / np.asarray(self.image["raster_um_per_px"], float)

    def image_px_to_local(self, xy_px) -> np.ndarray:
        """(x, y) in the original image -> local micrometres.

        Pixel centres sit at integer coordinates in both the image and the
        raster, so half a pixel is carried across the change of resolution.
        """
        ratio = self._image_ratio()
        raster = (np.atleast_2d(np.asarray(xy_px, float)) + 0.5) * ratio - 0.5
        return raster * self.local_um_per_px

    def local_to_image_px(self, xy_local_um) -> np.ndarray:
        ratio = self._image_ratio()
        raster = np.atleast_2d(np.asarray(xy_local_um, float)) / self.local_um_per_px
        return (raster + 0.5) / ratio - 0.5

    def image_px_to_frame(self, xy_px, warp: bool = True) -> np.ndarray:
        return self.to_frame(self.image_px_to_local(xy_px), warp=warp)

    def frame_to_image_px(self, xy_frame_um, warp: bool = True) -> np.ndarray:
        return self.local_to_image_px(self.from_frame(xy_frame_um, warp=warp))

    # ---- a measure of the warp itself --------------------------------------
    def displacement_um(self, xy_frame_um) -> np.ndarray:
        """How far stage B moves the image at each frame position."""
        xy = np.atleast_2d(np.asarray(xy_frame_um, float))
        return np.linalg.norm(_chunked(self.pull, xy) - xy, axis=1)


def _chunked(model: RBFInterpolator, xy: np.ndarray, size: int = 200_000) -> np.ndarray:
    """Evaluate a spline field over many points without one huge kernel matrix."""
    if len(xy) <= size:
        return model(xy)
    return np.concatenate([model(xy[i:i + size]) for i in range(0, len(xy), size)])


class Transforms:
    """Every IF section's transform for one run."""

    def __init__(self, sections: dict[str, SectionTransform], frame: dict, dnb_um: float,
                 origin: str):
        self.sections = sections
        self.frame = frame                # x0_um, y0_um, um_per_px, height, width
        self.dnb_um = dnb_um
        self.origin = origin              # "stored" or "rebuilt"

    def __getitem__(self, name: str) -> SectionTransform:
        try:
            return self.sections[name]
        except KeyError:
            raise KeyError(f"no IF section named {name!r}; this run has {list(self.sections)}") from None

    def __iter__(self):
        return iter(self.sections.values())

    @property
    def names(self) -> list[str]:
        return list(self.sections)

    def frame_um_to_st_units(self, xy_frame_um) -> np.ndarray:
        """Frame micrometres -> the units of the registered AnnData coordinates."""
        return np.asarray(xy_frame_um, float) / self.dnb_um

    def st_units_to_frame_um(self, xy_units) -> np.ndarray:
        return np.asarray(xy_units, float) * self.dnb_um

    @classmethod
    def load(cls, cfg: Config) -> "Transforms":
        """The transforms of a finished run: stored (0.2+) or rebuilt and verified."""
        path = cfg.output_dir / TRANSFORMS_FILE
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            origin = "stored"
        else:
            document = rebuild(cfg)
            origin = "rebuilt"
        sections = {}
        for name, body in document["sections"].items():
            sections[name] = SectionTransform(
                name=name,
                linear=np.asarray(body["linear"], float),
                offset=np.asarray(body["offset"], float),
                smoothing=float(body["tps"]["smoothing"]),
                source_um=np.asarray(body["tps"]["source_um"], float).reshape(-1, 2),
                target_um=np.asarray(body["tps"]["target_um"], float).reshape(-1, 2),
                local_um_per_px=float(document["local_um_per_px"]),
                z_um=float(body.get("z_um", 0.0)),
                neighbours=list(body.get("neighbours", [])),
                one_sided=bool(body.get("one_sided", False)),
                image=body.get("image"),
            )
        return cls(sections, document["frame"], cfg.st_scaffold.dnb_um, origin)


def rebuild(cfg: Config) -> dict:
    """Rebuild the transforms of a run that did not store them (gcalign 0.1).

    The anchors are recomputed from the run's own stage outputs with the
    pipeline's own matching code, and the stiffness is the one the run chose and
    recorded. The result is accepted only if it reproduces the anchor count and
    the fitted residual the run reported.
    """
    from . import match_warp

    root = cfg.output_dir
    globals_json = json.loads((root / "02_global_align/global_transforms.json").read_text())
    report = json.loads((root / "03_gc_match/match_report.json").read_text())
    grid_meta = globals_json["grid"]
    grid = (grid_meta["x0_um"], grid_meta["y0_um"], grid_meta["height"], grid_meta["width"])
    stored = np.load(root / "03_gc_match/plane_gc_labels.npz")
    st_labels = {name: stored[name] for name in cfg.st_names}
    centroids = match_warp.per_slice_centroids(cfg)
    sections = {}
    for name in cfg.if_names:
        entry = globals_json["sections"][name]
        found = match_warp.section_anchors(cfg, name, entry, st_labels, centroids, grid,
                                           verbose=False)
        src, dst = found["src"], found["dst"]
        smoothing = float(report[name]["tps_smoothing"])
        residual = float(np.median(np.linalg.norm(_tps(src, dst, smoothing)(src) - dst, axis=1)))
        if (len(src) != report[name]["n_anchor_pairs"]
                or abs(residual - report[name]["tps_residual_median_um"]) > 1e-6):
            raise RuntimeError(
                f"{name}: the rebuilt warp does not reproduce this run's report "
                f"({len(src)} anchors, residual {residual:.6f} um, against "
                f"{report[name]['n_anchor_pairs']} and {report[name]['tps_residual_median_um']:.6f}); "
                "re-run stage `match` to store the transforms")
        sections[name] = {"linear": entry["linear"], "offset": entry["offset"],
                          "tps": {"smoothing": smoothing,
                                  "source_um": src.tolist(), "target_um": dst.tolist()}}
    return match_warp.transforms_document(cfg, grid_meta, sections)
