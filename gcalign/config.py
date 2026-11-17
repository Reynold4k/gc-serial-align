"""Experiment description: everything the pipeline needs, in one YAML file.

The section list is the heart of it. You write down your sections in the order
you cut them, saying which are immunofluorescence and which are spatial
transcriptomics, and the pipeline derives the rest: which Stereo-seq section
flanks each IF section, the z coordinate of every plane, and the plane order the
viewer uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Section:
    index: int          # cryostat section number, 1-based, in cutting order
    kind: str           # "IF" or "ST"
    name: str           # label shown in the viewer, e.g. "IF3" or "D1"
    source: str         # CZI filename for IF, chip/slice id for ST


@dataclass
class IFImages:
    dir: Path
    um_per_px: float
    channels: dict[int, str]
    gc_channel: str
    tissue_channels: dict[str, float]
    min_gc_px: int = 30
    filename_template: str = "Section{index}_stitched.czi"
    # Multiplier on the Otsu tissue threshold. Leave at 1.0 unless a section has
    # an illumination or staining gradient across it, in which case a global
    # threshold drops the dim side of the section; ~0.6 recovers it.
    tissue_threshold_scale: float = 1.0
    # Which three channels the figures show as red, green and blue. Leave unset
    # and the fluorophore in each channel name decides (594 -> red, 488 -> green,
    # 647 -> blue); a channel can be left out with `null`.
    display_rgb: list[str | None] | None = None
    # "czi" (Zeiss mosaic) or "tiff" (a plain or OME multi-channel TIFF);
    # "auto" goes by the file extension.
    reader: str = "auto"


@dataclass
class STScaffold:
    h5ad: Path
    coord_key: str = "align_deformable"
    slice_col: str = "slice_short"
    gc_id_col: str = "gc_match_curated_final_id"
    cluster_col: str | None = "leiden_merged"
    dnb_um: float = 0.5


@dataclass
class Align:
    grid_um: float = 10.0
    margin_px: int = 45
    coarse_angle_step_deg: int = 5
    n_orientation_candidates: int = 6
    scale_search: tuple[float, float, float] = (0.94, 1.06, 0.02)
    icp_match_radius_um: float = 420.0
    warp_match_radius_um: float = 260.0
    close_match_um: float = 150.0
    tps_smoothing_grid: list[float] = field(
        default_factory=lambda: [1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6])
    # Two neighbouring GCs that touch on one IF section come through the
    # connected-component step as a single object, and then only one of them can
    # be matched. Split such an object using the flanking Stereo-seq regions as
    # markers - they say how many structures are really there and where.
    split_touching_gcs: bool = True
    split_min_frac: float = 0.06      # each claimant must cover this much of the object
    split_min_px: int = 30            # and at least this many pixels


@dataclass
class Columns:
    z_oversample: int = 4
    sigma_xy_px: float = 1.8
    sigma_z_sub: float = 5.0
    target_faces: int = 700
    shell_faces: int = 300
    pad_z: int = 3
    pad_xy: int = 8
    min_planes: int = 2
    min_px_per_plane: int = 8


@dataclass
class Viewer:
    title: str = "GC columns"
    subtitle: str = ""
    slab_half_um: float = 2.0
    st_hex: str = "#FF1493"        # the GC cluster's colour in your own atlas
    if_hex: str = "#0F52BA"        # sapphire
    default_z_exaggeration: int = 5
    filename: str = "gc_columns.html"
    # "aurora": even-lightness identity colours tuned for the black theme.
    # "hsv": the saturated palette of gcalign 0.1 and the four-section atlas.
    palette: str = "aurora"


@dataclass
class Config:
    name: str
    output_dir: Path
    sections: list[Section]
    section_thickness_um: float
    if_images: IFImages
    st_scaffold: STScaffold
    align: Align = field(default_factory=Align)
    columns: Columns = field(default_factory=Columns)
    viewer: Viewer = field(default_factory=Viewer)

    # ---- everything below is derived, never written by hand ----------------
    @property
    def grid_um(self) -> float:
        return self.align.grid_um

    @property
    def if_sections(self) -> list[Section]:
        return [s for s in self.sections if s.kind == "IF"]

    @property
    def st_sections(self) -> list[Section]:
        return [s for s in self.sections if s.kind == "ST"]

    @property
    def st_names(self) -> list[str]:
        return [s.name for s in self.st_sections]

    @property
    def if_names(self) -> list[str]:
        return [s.name for s in self.if_sections]

    @property
    def planes(self) -> list[tuple[str, str, int]]:
        """(name, kind, section index) in true cutting order."""
        return [(s.name, s.kind, s.index) for s in sorted(self.sections, key=lambda s: s.index)]

    @property
    def plane_names(self) -> list[str]:
        return [name for name, _, _ in self.planes]

    @property
    def n_planes(self) -> int:
        return len(self.sections)

    def section(self, name: str) -> Section:
        for item in self.sections:
            if item.name == name:
                return item
        raise KeyError(f"no section named {name!r}; the config lists {self.plane_names}")

    def z_um(self, section_index: int) -> float:
        first = min(s.index for s in self.sections)
        return (section_index - first) * self.section_thickness_um

    def z_of(self, name: str) -> float:
        return self.z_um(self.section(name).index)

    @property
    def z_span_um(self) -> float:
        return self.z_um(max(s.index for s in self.sections))

    def if_path(self, section: Section) -> Path:
        """The image file of one IF section."""
        return self.if_images.dir / self.if_images.filename_template.format(
            index=section.index, source=section.source, name=section.name)

    @property
    def rgb_channels(self) -> list[str | None]:
        """The channels shown as red, green and blue, in that order."""
        names = list(self.if_images.channels.values())
        if self.if_images.display_rgb is not None:
            chosen = list(self.if_images.display_rgb)
            if len(chosen) != 3:
                raise ValueError("if_images.display_rgb takes exactly three entries (red, green, blue)")
            for item in chosen:
                if item is not None and item not in names:
                    raise ValueError(f"display_rgb names {item!r}, which is not one of the channels {names}")
            return chosen
        return default_rgb(names)

    @property
    def neighbours(self) -> dict[int, list[str]]:
        """For each IF section, the Stereo-seq sections immediately either side.

        An IF section between two ST sections gets both, and the disagreement
        between the two independent answers becomes its error bar. The first and
        last sections only have one neighbour, so they are flagged as such.
        """
        ordered = sorted(self.sections, key=lambda s: s.index)
        out: dict[int, list[str]] = {}
        for position, section in enumerate(ordered):
            if section.kind != "IF":
                continue
            flanking = []
            for step in (-1, 1):
                cursor = position + step
                while 0 <= cursor < len(ordered):
                    if ordered[cursor].kind == "ST":
                        flanking.append(ordered[cursor].name)
                        break
                    cursor += step
            out[section.index] = flanking
        return out

    def one_sided(self, section_index: int) -> bool:
        return len(self.neighbours.get(section_index, [])) < 2


def _as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


# Fluorophores by where they are conventionally displayed. Anything not listed
# is placed in whichever of red / green / blue is still free, in channel order.
# Wavelengths match anywhere in a word ("AF594"); dye names must be the word.
_RGB_HINTS = (
    (0, ("594", "568", "555", "546"), ("cy3", "tritc", "texasred", "mcherry", "rfp", "pe")),
    (1, ("488", "514"), ("fitc", "gfp", "cy2")),
    (2, ("647", "633", "635", "640", "660", "680"), ("cy5", "apc")),
)
_NUCLEAR = ("dapi", "hoechst", "draq5", "draq7")


def default_rgb(names: list[str]) -> list[str | None]:
    """Red, green and blue from the channel names; nuclear counterstains stay out."""
    def words(name: str) -> list[str]:
        return name.lower().replace("-", "_").replace(" ", "_").split("_")

    out: list[str | None] = [None, None, None]
    pool = [n for n in names if not any(w in _NUCLEAR for w in words(n))]
    for name in list(pool):
        for slot, wavelengths, dyes in _RGB_HINTS:
            hit = any(tag in word for word in words(name) for tag in wavelengths) \
                or any(word == tag for word in words(name) for tag in dyes)
            if hit and out[slot] is None:
                out[slot] = name
                pool.remove(name)
                break
    for name in pool:
        if None not in out:
            break
        out[out.index(None)] = name
    return out


def _block(klass, raw: dict, where: str):
    """Build one optional config block, naming the key if one is misspelt."""
    known = {f.name for f in klass.__dataclass_fields__.values()}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} under `{where}`; "
                         f"it understands {sorted(known)}")
    return klass(**raw)


def load(path: str | Path, output_dir: str | Path | None = None) -> Config:
    """Read an experiment description.

    `output_dir` overrides the one written in the file, which is how a finished
    run that has been moved or copied is opened without editing its config.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "output_dir", "sections", "section_thickness_um", "if_images", "st_scaffold"):
        if key not in raw:
            raise ValueError(f"{path}: the config has no `{key}`")
    sections = []
    for item in raw["sections"]:
        kind = item["kind"].upper()
        name = item.get("name") or (f"IF{item['index']}" if kind == "IF" else str(item["source"]))
        sections.append(Section(index=int(item["index"]), kind=kind, name=name,
                                source=str(item.get("source", name))))
    if not any(s.kind == "IF" for s in sections):
        raise ValueError("config lists no IF sections")
    if not any(s.kind == "ST" for s in sections):
        raise ValueError("config lists no ST sections")
    if len({s.index for s in sections}) != len(sections):
        raise ValueError("two sections share the same index")
    if len({s.name for s in sections}) != len(sections):
        raise ValueError("two sections share the same name")

    images = raw["if_images"]
    scaffold = raw["st_scaffold"]
    cfg = Config(
        name=raw["name"],
        output_dir=_as_path(raw["output_dir"]),
        sections=sections,
        section_thickness_um=float(raw["section_thickness_um"]),
        if_images=IFImages(
            dir=_as_path(images["dir"]),
            um_per_px=float(images["um_per_px"]),
            channels={int(k): str(v) for k, v in images["channels"].items()},
            gc_channel=str(images["gc_channel"]),
            tissue_channels={str(k): float(v) for k, v in images["tissue_channels"].items()},
            min_gc_px=int(images.get("min_gc_px", 30)),
            tissue_threshold_scale=float(images.get("tissue_threshold_scale", 1.0)),
            filename_template=str(images.get("filename_template", "Section{index}_stitched.czi")),
            display_rgb=(None if images.get("display_rgb") is None
                         else [None if v is None else str(v) for v in images["display_rgb"]]),
            reader=str(images.get("reader", "auto")).lower(),
        ),
        st_scaffold=STScaffold(
            h5ad=_as_path(scaffold["h5ad"]),
            coord_key=str(scaffold.get("coord_key", "align_deformable")),
            slice_col=str(scaffold.get("slice_col", "slice_short")),
            gc_id_col=str(scaffold.get("gc_id_col", "gc_match_curated_final_id")),
            cluster_col=scaffold.get("cluster_col", "leiden_merged"),
            dnb_um=float(scaffold.get("dnb_um", 0.5)),
        ),
    )
    for block, klass in (("align", Align), ("columns", Columns), ("viewer", Viewer)):
        if block in raw and raw[block]:
            setattr(cfg, block, _block(klass, raw[block], block))
    if output_dir is not None:
        cfg.output_dir = _as_path(output_dir)
    channels = list(cfg.if_images.channels.values())
    if cfg.if_images.gc_channel not in channels:
        raise ValueError(f"gc_channel {cfg.if_images.gc_channel!r} is not one of the channels {channels}")
    stray = sorted(set(cfg.if_images.tissue_channels) - set(channels))
    if stray:
        raise ValueError(f"tissue_channels names {stray}, which are not among the channels {channels}")
    if cfg.if_images.reader not in ("auto", "czi", "tiff"):
        raise ValueError("if_images.reader must be auto, czi or tiff")
    from . import theme
    if cfg.viewer.palette not in theme.PALETTES:
        raise ValueError(f"viewer.palette must be one of {theme.PALETTES}")
    cfg.rgb_channels                      # validates display_rgb
    return cfg


def stage_dir(cfg: Config, name: str) -> Path:
    path = cfg.output_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path
