"""A made-up experiment with a known answer, for testing only.

The pipeline is tested on data where the truth is known exactly: a tissue
outline and a scatter of blob-like structures are defined analytically in a
world frame, "transcriptomics" sections sample them on a bin lattice, and each
"IF" section is rendered through a known mirror + rotation + scale + shift and a
smooth elastic distortion into a multi-channel TIFF. Running the pipeline on
this must give the mirror state, the rotation and the point mapping back.

None of this is used in any figure or document; it exists so that a change to
the registration can be checked against ground truth in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import tifffile
import yaml

DNB_UM = 0.5
BIN_UM = 10.0
IMAGE_UM = 2.0
CHANNELS = {0: "RING_AF647", 1: "BODY_AF594", 2: "BLOB_AF488", 3: "DAPI"}


@dataclass
class Scene:
    """The world: one tissue outline and structures that change slowly with z."""
    centre: np.ndarray
    radius: float
    harmonics: np.ndarray                 # (k, 2): amplitude, phase of the outline wobble
    blob_xy: np.ndarray                   # (n, 2) centres at z = 0
    blob_r: np.ndarray                    # (n,) radii at z = 0
    blob_drift: np.ndarray                # (n, 2) um of centre shift per um of z
    blob_growth: np.ndarray               # (n,) fractional radius change per um of z
    blob_aspect: np.ndarray               # (n,) axis ratio
    blob_angle: np.ndarray                # (n,) orientation, radians

    def tissue(self, xy: np.ndarray) -> np.ndarray:
        d = xy - self.centre
        phi = np.arctan2(d[..., 1], d[..., 0])
        edge = self.radius * (1 + sum(a * np.cos((k + 2) * phi + p)
                                      for k, (a, p) in enumerate(self.harmonics)))
        return np.hypot(d[..., 0], d[..., 1]) < edge

    def blob_centres(self, z: float) -> np.ndarray:
        return self.blob_xy + self.blob_drift * z

    def blob_radii(self, z: float) -> np.ndarray:
        return self.blob_r * (1 + self.blob_growth * z)

    def blob_distance(self, xy: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
        """Per point: normalised distance to the nearest blob (1 = its edge) and its index."""
        centres, radii = self.blob_centres(z), self.blob_radii(z)
        best = np.full(xy.shape[:-1], np.inf)
        owner = np.zeros(xy.shape[:-1], np.int32)
        for index, (c, r, aspect, angle) in enumerate(
                zip(centres, radii, self.blob_aspect, self.blob_angle)):
            d = xy - c
            u = d[..., 0] * np.cos(angle) + d[..., 1] * np.sin(angle)
            v = -d[..., 0] * np.sin(angle) + d[..., 1] * np.cos(angle)
            rho = np.hypot(u / (r * aspect), v / (r / aspect))
            closer = rho < best
            best = np.where(closer, rho, best)
            owner = np.where(closer, index, owner)
        return best, owner


@dataclass
class Truth:
    """How one IF section was put on its slide."""
    name: str
    index: int
    z_um: float
    mirror: bool
    angle_deg: float
    scale: float
    linear: np.ndarray
    offset: np.ndarray
    elastic_um: float
    elastic_period_um: float
    scene_centre: np.ndarray
    image_shape: tuple[int, int]

    def to_frame(self, xy_image_px: np.ndarray) -> np.ndarray:
        """Image pixels -> world micrometres, exactly as the image was rendered."""
        placed = np.atleast_2d(xy_image_px).astype(float) * IMAGE_UM @ self.linear.T + self.offset
        return placed + self.displacement(placed)

    def displacement(self, placed: np.ndarray) -> np.ndarray:
        d = placed - self.scene_centre
        k = 2 * np.pi / self.elastic_period_um
        return self.elastic_um * np.stack([np.sin(k * d[..., 1]), np.sin(k * d[..., 0])], axis=-1)


@dataclass
class Experiment:
    config: Path
    output_dir: Path
    scene: Scene
    truth: dict[str, Truth]
    planes: list[tuple[str, str, int]]
    n_blobs: int
    files: dict[str, Path] = field(default_factory=dict)


def make_scene(rng: np.random.Generator, n_blobs: int = 26) -> Scene:
    centre = np.array([3000.0, 2600.0])
    radius = 1150.0
    harmonics = np.array([[0.07, rng.uniform(0, 6.28)], [0.045, rng.uniform(0, 6.28)],
                          [0.03, rng.uniform(0, 6.28)]])
    blobs: list[np.ndarray] = []
    radii: list[float] = []
    attempts = 0
    while len(blobs) < n_blobs and attempts < 20000:
        attempts += 1
        r = rng.uniform(55, 120)
        p = centre + rng.uniform(-1, 1, 2) * radius
        if np.hypot(*(p - centre)) > radius * 0.80:
            continue
        if any(np.hypot(*(p - q)) < (r + s) * 1.9 + 60 for q, s in zip(blobs, radii)):
            continue
        blobs.append(p)
        radii.append(r)
    n = len(blobs)
    return Scene(
        centre=centre, radius=radius, harmonics=harmonics,
        blob_xy=np.array(blobs), blob_r=np.array(radii),
        blob_drift=rng.uniform(-0.25, 0.25, (n, 2)),
        blob_growth=rng.uniform(-0.002, 0.002, n),
        blob_aspect=rng.uniform(0.85, 1.2, n),
        blob_angle=rng.uniform(0, np.pi, n),
    )


def write_scaffold(path: Path, scene: Scene, st_planes: list[tuple[str, float]],
                   coord_key: str = "aligned") -> None:
    """A minimal AnnData-shaped HDF5 file: what stage `scaffold` reads, nothing else."""
    lo = scene.centre - scene.radius * 1.25
    hi = scene.centre + scene.radius * 1.25
    xs = np.arange(np.floor(lo[0] / BIN_UM), np.ceil(hi[0] / BIN_UM)) * BIN_UM
    ys = np.arange(np.floor(lo[1] / BIN_UM), np.ceil(hi[1] / BIN_UM)) * BIN_UM
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    grid = grid[scene.tissue(grid)]
    coords, section, gc_id, names = [], [], [], []
    for code, (name, z) in enumerate(st_planes):
        rho, owner = scene.blob_distance(grid, z)
        ids = np.where(rho < 1.0, owner + 1, 0).astype(np.int32)
        coords.append(grid / DNB_UM)
        section.append(np.full(len(grid), code, np.int8))
        gc_id.append(ids)
        names.extend(f"{name}:{int(x)}:{int(y)}" for x, y in grid / DNB_UM)
    coords = np.concatenate(coords)
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=np.array(names, dtype="S"))
        group = obs.create_group("slice")
        group.create_dataset("codes", data=np.concatenate(section))
        group.create_dataset("categories", data=np.array([n for n, _ in st_planes], dtype="S"))
        ids = np.concatenate(gc_id)
        obs.create_dataset("gc_id", data=ids)
        cluster = obs.create_group("cluster")
        cluster.create_dataset("codes", data=(ids > 0).astype(np.int8))
        cluster.create_dataset("categories", data=np.array(["other", "GC"], dtype="S"))
        obsm = handle.create_group("obsm")
        obsm.create_dataset("spatial", data=np.rint(coords).astype(np.int32))
        obsm.create_dataset(coord_key, data=coords.astype(np.float32))


def render_if(scene: Scene, truth: Truth, rng: np.random.Generator) -> np.ndarray:
    """One IF section as (channel, row, col) uint16."""
    height, width = truth.image_shape
    rows, cols = np.indices((height, width))
    world = truth.to_frame(np.stack([cols, rows], axis=-1).reshape(-1, 2)).reshape(height, width, 2)
    tissue = scene.tissue(world).astype(np.float32)
    rho, _ = scene.blob_distance(world, truth.z_um)
    blob = (rho < 1.0).astype(np.float32) * tissue
    ring = ((rho >= 1.0) & (rho < 1.7)).astype(np.float32) * tissue

    def noisy(image: np.ndarray, sd: float = 0.06) -> np.ndarray:
        return image * (1 + rng.normal(0, sd, image.shape)) + rng.normal(0, 12, image.shape)

    stack = np.stack([
        noisy(2400 * ring + 320 * tissue),              # a mantle-like ring round each structure
        noisy(1500 * tissue),                           # present across the section
        noisy(3200 * blob + 260 * tissue),              # the structure marker
        noisy(900 * tissue),                            # a nuclear counterstain
    ])
    return np.clip(stack + 40, 0, 65535).astype(np.uint16)


def make_truth(scene: Scene, name: str, index: int, z_um: float, mirror: bool, angle_deg: float,
               scale: float, elastic_um: float) -> Truth:
    side = int(np.ceil(scene.radius * 2 * 1.45 / IMAGE_UM))
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    linear = scale * rot @ (np.diag([-1.0, 1.0]) if mirror else np.eye(2))
    image_centre = np.array([side / 2, side / 2]) * IMAGE_UM
    offset = scene.centre - linear @ image_centre
    return Truth(name, index, z_um, mirror, angle_deg, scale, linear, offset,
                 elastic_um, 2600.0, scene.centre.copy(), (side, side))


def build(root: Path, seed: int = 7, thickness_um: float = 10.0,
          layout: tuple[tuple[int, str], ...] = ((1, "IF"), (2, "ST"), (3, "IF"), (4, "ST"), (5, "IF")),
          placements: tuple[tuple[bool, float, float], ...] = ((False, 28.0, 1.02),
                                                               (True, 205.0, 0.98),
                                                               (False, 311.0, 1.00)),
          elastic_um: float = 22.0) -> Experiment:
    """Write the images, the scaffold and the config; return them with the truth."""
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scene = make_scene(rng)
    first = layout[0][0]
    planes, sections, st_planes, truth = [], [], [], {}
    if_count = 0
    for index, kind in layout:
        z = (index - first) * thickness_um
        if kind == "ST":
            name = f"S{index}"
            st_planes.append((name, z))
            sections.append({"index": index, "kind": "ST", "source": name, "name": name})
        else:
            name = f"IF{index}"
            mirror, angle, scale = placements[if_count % len(placements)]
            if_count += 1
            truth[name] = make_truth(scene, name, index, z, mirror, angle, scale, elastic_um)
            sections.append({"index": index, "kind": "IF", "source": index})
        planes.append((name, kind, index))

    files = {}
    for name, item in truth.items():
        path = root / "images" / f"Section{item.index}.tif"
        tifffile.imwrite(path, render_if(scene, item, rng), photometric="minisblack",
                         metadata={"axes": "CYX"})
        files[name] = path
    scaffold = root / "scaffold.h5ad"
    write_scaffold(scaffold, scene, st_planes)

    config = {
        "name": "synthetic_test",
        "output_dir": str(root / "out"),
        "section_thickness_um": thickness_um,
        "sections": sections,
        "if_images": {
            "dir": str(root / "images"),
            "filename_template": "Section{index}.tif",
            "um_per_px": IMAGE_UM,
            "channels": CHANNELS,
            "gc_channel": "BLOB_AF488",
            "tissue_channels": {"BODY_AF594": 1.0, "RING_AF647": 0.6, "BLOB_AF488": 0.3},
        },
        "st_scaffold": {"h5ad": str(scaffold), "coord_key": "aligned", "slice_col": "slice",
                        "gc_id_col": "gc_id", "cluster_col": "cluster", "dnb_um": DNB_UM},
        "align": {"grid_um": BIN_UM},
        "viewer": {"title": "Synthetic test", "filename": "viewer.html"},
    }
    path = root / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return Experiment(path, root / "out", scene, truth, planes, len(scene.blob_xy), files)
