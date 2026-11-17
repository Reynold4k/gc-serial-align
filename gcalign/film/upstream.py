"""Act 1's input: the transcriptomics registration as it actually proceeded.

gcalign starts from spatial-transcriptomics sections that were already
registered to each other (with Spateo). A finished run keeps only the answer -
``00_scaffold/st_scaffold.npz`` - but the Spateo run that produced it stored
every iteration of its three stages, and act 1 replays them:

    <spateo_dir>/<source>_gc_bin20_aligned_all_stages.h5ad
        obsm/spatial                          the unregistered chip coordinates
        obsm/<coord_key>                      the final registered coordinates
        uns/iter_rigid/align_rigid/<i>        one dataset per stored iteration
        uns/iter_nonrigid/align_nonrigid/<i>
        uns/iter_deformable/align_deformable/<i>

``<source>`` is each ST section's ``source`` in the config (its chip id). A
section without stored iterations is the reference: it never moves.

Nothing else is needed. Which germinal centre a bin belongs to comes from the
run itself: every bin of the frozen scaffold is found in the Spateo file by its
chip coordinate, and the scaffold's own ``gc_id`` colours it. The join is exact
- and checked: the Spateo file's final coordinates must land on the scaffold's
frozen coordinates, or the two do not belong together and act 1 is refused.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PATTERN = "{source}_gc_bin20_aligned_all_stages.h5ad"
STAGE_ORDER = ("rigid", "nonrigid", "deformable")
ENV_DIR = "GCALIGN_SPATEO_DIR"
ENV_PATTERN = "GCALIGN_SPATEO_PATTERN"


@dataclass
class Upstream:
    """Where the Spateo run with its stored iterations lives."""
    spateo_dir: Path
    pattern: str = DEFAULT_PATTERN

    def __post_init__(self):
        self.spateo_dir = Path(self.spateo_dir).expanduser()

    @classmethod
    def from_env(cls, environ=None) -> "Upstream | None":
        """``GCALIGN_SPATEO_DIR`` (and optionally ``GCALIGN_SPATEO_PATTERN``)."""
        environ = os.environ if environ is None else environ
        folder = environ.get(ENV_DIR, "").strip()
        if not folder:
            return None
        return cls(Path(folder), environ.get(ENV_PATTERN, "").strip() or DEFAULT_PATTERN)

    @classmethod
    def coerce(cls, value) -> "Upstream | None":
        if value is None or isinstance(value, Upstream):
            return value
        return cls(Path(value))

    def path(self, section) -> Path:
        return self.spateo_dir / self.pattern.format(source=section.source, name=section.name)


def _stages(handle) -> list[tuple[str, str, list[str]]]:
    """[(stage, dataset group path, iteration keys)] in the order they ran."""
    found = []
    uns = handle.get("uns")
    if uns is None:
        return found
    for group in uns:
        if not group.startswith("iter_"):
            continue
        stage = group[len("iter_"):]
        node = uns[group]
        coord = f"align_{stage}" if f"align_{stage}" in node else None
        if coord is None:          # any numbered group of (n, 2) arrays will do
            for key in node:
                child = node[key]
                if hasattr(child, "keys") and len(child) and all(k.isdigit() for k in child):
                    coord = key
                    break
        if coord is None:
            continue
        keys = sorted(node[coord], key=int)
        found.append((stage, f"uns/{group}/{coord}", keys))
    rank = {s: i for i, s in enumerate(STAGE_ORDER)}
    found.sort(key=lambda item: (rank.get(item[0], len(rank)), item[0]))
    return found


class SpateoReplay:
    """The stored Spateo iterations, joined to the run's frozen scaffold.

    Files are opened lazily and per process, so frames can be rendered in
    forked worker processes (HDF5 handles must not cross a fork)."""

    def __init__(self, run, upstream: Upstream, tissue_stride: int = 5, log=print):
        import h5py

        self.upstream = upstream
        cfg = run.cfg
        dnb = float(cfg.st_scaffold.dnb_um)
        self.dnb_um = dnb
        scaffold = run.scaffold
        slice_short = np.asarray(scaffold["slice_short"])
        gc_id = np.asarray(scaffold["gc_id"])
        chip = np.asarray(scaffold["chip_um"], np.float64)
        world = np.asarray(scaffold["world_um"], np.float64)

        self.slices = []
        sequence = None
        for section in sorted(cfg.st_sections, key=lambda s: s.index):
            path = upstream.path(section)
            if not path.exists():
                raise FileNotFoundError(
                    f"act 1: no Spateo file for {section.name} at {path} "
                    f"(set the directory with --spateo-dir / {ENV_DIR}, the file name "
                    f"pattern with {ENV_PATTERN}; '{{source}}' is the config's source)")
            keep = slice_short == section.name
            rows = np.flatnonzero(keep)
            with h5py.File(path, "r") as handle:
                spatial = np.asarray(handle["obsm"]["spatial"]).astype(np.int64)
                coord_key = cfg.st_scaffold.coord_key
                final = np.asarray(handle["obsm"][coord_key], np.float64) \
                    if coord_key in handle["obsm"] else spatial.astype(np.float64)
                stages = _stages(handle)
            # join scaffold bins to Spateo rows by their chip coordinate
            key_h = (spatial[:, 0] << 32) | spatial[:, 1]
            sc = np.rint(chip[rows] / dnb).astype(np.int64)
            key_s = (sc[:, 0] << 32) | sc[:, 1]
            order = np.argsort(key_h)
            pos = np.searchsorted(key_h[order], key_s)
            pos = np.clip(pos, 0, len(order) - 1)
            hit = key_h[order][pos] == key_s
            if not hit.all():
                raise ValueError(
                    f"act 1: {int((~hit).sum())} of {len(hit)} scaffold bins of {section.name} "
                    f"are not in {path.name}; the Spateo run does not belong to this run")
            h5_rows = order[pos]
            deviation = float(np.abs(final[h5_rows] * dnb - world[rows]).max()) if len(rows) else 0.0
            if deviation > 1.0:
                raise ValueError(
                    f"act 1: {path.name} ends {deviation:.1f} um away from the run's frozen "
                    f"scaffold for {section.name}; it is not the registration this run used")
            moving = bool(stages)
            if moving:
                stage_keys = [(s, keys) for s, _, keys in stages]
                if sequence is None:
                    sequence = stage_keys
                elif [(s, len(k)) for s, k in sequence] != [(s, len(k)) for s, k in stage_keys]:
                    raise ValueError("act 1: the sections stored different iteration sequences")
            gid = gc_id[rows]
            gsel = np.flatnonzero(gid > 0)
            tsel = np.flatnonzero(gid <= 0)[::max(int(tissue_stride), 1)]
            self.slices.append({
                "name": section.name, "source": section.source, "path": str(path),
                "moving": moving, "groups": {s: g for s, g, _ in stages},
                "rows": h5_rows,                       # scaffold order -> Spateo row
                "start": spatial[h5_rows].astype(np.float32) * dnb,
                "final": world[rows].astype(np.float32),
                "gid": gid, "tsel": tsel, "gsel": gsel,
                "n_bins_spateo": int(len(spatial)), "n_bins_run": int(len(rows)),
                "max_final_deviation_um": deviation,
            })
            log(f"  {section.name}: {len(rows):,} bins, {len(gsel):,} in GCs, "
                + ("moving" if moving else "reference"), flush=True)
        if sequence is None:
            raise ValueError("act 1: none of the Spateo files stored its iterations")
        self.stages = [s for s, _ in sequence]
        self.stage_counts = {s: len(k) for s, k in sequence}
        self.steps = [(s, k) for s, keys in sequence for k in keys]
        self.n_iter = len(self.steps)
        self._handles: dict = {}
        self._pid = None

    # ---- reading ------------------------------------------------------------
    def _handle(self, path: str):
        import h5py
        if self._pid != os.getpid():                  # first use in this process
            self._handles = {}
            self._pid = os.getpid()
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def coords(self, slice_: dict, step: int | str) -> np.ndarray:
        """Grid micrometres of every run bin of one section at a stored step.

        ``step`` is an index into ``steps``, or "start" / "final"."""
        if not slice_["moving"] or step == "start":
            return slice_["start"] if slice_["moving"] else slice_["final"]
        if step == "final":
            return slice_["final"]
        stage, key = self.steps[int(step)]
        data = self._handle(slice_["path"])[f"{slice_['groups'][stage]}/{key}"][...]
        return np.asarray(data, np.float32)[slice_["rows"]] * self.dnb_um

    def stage_of(self, step: int) -> tuple[str, int, int]:
        """(stage, index within the stage starting at 1, stage length)."""
        stage, _ = self.steps[step]
        first = next(i for i, (s, _) in enumerate(self.steps) if s == stage)
        return stage, step - first + 1, self.stage_counts[stage]

    def describe(self) -> dict:
        return {
            "spateo_dir": str(self.upstream.spateo_dir),
            "pattern": self.upstream.pattern,
            "iterations": self.n_iter,
            "stage_counts": self.stage_counts,
            "sections": [{
                "name": s["name"], "source": s["source"], "file": Path(s["path"]).name,
                "moving": s["moving"], "bins_in_spateo_file": s["n_bins_spateo"],
                "bins_in_run_scaffold": s["n_bins_run"],
                "bins_drawn": int(len(s["tsel"]) + len(s["gsel"])),
                "max_final_deviation_from_scaffold_um": s["max_final_deviation_um"],
            } for s in self.slices],
        }
