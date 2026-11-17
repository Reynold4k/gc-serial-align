"""The whole pipeline on an experiment whose answer is known (tests/synthetic.py)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from gcalign import cli, config, replay
from gcalign.runio import Run
from gcalign.transforms import Transforms


def mapping_error(experiment, transforms, name, warp=True, n=3000, seed=0):
    """Distance between where the pipeline puts image pixels and where they truly are."""
    truth = experiment.truth[name]
    height, width = truth.image_shape
    pixels = np.random.default_rng(seed).uniform(0, [width, height], (n, 2))
    world = truth.to_frame(pixels)
    inside = experiment.scene.tissue(world)
    found = transforms[name].image_px_to_frame(pixels[inside], warp=warp)
    return np.linalg.norm(found - world[inside], axis=1)


def auc(score, label):
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    ranks = np.empty(len(score))
    ranks[np.argsort(score, kind="mergesort")] = np.arange(1, len(score) + 1)
    positives = label.sum()
    return (ranks[label].sum() - positives * (positives + 1) / 2) / (positives * (len(label) - positives))


def test_check_accepts_the_inputs(experiment, capsys):
    assert cli.main(["check", "--config", str(experiment.config)]) == 0
    assert "A run can start" in capsys.readouterr().out


def test_orientation_is_recovered(finished):
    cfg = config.load(finished.config)
    sections = json.loads((cfg.output_dir / "02_global_align/global_transforms.json").read_text())["sections"]
    for name, truth in finished.truth.items():
        entry = sections[name]
        assert entry["flip"] == truth.mirror, f"{name}: wrong mirror state"
        relative = np.array(entry["linear"]) @ np.linalg.inv(truth.linear)
        rotation = np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))
        scale = np.sqrt(abs(np.linalg.det(relative)))
        assert abs(rotation) < 1.0, f"{name}: {rotation:.2f} deg off"
        assert abs(scale - 1) < 0.02, f"{name}: scale off by {scale:.4f}"
        assert entry["n_matched"] == entry["n_if_gc"] == finished.n_blobs


def test_the_structures_not_the_outline_pick_the_orientation(finished):
    """Every candidate's scores are stored, and the winner is the one the structures agree with."""
    run = Run.open(finished.config)
    search = run.orientation_search
    assert search["close_match_um"] == run.cfg.align.close_match_um
    for name, truth in finished.truth.items():
        candidates = search["sections"][name]["candidates"]
        winner = candidates[search["sections"][name]["winner"]]
        assert winner["flip"] == truth.mirror
        assert winner["n_close"] == max(c["n_close"] for c in candidates)
        wrong_mirror = [c for c in candidates if c["flip"] != truth.mirror]
        assert wrong_mirror, "the search should have carried a mirrored candidate too"
        assert all(c["n_close"] < winner["n_close"] for c in wrong_mirror)
        # the outline alone could not have told them apart this clearly
        assert max(c["tissue_dice"] for c in wrong_mirror) > 0.8


def test_points_land_where_they_truly_are(finished):
    transforms = Transforms.load(config.load(finished.config))
    assert transforms.origin == "stored"
    for name in finished.truth:
        rigid = mapping_error(finished, transforms, name, warp=False)
        warped = mapping_error(finished, transforms, name, warp=True)
        assert np.median(warped) < 6.0, f"{name}: median error {np.median(warped):.1f} um"
        assert np.percentile(warped, 95) < 25.0
        assert np.median(warped) < np.median(rigid) / 3, "the warp should remove most of the distortion"


def test_transforms_round_trip(finished):
    transforms = Transforms.load(config.load(finished.config))
    points = np.random.default_rng(3).uniform(1500, 4000, (500, 2))
    for t in transforms:
        back = t.to_frame(t.from_frame(points))
        assert np.abs(back - points).max() < 1e-4
        pixels = t.frame_to_image_px(points)
        assert np.abs(t.image_px_to_frame(pixels) - points).max() < 1e-4


def test_stored_transforms_equal_rebuilt_ones(finished, tmp_path):
    """A run from before transforms were stored gets exactly the same ones."""
    cfg = config.load(finished.config)
    old = tmp_path / "as_0_1"
    shutil.copytree(cfg.output_dir, old)
    (old / "03_gc_match/section_transforms.json").unlink()
    (old / "02_global_align/icp_trajectory.json").unlink()
    stored = Transforms.load(cfg)
    rebuilt = Transforms.load(config.load(finished.config, output_dir=old))
    assert rebuilt.origin == "rebuilt"
    for name in stored.names:
        assert np.array_equal(stored[name].source_um, rebuilt[name].source_um)
        assert np.array_equal(stored[name].target_um, rebuilt[name].target_um)
        assert stored[name].smoothing == rebuilt[name].smoothing
        steps = replay.rebuild(config.load(finished.config, output_dir=old), name)["steps"]
        kept = json.loads((cfg.output_dir / "02_global_align/icp_trajectory.json").read_text())[name]["steps"]
        assert steps == kept


def test_every_structure_keeps_one_identity(finished):
    run = Run.open(finished.config)
    assert len(run.gc_ids) == finished.n_blobs
    assert (run.presence.n_planes == run.cfg.n_planes).all()
    metrics = run.metrics
    assert len(metrics) == finished.n_blobs
    assert (metrics.mesh_open_edges == 0).all()
    ratio = metrics.mesh_volume_um3 / metrics.volume_um3
    assert ratio.between(0.97, 1.03).all()


def test_volumes_match_the_analytic_scene(finished):
    """Cavalieri on the registered outlines against the volume the scene defines."""
    run = Run.open(finished.config)
    thickness = run.cfg.section_thickness_um
    presence = run.presence.set_index("merged_gc_id")
    measured = run.metrics.set_index("merged_gc_id").volume_um3
    # the volume is exactly the outlines' areas times the section thickness
    areas = sum(presence[f"{plane.name}_area_um2"] for plane in run.planes)
    assert np.allclose(measured, (areas * thickness)[measured.index])
    # on the transcriptomics planes the outline is the scene's own, up to the bin size
    for plane in run.planes:
        if plane.kind != "ST":
            continue
        truth = np.pi * finished.scene.blob_radii(plane.z_um) ** 2
        found = presence[f"{plane.name}_area_um2"]
        assert np.allclose(found[run.gc_ids], truth[[g - 1 for g in run.gc_ids]], rtol=0.10)
    # IF outlines come from thresholding a blurred stain, so they run slightly
    # large; the total must still be the scene's volume, not a multiple of it
    expected = sum(np.pi * finished.scene.blob_radii(plane.z_um) ** 2 * thickness
                   for plane in run.planes)
    assert 0.95 < measured.sum() / expected.sum() < 1.25


def test_run_reader_serves_every_plane(finished):
    run = Run.open(finished.config)
    assert run.names == [name for name, _, _ in finished.planes]
    for plane in run.planes:
        assert run.labels(plane.name).shape == run.grid.shape
        tissue = run.tissue(plane.name)
        assert tissue.shape == run.grid.shape and tissue.sum() > 1000
        assert run.tissue_is_stored(plane.name)
        if plane.kind == "IF":
            image = run.rgb(plane.name)
            assert image.shape == run.grid.shape + (3,) and 0 <= image.min() and image.max() <= 1
            assert run.icp(plane.name)["origin"] == "stored by stage align"
    assert set(run.identity) == set(run.gc_ids)


def test_processes_do_not_change_the_answer(finished, tmp_path):
    cfg = config.load(finished.config)
    other = tmp_path / "parallel"
    for stage in ("00_scaffold", "01_if_gc"):
        shutil.copytree(cfg.output_dir / stage, other / stage)
    assert cli.main(["run", "--config", str(finished.config), "--output-dir", str(other),
                     "--from", "align", "--to", "match", "--jobs", "3"]) == 0
    for relative in ("02_global_align/global_transforms.json", "02_global_align/icp_trajectory.json",
                     "03_gc_match/match_report.json", "03_gc_match/section_transforms.json"):
        assert (other / relative).read_bytes() == (cfg.output_dir / relative).read_bytes(), relative
    a = np.load(cfg.output_dir / "03_gc_match/plane_gc_labels.npz")
    b = np.load(other / "03_gc_match/plane_gc_labels.npz")
    assert all(np.array_equal(a[key], b[key]) for key in a.files)


def test_validation_reports_what_the_data_did(finished):
    assert cli.main(["run", "--config", str(finished.config), "--only", "validate"]) == 0
    report = Run.open(finished.config).validation
    orientation = report["stage_A_orientation"]
    mirrored = [name for name, truth in finished.truth.items() if truth.mirror]
    assert orientation["all_sections_same_handedness"] is False
    assert orientation["sections_with_minority_handedness"] == mirrored
    identity = report["stage_C_identity"]
    assert identity["n_planes"] == 5 and identity["n_on_all_planes"] == finished.n_blobs
    assert report["stage_D_columns"]["watertight"] == finished.n_blobs
    assert report["gcalign_version"]


def test_transfer_reads_the_stain_at_the_bins(finished):
    assert cli.main(["transfer", "--config", str(finished.config)]) == 0
    cfg = config.load(finished.config)
    table = np.load(cfg.output_dir / "06_transfer/st_bins_if.npz")
    meta = json.loads((cfg.output_dir / "06_transfer/st_bins_if.json").read_text())
    inside = table["gc_id"] > 0
    assert auc(table["BLOB_AF488"], inside) > 0.97          # the structure marker follows the structures
    assert auc(table["BODY_AF594"], inside) < 0.65          # a uniform stain does not
    assert table["obs_names"][0].startswith("S2:")
    assert set(np.unique(table["if_before"])) == {"IF1", "IF3"}
    agree = (table["if_gc_before"][inside] == table["gc_id"][inside]).mean()
    assert agree > 0.85
    assert set(meta["scale"]) == set(cfg.if_names)


def test_export_agrees_with_the_pipeline_raster(finished, tmp_path):
    out = tmp_path / "IF3_10um.ome.tif"
    assert cli.main(["export", "--config", str(finished.config), "--section", "IF3",
                     "--um-per-px", "10", "--out", str(out)]) == 0
    import tifffile
    image = tifffile.imread(out)
    side = json.loads(out.with_suffix("").with_suffix(".json").read_text())
    run = Run.open(finished.config)
    assert image.shape[0] == 4 and side["channels"] == run.channel_names
    # put the export on the pipeline grid and compare the structure marker
    col0 = int(round((side["x0_um"] - run.grid.x0) / run.grid.um))
    row0 = int(round((side["y0_um"] - run.grid.y0) / run.grid.um))
    rows, cols = image.shape[1:]
    mine = image[run.channel_names.index("BLOB_AF488")].astype(float)
    theirs = run.channel("IF3", "BLOB_AF488")[row0:row0 + rows, col0:col0 + cols].astype(float)
    mask = run.tissue("IF3")[row0:row0 + rows, col0:col0 + cols]
    assert np.corrcoef(mine[mask], theirs[mask])[0, 1] > 0.9


def test_map_command_moves_points(finished, tmp_path):
    truth = finished.truth["IF3"]
    height, width = truth.image_shape
    pixels = np.random.default_rng(5).uniform(0, [width, height], (400, 2))
    pixels = pixels[finished.scene.tissue(truth.to_frame(pixels))][:60]     # points on the section
    points = tmp_path / "points.csv"
    pd.DataFrame(pixels, columns=["x", "y"]).to_csv(points, index=False)
    out = tmp_path / "mapped.csv"
    assert cli.main(["map", "--config", str(finished.config), "--section", "IF3",
                     "--points", str(points), "--out", str(out)]) == 0
    mapped = pd.read_csv(out)
    error = np.linalg.norm(mapped[["x_frame_um", "y_frame_um"]].to_numpy() - truth.to_frame(pixels), axis=1)
    assert np.median(error) < 6 and error.max() < 30
    assert np.allclose(mapped.x_anndata * 0.5, mapped.x_frame_um)
    # and back again
    back = tmp_path / "back.csv"
    mapped[["x_frame_um", "y_frame_um"]].rename(columns={"x_frame_um": "x", "y_frame_um": "y"}).to_csv(
        tmp_path / "frame.csv", index=False)
    assert cli.main(["map", "--config", str(finished.config), "--section", "IF3", "--from", "frame",
                     "--points", str(tmp_path / "frame.csv"), "--out", str(back)]) == 0
    returned = pd.read_csv(back)[["x_image_px", "y_image_px"]].to_numpy()
    assert np.abs(returned - pixels).max() < 1e-3


@pytest.fixture(scope="session")
def skipped(tmp_path_factory):
    """Sections 3 and 5 were not kept: the stack has gaps of two different sizes."""
    import synthetic
    experiment = synthetic.build(
        tmp_path_factory.mktemp("skipped"), seed=11,
        layout=((1, "IF"), (2, "ST"), (4, "IF"), (6, "ST"), (7, "IF")))
    assert cli.main(["run", "--config", str(experiment.config), "--to", "columns"]) == 0
    return experiment


def test_skipped_sections_keep_their_true_depth(skipped):
    run = Run.open(skipped.config)
    assert [plane.z_um for plane in run.planes] == [0, 10, 30, 50, 60]
    build = json.loads((run.root / "04_columns/column_build_report.json").read_text())
    assert build["plane_slots"] == [0, 1, 3, 5, 6] and build["bridged_slots"] == [2, 4]
    assert build["z_step_um"] == 10.0
    columns = np.load(run.root / "04_columns/gc_columns.npz")
    top = max(float(columns[f"v{gid}"][:, 2].max()) for gid in columns["gc_ids"])
    assert 55 < top < 80                                   # the meshes reach the last section's z
    thickness = run.cfg.section_thickness_um
    measured = run.metrics.set_index("merged_gc_id").volume_um3
    presence = run.presence.set_index("merged_gc_id")
    kept = sum(presence[f"{plane.name}_area_um2"] for plane in run.planes) * thickness
    # seven slices of depth, of which five were measured: the two bridged ones add volume
    ratio = (measured / kept[measured.index]).median()
    assert 1.3 < ratio < 1.5
    expected = sum(np.pi * skipped.scene.blob_radii(z) ** 2 * thickness
                   for z in np.arange(0, 61, thickness))
    assert 0.95 < measured.sum() / expected[[g - 1 for g in measured.index]].sum() < 1.25
    assert (run.metrics.mesh_open_edges == 0).all()
    ratio_mesh = run.metrics.mesh_volume_um3 / run.metrics.volume_um3
    assert ratio_mesh.between(0.96, 1.04).all()


def test_a_run_without_any_3d_surface_still_finishes(finished, tmp_path):
    """No structure spans enough sections: the later stages must not crash on empty tables."""
    from gcalign import columns, validate, viewer

    cfg = config.load(finished.config)
    copy = tmp_path / "no_columns"
    for stage in ("00_scaffold", "01_if_gc", "02_global_align", "03_gc_match"):
        shutil.copytree(cfg.output_dir / stage, copy / stage)
    cfg = config.load(finished.config, output_dir=copy)
    cfg.columns.min_planes = 99
    columns.run(cfg)
    viewer.run(cfg)
    validate.run(cfg)
    run = Run(cfg)
    assert len(run.metrics) == 0 and run.column_ids == []
    report = run.validation
    assert report["stage_D_columns"]["n_columns"] == 0
    assert report["stage_D_columns"]["mesh_over_voxel_volume"]["median"] is None
    assert (copy / "05_viewer" / cfg.viewer.filename).stat().st_size > 100_000
