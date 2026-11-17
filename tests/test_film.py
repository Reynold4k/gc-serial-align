"""The film renders any finished run, not only the experiment it was made for.

The synthetic experiment has five planes with other names (IF1, S2, IF3, S4,
IF5), three IF sections, other channel names and no Spateo run, so the film has
to open on act 2 and derive its whole storyboard from the run.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

SIZE = (480, 270)


@pytest.fixture(scope="module")
def film(finished):
    from gcalign import load
    from gcalign.film.story import Film
    from gcalign.runio import Run

    run = Run(load(finished.config))
    return Film(run, size=SIZE, fps=12, preset="short", quality="draft", log=None)


def _wide(film, act):
    """A moment of the act where the frame is a wide view (the matching
    close-up is full-bleed by design, so its corners show tissue)."""
    shots = [s for s in film.shots if s.act == act]
    assert shots, f"no shot for act {act}"
    s = shots[0]
    if s.kind == "insert":
        return s.start + s.params["phases"]["fly"] * 0.5
    return s.start + s.length * 0.5


def test_storyboard_follows_the_run(film):
    A = film.A
    assert A.names == ["IF1", "S2", "IF3", "S4", "IF5"]
    assert A.if_names == ["IF1", "IF3", "IF5"]
    assert sorted(film.act_bounds) == [2, 3, 4, 5]          # no Spateo run: no act 1
    inserted = [s.params["name"] for s in film.shots if s.act == 3]
    assert inserted == A.if_names                            # one insertion per IF section
    assert sum(s.kind == "insert" for s in film.shots) == 1  # the short edit matches one in full
    for name in A.if_names:
        local = A.local[name]
        assert local["n_icp"] == len(local["steps"]) - 1
    z = [A.z[n] for n in A.names]
    assert z == sorted(z) and z[0] == 0.0
    assert film.n_frames == int(round(film.duration * film.fps))
    record = film.record(None, None, None)
    json.dumps(record)                                       # serialisable as written


@pytest.mark.parametrize("act", [2, 3, 4, 5])
def test_frames_are_drawn_on_black(film, act):
    frame = np.asarray(film.render(_wide(film, act)))
    assert frame.shape == (SIZE[1], SIZE[0], 3)
    assert frame.max() > 60 and (frame.max(axis=2) > 20).mean() > 0.01, "blank frame"
    for corner in (frame[:4, :4], frame[:4, -4:], frame[-4:, :4], frame[-4:, -4:]):
        assert corner.max() == 0, "corners must be pure black"


def test_icp_frames_show_stored_iterations(film):
    shot = next(s for s in film.shots if s.kind == "insert")
    ph = shot.params["phases"]
    t = shot.start + ph["fly"] + ph["dive"] + ph["icp"] * 0.99
    frame = np.asarray(film.render(t))
    assert frame.max() > 60


def test_encode_a_short_clip(finished, tmp_path):
    from gcalign.film import find_ffmpeg, probe, run

    try:
        find_ffmpeg()
    except RuntimeError:
        pytest.skip("no ffmpeg available")
    out = run(finished.config, tmp_path / "clip.mp4", preset="short", size=(320, 180), fps=12,
              acts=[4], quality="draft", workers=1, log=None)
    assert out.exists() and out.stat().st_size > 0
    record = json.loads(out.with_name("clip_validation.json").read_text())
    info = probe(out)
    assert info["frames"] == record["film"]["frames"] > 0
