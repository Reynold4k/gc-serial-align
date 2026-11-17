"""The small pieces the registration is made of, each against a known answer."""
from __future__ import annotations

import numpy as np
import pytest

from gcalign import columns, global_align, parallel
from gcalign.validate import angular_spread_deg, orientation_summary


def test_similarity_fit_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 1000, (30, 2))
    theta = np.deg2rad(37.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = 1.07 * src @ rot.T + np.array([120.0, -45.0])
    r, s, t = global_align.similarity_from_pairs(src, dst)
    assert np.allclose(r, rot, atol=1e-10) and abs(s - 1.07) < 1e-10
    assert np.allclose(t, [120.0, -45.0], atol=1e-7)


def test_similarity_fit_never_returns_a_reflection():
    src = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1]])
    dst = src * [-1, 1]                                 # a mirror image
    r, _, _ = global_align.similarity_from_pairs(src, dst)
    assert np.linalg.det(r) > 0


def test_matching_respects_the_radius():
    src = np.array([[0.0, 0], [100, 0], [1000, 1000]])
    dst = np.array([[101.0, 0], [1, 0], [5000, 5000]])
    rows, cols, dists = global_align.match_pairs(src, dst, 50.0)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0)]
    assert np.allclose(sorted(dists), [1, 1])


def test_fft_correlation_finds_a_shift():
    fixed = np.zeros((120, 160), bool)
    fixed[30:70, 40:110] = True
    moving = np.roll(fixed, (-9, 14), axis=(0, 1))
    dy, dx, _ = global_align.Correlator(fixed).best_shift(moving)
    assert (dy, dx) == (9, -14)


def test_icp_converges_and_keeps_its_steps():
    rng = np.random.default_rng(1)
    target = rng.uniform(0, 3000, (40, 2))
    theta = np.deg2rad(4.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    source = (target - 1500) @ rot.T * 0.98 + 1500 + [40, -30]
    linear, offset, history, steps = global_align.run_icp(
        source, target, np.arange(40), np.eye(2), np.zeros(2), 420.0)
    assert np.abs(source @ linear.T + offset - target).max() < 1e-6
    assert len(steps) == len(history) + 1 and steps[0]["iteration"] == 0
    assert history[-1][1] < history[0][1]
    assert np.allclose(steps[-1]["linear"], linear)


def test_angular_spread_wraps_round_the_circle():
    assert angular_spread_deg([355, 2, 358]) == pytest.approx(7)
    assert angular_spread_deg([10, 190]) == pytest.approx(180)
    assert angular_spread_deg([42]) == 0


def test_orientation_summary_reports_what_happened():
    same = orientation_summary({"IF1": {"mirror": False, "angle_deg": 353.0},
                                "IF3": {"mirror": False, "angle_deg": 356.0}})
    assert same["all_sections_same_handedness"] and same["angle_spread_deg"] == 3.0
    assert "3 deg" in same["note"] and "without mirroring" in same["note"]
    odd = orientation_summary({"IF1": {"mirror": False, "angle_deg": 10.0},
                               "IF3": {"mirror": True, "angle_deg": 12.0},
                               "IF5": {"mirror": False, "angle_deg": 11.0}})
    assert not odd["all_sections_same_handedness"]
    assert odd["sections_with_minority_handedness"] == ["IF3"]


def test_gap_free_stack_is_left_alone():
    masks = np.random.default_rng(0).random((4, 20, 20)) > 0.5
    out, filled = columns.fill_unmeasured(masks, [0, 1, 2, 3])
    assert out is masks and filled == []


def test_skipped_sections_are_bridged_at_their_true_spacing():
    yy, xx = np.mgrid[0:80, 0:80]
    def disc(radius):
        return np.hypot(xx - 40, yy - 40) < radius
    # a cone measured at slots 0 and 4: the bridged slices must shrink evenly
    out, filled = columns.fill_unmeasured(np.stack([disc(30), disc(10)]), [0, 4])
    assert filled == [1, 2, 3] and out.shape[0] == 5
    radii = [np.sqrt(layer.sum() / np.pi) for layer in out]
    assert np.allclose(radii, [30, 25, 20, 15, 10], atol=0.8)


def test_an_absence_that_was_measured_is_not_filled():
    yy, xx = np.mgrid[0:40, 0:40]
    disc = np.hypot(xx - 20, yy - 20) < 8
    out, filled = columns.fill_unmeasured(np.stack([disc, np.zeros_like(disc)]), [0, 3])
    assert filled == [1, 2] and not out[1:].any()


def test_stack_layout_follows_the_section_numbers(tmp_path):
    from test_config import write
    from gcalign import config
    sections = [{"index": 2, "kind": "ST", "source": "A", "name": "A"},
                {"index": 4, "kind": "IF", "source": 4},
                {"index": 6, "kind": "ST", "source": "B", "name": "B"},
                {"index": 12, "kind": "IF", "source": 12}]
    cfg = config.load(write(tmp_path, sections=sections))
    assert columns.stack_layout(cfg) == ([0, 1, 2, 5], 20.0)


def _square(value):
    return value * value


def test_parallel_sections_keep_their_order():
    arguments = [(n,) for n in range(7)]
    assert parallel.run_sections(_square, arguments, jobs=1) == [n * n for n in range(7)]
    assert parallel.run_sections(_square, arguments, jobs=3) == [n * n for n in range(7)]
    assert parallel.resolve_jobs(8, 3) == 3 and parallel.resolve_jobs(1, 3) == 1
