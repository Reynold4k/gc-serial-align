from __future__ import annotations

import pytest
import yaml

from gcalign import config


def write(tmp_path, **changes):
    base = {
        "name": "t", "output_dir": str(tmp_path / "out"), "section_thickness_um": 10.0,
        "sections": [{"index": 1, "kind": "IF", "source": 1},
                     {"index": 2, "kind": "ST", "source": "A", "name": "A"},
                     {"index": 3, "kind": "IF", "source": 3},
                     {"index": 4, "kind": "ST", "source": "B", "name": "B"},
                     {"index": 5, "kind": "IF", "source": 5}],
        "if_images": {"dir": str(tmp_path), "um_per_px": 0.8,
                      "channels": {0: "IgD_AF647", 1: "CD4_AF594", 2: "GL7_AF488", 3: "DAPI"},
                      "gc_channel": "GL7_AF488", "tissue_channels": {"CD4_AF594": 1.0}},
        "st_scaffold": {"h5ad": str(tmp_path / "x.h5ad")},
    }
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return path


def test_layout_is_derived_from_the_section_list(tmp_path):
    cfg = config.load(write(tmp_path))
    assert cfg.plane_names == ["IF1", "A", "IF3", "B", "IF5"]
    assert cfg.neighbours == {1: ["A"], 3: ["A", "B"], 5: ["B"]}
    assert cfg.one_sided(1) and cfg.one_sided(5) and not cfg.one_sided(3)
    assert [cfg.z_um(i) for _, _, i in cfg.planes] == [0, 10, 20, 30, 40]
    assert cfg.z_span_um == 40 and cfg.n_planes == 5


def test_z_starts_at_the_first_section_whatever_its_number(tmp_path):
    sections = [{"index": 7, "kind": "ST", "source": "A", "name": "A"},
                {"index": 8, "kind": "IF", "source": 8},
                {"index": 11, "kind": "ST", "source": "B", "name": "B"}]
    cfg = config.load(write(tmp_path, sections=sections))
    assert cfg.z_of("A") == 0 and cfg.z_of("IF8") == 10 and cfg.z_of("B") == 40


def test_display_channels_follow_the_fluorophores(tmp_path):
    cfg = config.load(write(tmp_path))
    assert cfg.rgb_channels == ["CD4_AF594", "GL7_AF488", "IgD_AF647"]
    chosen = config.load(write(tmp_path, if_images={"display_rgb": ["GL7_AF488", None, "DAPI"]}))
    assert chosen.rgb_channels == ["GL7_AF488", None, "DAPI"]


def test_default_rgb_leaves_the_counterstain_out():
    assert config.default_rgb(["DAPI", "a", "b"]) == ["a", "b", None]
    assert config.default_rgb(["PE_CD4", "x_AF488", "y_Cy5"]) == ["PE_CD4", "x_AF488", "y_Cy5"]
    # a dye name inside a longer word is not a dye
    assert config.default_rgb(["Peanut_AF488", "x_AF594"])[:2] == ["x_AF594", "Peanut_AF488"]


@pytest.mark.parametrize("changes, message", [
    ({"if_images": {"gc_channel": "nope"}}, "gc_channel"),
    ({"if_images": {"tissue_channels": {"nope": 1.0}}}, "tissue_channels"),
    ({"if_images": {"display_rgb": ["a", "b"]}}, "three"),
    ({"align": {"gird_um": 5}}, "gird_um"),
    ({"viewer": {"palette": "neon"}}, "palette"),
    ({"sections": [{"index": 1, "kind": "IF", "source": 1}]}, "no ST"),
    ({"sections": [{"index": 1, "kind": "IF", "source": 1},
                   {"index": 1, "kind": "ST", "source": "A"}]}, "same index"),
])
def test_mistakes_are_named(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        config.load(write(tmp_path, **changes))


def test_output_dir_can_be_overridden(tmp_path):
    cfg = config.load(write(tmp_path), output_dir=tmp_path / "elsewhere")
    assert cfg.output_dir == tmp_path / "elsewhere"
