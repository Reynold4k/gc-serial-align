"""Shared fixtures: one synthetic experiment, run once per test session."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import synthetic  # noqa: E402


@pytest.fixture(scope="session")
def experiment(tmp_path_factory):
    """Images, scaffold and config with a known answer (see tests/synthetic.py)."""
    return synthetic.build(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="session")
def finished(experiment):
    """The same experiment after the registration stages have run on it."""
    from gcalign import cli

    assert cli.main(["run", "--config", str(experiment.config), "--to", "columns"]) == 0
    return experiment
