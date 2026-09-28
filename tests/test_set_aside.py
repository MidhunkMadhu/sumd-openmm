"""An existing output_dir is renamed to <output_dir>_oldNNNNN before a new run."""

import pytest

from sumd_openmm.cli import _set_aside
from sumd_openmm.config import ConfigError


def _inp(tmp_path, extra=""):
    inp = tmp_path / "run.inp"
    inp.write_text("topology_file = top.parm7\noutput_dir = sumd_run\n" + extra)
    (tmp_path / "top.parm7").write_text("")
    return str(inp)


def test_first_free_number_is_used(tmp_path):
    inp = _inp(tmp_path)
    out = tmp_path / "sumd_run"
    for n in (1, 2):
        (tmp_path / ("sumd_run_old%05d" % n)).mkdir()
    out.mkdir()
    (out / "progress.log").write_text("earlier run")
    old = _set_aside(str(out), inp)
    assert old == str(tmp_path / "sumd_run_old00003")
    assert (tmp_path / "sumd_run_old00003" / "progress.log").read_text() == "earlier run"
    assert not out.exists()


def test_input_inside_output_dir_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inp = _inp(tmp_path, "coordinate_file = sumd_run/final_state.rst7\n")
    out = tmp_path / "sumd_run"
    out.mkdir()
    (out / "final_state.rst7").write_text("")
    with pytest.raises(ConfigError, match="coordinate_file = sumd_run/final_state.rst7"):
        _set_aside(str(out), inp)
    assert out.exists() and not (tmp_path / "sumd_run_old00001").exists()
