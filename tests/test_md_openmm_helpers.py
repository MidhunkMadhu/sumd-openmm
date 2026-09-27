"""
The bundled MD_openmm helpers: present, importable, and parsing a real .inp
through MD_openmm's own build_inputs(). Optionally, that they are in sync
with an MD_openmm checkout (set MD_OPENMM_DIR).
"""

import filecmp
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(ROOT, "src", "sumd_openmm", "md_openmm")

REQUIRED = ("read_key_value_file", "build_inputs", "get_genvel", "normalize_none", "get_str",
            "get_bool", "get_int", "detect_coordinate_file_type", "load_coordinate_file_auto",
            "write_amber_restart", "use_velocities_from_coordinate_object",
            "barostat", "rewrap", "vfswitch", "read_crd")


def test_bundled_files_present():
    for f in ("production_helpers.py", "PROVENANCE.txt"):
        assert os.path.isfile(os.path.join(PKG, f)), f
    assert not [f for f in os.listdir(PKG) if f.startswith("omm_")]
    assert "commit:" in open(os.path.join(PKG, "PROVENANCE.txt")).read()


def test_no_bare_md_openmm_imports():
    """Inside the package every omm_* import must be package-relative."""
    for f in os.listdir(PKG):
        if f.endswith(".py"):
            for line in open(os.path.join(PKG, f)):
                s = line.strip()
                assert not (s.startswith("from omm_") or s.startswith("import omm_")), (f, s)


def test_helpers_import_and_parse_example_inp():
    pytest.importorskip("openmm")
    from sumd_openmm.omm_setup import load_production_helpers

    prod = load_production_helpers()
    missing = [f for f in REQUIRED if not hasattr(prod, f)]
    assert not missing, missing

    cfg = prod.read_key_value_file(os.path.join(ROOT, "examples", "binding_distance.inp"))
    cfg.setdefault("nstep", "0")
    inputs = prod.build_inputs(cfg)
    assert inputs.dt == 0.002 and inputs.temp == 300.0


@pytest.mark.skipif(not os.environ.get("MD_OPENMM_DIR"), reason="set MD_OPENMM_DIR to check sync")
def test_in_sync_with_md_openmm_checkout(tmp_path):
    """Re-vendoring from the checkout must reproduce the bundled files exactly."""
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "vendor_md_openmm.py"),
                    os.environ["MD_OPENMM_DIR"], "--dest", str(tmp_path)], check=True)
    for f in os.listdir(PKG):
        if f.endswith(".py"):
            assert filecmp.cmp(os.path.join(PKG, f), str(tmp_path / f), shallow=False), \
                "%s differs from MD_openmm: re-run tools/vendor_md_openmm.py" % f
