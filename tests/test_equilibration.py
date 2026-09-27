"""
CHARMM-GUI protocols: parsing, restraint energies and the staged run.

The folder tests use a CHARMM-GUI download (tests/charmm-gui, or
SUMD_CHARMMGUI_DIR) holding amber/ and gromacs/; they are skipped without it.
"""

import os

import numpy as np
import pytest

from sumd_openmm import charmmgui

HERE = os.path.dirname(os.path.abspath(__file__))
CGUI = os.environ.get("SUMD_CHARMMGUI_DIR", os.path.join(HERE, "charmm-gui"))
needs_cgui = pytest.mark.skipif(not os.path.isdir(os.path.join(CGUI, "amber")),
                                reason="no CHARMM-GUI download at %s" % CGUI)


MDIN = """title line that is not a restraint group
 &cntrl
    imin=0, irest=1, ntx=5,     ! comment
    nstlim=250000, dt=0.002,
    ntp=3, csurften=3, pres0=1.0,
    ntr=1, nmropt=1,
 /
 &ewald
    vdwmeth = 0,
 /
 &wt
    type='END'
 /
DISANG=x.rest
LISTIN=POUT
LISTOUT=POUT
&end
Protein posres
1.0
RES 1 2
END
Membrane posres
0.5
FIND
P * * POPC
SEARCH
RES 1 4
END
END
"""


def test_mdin_and_group_input():
    values, rest = charmmgui.parse_mdin(MDIN)
    assert values["nstlim"] == "250000" and values["vdwmeth"] == "0" and values["type"] == "END"
    groups = charmmgui.parse_group_input(rest)
    assert [(g[0], g[1]) for g in groups] == [("Protein posres", 1.0), ("Membrane posres", 0.5)]
    names = np.array(["N", "CA", "N", "P", "P", "C1"])
    resnames = np.array(["ALA", "ALA", "GLY", "POPC", "POPC", "POPC"])
    residues = np.array([1, 1, 2, 3, 4, 5])
    types = np.array(["X"] * 6)
    atoms = [charmmgui._group_atoms(g[2], g[3], residues, names, resnames, types) for g in groups]
    assert atoms[0].tolist() == [0, 1, 2]
    assert atoms[1].tolist() == [3, 4]                    # residue 5 is outside SEARCH RES 1 4


def test_rst_flat_bottom_parameters():
    text = """&rst
    iat=1, 2, 3, 4,
    r1=-132.5, r2=-122.5,
    r3=-117.5, r4=-107.5,
    rk2=FC,
    rk3=FC,
/"""
    t = charmmgui.parse_rst(text, 250.0)
    assert t.atoms.tolist() == [[0, 1, 2, 3]]
    c, l1, lo, hi, u4, klo, khi = t.params[0]
    assert np.degrees([c, l1, lo, hi, u4]) == pytest.approx([-120, -12.5, -2.5, 2.5, 12.5])
    assert klo == khi == pytest.approx(250 * 4.184)


def test_gromacs_preprocessor(tmp_path):
    (tmp_path / "mol.itp").write_text("""
[ moleculetype ]
MOL 3
[ atoms ]
1 C 1 MOL C1 1 0 12
2 C 1 MOL C2 1 0 12
#ifdef POSRES
[ position_restraints ]
1 1 FC_BB FC_BB FC_BB
2 1 0.0 0.0 FC_Z
#endif
#ifdef DIHRES
[ dihedral_restraints ]
1 2 1 2 1 60.0 5.0 DIHRES_FC
#endif
""")
    (tmp_path / "wat.itp").write_text("[ moleculetype ]\nWAT 2\n[ atoms ]\n1 O 1 WAT O 1 0 16\n")
    (tmp_path / "topol.top").write_text('#include "mol.itp"\n#include "wat.itp"\n'
                                        "[ system ]\nx\n[ molecules ]\nWAT 3\nMOL 2\n")
    defines = charmmgui.mdp_defines({"define": "-DPOSRES -DFC_BB=100 -DFC_Z=40 -DDIHRES -DDIHRES_FC=8"})
    positions, torsions, total = charmmgui._TopologyReader(
        str(tmp_path / "topol.top"), [str(tmp_path)], defines).restraints()
    assert total == 7
    assert positions[0].atoms.tolist() == [3, 4, 5, 6]    # after 3 waters
    assert positions[0].k.tolist() == [[50, 50, 50], [0, 0, 20]] * 2   # E = 1/2 k dx^2
    assert torsions[0].atoms.tolist() == [[3, 4, 3, 4], [5, 6, 5, 6]]
    assert torsions[0].params[0, 5] == 4
    none = charmmgui._TopologyReader(str(tmp_path / "topol.top"), [str(tmp_path)], {}).restraints()
    assert none[0] == [] and none[1] == []
    with pytest.raises(charmmgui.CharmmGuiError, match="FC_BB"):
        charmmgui._TopologyReader(str(tmp_path / "topol.top"), [str(tmp_path)], {"POSRES": True})


def _flat_bottom(theta, c, l1, lo, hi, u4, klo, khi):
    d = (theta - c + np.pi) % (2 * np.pi) - np.pi
    if d < l1:
        return klo * ((l1 - lo) ** 2 + 2 * (l1 - lo) * (d - l1))
    if d < lo:
        return klo * (d - lo) ** 2
    if d > u4:
        return khi * ((u4 - hi) ** 2 + 2 * (u4 - hi) * (d - u4))
    if d > hi:
        return khi * (d - hi) ** 2
    return 0.0


def test_torsion_restraint_energy_matches_definition():
    openmm = pytest.importorskip("openmm")
    from sumd_openmm.equilibration import _restraint_forces

    params = np.array([np.radians(170.0), np.radians(-12.5), np.radians(-2.5), np.radians(2.5),
                       np.radians(12.5), 1046.0, 900.0])
    stage = charmmgui.Stage("s", torsions=[charmmgui.TorsionRestraints(
        "t", np.array([[0, 1, 2, 3]]), params[None, :])])
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    system.addForce(_restraint_forces(stage, np.zeros((4, 3)))[0])
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    for degrees in (170, 175, 180, -175, 150, -150, 10, 160, 185):
        phi = np.radians(degrees)
        pos = [[1, 0, 0], [0, 0, 0], [0, 0, 1], [np.cos(phi), np.sin(phi), 1]]
        context.setPositions(np.asarray(pos, float) * 0.1)
        energy = context.getState(getEnergy=True).getPotentialEnergy()._value
        # OpenMM's torsion of this geometry is +phi
        assert energy == pytest.approx(_flat_bottom(phi, *params), abs=1e-6), degrees


@needs_cgui
@pytest.mark.parametrize("fmt", ["amber", "gromacs"])
def test_charmm_gui_protocols(fmt):
    p = charmmgui.read(CGUI, fmt)
    assert [s.name for s in p.stages] == ["step6.0_minimization"] + [
        "step6.%d_equilibration" % i for i in range(1, 7)]
    assert p.stages[0].minimize and p.stages[1].generate_velocities
    assert [s.barostat for s in p.stages[1:]] == [None, None, "membrane", "membrane", "membrane", "membrane"]
    assert [s.dt_ps for s in p.stages[1:]] == [0.001] * 3 + [0.002] * 3
    assert (p.cutoff_nm, p.switch_nm, p.dispersion_correction) == (1.2, 1.0, False)
    assert p.production["pcouple"] == "yes" and p.production["p_type"] == "membrane"
    if fmt == "amber":                                   # kcal/mol/A^2 -> kJ/mol/nm^2
        protein = [s.positions[0].k[0, 0] / 418.4 for s in p.stages]
        assert protein == pytest.approx([10, 10, 5, 2.5, 1, 0.5, 0.1])
        dihedral = [s.torsions[0].params[0, 5] / 4.184 if s.torsions else 0 for s in p.stages]
        assert dihedral == pytest.approx([250, 250, 100, 50, 50, 25, 0])
    else:                                                # E = 1/2 k: stored k is half the mdp value
        backbone = [s.positions[0].k.max() * 2 for s in p.stages]
        assert backbone == pytest.approx([4000, 4000, 2000, 1000, 500, 200, 50])
        assert p.stages[-1].torsions == []


@needs_cgui
def test_staged_run_resumes(tmp_path):
    pytest.importorskip("openmm")
    from sumd_openmm.equilibration import Equilibration

    p = charmmgui.read(CGUI, "amber")
    p.stages = p.stages[:3]
    messages = []
    job = Equilibration(p, str(tmp_path), "auto", seed=3, log=messages.append, test=True)
    job.run()
    job.close()
    assert os.path.exists(tmp_path / "equilibrated.xml")
    assert sum("finished on" in m for m in messages) == 3
    os.remove(tmp_path / "step6.2_equilibration.xml")          # as if the job was killed there
    messages.clear()
    job = Equilibration(p, str(tmp_path), "auto", seed=3, log=messages.append, test=True)
    job.run()
    job.close()
    assert sum("skipping" in m for m in messages) == 2
    assert sum("finished on" in m for m in messages) == 1
    log = open(tmp_path / "equilibration.log").read()
    assert log.count("stage 3/3 started") == 2


@needs_cgui
def test_production_settings_follow_protocol():
    from sumd_openmm.equilibration import production_settings

    p = charmmgui.read(CGUI, "gromacs")
    cfg = production_settings(p, "equilibrated.xml", {"temp": "300"})
    assert cfg["force_field"] == "GROMACS" and cfg["coordinate_file"] == "equilibrated.xml"
    assert cfg["vdw"] == "Force-switch" and cfg["r_on"] == "1.0" and cfg["genvel"] == "no"
    assert cfg["temp"] == "300" and cfg["p_type"] == "membrane"
    with pytest.raises(ValueError, match="coordinate_file"):
        production_settings(p, "x.xml", {"coordinate_file": "other.rst7"})
