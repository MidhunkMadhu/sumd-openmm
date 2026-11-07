"""The shared restart export preserves 0.9.1's whole-molecule PBC convention."""
import numpy as np
import pytest


def test_restart_matches_openmm_wrapped_positions_and_velocities(tmp_path):
    openmm = pytest.importorskip('openmm')
    from openmm import unit
    from parmed.amber import Rst7
    from sumd_openmm.backend import write_restart
    system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(nonbonded.CutoffPeriodic)
    nonbonded.setCutoffDistance(.5)
    for _ in range(3):
        system.addParticle(12)
        nonbonded.addParticle(0, .3, 0)
    system.addForce(nonbonded)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, .1, 100)
    system.addForce(bond)
    box = np.array([[2., 0, 0], [.4, 2, 0], [.1, .2, 2]])
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in box])
    context = openmm.Context(system, openmm.VerletIntegrator(.001),
                             openmm.Platform.getPlatformByName('Reference'))
    positions = np.array([[-.2, .3, .3], [-.1, .3, .3], [2.3, 2.5, 2.7]])
    velocities = np.array([[.1, .2, .3], [.2, .1, .4], [.3, .2, .1]])
    context.setPositions(positions)
    context.setVelocities(velocities)
    wrapped = context.getState(getPositions=True, enforcePeriodicBox=True)
    packet = dict(pos=positions, vel=velocities, box=box, time=1.25)
    path = tmp_path/'final.rst7'
    write_restart(str(path), packet, True, context.getMolecules())
    exported = Rst7.open(str(path))
    expected = wrapped.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    assert np.asarray(exported.coordinates).reshape(-1, 3) == pytest.approx(expected, abs=1e-6)
    assert np.asarray(exported.velocities).reshape(-1, 3) == pytest.approx(velocities * 10, abs=2e-6)
    assert np.array_equal(packet['pos'], positions)
