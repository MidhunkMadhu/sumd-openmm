#!/usr/bin/env python3
"""Run a matched deterministic three-way comparison on an allocated GPU node.

The reference is the original 0.9.1 source, using its default Langevin
integrator at zero friction. Both OpenMM versions receive the same native
State, including velocities; GROMACS receives the corresponding GRO.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from sumd_openmm.backend import is_portable_state, read_state
from sumd_openmm.dcdtools import read_frames
from sumd_openmm.inspect_run import inspect

ROOT = Path(__file__).resolve().parents[1]


def state_packet(path):
    if is_portable_state(path):
        return read_state(path)
    from openmm import XmlSerializer, unit
    state = XmlSerializer.deserialize(Path(path).read_text())
    return dict(pos=np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)),
                vel=np.asarray(state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)),
                epot=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def numeric_comparison(left, right, coordinate_tolerance=.001, energy_tolerance=.001):
    left, right = Path(left), Path(right)
    a, b = read_frames(left / 'sumd_traj.dcd'), read_frames(right / 'sumd_traj.dcd')
    if a.shape != b.shape or not a.size:
        raise ValueError('trajectory shapes differ or no frames were saved: %s, %s' % (a.shape, b.shape))
    p, q = state_packet(left / 'final_state.xml'), state_packet(right / 'final_state.xml')
    error = np.asarray(a, float) - b
    result = dict(shape=list(a.shape), bit_identical_saved_coordinates=bool(np.array_equal(a, b)),
                  max_coordinate_difference_A=float(np.max(abs(error))),
                  rms_coordinate_difference_A=float(np.sqrt(np.mean(error**2))),
                  final_potential_difference_kJmol=float(abs(p['epot'] - q['epot'])),
                  final_velocity_difference_nm_ps=float(np.max(abs(p['vel'] - q['vel']))),
                  audits=[inspect(str(left)), inspect(str(right))])
    from parmed.amber import Rst7
    left_restart, right_restart = (Rst7.open(str(p / 'final_state.rst7')) for p in (left, right))
    result['restart_max_coordinate_difference_A'] = float(np.max(abs(
        np.asarray(left_restart.coordinates) - np.asarray(right_restart.coordinates))))
    public = lambda directory: {str(p.relative_to(directory)) for p in directory.rglob('*')
                                if p.is_file() and 'engine_work' not in p.relative_to(directory).parts}
    result['same_public_filenames'] = public(left) == public(right)
    result['csv_columns'] = {}
    for filename in sorted(public(left) & public(right)):
        if filename.endswith('.csv'):
            with open(left / filename) as x, open(right / filename) as y:
                ax, by = next(csv.reader(x)), next(csv.reader(y))
            result['csv_columns'][filename] = dict(identical=ax == by,
                                                   only_left=[c for c in ax if c not in by],
                                                   only_right=[c for c in by if c not in ax])
    result['passed'] = bool(result['same_public_filenames']
                            and result['max_coordinate_difference_A'] < coordinate_tolerance
                            and result['restart_max_coordinate_difference_A'] < coordinate_tolerance
                            and result['final_potential_difference_kJmol'] < energy_tolerance
                            and all(r['final_traj_identical_to_path_windows']
                                    and r['max_dcd_vs_state_mismatch_A'] < 1e-4
                                    for r in result['audits']))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-source', required=True)
    parser.add_argument('--gmx', default='gmx_mpi')
    parser.add_argument('--cpu', action='store_true', help='local preflight; cluster validation defaults to GPU')
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    fixtures = ROOT / 'validation/gromacs'
    for name in ('argon.top', 'argon.gro', 'production.mdp'):
        shutil.copyfile(fixtures / name, work / name)
    import parmed
    gro = parmed.load_file(str(work / 'argon.gro'))
    from openmm import System, Context, VerletIntegrator, Platform, XmlSerializer, unit
    from openmm.app import GromacsGroFile
    system = System()
    for atom in gro.atoms:
        system.addParticle(atom.mass)
    box = GromacsGroFile(str(work / 'argon.gro')).getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*box)
    context = Context(system, VerletIntegrator(.0001), Platform.getPlatformByName('Reference'))
    context.setPositions(gro.coordinates * unit.angstrom)
    context.setVelocities(gro.velocities * unit.angstrom / unit.picosecond)
    initial = context.getState(getPositions=True, getVelocities=True, getParameters=True)
    (work / 'initial_state.xml').write_text(XmlSerializer.serialize(initial))
    del context
    os.chdir(work)
    report = dict(status='running', passed=False, commands=[], comparisons={},
                  coordinate_tolerance_A=.001, energy_tolerance_kJmol=.001,
                  reference_source=str(Path(args.reference_source).resolve()),
                  slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'),
                  reference_note='0.9.1 source; isolated Dardel copy includes only the membrane-pressure logging fix. '
                                 'This deterministic fixture has no barostat, so that fix is inactive.')
    report['source_hashes'] = {}
    for label, folder in [('reference', Path(args.reference_source)), ('current', ROOT / 'src')]:
        report['source_hashes'][label] = {
            name: hashlib.sha256((folder / 'sumd_openmm' / name).read_bytes()).hexdigest()
                  if (folder / 'sumd_openmm' / name).exists() else None
            for name in ('engine.py', 'omm_setup.py', 'cli.py', 'gromacs.py', 'executors.py', 'backend.py')}

    def save():
        Path('reference_091.json').write_text(json.dumps(report, indent=2) + '\n')

    for name, engine, reference in (('reference_091', 'openmm', True),
                                     ('current_openmm', 'openmm', False),
                                     ('current_gromacs', 'gromacs', False)):
        template = (fixtures / (engine + '.inp')).read_text()
        lines = [line for line in template.splitlines()
                 if not line.startswith(('engine =', 'gromacs_', 'integrator =', 'output_dir ='))]
        text = '\n'.join(lines).replace('system.top', 'argon.top').replace('system.gro',
                       'argon.gro' if engine == 'gromacs' else 'initial_state.xml')
        if not args.cpu:
            text = text.replace('platform = CPU', 'platform = GPU' if engine == 'gromacs' else 'platform = HIP')
        text += '\nfric_coeff = 0\noutput_dir = %s\n' % name
        if not reference:
            text += 'engine = %s\n' % engine
        if engine == 'gromacs':
            text += ('gromacs_mdp = production.mdp\ngromacs_threads = auto\n'
                     'gromacs_executable = %s\n' % args.gmx)
            if not args.cpu:
                text += 'gromacs_mdrun_args = -pme cpu -bonded cpu -update cpu\n'
        Path(name + '.inp').write_text(text + '\n')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(args.reference_source).resolve()) if reference else str(ROOT / 'src')
        if reference:
            probe = subprocess.run([sys.executable, '-c',
                'import json,sumd_openmm; print(json.dumps(dict(version=sumd_openmm.__version__, '
                'source=sumd_openmm.__file__)))'], env=env, text=True, capture_output=True, check=True)
            report['reference_runtime'] = json.loads(probe.stdout)
            assert report['reference_runtime']['version'] == '0.9.1'
        command = [sys.executable, '-m', 'sumd_openmm', name + '.inp', '--overwrite']
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with open('reference_091.log', 'a') as log:
            log.write('$ ' + ' '.join(command) + '\n' + result.stdout + '\n')
        report['commands'].append(dict(name=name, returncode=result.returncode))
        if result.returncode:
            report['status'] = 'failed'
            save()
            raise RuntimeError('%s failed: %s' % (name, result.stdout[-4000:]))
        save()
    for left, right in (('reference_091', 'current_openmm'), ('reference_091', 'current_gromacs'),
                         ('current_openmm', 'current_gromacs')):
        comparison = numeric_comparison(left, right)
        report['comparisons'][left + '_vs_' + right] = comparison
        if not comparison['passed']:
            report['status'] = 'failed'
            save()
            raise AssertionError(report['comparisons'])
        save()
    # The two current backends must publish identical column schemas.
    assert all(c['identical'] for c in report['comparisons']['current_openmm_vs_current_gromacs']['csv_columns'].values())
    report.update(status='completed', passed=True)
    save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
