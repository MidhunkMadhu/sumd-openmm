"""Native Amber cMD/GaMD backend, built from the supervisedmdamber workflow.

Simulation validation is intentionally pending. No Amber executable is
imported or run by importing this module. A selected backend performs a
zero-step energy/state initialization when a user actually starts a run.
"""
import copy
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import numpy as np

from .amber_io import read_restart, state_bundle, unpack_bundle
from .amber_io import read_mdout, read_trajectory
from .backend import read_state, write_state, wrapped_positions
from .config import ConfigError
from .dcdtools import DCDWriter
from .energylog import COLUMNS, ENERGY_TERMS, format_row
# Process helpers do not load or require a GROMACS executable.
from .gromacs import subprocess_environment, thread_count
from .selection import load_structure, _molecule_ids
from .parallel import visible_devices

KJ_PER_KCAL = 4.184
BOLTZMANN_KJ = 0.00831446261815324


def maxwell_boltzmann(masses, temperature, seed):
    """Velocities (nm/ps) at `temperature` with zero center-of-mass motion."""
    rng = np.random.default_rng(int(seed))
    velocity = np.zeros((len(masses), 3))
    real = masses > 0
    velocity[real] = rng.normal(size=(real.sum(), 3)) * np.sqrt(
        BOLTZMANN_KJ * temperature / masses[real, None])
    if real.sum() > 1:
        velocity[real] -= np.average(velocity[real], axis=0, weights=masses[real])
    return velocity
NATIVE_FORMAT = 'amber-restart-bundle-zip-v1'


class AmberSnap:
    def __init__(self, packet):
        self.state = packet
        self.step, self.epot = packet['step'], packet['epot']


class AmberEngine:
    name = 'amber'

    def __init__(self, cfg, scfg, seed, outdir, rank=0, device=None, log=print):
        try:
            import f90nml
            import netCDF4  # Ensures ParmEd has a supported NetCDF implementation.
        except ImportError as exc:
            raise ConfigError('Amber requires the optional dependencies: pip install "sumd-openmm[amber]"') from exc
        self.cfg, self.scfg, self.seed, self.device, self.log = cfg, scfg, int(seed), device, log
        if cfg.get('force_field', 'AMBER').upper() != 'AMBER':
            raise ConfigError('engine = amber requires an Amber/Chamber topology')
        for key in ('topology_file', 'coordinate_file', 'amber_mdin', 'amber_executable'):
            if not cfg.get(key):
                raise ConfigError('engine = amber requires %s' % key)
        for key in ('topology_file', 'coordinate_file', 'amber_mdin'):
            if not Path(cfg[key]).is_file():
                raise ConfigError('%s does not exist: %s' % (key, cfg[key]))
        self.topfile = str(Path(cfg['topology_file']).resolve())
        self.crdfile = str(Path(cfg['coordinate_file']).resolve())
        self.template = f90nml.read(cfg['amber_mdin'])
        self.control = self.template.get('cntrl')
        if self.control is None or not isinstance(self.control, dict):
            raise ConfigError('amber_mdin needs one &cntrl namelist')
        if int(self.control.get('imin', 0)) != 0:
            raise ConfigError('amber_mdin must describe dynamics, not minimization')
        if any(int(self.control.get(k, 0)) for k in ('icfe', 'ifsc', 'nmropt', 'numexchg')) or 'wt' in self.template:
            raise ConfigError('Amber free-energy, replica-exchange and time-dependent protocols are not supported')
        self.dt_ps = float(self.control.get('dt', .001))
        self.temperature_K = float(self.control.get('temp0', 300))
        if self.dt_ps <= 0 or self.temperature_K <= 0:
            raise ConfigError('Amber dt and temp0 must be positive')
        for key, actual in (('dt', self.dt_ps), ('temp', self.temperature_K)):
            if key in cfg and not np.isclose(float(cfg[key]), actual, rtol=0, atol=1e-10):
                raise ConfigError('%s differs from amber_mdin' % key)
        self.periodic = int(self.control.get('ntb', 1)) != 0
        self.gamd = scfg.method == 'GaMD'
        if self.gamd != (int(self.control.get('igamd', 0)) > 0):
            raise ConfigError('method and amber_mdin igamd disagree')
        gamd = None
        if self.gamd:
            path = cfg.get('amber_gamd_restart')
            if not path or not Path(path).is_file():
                raise ConfigError('GaMD requires amber_gamd_restart pointing to prepared gamd-restart.dat')
            if int(self.control.get('irest_gamd', 0)) != 1 or any(
                    int(self.control.get(k, 0)) for k in ('ntcmdprep', 'ntcmd', 'ntebprep', 'nteb')):
                raise ConfigError('GaMD must already be equilibrated: irest_gamd=1 and preparation/equilibration steps=0')
            gamd = Path(path).read_bytes()
            if not gamd:
                raise ConfigError('amber_gamd_restart is empty')
        self.reference = cfg.get('amber_reference')
        if self.reference:
            self.reference = str(Path(self.reference).resolve())
            if not Path(self.reference).is_file():
                raise ConfigError('amber_reference does not exist')
        if int(self.control.get('ntr', 0)) and not self.reference:
            raise ConfigError('Amber positional restraints require amber_reference')
        if int(self.control.get('ntr', 0)) and not self.control.get('restraintmask'):
            raise ConfigError('Amber positional restraints require restraintmask; legacy group cards are unsupported')
        if int(self.control.get('ibelly', 0)) and not self.control.get('bellymask'):
            raise ConfigError('Amber belly groups require bellymask; legacy group cards are unsupported')
        self.command = shlex.split(cfg['amber_executable'])
        if not self.command or shutil.which(self.command[0]) is None:
            raise ConfigError('Amber executable not found; load its module or set amber_executable')
        self.command[0] = shutil.which(self.command[0])
        self.args = shlex.split(cfg.get('amber_args', ''))
        if {'-i', '-o', '-p', '-c', '-r', '-x', '-v', '-inf', '-ref', '-gamd'}.intersection(self.args):
            raise ConfigError('amber_args overrides a SuMD-managed file argument')
        self.threads = thread_count(cfg, os.environ, key='amber_threads')
        self.env = subprocess_environment(os.environ, self.threads)
        visible = visible_devices(os.environ)
        if device is not None and (visible is None or len(visible) != 1):
            executable = ' '.join(self.command).lower()
            platform = cfg.get('platform', '').lower()
            if '.hip' in executable or platform == 'hip':
                self.env['HIP_VISIBLE_DEVICES'] = str(device)
            elif '.cuda' in executable or platform == 'cuda':
                cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
                devices = cuda.split(',') if cuda else None
                self.env['CUDA_VISIBLE_DEVICES'] = devices[int(device)] if devices else str(device)
        self.platform_name = 'Amber ' + Path(self.command[-1]).name
        self.work = Path(outdir).resolve() / 'engine_work' / ('rank_%03d' % rank)
        self.work.mkdir(parents=True, exist_ok=True)
        structure = load_structure(self.topfile, log)
        self.natoms = len(structure.atoms)
        self.mass = sum(atom.mass for atom in structure.atoms)
        ids = np.asarray(_molecule_ids(structure))
        self._molecules = [np.flatnonzero(ids == i) for i in np.unique(ids)]
        self._reseed = self._dynamics_seed = None
        self._window = 'initial'
        raw = read_restart(self.crdfile)
        if len(raw['pos']) != self.natoms:
            raise ConfigError('Amber topology and restart atom counts differ')
        raw_genvel = cfg.get('genvel', 'no').strip().lower()
        if raw_genvel not in ('yes', 'true', '1', 'no', 'false', '0'):
            raise ConfigError('Amber genvel must be yes or no')
        self.initial_genvel = raw_genvel in ('yes', 'true', '1')
        if raw['vel'] is None and not self.initial_genvel:
            raise ConfigError('Amber restart has no velocities; set genvel = yes to generate them')
        restart = Path(self.crdfile).read_bytes()
        if self.initial_genvel:
            # pmemd cannot generate velocities without integrating a step, so draw
            # Maxwell-Boltzmann velocities here and start from the input coordinates.
            masses = np.array([atom.mass for atom in structure.atoms], float)
            raw['vel'] = maxwell_boltzmann(masses, self.temperature_K, self.seed)
            from .backend import write_restart
            write_restart(self.work/'initial.rst7', dict(raw, step=0), self.periodic)
            restart = (self.work/'initial.rst7').read_bytes()
        raw.update(step=0, time=0.0, epot=0.0, params={}, engine=self.name,
                   native_format=NATIVE_FORMAT, native=state_bundle(restart, gamd))
        self._packet = raw
        self._packet['epot'] = self._initial_energy()
        log('Amber backend: %s; %d atoms; device %s; %d CPU thread(s); simulation validation pending'
            % (self.platform_name, self.natoms, device or 'scheduler/default', self.threads))

    def _call(self, args, mdout='window.mdout'):
        command = self.command + list(map(str, args))
        result = subprocess.run(command, cwd=self.work, env=self.env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with open(self.work/'commands.log', 'a') as handle:
            handle.write('$ ' + shlex.join(command) + '\n' + result.stdout + '\n')
        if result.returncode:
            mdout = self.work/mdout
            detail = mdout.read_text(errors='replace')[-5000:] if mdout.exists() else result.stdout[-5000:]
            raise RuntimeError('Amber failed (%d); see %s:\n%s' %
                               (result.returncode, self.work/'commands.log', detail))

    def _write_parent(self):
        bundle = unpack_bundle(self._packet['native'])
        (self.work/'parent.rst7').write_bytes(bundle['restart.rst7'])
        if self.gamd:
            if 'gamd-restart.dat' not in bundle:
                raise ValueError('Amber GaMD state lacks gamd-restart.dat')
            (self.work/'gamd-restart.dat').write_bytes(bundle['gamd-restart.dat'])

    def _initial_energy(self):
        """Potential energy (kJ/mol) of the input state, which itself is kept unchanged.

        pmemd integrates at least one step even with nstlim = 0. Its first
        record (NSTEP = 1 with irest = 1) holds the energy of the input
        coordinates; the advanced restart it writes is discarded.
        """
        self._write_parent()
        nml = copy.deepcopy(self.template)
        nml['cntrl'].update(nstlim=1, dt=self.dt_ps, ntpr=1, ntwx=0, ntwr=1, ntwv=0, ntwf=0,
                            ntxo=2, ioutfm=1, iwrap=0, irest=1, ntx=5, ig=int(self.seed))
        with open(self.work/'initial.mdin', 'w') as handle:
            handle.write('SuMD initial energy\n')
            nml.end_comma = True
            nml.write(handle)
        args = ['-O', '-i', 'initial.mdin', '-o', 'initial.mdout', '-p', self.topfile,
                '-c', 'parent.rst7', '-r', 'initial.out.rst7', '-inf', 'initial.mdinfo']
        if self.reference:
            args += ['-ref', self.reference]
        if self.gamd:
            args += ['-gamd', 'initial.gamd.log']
        self._call(args + self.args, 'initial.mdout')
        records = read_mdout(self.work/'initial.mdout')
        if not records or 'EPTOT' not in records[min(records)]:
            raise RuntimeError('Amber initial energy record is missing')
        epot = records[min(records)]['EPTOT']*KJ_PER_KCAL
        if not np.isfinite(epot):
            raise RuntimeError('Amber wrote a non-finite initial potential energy')
        for name in ('initial.out.rst7', 'initial.mdinfo'):
            if (self.work/name).exists():
                (self.work/name).unlink()
        return epot

    def _execute(self, steps, sample):
        parent = self._packet
        self._write_parent()
        nml = copy.deepcopy(self.template)
        control = nml['cntrl']
        generate = self._reseed is not None
        seed = self._dynamics_seed if self._dynamics_seed is not None else self.seed
        if generate and self._reseed is not None:
            seed = self._reseed
        control.update(nstlim=int(steps), dt=self.dt_ps, ntpr=int(sample), ntwx=int(sample),
                       ntwr=max(1, int(steps)), ntwv=0, ntwf=0, ntxo=2, ioutfm=1,
                       iwrap=0, irest=0 if generate else 1, ntx=1 if generate else 5,
                       ig=int(seed))
        if generate:
            control['tempi'] = self.temperature_K
        with open(self.work/'window.mdin', 'w') as handle:
            handle.write('SuMD native Amber window\n')
            nml.end_comma = True
            nml.write(handle)
        for name in ('window.rst7', 'window.nc', 'window.mdout', 'window.mdinfo', 'window.gamd.log'):
            path = self.work/name
            if path.exists():
                path.unlink()
        args = ['-O', '-i', 'window.mdin', '-o', 'window.mdout', '-p', self.topfile,
                '-c', 'parent.rst7', '-r', 'window.rst7', '-x', 'window.nc', '-inf', 'window.mdinfo']
        if self.reference:
            args += ['-ref', self.reference]
        if self.gamd:
            args += ['-gamd', 'window.gamd.log']
        self._call(args + self.args)
        records = read_mdout(self.work/'window.mdout')
        if steps not in records or not all(k in records[steps] for k in ('EPTOT', 'EKTOT', 'TEMP(K)')):
            raise RuntimeError('Amber final instantaneous energy record is missing')
        final = read_restart(self.work/'window.rst7')
        if final['vel'] is None or final['pos'].shape != (self.natoms, 3):
            raise RuntimeError('Amber final restart lacks complete coordinates and velocities')
        if self.periodic and (not np.isfinite(final['box']).all() or np.linalg.det(final['box']) <= 0):
            raise RuntimeError('Amber periodic restart lacks a valid box')
        native_time = final['time']
        final.update(step=parent['step'] + steps, time=parent['time'] + steps*self.dt_ps,
                     epot=records[steps]['EPTOT']*KJ_PER_KCAL,
                     params=dict(ig=int(seed), native_time_ps=native_time), engine=self.name,
                     native_format=NATIVE_FORMAT,
                     native=state_bundle((self.work/'window.rst7').read_bytes(),
                           (self.work/'gamd-restart.dat').read_bytes() if self.gamd else None))
        if not all(np.all(np.isfinite(final[key])) for key in ('pos', 'vel', 'box')):
            raise RuntimeError('Amber wrote non-finite state arrays')
        if not np.isfinite(final['epot']):
            raise RuntimeError('Amber wrote a non-finite potential energy')
        self._packet = final
        self._reseed = self._dynamics_seed = None
        archive = self.work/'native_windows'
        archive.mkdir(exist_ok=True)
        for suffix in ('mdin', 'mdout', 'gamd.log'):
            source = self.work/('window.' + suffix)
            if source.exists():
                shutil.copyfile(source, archive/(self._window + '.' + suffix))
        return records

    def prepare_window(self, cycle, walker):
        master = self.scfg.random_seed if self.scfg.random_seed is not None else self.seed
        stream = np.random.SeedSequence([int(master), int(cycle), int(walker), 0x414D4252])
        self._dynamics_seed = (self._reseed if self._reseed is not None else
                               int(stream.generate_state(1)[0] % (2**31 - 1)) + 1)
        self._window = 'c%05d_w%d' % (cycle, walker)
        return self._dynamics_seed

    def molecules(self):
        return self._molecules

    def snapshot(self):
        return AmberSnap(copy.deepcopy(self._packet))

    def packet(self, snap):
        return copy.deepcopy(snap.state)

    def time_ps(self, snap):
        return snap.state['time']

    def restore(self, snap, check=True):
        self._packet = self.packet(snap)
        self._reseed = self._dynamics_seed = None
        return None  # Energy has not been independently reevaluated.

    def restore_packet(self, packet, check=True):
        return self.restore(AmberSnap(packet), check)

    def reseed(self, seed):
        self._reseed = int(seed)

    def save_state(self, snap, path):
        packet = snap.state
        write_state(path, packet, self.name, packet['native'], NATIVE_FORMAT)

    def load_state(self, path, step, epot):
        packet = read_state(path)
        if packet['engine'] != self.name or packet['native_format'] != NATIVE_FORMAT:
            raise ValueError('not a supported Amber SuMD state')
        unpack_bundle(packet['native'])
        return AmberSnap(packet)

    def current_cvs(self, evaluator):
        p = self._packet
        positions = wrapped_positions(p['pos'], p['box'], self._molecules) if self.periodic else p['pos']
        return evaluator(positions*10, p['box']*10 if self.periodic else None)

    def run_window(self, n_samples, steps_per_sample, evaluator, dcd_path, stride,
                   log_path=None, log_stride=0):
        start_step, start_time = self._packet['step'], self._packet['time']
        begun = time.time()
        records = self._execute(n_samples*steps_per_sample, steps_per_sample)
        positions, native_times, boxes = read_trajectory(self.work/'window.nc')
        if len(positions) != n_samples or len(native_times) != n_samples:
            raise RuntimeError('Amber trajectory frame count differs from the requested window')
        if not np.allclose(native_times-native_times[0], np.arange(n_samples)*steps_per_sample*self.dt_ps,
                           rtol=0, atol=max(1e-6, self.dt_ps*.1)):
            raise RuntimeError('Amber trajectory sampling interval differs from cv_sample_ps')
        if self.periodic and boxes is None:
            raise RuntimeError('Amber periodic trajectory lacks boxes')
        values = []
        with open(dcd_path, 'wb') as output, open(log_path or os.devnull, 'w') as energy:
            writer = DCDWriter(output, self.natoms, self.dt_ps, start_step+steps_per_sample*stride,
                               steps_per_sample*stride, self.periodic)
            if log_path:
                energy.write(','.join(COLUMNS + list(ENERGY_TERMS) + list(evaluator.names)) + '\n')
            for index, coords in enumerate(positions, 1):
                box = boxes[index-1] if self.periodic else None
                coords = wrapped_positions(coords, box, self._molecules) if self.periodic else coords
                row = evaluator(coords*10, box*10 if self.periodic else None)
                values.append(row)
                if index % stride == 0:
                    writer.write(coords, box)
                if log_path and index % log_stride == 0:
                    local_step = index*steps_per_sample
                    if local_step not in records:
                        raise RuntimeError('Amber energy and coordinate sampling do not align')
                    energy.write(self._energy_row(records[local_step], start_step+local_step,
                                  start_time+local_step*self.dt_ps, box, row, time.time()-begun))
        return np.arange(1, n_samples+1)*steps_per_sample*self.dt_ps, np.asarray(values)

    def _energy_row(self, record, step, time_ps, box, metrics, wall):
        potential, kinetic = record['EPTOT']*KJ_PER_KCAL, record['EKTOT']*KJ_PER_KCAL
        volume = abs(np.linalg.det(box)) if box is not None else None
        density = self.mass*1.66053906660e-3/volume if volume else None
        terms = {}
        # Chamber (CHARMM) topologies report Urey-Bradley and impropers separately.
        groups = {'E_bond': ('BOND',), 'E_angle': ('ANGLE', 'UB'),
                  'E_dihedral': ('DIHED', 'IMP', 'CMAP'), 'E_vdw': ('VDWAALS', '1-4 NB')}
        if self.scfg.energy_terms:
            for name, native in groups.items():
                present = [record[key] for key in native if key in record]
                if present:
                    terms[name] = sum(present)*KJ_PER_KCAL
            # Amber EELEC contains reciprocal contributions. Do not mislabel it
            # as direct electrostatics; preserve the remaining energy as OTHER.
            terms['E_other'] = potential - sum(terms.values())
        return format_row(step, time_ps, potential, kinetic, record['TEMP(K)'],
                          record.get('PRESS'), volume, density, wall, terms, metrics)
