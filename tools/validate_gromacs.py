#!/usr/bin/env python3
"""Reproducible small-system validation. Run on a compute node for --gpu.

Runs both CLIs, checks their public outputs, tests GROMACS continuation,
rejection/reseeding and checkpoint branching. Produces validation.json.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from sumd_openmm.backend import read_state
from sumd_openmm.dcdtools import read_frames
from sumd_openmm.inspect_run import inspect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'validation'/'gromacs'))
from compare_outputs import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gmx', default='gmx')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--systems', nargs='+', choices=('system', 'argon'),
                    default=('system', 'argon'), help='fixtures to compare (argon exercises nonbonded GPU work)')
    ap.add_argument('--work-dir', default=str(ROOT/'validation'/'gromacs'/'results'))
    args = ap.parse_args()
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    fixtures = ROOT/'validation'/'gromacs'
    for name in ('system.top','system.gro','argon.top','argon.gro','production.mdp'):
        shutil.copyfile(fixtures/name, work/name)
    os.chdir(work)
    commands = []
    report={'gpu_requested':args.gpu,'comparisons':{},'commands':commands,
            'status':'running','passed':False}

    def save_report():
        Path('validation.json').write_text(json.dumps(report,indent=2)+'\n')

    save_report()

    def run(input_name, *extra):
        cmd = [sys.executable, '-m', 'sumd_openmm', input_name, *extra]
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        commands.append({'command': cmd, 'returncode': result.returncode})
        with open('validation.log','a') as fh:
            fh.write('$ '+' '.join(cmd)+'\n'+result.stdout+'\n')
        print(result.stdout, flush=True)
        if result.returncode:
            report['status']='failed'
            report['failed_command']=cmd
            save_report()
            raise RuntimeError('validation command failed: '+' '.join(cmd))
        save_report()

    def config(name, engine, system='system', cycles=2, extra=''):
        s=(fixtures/(engine+'.inp')).read_text()
        s=s.replace('system.top',system+'.top').replace('system.gro',system+'.gro')
        s=s.replace('max_cycles = 2','max_cycles = '+str(cycles))
        s=s.replace('output_dir = '+engine+'_output','output_dir = '+name)
        if args.gpu:
            s=s.replace('platform = CPU','platform = '+('HIP' if engine=='openmm' else 'GPU'))
        if engine=='gromacs':
            s+='\ngromacs_executable = '+args.gmx+'\n'
            if args.gpu:
                s+='gromacs_mdrun_args = -pme cpu -bonded cpu -update cpu\n'
        Path(name+'.inp').write_text(s+extra)
        return name+'.inp'

    for system in args.systems:
        for engine in ('openmm','gromacs'):
            name=engine+'_'+system
            run(config(name,engine,system),'--overwrite')
        report['comparisons'][system]=compare('openmm_'+system,'gromacs_'+system)
        if not report['comparisons'][system]['passed']:
            report['status']='failed'
        save_report()
        assert report['comparisons'][system]['passed'], report['comparisons'][system]
    # Restart/retry checks use the interacting argon system.
    if 'argon' not in args.systems:
        report['commands']=commands
        report['passed']=True
        report['status']='completed'
        save_report()
        print(json.dumps(report,indent=2))
        return
    # Continuing the saved checkpoint must equal an uninterrupted three-window run.
    run('gromacs_argon.inp','--extend','1')
    run(config('gromacs_uninterrupted','gromacs','argon',cycles=3),'--overwrite')
    continued=read_frames('gromacs_argon/sumd_traj.dcd')
    full=read_frames('gromacs_uninterrupted/sumd_traj.dcd')
    report['continuation_bit_identical']=bool(np.array_equal(continued,full))
    if not report['continuation_bit_identical']:
        report['status']='failed'
    save_report()
    assert report['continuation_bit_identical']
    # A constant supervised distance forces rejection, regardless of velocities.
    # For this deterministic fixture, keep must reproduce the rejected branch;
    # reassignment must produce another path from the same accepted parent.
    for retry in ('keep','reassign'):
        name='gromacs_retry_'+retry
        inp=config(name,'gromacs','argon',cycles=2)
        s=Path(inp).read_text().replace('metric_1_b = indices:2','metric_1_b = indices:0')
        s=s.replace('max_retries_per_parent = 0','max_retries_per_parent = 3')
        s=s.replace('retry_velocities = keep','retry_velocities = '+retry)
        s+='\nkeep_rejected_dcd = yes\nrequire_significant_slope = yes\n'
        if retry=='reassign':
            s+='\nwalkers = 2\nwalker_acceptance = sumd\n'
        Path(inp).write_text(s)
        run(inp,'--overwrite')
        audit=inspect(name)
        assert audit['final_traj_identical_to_path_windows'] in (True,None)
        attempts = [json.loads(line) for line in Path(name,'windows.jsonl').read_text().splitlines()]
        assert len(attempts) == (4 if retry == 'reassign' else 2)
        assert all(not row['accepted'] and row['start_accepted_step'] == 0 for row in attempts)
        first = read_frames(Path(name,'windows/rejected/c00001_w0.dcd'))
        second = read_frames(Path(name,'windows/rejected/c00002_w0.dcd'))
        assert bool(np.array_equal(first, second)) == (retry == 'keep')
        report[retry+'_audit']=audit
    report['commands']=commands
    report['passed']=True
    report['status']='completed'
    save_report()
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
