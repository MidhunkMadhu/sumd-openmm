#!/bin/bash -l
#SBATCH --job-name=sumd-091-bound-verify
#SBATCH --account=naiss2025-3-3-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=verify-091-%j.out
#SBATCH --error=verify-091-%j.err
set -euo pipefail
module load PDC/26.03 gromacs/2026.1-gpu-cpe2603-rocm7.0.2
VALIDATION_ROOT=/cfs/klemming/projects/supr/carlssonlab/midhun/apps/sumd-validation-20261107
VALIDATION_PYTHON="$VALIDATION_ROOT/venv/bin/python"
VALIDATION_SOURCE="$VALIDATION_ROOT/development"
REFERENCE_SOURCE="$VALIDATION_ROOT/release-0.9.1/src"
VERIFY_DIR="$VALIDATION_ROOT/verify_091_$SLURM_JOB_ID"
export VALIDATION_ROOT VALIDATION_SOURCE VERIFY_DIR
export PYTHONPATH="$VALIDATION_SOURCE/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export OPENMM_CPU_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
mkdir -p "$VERIFY_DIR"
cd "$VERIFY_DIR"
bound() {
    srun --exclusive --exact --nodes=1 --ntasks="$1" --cpus-per-task=8 --gpus-per-task=1 \
        --hint=nomultithread --cpu-bind=verbose,cores --gpu-bind=verbose,single:1 "${@:2}"
}
bound 8 "$VALIDATION_PYTHON" "$VALIDATION_SOURCE/tools/audit_slurm_binding.py" "$VERIFY_DIR/binding.json"
bound 1 "$VALIDATION_PYTHON" "$VALIDATION_SOURCE/tools/validate_reference_091.py" \
    --reference-source "$REFERENCE_SOURCE" --gmx gmx_mpi --work-dir "$VERIFY_DIR/deterministic"
"$VALIDATION_PYTHON" - <<'PY'
import os
from pathlib import Path
from sumd_openmm.config import read_key_value_file
root = Path(os.environ['VALIDATION_ROOT'])
work = Path(os.environ['VERIFY_DIR'])
base = read_key_value_file(str(root/'real-system/baseline.inp'))
for key in ('topology_file','coordinate_file','gmx_include'):
    base[key] = str(root/'real-system'/base[key])
base.update(parallel='mpi', mpi_mode='multi_gpu', walkers='8', walker_score='smscore',
            window_ps='0.04', max_cycles='2', check_stride='1')
for name, engine in [('reference_091','reference'), ('current_openmm','openmm'), ('current_gromacs','gromacs')]:
    cfg = dict(base, output_dir=name)
    if engine != 'reference':
        cfg['engine'] = engine
    if engine == 'gromacs':
        cfg.update(gromacs_threads='auto', gromacs_executable='gmx_mpi',
                   gromacs_mdp=str(root/'real-system/gpu_production.mdp'))
    (work/(name+'.inp')).write_text('\n'.join('%s = %s' % item for item in cfg.items())+'\n')
PY
for name in reference_091 current_openmm current_gromacs; do
    if [[ "$name" == reference_091 ]]; then
        export PYTHONPATH="$REFERENCE_SOURCE"
    else
        export PYTHONPATH="$VALIDATION_SOURCE/src"
    fi
    bound 8 "$VALIDATION_PYTHON" -m sumd_openmm "$name.inp"
    bound 8 "$VALIDATION_PYTHON" -m sumd_openmm "$name.inp" --continue --extend 1
done
export PYTHONPATH="$VALIDATION_SOURCE/src:$VALIDATION_SOURCE/tools"
"$VALIDATION_PYTHON" - <<'PY'
import json, re
from pathlib import Path
from validate_reference_091 import numeric_comparison, state_packet
from sumd_openmm.inspect_run import inspect
energy = abs(state_packet('reference_091/final_state.xml')['epot'])
comparison = numeric_comparison('reference_091','current_openmm', energy_tolerance=max(.001, energy*1e-6))
comparison['energy_tolerance_kJmol'] = max(.001, energy*1e-6)
assert comparison['passed'], comparison
gromacs = inspect('current_gromacs')
assert gromacs['windows_simulated'] == 24
assert gromacs['final_traj_identical_to_path_windows']
assert gromacs['max_dcd_vs_state_mismatch_A'] < 1e-4
threads = {}
for p in sorted(Path('current_gromacs/engine_work').glob('rank_*/commands.log')):
    counts = sorted(set(map(int, re.findall(r'Using\s+(\d+)\s+OpenMP thread',p.read_text()))))
    assert counts == [8], (p, counts)
    threads[p.parent.name] = counts
assert len(threads) == 8
windows = [json.loads(line) for line in Path('current_gromacs/windows.jsonl').read_text().splitlines()]
first = [row for row in windows if row['cycle'] == 1]
noise_seeds = [row['dynamics_seed'] for row in first]
assert len(first) == 8 and len(set(noise_seeds)) == 8
cv = [row['metric_final']['tm3_tm6_dror'] for row in first]
spread = max(cv) - min(cv)
assert spread > 1e-4, ('stochastic walkers differ only at roundoff level', cv)
report = dict(passed=True, binding=json.loads(Path('binding.json').read_text()),
              deterministic=json.loads(Path('deterministic/reference_091.json').read_text()),
              real_openmm_091=comparison, real_gromacs=gromacs, gromacs_thread_counts=threads,
              first_cycle_noise_seeds=noise_seeds, first_cycle_tm3_tm6_spread_A=spread)
Path('verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
