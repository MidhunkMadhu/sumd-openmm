# Native engine backends

`engine = openmm` is the default. `force_field = AMBER` alone still means
OpenMM reading an Amber topology; select `engine = amber` to run Amber.
The shared supervisor, selections, fitting, CVs, walker selection and
public output writers are used by all backends.

## GROMACS: tested on Dardel

Set `engine = gromacs`, provide a native `gromacs_mdp`, and select its
executable with `gromacs_executable`. The MDP controls the physics; SuMD
controls window length, sampling, restart handling and random streams.
`dt` and `temp`, when also specified in the SuMD input, must match the MDP.
The backend samples every `cv_sample_ps` and thins the public DCD using
`dcd_stride`. It retains native checkpoints in accepted-state envelopes.

[The Dardel Slurm example](../examples/gromacs_dardel.slurm) gives each
walker one GPU and eight physical CPU cores. See [parallel execution](PARALLEL.md).

[Dardel verification against 0.9.1](../validation/gromacs/reference_091_dardel.json)
records CPU/GPU bindings, numerical comparisons and continuation checks.
The matched 64-atom deterministic GPU benchmark produced bit-identical
saved coordinates across the 0.9.1 reference, current OpenMM and GROMACS;
the final GROMACS potential differed by about 1.4e-6 kJ/mol. This does not
establish bit-identical stochastic production paths across engines.

## Amber

The Amber adapter follows the native-window/restart workflow in
[supervisedmdamber](https://github.com/MidhunkMadhu/supervisedmdamber)
(reference commit `5c341725807cdc1469af94b24d21ba10cf7aa1ce`). It runs one
GPU pmemd per walker: `pmemd.cuda` on NVIDIA and `pmemd.hip` on AMD GPUs.
Classical MD is tested; GaMD is not.

pmemd integrates at least one step even with `nstlim = 0`. The run therefore
starts from the input restart unchanged and takes its initial potential
energy from pmemd's first record, which is evaluated at the input
coordinates. With `genvel = yes`, Maxwell-Boltzmann velocities are drawn at
`temp0` before the first window.

pmemd reports the energy of the coordinates entering a step. A window's
energy-log row at step n is therefore the energy at step n - 1, one step
behind its CV columns.

Install its optional Python readers/parser:

```bash
python -m pip install '.[amber]'
```

The Amber executable is installed separately. Choose it explicitly; for
example `amber_executable = pmemd.hip` for a compatible AMD build,
`pmemd.cuda` for NVIDIA, or `pmemd`/`sander` for CPU execution. The adapter
does not need a GROMACS executable. A command prefix may be included in
`amber_executable`; resource allocation remains the caller's responsibility.

| Setting | Meaning |
| --- | --- |
| `engine = amber` | Select the native Amber backend |
| `force_field = AMBER` | Amber/Chamber topology |
| `topology_file` | Native `.parm7`/`.prmtop` |
| `coordinate_file` | Prepared Amber ASCII or NetCDF restart |
| `amber_mdin` | Native production `mdin`, including `&cntrl` and optional `&ewald` |
| `amber_executable` | Executable or explicit command prefix; required |
| `amber_threads = auto` | Slurm per-task CPU budget, then OpenMP default, then one |
| `amber_reference` | Native positional-restraint reference, required with `ntr > 0` |
| `amber_args` | Additional flags; managed file arguments cannot be overridden |
| `amber_gamd_restart` | Prepared `gamd-restart.dat`, required for GaMD |

The native MDIN controls constraints, cutoff/switching, thermostat and
barostat. SuMD overrides `nstlim`, sampling/output intervals, native file
formats, `ig`, and restart/velocity-generation controls for each window.
Native coordinates are kept unwrapped (`iwrap=0`); the shared writers wrap
whole molecules for public DCD and restart exports. Amber energies are
converted from kcal/mol to the common kJ/mol tables.

Known bonded, angle, dihedral and van der Waals terms are exported. Amber
`EELEC` includes reciprocal contributions and is not mislabeled as direct
electrostatics: it and remaining potential terms are collected in `E_other`.
Unavailable common decompositions remain blank. Full native `mdout` and
resolved `mdin` files are retained under `engine_work/rank_*/native_windows/`.

The adapter expects fixed production protocols. Free-energy, replica-exchange
and time-dependent restraint/temperature protocols are rejected. Positional
and belly restraints must use namelist masks, not legacy group cards. Loading
the adapter for a real run includes a zero-step native initialization to
obtain its initial energy and restart state; this was not executed during
the build-only work.

### Classical MD and GaMD

For cMD, use `method = cMD` with `igamd` absent or zero. Start with a
prepared restart containing velocities and `genvel = no`, matching the
reference driver's `irest=1, ntx=5`. Use `retry_velocities = keep` to retain
the parent's velocities. Reassignment requests native Amber velocity
generation. Thermostat seeds are independent across walker/cycle attempts.

For GaMD, set `method = GaMD`, use a prepared native GaMD MDIN with
`igamd > 0`, `irest_gamd = 1`, and zero preparation/equilibration counts,
and provide `amber_gamd_restart`. Both the native restart and GaMD statistics
are embedded in each accepted-state bundle, including MPI transfers.
Continuing or branching restores the accepted parent's statistics.
`final_state.rst7` alone is a coordinate/velocity interchange file; resume
GaMD through `--continue` so that the statistics are retained.

### Match the reference RMSD supervision

The reference's five frame indices and frame-index regression are already
implemented by `slope_points = 5`. Use `rmsd_displacement` with separate
fit and ligand masks to reproduce its fit-then-measure workflow. This applies
to GROMACS too; CV evaluation does not depend on the MD executable.

See the [native Amber example](../examples/native_amber/run.inp) and the
[corresponding GROMACS example](../examples/native_gromacs/run.inp).
An [Amber Dardel submission template](../examples/amber_dardel.slurm) uses
the same per-walker CPU/GPU binding; it has not been submitted or run.
