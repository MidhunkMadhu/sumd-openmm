# Changelog

The third number of the version goes up with every push to GitHub; the
second marks a larger release. `sumd-openmm --version` prints the
installed version; install a given version with `@vX.Y.Z` in place of
`@main`.

## 1.0.0 (2026-11-07)

Added
- Interchangeable OpenMM and native GROMACS backends with shared CVs,
  supervision, DCD trajectories, energy columns and versioned state envelopes.
- GROMACS CPU allocation from Slurm, one GPU per walker, independent
  stochastic noise streams, native checkpoint branching and continuation.
- Native Amber cMD/GaMD adapter (pmemd.cuda, pmemd.hip) and optional
  dependencies. GaMD has not been run.
- Dardel reference comparisons against 0.9.1, binding audits, and native
  Amber/GROMACS examples following the supervisedmdamber RMSD workflow.

Fixed
- Membrane and anisotropic pressure logging handles vector pressure.
- Shared restart exports preserve whole-molecule periodic wrapping.
- External engines do not inherit the controller's MPI/Slurm rank identity.
- Amber starts: pmemd integrates a step even with nstlim = 0, so the input
  state is kept and only the energy of its first record is used.
  genvel = yes draws Maxwell-Boltzmann velocities instead.
- GROMACS windows keep the MDP's nstcalcenergy (reduced to divide the CV
  sample interval) instead of 1, which cost 3-9x of the GPU throughput.
- GROMACS energy rows include CMAP in E_dihedral.
- Amber energy rows put Urey-Bradley in E_angle and impropers in E_dihedral.
- GROMACS and Amber energy rows use the OpenMM number formats.
- GROMACS velocity reassignment no longer re-reads the topology.

Compatibility
- OpenMM remains the default engine. Legacy OpenMM State XML is readable;
  new accepted states use the common envelope with native restart data.
- The Dardel production installation is not upgraded by this local release.

## 0.9.1 (2026-11-05)

Changed
- Energy logs are written in the layout of an Amber `mdout`: the input
  file, numbered sections, a labelled block per record, averages,
  fluctuations and timings, in kcal/mol, Angstrom^3, g/mL and bar. This
  applies to each window's `.log`, `sumd_energy.log` and the new
  `check/sumd_energy.log`, and `mdout_analyzer.py` reads them.
- The table these logs held moves to `.csv` beside each `.log`, and
  `sumd_energy.csv` joins the final path. `--continue` still reads windows
  of version 0.9.0.
- `energy_terms` is on by default.

Added
- Instantaneous pressure from the barostat (`PRESS`, `Pressure_bar`) and
  wall time (`Wall_s`, ns/day) in the energy logs.

Fixed
- Topologies with a virtual site parmed cannot build can be read for
  selections, such as the type-3 lone pair CGenFF writes for a halogen.
  Such a site is read as a bond to its parent atom, and `progress.log`
  says so once per topology.
- Virtual sites in `[ virtual_sites2 ]`, `[ virtual_sites4 ]` and
  `[ virtual_sitesn ]` belong to the molecule of their parent atom. They
  were separate molecules, which split a selection containing one.

## 0.9.0 (2026-11-04)

Added
- Energy logs, like a GROMACS `md.log` or an Amber `mdout`: every window
  writes `windows/.../cCCCCC_wW.log` beside its DCD, with step, time,
  potential, kinetic and total energy, temperature, volume, density and
  the metrics.
- `sumd_energy.log` joins the windows of the final path, one row per
  frame of `sumd_traj.dcd`, and adds `Cycle`, `AcceptedStep` and `Walker`.
  `check/sumd_energy_log.csv` holds the same for a running job, and
  `run_summary.json` gives each column's mean and standard deviation.
- `log_stride` sets how often rows are written (default: once per DCD
  frame; 0 turns the logs off).
- `energy_terms = yes` adds the potential energy by term: `E_bond`,
  `E_angle`, `E_dihedral`, `E_vdw`, `E_elec_direct` or `E_nonbonded`,
  `E_elec_recip`, `E_other`.
- `log_stride` and `energy_terms` may change with `--continue`, and runs
  from earlier versions can be continued; their windows have no log.

## 0.8.0 (2026-09-29)

Changed
- A run whose `output_dir` holds an earlier run renames that folder to
  `<output_dir>_old00001` (or the first free number: `_old00002`, ...)
  and starts, instead of stopping with `output_dir exists`. The new
  `progress.log` names the renamed folder. `--overwrite` still deletes
  it. A run with an input file inside that folder stops with a message
  naming the input.

## 0.7.2 (2026-09-28)

Documentation
- README: "Install from source" explains the clone and editable install
  for changing the code; "Checking the installation" describes
  `--test`, `--version` and MDAnalysis.

## 0.7.1 (2026-09-28)

Documentation
- The README install section has one part each for CPU only, NVIDIA
  GPUs (CUDA), AMD GPUs (HIP) and several GPUs with MPI (mpi4py).

## 0.7.0 (2026-09-28)

Changed
- Cycle lines of `progress.log` are labelled fields instead of table
  columns: `Cycle=1  Window=4  CV1=31.41 A (-0.84)`, with
  `AcceptedStep=` and `Result=` for single-walker SuMD. Lines at the top
  name each CV (`CV1 = ligand_fitted_rmsd (A)`).

## 0.6.2 (2026-09-28)

- The 0.6.1 changes below (`check/` snapshots, partial results,
  `--continue`, CUDA and force-switch fixes) are on `main`. No other
  code changes.

## 0.6.1 (2026-09-28)

Added
- `check/`, replaced every 10th AcceptedStep while the run goes on, and
  at its end: the trajectory from the start to that AcceptedStep with every 10th frame
  (`check/sumd_traj_stride10.dcd`), its OpenMM State
  (`check/latest_accepted_step.xml`) and `check/latest.json`. The new
  `check_stride` key sets the thinning (`0` turns it off) and
  `check_every` how often it is written. `dcdtools.concat_dcds` takes a
  `stride`.
- A run stopped by an error, Ctrl-C or SIGTERM (a Slurm time limit)
  still writes `sumd_traj.dcd`, `final_state.xml`/`.rst7`,
  `final_path.txt` and `run_summary.json` for the steps accepted so far,
  marked `"partial": true`.
- `--continue` and `--extend N` carry a run on in its `output_dir` after
  `max_cycles`, convergence (with a new target), a crash, a time limit
  or a `STOP` file. The run saves `resume.json` every cycle; changes
  that would alter the meaning of the saved AcceptedSteps are refused.

Fixed
- A GPU platform that loads but cannot run, such as CUDA with a toolkit
  newer than the NVIDIA driver (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`), is
  detected with a small Context before use. The run uses another GPU
  platform if one works, and otherwise stops with the cause and the
  `cuda-version` to install; it never falls back to the CPU. `--test`
  reports the same.
- The install command pins `cuda-version` to the CUDA version the NVIDIA
  driver supports.
- `vdw = Force-switch` with `lj_lrc = yes` no longer aborts the process
  ("Long range correction did not converge"). The switched energy is now
  zero beyond `r_off`, which also removes a spurious 1-4 Lennard-Jones
  energy for pairs past the cutoff. CHARMM-GUI protocols with force
  switching and a dispersion correction hit this at production start.
- `p_type = anisotropic` works in production (Monte Carlo anisotropic
  barostat). CHARMM-GUI protocols with Amber `ntp = 2` or GROMACS
  `pcoupltype = anisotropic` failed after equilibration.
- `lj_lrc = no` removes the dispersion correction that Amber and GROMACS
  topologies get by default; leaving the key out keeps that default.
- `pcouple`, `lj_lrc` and `rest` accept any case and yes/no, true/false,
  on/off, 1/0; `pcouple = Yes` used to run without a barostat, and an
  invalid value is now an error.
- The barostat seed is set from the run's seed, so constant-pressure runs
  are reproducible from `random_seed`.
- The force switch keeps a non-periodic cutoff for non-periodic systems.
- CHARMM `toppar_file` lists resolve relative paths from the list's
  directory, as in CHARMM-GUI's `toppar.str`, falling back to the
  current directory.
- OpenMM XML systems with a membrane or anisotropic barostat are reported
  as constant pressure.

Changed
- A barostat chosen by MD_openmm's defaults (`pcouple` and `p_type` left
  out, giving a membrane barostat) is reported in the log.

## 0.6.0 (2026-09-28)

Added
- MIT License.

Changed
- Pressure coupling, CHARMM force switching, molecule rewrapping and
  CHARMM file reading are implemented in `sumd_openmm/omm_utils.py`;
  energies and forces agree with the previous implementation to within
  floating-point precision. Rewrapping also joins molecules split across
  several box edges.
- `rest = yes` is ignored with a warning for every force field.

## 0.5.0 (2026-09-28)

Changed
- `progress.log` prints the cycles as a table: time, cycle, walker kept
  and each metric with its change. The AcceptedStep and result columns
  appear only when cycles can be rejected (one walker, or
  `walker_acceptance = sumd`); notes mark a different starting
  AcceptedStep, a return after the retry limit, and convergence.
- The retry limit with `on_retry_exhaustion = step_back` is reported as
  `back to AcceptedStep N`.

## 0.4.0 (2026-09-27)

Added
- `windows.csv`: one row per window of every walker, kept or not, with
  its outcome, start and new AcceptedStep, retries used, final metric
  values, slope and score.
- `cv_samples.csv`: every metric sample of every window;
  `write_cv_samples = no` omits it.

Changed
- The saved points of the path are called AcceptedSteps:
  `accepted_steps.csv` (was `nodes.csv`; columns `accepted_step`,
  `parent_accepted_step`, `md_step`, `trajectory`, `restart_file`),
  `accepted_steps/accepted_step_NNNNNN.xml` (was `states/node_NNNNNN.xml`),
  and `final_accepted_step`, `best_accepted_step`, `n_accepted_steps` in
  `run_summary.json`. `sumd-inspect` still reads earlier runs.
- Progress lines read `from AcceptedStep 6 ... -> AcceptedStep 7`, and
  rejections `(1 of 8 retries from AcceptedStep 7 used)`.

## 0.3.1 (2026-09-27)

Changed
- Progress lines no longer list every walker's value; `windows.jsonl`
  keeps them.
- The start of `progress.log` describes each metric on one short line
  (selections, atom counts, residue range, initial value) instead of the
  full selection record, which stays in `run_summary.json`.

## 0.3.0 (2026-09-27)

Changed
- Each progress line shows the cycle out of `max_cycles`, the walker
  kept, every metric at the end of the kept window with its change, and
  the supervised metric reached by every walker.

## 0.2.0 (2026-09-27)

Added
- AMD GPUs through OpenMM's HIP platform; `platform = auto` picks the
  fastest platform and a GPU request never falls back to the CPU.
- `sumd-openmm --test`: checks the installation (platforms, devices,
  forces, mpi4py's MPI library) on a built-in system; `sumd-openmm
  run.inp --test` runs and audits two short cycles of the real system.
- `equilibration = charmm-gui`: runs the restrained equilibration of a
  CHARMM-GUI Amber or GROMACS folder in OpenMM before SuMD.
- `contacts` metric; `reference_a` and `reference_fit` masks for RMSD.
- `walker_acceptance`, and `retry_velocities = auto`, which keeps the
  selected walker's velocities in multiple-walker runs.
- `sumd-openmm[mpi]` install extra.
- User guide, troubleshooting guide, generic GPU job script, tested
  Dardel setup, ligand-binding example system.

Changed
- Multiple-walker runs continue from the best walker every cycle
  (Deganutti et al. 2025); `walker_acceptance = sumd` repeats a cycle
  without progress.
- A chain run ends at its latest state; `run_summary.json` also names
  the most advanced state.
- `vdw = Force-switch` also switches Amber and GROMACS topologies.
- Configuration errors and a missing mpi4py print one line.
- The output of `--dry-run` no longer blocks the following run.
- Keys without effect (`continuation`, `restraint_file`, `restraint_k`,
  `barostat_freq`, `pressure`) warn as unknown; unsupported `p_type`
  values stop with a message.

## 0.1.0

First release: SuMD and multiple-walker SuMD on OpenMM with the bundled
MD_openmm setup.
