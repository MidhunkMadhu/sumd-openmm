# Changelog

Versions follow [semantic versioning](https://semver.org): the third
number changes for bug fixes, the second for new features or changed
behaviour. `sumd-openmm --version` prints the installed version; install
a given version with `@vX.Y.Z` in place of `@main`.

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
