# Outputs

An **AcceptedStep** is a saved point of the SuMD path: coordinates,
velocities and box at the end of a kept window. AcceptedStep 0 is the starting
structure; every kept window adds one. A **window** is one short
simulation of one walker, kept or not.

| File | Contents |
| --- | --- |
| `progress.log` | Timestamped run log: metric summary, one line per cycle, warnings |
| `accepted_steps.csv` | One row per AcceptedStep |
| `windows.csv` | One row per window of every walker, kept or not |
| `cv_samples.csv` | Every metric sample of every window (`write_cv_samples = no` omits it) |
| `windows.jsonl` | Full record of every window, one JSON object per line |
| `accepted_steps/accepted_step_NNNNNN.xml` | OpenMM State (coordinates, velocities, box) of each AcceptedStep; a restart point |
| `windows/accepted/cCCCCC_wW.dcd`, `.log` | Trajectory and energy log of each kept window (cycle C, walker W) |
| `windows/rejected/` | Trajectories and energy logs of other windows, with `keep_rejected_dcd = yes` |
| `sumd_traj.dcd` | The kept windows of the final path, joined in order |
| `sumd_energy.log` | Energy log of the final path, joined in the same order |
| `check/sumd_traj_strideN.dcd` | Updated every 10th AcceptedStep (`check_every`) and at the end: the path to it, every Nth frame (`check_stride`, default 10) |
| `check/sumd_energy_log.csv` | Energy log of that path, every row |
| `check/latest_accepted_step.xml` | That AcceptedStep's OpenMM State |
| `check/latest.json` | Which AcceptedStep `check/` holds, its path, frame count and metric values |
| `final_path.txt` | AcceptedSteps of the final path, with their metric values and trajectories |
| `final_state.xml`, `final_state.rst7` | Last structure of the final path, as OpenMM State and Amber restart |
| `resume.json` | State of the run after its last completed cycle, for `--continue` |
| `run_summary.json` | Input, resolved selections, seeds, platform, stop reason, final and best AcceptedStep |
| `ranks/rank_NNN.log` | Log of each MPI rank other than 0 |
| `FATAL_rank_NNN.txt` | Error details if an MPI rank aborts |

## accepted_steps.csv

| Column | Meaning |
| --- | --- |
| `accepted_step`, `parent_accepted_step` | This AcceptedStep and the AcceptedStep its window started from |
| `cycle`, `walker` | Cycle and walker that produced it |
| one column per metric | Metric value at this AcceptedStep |
| `time_ps`, `md_step` | Simulation time and MD integration step |
| `epot_kJmol` | Potential energy |
| `reason` | Why the window was kept |
| `trajectory`, `restart_file` | Its window trajectory and its saved State |
| `band`, `cell` | Stratified seeding only |

## windows.csv

| Column | Meaning |
| --- | --- |
| `cycle`, `walker` | The window |
| `start_accepted_step` | AcceptedStep it started from |
| `outcome` | `kept`, `converged`, `rejected` (the best window of a cycle that kept none) or `not kept` |
| `new_accepted_step` | AcceptedStep created from it, if kept |
| `retries_used` | Rejections from `start_accepted_step` so far |
| one column per metric | Value at the end of the window |
| `slope`, `score` | Trend of the supervised quantity and the walker score used for selection |
| `wallclock_s` | Wall time of the window |

## cv_samples.csv

`cycle`, `walker`, `t_ps` (time within the window) and one column per
metric, for every sample of every window: the whole sampled history,
kept or not, for plots of all attempts. It grows with windows × samples
per window; `write_cv_samples = no` omits it.

## windows.jsonl

Everything recorded for a window: the columns of `windows.csv`, the MPI
rank, host and GPU, seed and start mode, the energy change on restart,
the sample times and every metric series, continuous torsion series,
slope statistics and the stratification cell. Undefined values are JSON
`null`.

## A run that stops early

If the run stops with an error, Ctrl-C, or SIGTERM (as sent by Slurm at
the time limit, or by `scancel`), it still writes `sumd_traj.dcd`,
`final_state.xml`/`.rst7`, `final_path.txt`, `check/` and
`run_summary.json` for the newest AcceptedStep whose State was saved
(the most advanced one with stratified seeding), then exits with the
error. `run_summary.json` then has `"partial": true` and a `stop_reason`
such as `terminated (signal 15)` or `error: ...`. After a GPU error
`final_state.rst7` may be missing; the XML is still written. SIGKILL,
a node failure or a power cut leave nothing to do this, so `check/` and
the per-step files in `accepted_steps/` and `windows/accepted/` remain
the record. Under MPI, rank 0 acts on SIGTERM only when the current
batch returns, so when windows are long ask Slurm for an early warning with
`#SBATCH --signal=TERM@300` (SIGTERM to the job steps 300 s before the limit).

## Energy logs

Each window writes a CSV energy log, the counterpart of a GROMACS
`md.log` or an Amber `mdout`, next to its DCD. A row is written every
`log_stride`-th metric sample, by default once per DCD frame, so row N of
`sumd_energy.log` is frame N of `sumd_traj.dcd`, and the last row of a
window is the saved State of its AcceptedStep. `sumd_energy.log` holds
only the windows of the final path, in order: a kept window that a
`step_back` abandoned is left out, as it is from the trajectory.
`run_summary.json` gives the mean and standard deviation of each column
under `energy_log`.

| Column | Contents |
| --- | --- |
| `Step`, `Time_ps` | MD step and simulation time, continuous along the path |
| `Cycle`, `AcceptedStep`, `Walker` | The window the row belongs to (joined logs only) |
| `Epot_kJmol`, `Ekin_kJmol`, `Etot_kJmol` | Potential, kinetic and total energy |
| `Temp_K` | Instantaneous temperature, as in OpenMM's `StateDataReporter` |
| `Volume_nm3`, `Density_gmL` | Box volume and density; empty without periodic boundaries |
| one column per metric | Metric value at that sample, as in `cv_samples.csv` |

`energy_terms = yes` adds the potential energy by term, after
`Density_gmL`. The terms add up to `Epot_kJmol`.

| Column | Contents |
| --- | --- |
| `E_bond` | Bonds, including Urey-Bradley terms |
| `E_angle` | Angles |
| `E_dihedral` | Proper and improper dihedrals, and CMAP |
| `E_vdw` | Lennard-Jones, including 1-4 pairs, when it is held apart from electrostatics (`vdw = Force-switch`, NBFIX tables) |
| `E_elec_direct` | Real-space electrostatics, including 1-4 pairs, with `vdw = Force-switch` |
| `E_nonbonded` | Real-space electrostatics and Lennard-Jones together, in place of `E_elec_direct`, when OpenMM holds them in one force |
| `E_elec_recip` | Reciprocal-space electrostatics (PME, Ewald) |
| `E_nonbonded_recip` | Reciprocal-space electrostatics and Lennard-Jones, with LJPME |
| `E_other` | Any other force |

A column appears only if the system has that term. Each row costs extra
energy evaluations, one for the totals and one per term, so a small
`log_stride` slows a run; `log_stride = 0` turns the logs off.

## check/

A look at a run that is still going. Every `check_every`-th AcceptedStep
(10, 20, 30, ... by default) and once more when the run ends, `check/`
is replaced with the trajectory from the starting structure to that
AcceptedStep, keeping every `check_stride`-th frame (10 by default, so a
tenth of the data), and that AcceptedStep's State. The trajectory is
`sumd_traj.dcd` of that path sliced `[::check_stride]`; at the end of a
chain run it is the final trajectory thinned. Load it with the
topology, e.g. `vmd system.parm7 check/sumd_traj_stride10.dcd`; the XML
is a restart point like the files in `accepted_steps/`. Both are written
to temporary names and renamed, so a reader never sees a half-written
file. With stratified seeding the newest AcceptedStep is not always the
most advanced; `latest.json` says which one it is. `check_stride = 1`
keeps every frame, `check_every = 1` updates it after every AcceptedStep,
and `check_stride = 0` turns `check/` off. Writing it
rereads the path's frames each time, so for long paths of large systems
a larger stride keeps it cheap. A failure to write it is logged as a
warning and the run continues.

A run that stops early can be continued with `--continue`
([user guide](USER_GUIDE.md#continuing-a-run)); `run_summary.json`
then lists each continuation under `continuations`.

## Checks

`sumd-inspect OUTPUT_DIR` checks that `sumd_traj.dcd` is exactly the
kept windows of the final path, that the last frame of every kept window
equals its saved AcceptedStep, the energy changes on restart, and summarizes the
metric values reached.

## Equilibration

With `equilibration = charmm-gui`, `equilibration_dir` (outside
`output_dir`, so `--overwrite` keeps it) holds `equilibration.log`,
`protocol.json`, per-stage `.log`, `.dcd` and `.xml` files and
`equilibrated.xml`/`.rst7`; `run_summary.json` records it under
`equilibration`. See [equilibration](EQUILIBRATION.md).
