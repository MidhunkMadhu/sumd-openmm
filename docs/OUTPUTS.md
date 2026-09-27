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
| `windows/accepted/cCCCCC_wW.dcd` | Trajectory of each kept window (cycle C, walker W) |
| `windows/rejected/` | Trajectories of other windows, with `keep_rejected_dcd = yes` |
| `sumd_traj.dcd` | The kept windows of the final path, joined in order |
| `final_path.txt` | AcceptedSteps of the final path, with their metric values and trajectories |
| `final_state.xml`, `final_state.rst7` | Last structure of the final path, as OpenMM State and Amber restart |
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
