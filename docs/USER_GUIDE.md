# User guide

This guide explains how a sumd-openmm run proceeds and every key of its
input file. [Configuration](CONFIG_REFERENCE.md) has the same keys as
compact tables; [troubleshooting](TROUBLESHOOTING.md) covers installation
and cluster problems.

## How a run proceeds

1. The system is built in OpenMM from the input files, and every
   collective variable (metric) is evaluated once on the starting
   structure.
2. A **window** of `window_ps` of ordinary molecular dynamics runs. Every
   `cv_sample_ps` the metrics are measured.
3. The measurements of the supervised metric are fitted to a straight
   line. A slope in the requested direction (or a value already past
   the target) accepts the window: its final state becomes the start of
   the next window. Otherwise the window is rejected and repeated from
   the same state with new velocities.
4. With several **walkers**, every cycle runs that many windows from the
   same state and continues from the best one.
5. The run ends when the supervised metric reaches its target, after
   `max_cycles` cycles, or when a file named `STOP` appears in
   `output_dir`.

The accepted windows, joined in order, form the trajectory
`sumd_traj.dcd`. Every accepted state is saved, so any point of the path
can seed further simulations.

## Reading the progress log

`output_dir/progress.log` (also printed to the screen) has one line per
cycle:

```
cycle 7/100 from node 6: walker w3 of 8 accepted -> node 7 | ligand_rmsd 28.41 A (-0.60) | ligand_rmsd by walker: w0=29.10 w1=28.95 w2=29.40 w3=28.41* ...
```

| Part | Meaning |
| --- | --- |
| `cycle 7/100` | Cycle number and `max_cycles` |
| `from node 6` | Saved state the windows started from |
| `walker w3 of 8 accepted -> node 7` | Walker whose window was kept (`w3` matches the `_w3` window files and the `walker` column of `nodes.csv`) and the new state it produced. `converged` when the target is reached; `rejected` with the retry count when no window is kept |
| `ligand_rmsd 28.41 A (-0.60)` | Every metric at the end of the kept window, and its change from the starting state |
| `by walker: … w3=28.41*` | The first supervised metric at the end of every walker's window; `*` marks the one kept |

With several walkers and the default `walker_acceptance = always`, every
cycle is accepted: the best walker is always continued, even when the
metric moved the wrong way. Progress shows in the metric values.
`nodes.csv` holds the same values for every saved state.

## Workflow

```bash
sumd-openmm run.inp --dry-run    # check selections and initial metric values
sumd-openmm run.inp --test       # two short cycles, audited, with speed
sumd-openmm run.inp              # the run
sumd-inspect sumd_run            # audit the finished run
```

| Option | Effect |
| --- | --- |
| `--dry-run` | Builds the system and reports what every selection matched and each metric's initial value; no dynamics. A following run replaces its output |
| `--test` | Two cycles of ten samples in `<output_dir>_test`, then checks of the saved states and trajectories and the simulation speed. Without an input file, checks the OpenMM installation on a built-in system |
| `--overwrite` | Replaces an existing `output_dir` |
| `--version` | Prints the version |

To stop a running job cleanly, create an empty file named `STOP` in
`output_dir`; the run finishes the current cycle and writes its final
trajectory and state.

## The input file

One `key = value` per line; `#` starts a comment. Keys are
case-sensitive. An unknown key produces a warning, so a misspelt key
does not pass unnoticed. Units: MD settings use nanometres, picoseconds,
kelvin and bar; metrics use ångström and degrees.

## System

| Key | Default | Meaning |
| --- | --- | --- |
| `force_field` | `AMBER` | Format of the topology: `AMBER` (prmtop/parm7), `CHARMM` (psf), `GROMACS` (top), or `OPENMM_XML` (a serialized OpenMM System) |
| `topology_file` | required | The topology: `.parm7`/`.prmtop`, `.psf`, `.top`, or for `OPENMM_XML` a PDB or PDBx file |
| `coordinate_file` | required (optional for `OPENMM_XML`) | Starting coordinates, recognized by extension: Amber `.rst7`, `.rst`, `.inpcrd`, `.crd`, `.ncrst`; GROMACS `.gro`; CHARMM `.cor` or `.pdb`; OpenMM State `.xml`; OpenMM checkpoint `.chk`. Should be an equilibrated structure |
| `toppar_file` | none | CHARMM only: a stream file listing the parameter files |
| `gmx_include` | `toppar` | GROMACS only: folder searched for `#include` files |
| `system_xml` | none | `OPENMM_XML` only: the serialized System |
| `genvel` | `yes` | `yes` draws new velocities at `temp`; `no` keeps the velocities stored in `coordinate_file` (required to continue an equilibrated run; the file must contain velocities) |
| `rewrap_coordinates` | `yes` | Places every molecule whole inside the periodic box before the run |
| `reset_step_and_time` | `no` | `yes` starts step and time counters at zero |

## Dynamics

| Key | Default | Meaning |
| --- | --- | --- |
| `dt` | required (`0.002` for `OPENMM_XML`) | Time step in ps. `0.002` with `cons = HBonds`; `0.004` needs hydrogen mass repartitioning in the topology |
| `temp` | `310` (`300` for `OPENMM_XML`) | Temperature in K (Langevin thermostat) |
| `fric_coeff` | `1` | Langevin friction in ps⁻¹. Amber's `gamma_ln` |
| `cons` | `HBonds` | Constrained bonds: `HBonds`, `AllBonds`, `HAngles` or `None`. Water is always rigid |
| `platform` | `CUDA` | `auto` (fastest available), `CUDA` (NVIDIA), `HIP` (AMD), `OpenCL`, `CPU`, `Reference`. A GPU request never falls back to the CPU |
| `precision` | `single` | GPU arithmetic: `single`, `mixed` or `double`. Also accepted as `cuda_precision` |

## Nonbonded interactions

| Key | Default | Meaning |
| --- | --- | --- |
| `coulomb` | `PME` | Electrostatics: `PME`, `LJPME`, `Ewald`, `CutoffPeriodic`, `CutoffNonPeriodic`, `NoCutoff` |
| `ewald_Tol` | `0.0005` | Relative error tolerance of PME/Ewald |
| `r_off` | `1.2` | Nonbonded cutoff in nm (Amber `cut`/10) |
| `vdw` | `Force-switch` | Lennard-Jones treatment. `Force-switch`: forces switched to zero between `r_on` and `r_off`, required by CHARMM force fields (Amber `fswitch`); applied to CHARMM topologies by default and to Amber and GROMACS topologies when written in the input. `Switch`: potential switching. `LJPME`: Lennard-Jones by PME. Any other value: plain cutoff at `r_off` |
| `r_on` | `1.0` | Start of switching in nm (Amber `fswitch`/10) |
| `lj_lrc` | `no` | `yes` adds the long-range dispersion correction |
| `e14scale` | `1.0` | Scale factor for 1-4 electrostatic interactions |

## Pressure

| Key | Default | Meaning |
| --- | --- | --- |
| `pcouple` | `yes` | `yes` adds a Monte Carlo barostat (constant pressure); `no` keeps the volume constant. Set it explicitly: the default is constant pressure |
| `p_type` | `membrane` | `membrane`: box area and height scaled separately, for bilayers; `isotropic`: all box edges scaled together, for soluble systems |
| `p_ref` | `1.0` | Pressure in bar |
| `p_XYMode` | `XYIsotropic` | Membrane only: `XYIsotropic` or `XYAnisotropic` scaling of the box area |
| `p_ZMode` | `ZFree` | Membrane only: `ZFree`, `ZFixed` or `ConstantVolume` |
| `p_tens` | `0` | Membrane only: surface tension in dyn/cm |
| `p_freq` | `100` | Steps between barostat moves |

## Keys read but not used

`rest` (restraints, which change the Hamiltonian; SuMD windows must be
unbiased), `nstep`, `nstout` and `nstdcd` belong to the MD_openmm
production driver and can stay in a shared input file; the window
settings below replace them.

## Equilibration before SuMD

| Key | Default | Meaning |
| --- | --- | --- |
| `equilibration` | `none` | `none`: `coordinate_file` is already equilibrated, by any protocol. `charmm-gui`: run the minimization and restrained equilibration stages of a CHARMM-GUI folder in OpenMM first; `force_field`, `topology_file` and `coordinate_file` then come from the folder |
| `charmm_gui_dir` | none | The CHARMM-GUI download, or its `amber/` or `gromacs/` folder |
| `charmm_gui_format` | `auto` | `amber` or `gromacs` when the download holds both |
| `equilibration_dir` | `equilibration` | Stage logs, trajectories and states; a finished equilibration here is reused by later runs |

See [equilibration](EQUILIBRATION.md).

## Windows and acceptance

| Key | Default | Meaning |
| --- | --- | --- |
| `window_ps` | `20` | Length of one window in ps |
| `cv_sample_ps` | `1` | Interval between metric measurements in ps; a multiple of `dt`, at least three per window. Every measurement copies coordinates off the GPU, so very short intervals slow large systems |
| `max_cycles` | `500` | Maximum number of cycles (a cycle is one window, or one batch of walkers) |
| `slope_points` | `all` | Points of the slope fit: `all` samples, or `5` evenly spaced samples |
| `require_significant_slope` | `no` | `yes` accepts a window only if its slope exceeds twice its standard error |
| `retry_velocities` | `auto` | Velocities for a repeated window: `auto` draws new ones for single-walker SuMD and keeps the selected walker's for multiple walkers; `reassign` always draws new ones; `keep` never does |
| `max_retries_per_parent` | `8` | Rejections allowed from one state before `on_retry_exhaustion` applies |
| `on_retry_exhaustion` | `accept_best` | `accept_best`: accept the best rejected attempt and continue from it; `step_back`: return to the previous accepted state |
| `random_seed` | random | Seed of velocities and thermostat noise, recorded in `run_summary.json`. GPU arithmetic is not bit-for-bit reproducible, so reruns agree statistically, not exactly |
| `method` | `cMD` | Conventional MD; the only method available |

## Metrics

Metrics are numbered `metric_1_*`, `metric_2_*`, … without gaps, in any
number. Each needs a type, its selections and a role.

| Key | Meaning |
| --- | --- |
| `metric_N_name` | Name used in logs and tables (letters, digits, underscores; default `metric_N`) |
| `metric_N_type` | `distance`, `distance_axis`, `mindist`, `contacts`, `angle`, `dihedral`, `rmsd`, `rmsd_displacement` |
| `metric_N_a`, `_b`, `_c`, `_d` | Atom selections; which ones depends on the type |
| `metric_N_role` | `supervise` (decides acceptance), `stratify` (spreads restarts, with `seeding = stratified`) or `monitor` (recorded only; default) |

| Type | Selections | Value |
| --- | --- | --- |
| `distance` | `a`, `b` | Distance in Å between the centres of geometry of `a` and `b` |
| `distance_axis` | `a`, `b`; `axis` | Component of that distance along `x`, `y`, `z` or a vector `ax ay az`; `signed = no` gives its absolute value |
| `mindist` | `a`, `b` | Shortest distance between any atom of `a` and any atom of `b` |
| `contacts` | `a`, `b`; `cutoff` | Number of atom pairs, one from each selection, closer than `cutoff` Å (default 4) |
| `angle` | `a`, `b`, `c`, one atom each | Angle in degrees at `b` |
| `dihedral` | `a`–`d`, one atom each | Torsion in degrees (−180 to 180); when supervised to increase or decrease it is followed continuously across ±180°, so full rotations count |
| `rmsd` | `a`; `reference` | RMSD in Å of `a` to the reference after superposing `a` itself (shape change) |
| `rmsd_displacement` | `a`, `fit`; `reference` | RMSD of `a` after superposing `fit` on the reference: for a ligand, `fit` is the receptor and the value measures the distance from the bound pose |

Further keys of a metric:

| Key | Meaning |
| --- | --- |
| `metric_N_reference` | Structure file (PDB, or anything ParmEd reads) with the target coordinates for RMSD types; atom names must correspond |
| `metric_N_reference_a`, `metric_N_reference_fit` | Selections applied to the reference instead of `a` and `fit`, when the reference numbers residues differently |
| `metric_N_direction` | Supervised metrics: `decrease`, `increase`, or `toward` (drive towards `target` from either side) |
| `metric_N_target` | Supervised metrics: the value that ends the run (or the stage) when crossed |
| `metric_N_tolerance` | `toward` only: how close to `target` counts as reached |
| `metric_N_target_delta` | Dihedrals only, instead of `target`: a change relative to the starting value, e.g. `180` for a full flip |
| `metric_N_bins` | Stratified metrics: bin edges (`5 10 15`) or `quantiles:N` |
| `metric_N_weight`, `metric_N_mu`, `metric_N_sigma` | `supervision = combined`: weight, and the mean and standard deviation used to standardize the metric (estimated from the first window when omitted) |
| `metric_N_expect_a` (`_b`, `_c`, `_d`) | Residue names the selection must match, e.g. `LIG` or `ASP/GLU`; the run stops if the selection resolves to anything else |

`selection_syntax` sets the language of all selections: `cpptraj`
(default; Amber masks such as `:45@CA`, `:LIG&!@H=`) or `vmd`
(MDAnalysis, such as `resid 45 and name CA`). A prefix changes it for one
selection: `cpptraj:`, `vmd:`, `indices:` (zero-based atom indices,
`0-9,15`) or `file:` (a file of such indices). A selection of several
atoms stands for its centre of geometry, except in `mindist` and
`contacts`.

## Supervising several metrics

| Key | Default | Meaning |
| --- | --- | --- |
| `supervision` | `single` | `single`: one supervised metric. `combined`: all supervised metrics summed into one standardized score. `multistep`: one metric at a time, in the order of `stages` |
| `stages` | none | `multistep` only: metric numbers in order, e.g. `2, 1` |

## Multiple walkers

| Key | Default | Meaning |
| --- | --- | --- |
| `walkers` | `1` | Windows run from the same state each cycle; `auto` uses one per MPI rank |
| `walker_score` | `slope` | How the best walker is chosen: `slope`; `smscore` (one metric; √(last value × mean)); `dmscore` (two or more metrics; summed relative change) |
| `walker_acceptance` | `always` | `always`: every cycle continues from the best walker (mwSuMD). `sumd`: with `walker_score = slope`, the cycle is repeated when even the best walker made no progress |

## Stratified seeding

| Key | Default | Meaning |
| --- | --- | --- |
| `seeding` | `chain` | `chain`: each cycle starts from the last accepted state. `stratified`: each cycle starts from a saved state chosen to cover bins of the stratified metrics near the most advanced progress |
| `band_width` | `1.0` | Width of the progress bands, in units of the supervised metric |
| `pool_per_cell` | `20` | States kept per band and bin |
| `frontier_bands` | `1` | Bands behind the most advanced one that may still be chosen |

## Parallel runs

| Key | Default | Meaning |
| --- | --- | --- |
| `parallel` | `serial` | `serial`: walkers one after another on one GPU. `mpi`: walkers on several GPUs, one MPI rank each (needs mpi4py) |
| `mpi_mode` | `multi_node` | `multi_gpu`: several ranks per node, one GPU each. `multi_node`: one rank per node |
| `gpu_devices` | `auto` | GPU for each rank on a node: `auto`, or a list such as `0,1,2,3` |

See [parallel use](PARALLEL.md).

## Output

| Key | Default | Meaning |
| --- | --- | --- |
| `output_dir` | `sumd_run` | Folder of all results |
| `dcd_stride` | `1` | Trajectory frames kept: every Nth metric sample (must divide the samples per window) |
| `keep_rejected_dcd` | `no` | `yes` keeps trajectories of rejected windows |
| `restore_check` | `yes` | Compares the energy of each restored state with the saved value |

The files are described in [outputs](OUTPUTS.md).
