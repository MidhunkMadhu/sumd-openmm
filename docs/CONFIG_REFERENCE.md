# Configuration reference

The input file uses `key = value` lines with `#` comments. Unrecognized
keys produce a warning. Selection masks and targets have no defaults.

## Simulation setup

The bundled MD_openmm setup handles Amber, CHARMM and GROMACS force fields.
Its `force_field`, `topology_file`, `coordinate_file`, `toppar_file`,
`gmx_include`, `dt`, `temp`, `fric_coeff`, `pcouple`, `p_type`,
`cons`, `coulomb`, `vdw`, `r_on`, `r_off`, `platform`,
`cuda_precision`, `genvel`, `continuation`, `rewrap_coordinates`,
`reset_step_and_time`, `lj_lrc` and `e14scale` keep their MD_openmm
meaning; the [user guide](USER_GUIDE.md) explains each with its default.
See also [setup](SETUP_LAYER.md).

`platform` is `auto`, `CUDA`, `HIP`, `OpenCL`, `CPU` or `Reference`.
`auto` takes the fastest available. A GPU platform that is not available
is replaced by another GPU platform (CUDA → HIP on AMD nodes), never by
the CPU; without a GPU platform the run stops. `precision` (`single`,
`mixed`, `double`; default `single`; also accepted as `cuda_precision`)
applies to every GPU platform.

`vdw = Force-switch` switches Lennard-Jones forces to zero between `r_on`
and `r_off`, as CHARMM force fields require and as Amber's `fswitch`
does. It applies to CHARMM topologies by default and to Amber and GROMACS
topologies when written in the input; otherwise Lennard-Jones is
truncated at `r_off`.

`force_field = OPENMM_XML` loads `system_xml` (an XmlSerializer System)
and `topology_file` (PDB or PDBx with initial positions).
`coordinate_file` optionally provides an OpenMM State XML, checkpoint,
PDB or PDBx. The System's existing barostat is used.

## Equilibration

| Key | Type; default | Meaning |
| --- | --- | --- |
| `equilibration` | none/charmm-gui; none | `none`: `coordinate_file` is already equilibrated. `charmm-gui`: run a CHARMM-GUI protocol first |
| `charmm_gui_dir` | path | CHARMM-GUI download, or its `amber/` or `gromacs/` folder |
| `charmm_gui_format` | auto/amber/gromacs; auto | Which folder to use when the download has both |
| `equilibration_dir` | path; `equilibration` | Stage outputs and `equilibrated.xml`; reused on later runs |

See [equilibration](EQUILIBRATION.md).

## Metrics

Declare consecutive groups `metric_1_*`, `metric_2_*`, and so on.
There is no limit on the count. Missing numbers and unknown metric
subkeys are errors.

| Key | Type; default | Meaning |
| --- | --- | --- |
| `metric_N_name` | string; `metric_N` | Unique log and CSV column name |
| `metric_N_type` | required choice | `distance`, `distance_axis`, `mindist`, `contacts`, `angle`, `dihedral`, `rmsd`, `rmsd_displacement` |
| `metric_N_a/b/c/d` | selections; type dependent | Atom selections; `angle` and `dihedral` require exactly one atom in each |
| `metric_N_fit` | selection; required for `rmsd_displacement` | Group fitted to the reference |
| `metric_N_reference` | path; required for RMSD types | Coordinate structure with corresponding atom names |
| `metric_N_reference_a` | selection; `metric_N_a` | Measured atoms in the reference, when its numbering differs |
| `metric_N_reference_fit` | selection; `metric_N_fit` | Fitted atoms in the reference, when its numbering differs |
| `metric_N_axis` | `x/y/z` or three numbers | Required for `distance_axis`; normalized internally |
| `metric_N_signed` | yes/no; yes | Preserve the axis component's sign |
| `metric_N_cutoff` | Å; 4.0 | `contacts`: atom pairs of `a` and `b` closer than this are counted |
| `metric_N_role` | supervise/stratify/monitor; monitor | Acceptance, seed selection or logging only |
| `metric_N_direction` | increase/decrease/toward; required if supervised | Requested change |
| `metric_N_target` | number; required if supervised | Strictly inside when crossed |
| `metric_N_target_delta` | number; optional | Increasing/decreasing torsion target relative to the initial value |
| `metric_N_tolerance` | positive number; required for toward | Allowed absolute deviation from target |
| `metric_N_bins` | edges or `quantiles:N`; required if stratifying | Stratification boundaries |
| `metric_N_weight` | number; 1 | Combined-score weight |
| `metric_N_mu/sigma` | numbers; from first window | Score standardization; sigma must be positive |
| `metric_N_expect_a/b/c/d` | residue names separated by `/`; absent | Assert the selection resolves to these names |

`selection_syntax = cpptraj | vmd` defaults to `cpptraj`. Selection prefixes
`cpptraj:`, `vmd:`, `indices:` and `file:` override it. Index lists
are zero based and can include inclusive ranges such as `10-25,40`.
Selections resolve once at startup. A many-atom distance selection uses
its centre of geometry. All geometric distances use angstroms.

## Supervision and seeding

| Key | Type; default | Constraint |
| --- | --- | --- |
| `supervision` | single/combined/multistep; single | One, at least two, or at least two supervised metrics respectively |
| `stages` | comma-separated metric numbers; absent | Required for multistep; each supervised metric once |
| `seeding` | chain/stratified; chain | Stratified needs at least one metric with `role = stratify` and bins |
| `band_width` | positive float; 1.0 | Frontier band width in the supervised progress quantity |
| `pool_per_cell` | positive integer; 20 | Live states retained in each cell |
| `frontier_bands` | integer; 1 | Additional bands allowed around best reached band |
| `walkers` | integer or auto; 1 | Auto means one per MPI rank |
| `walker_score` | slope/smscore/dmscore; slope | Score modes require multiple walkers; dmscore needs at least two supervised metrics |
| `walker_acceptance` | always/sumd; always | With chain seeding and several walkers, always continue from the best walker (mwSuMD); `sumd` repeats a `slope` batch whose best walker made no progress |

## Window control and output

| Key | Type; default | Constraint |
| --- | --- | --- |
| `window_ps` | float; 20 | Positive; divisible by `cv_sample_ps` |
| `cv_sample_ps` | float; 1 | Positive; multiple of `dt`; at least three samples per window |
| `max_cycles` | integer; 500 | Maximum number of window batches |
| `max_retries_per_parent` | integer; 8 | Nonnegative |
| `on_retry_exhaustion` | accept_best/step_back; accept_best | Behavior after chain retries |
| `retry_velocities` | auto/reassign/keep; auto | auto reassigns single-walker retries (SuMD) and keeps the selected walker's velocities for mwSuMD batches; reassign uses a logged seed |
| `require_significant_slope` | yes/no; no | Progress acceptance requires slope exceeding twice its standard error |
| `slope_points` | all/5; all | Five-point estimate requires at least eight samples |
| `random_seed` | integer; random | Seeds integrators and retries |
| `method` | cMD; cMD | GaMD is refused without a GaMD integrator |
| `output_dir` | path; `sumd_run` | Existing nonempty output requires `--overwrite` |
| `dcd_stride` | positive integer; 1 | Save every Nth sampled frame |
| `keep_rejected_dcd` | yes/no; no | Retain rejected trajectories |
| `restore_check` | yes/no; yes | Compare potential energies after restores |
| `parallel` | serial/mpi; serial | MPI requires mpi4py |
| `mpi_mode` | multi_node/multi_gpu; multi_node | One rank per node or one GPU per local rank |
| `gpu_devices` | auto or GPU list; auto | List is indexed by local MPI rank |
