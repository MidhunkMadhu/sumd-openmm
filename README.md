# sumd-openmm

Supervised molecular dynamics (SuMD) and multiple-walker SuMD (mwSuMD)
with OpenMM.

SuMD samples rare molecular events without adding any force to the
system. The simulation advances in short windows of ordinary, unbiased
molecular dynamics. After each window, user-defined collective variables
are checked: if they moved in the requested direction, the window is
kept and the next one continues from its end; otherwise the window is
discarded and the previous state is simulated again with new
velocities. mwSuMD runs several windows from the same state in each
cycle and continues from the best one.

sumd-openmm works with any system OpenMM can simulate from Amber,
CHARMM, GROMACS or OpenMM XML input: proteins, nucleic acids, lipids and
membranes, carbohydrates, small molecules, and complexes of any of
these. Collective variables are distances between atoms or centres of
geometry, distances along an axis, minimum distances between groups,
numbers of atomic contacts, angles, dihedrals and RMSDs. They are defined with cpptraj masks or
VMD-like selections, and each is driven to increase, to decrease or to
reach a target value. Any number can be combined, some supervising the
run, some choosing which states seed new windows, some only recorded.
Typical uses include binding and unbinding of small molecules, peptides
or ions; association of proteins, nucleic acids or both; domain and
loop motions; base-pair opening; lipid or solvent access; and side-
chain or backbone rotations.

Windows run inside one OpenMM process with no restarts between them.
Walkers run concurrently on several GPUs, on one node or across nodes,
with MPI. mwSuMD follows Deganutti et al. (2025): every batch continues from
its best walker by slope, SMscore or DMscore, with that walker's
velocities; see [the correspondence](docs/ALGORITHM.md#correspondence-with-mwsumd).

Only the selection of windows is supervised, not the dynamics, so
SuMD trajectories describe pathways, not rates or free energies. Use
states saved by a run to seed unbiased simulations for those.

## Install

To install from the public GitHub repository without cloning it yourself:

```bash
conda create -n sumd-openmm -c conda-forge python=3.11 openmm parmed numpy scipy pip git
conda activate sumd-openmm
python -m pip install "sumd-openmm @ git+https://github.com/MidhunkMadhu/sumd-openmm.git"
sumd-openmm --help
```

Git and pip fetch the package automatically. The repository must be public
for this command to work without GitHub credentials. The package is not yet
published on a Conda channel, so Conda installs OpenMM and its scientific
dependencies while pip installs sumd-openmm from GitHub.

If you have a local checkout and want an editable installation instead:

```bash
conda env create -f environment.yml
conda activate sumd-openmm
python -m pip install -e .
sumd-openmm --help
```

On AMD GPUs (Dardel, LUMI) OpenMM needs its HIP platform, which the
default CUDA build lacks; install `openmm-hip` instead of `openmm` as
described in [parallel use](docs/PARALLEL.md#amd-gpus-dardel-lumi).
`sumd-openmm --test` lists the platforms OpenMM can use, checks that
their forces agree, and runs two SuMD cycles of a built-in system.

An existing OpenMM environment can install the Python requirements with
`python -m pip install -e .`. Install `mpi4py` against the MPI library used to launch
parallel runs. VMD-like selections require `MDAnalysis`.

## Quick start

Create `run.inp` alongside a topology and coordinates. The selections in
this example use zero-based atom indices and must be replaced with indices
from your topology:

```ini
force_field = AMBER
topology_file = system.parm7
coordinate_file = system.rst7
platform = auto
temp = 300
dt = 0.002
window_ps = 20
cv_sample_ps = 1
max_cycles = 500
random_seed = 42
output_dir = sumd_run
metric_1_name = approach
metric_1_type = distance
metric_1_a = indices:0-9
metric_1_b = indices:100-109
metric_1_role = supervise
metric_1_direction = decrease
metric_1_target = 4
```

```bash
sumd-openmm run.inp --dry-run
sumd-openmm run.inp --test
sumd-openmm run.inp
sumd-inspect sumd_run
```

The input above starts from an already equilibrated system, whatever
software equilibrated it. To equilibrate a CHARMM-GUI system in OpenMM
first, add `equilibration = charmm-gui` and `charmm_gui_dir`; the
restrained stages of its Amber or GROMACS folder then run before SuMD,
each logged as it starts and ends. See [equilibration](docs/EQUILIBRATION.md).

The dry run prints each selection and its initial value. The test runs
two cycles of ten samples in `sumd_run_test`, audits the saved states
and trajectories, and reports the speed; run it on the node type the
job will use. All distances use
angstroms; angles use degrees; times use picoseconds.

| Type | Required selections | Value |
| --- | --- | --- |
| `distance` | `a, b` | Distance between centres of geometry |
| `distance_axis` | `a, b`, axis | Signed or unsigned component along an axis |
| `mindist` | `a, b` | Smallest atom-pair distance |
| `contacts` | `a, b`, cutoff | Number of atom pairs closer than `cutoff` (4 Å) |
| `angle` | `a, b, c` | Three-atom angle |
| `dihedral` | `a, b, c, d` | Four-atom torsion |
| `rmsd` | `a`, reference | Shape RMSD after fitting |
| `rmsd_displacement` | `a`, `fit`, reference | RMSD of `a` after fitting `fit` |

Supervised metrics require `direction = decrease | increase | toward` and
`target`. The `toward` direction also requires `tolerance`. Increasing
and decreasing torsions may use `target_delta` in place of `target`.
Other metrics may use `role = stratify` with `bins`, or `role = monitor`.
See [configuration](docs/CONFIG_REFERENCE.md), [equilibration](docs/EQUILIBRATION.md),
[algorithm](docs/ALGORITHM.md),
[outputs](docs/OUTPUTS.md), [parallel use](docs/PARALLEL.md),
[validation](docs/VALIDATION.md), and [setup](docs/SETUP_LAYER.md).
The [examples](examples) show different processes and selection languages.

### From supervisedmdamber

[`examples/ligand_binding_amber`](examples/ligand_binding_amber) is the
supervisedmdamber test system; its `run.inp` notes the Amber setting
each line reproduces. The ligand RMSD after fitting the receptor matches
cpptraj's `rms` then `rms nofit` to 1e-4 Å, and the switched
Lennard-Jones energy matches sander with `fswitch`.

| input_parameters.in | run.inp |
| --- | --- |
| `TIME_WINDOW_PS`, `TIMESTEP_FS` | `window_ps`, `dt` (ps) |
| `MDOUTSTEP`, `TRAJOUTSTEP` | `cv_sample_ps`, `dcd_stride` |
| `COLVARCUT` | `metric_1_target` |
| `MAXSTEP` | `max_cycles` |
| `FIT_TRAJ_STRING`, `FIT_PDB_STRING` | `metric_1_fit`, `metric_1_reference_fit` |
| `CALCRMSD_TRAJ_STRING`, `CALCRMSD_PDB_STRING` | `metric_1_a`, `metric_1_reference_a` |
| `PDBALIGN` | `metric_1_reference` |
| `INPUT_MDIN_FILE` | `temp`, `fric_coeff`, `cons`, `vdw`, `r_on`, `r_off`, `pcouple` |
| `AMBEREXE`, `MPIPREFIX` | `platform`; walkers use `parallel = mpi` |
| `METHOD=GaMD` | not available; OpenMM needs a GaMD integrator |

The five-point slope (`slope_points = 5`) and the acceptance rule are
those of supervisedmdamber. Unlike it, a parent is retried at most
`max_retries_per_parent` times.

### Selection syntax

`selection_syntax = cpptraj` uses ParmEd Amber masks; `:45@CA` selects
one atom in the 45th residue in file order. `selection_syntax = vmd` uses
MDAnalysis selections on a Universe built from the ParmEd structure.
Prefixes `cpptraj:`, `vmd:`, `indices:`, and `file:` override the
default per selection.

| Meaning | cpptraj | VMD-like |
| --- | --- | --- |
| Residue in file order | `:45` | `resnum 45` if topology numbering agrees |
| Named atom | `:45@CA` | `resid 45 and name CA` |
| Exclude matching names | `:LIG&!@H=` | `resname LIG and not name "H.*"` |

MDAnalysis syntax is VMD-like but not identical; `resid` uses topology
residue identifiers while `resnum` uses residue numbers. Check resolved
indices in the startup report. A selection of multiple atoms represents
its unweighted centre of geometry, except `mindist` which uses all atom
pairs.

### Methods

- Sabbadin D, Moro S. *J Chem Inf Model* **54**, 372-376 (2014).
  DOI: [10.1021/ci400766b](https://doi.org/10.1021/ci400766b).
- Cuzzolin A et al. *J Chem Inf Model* **56**, 687-705 (2016).
  DOI: [10.1021/acs.jcim.5b00702](https://doi.org/10.1021/acs.jcim.5b00702).
- Deganutti G et al. *eLife* **13**, RP96513 (2025).
  DOI: [10.7554/eLife.96513](https://doi.org/10.7554/eLife.96513).
