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
angles, dihedrals and RMSDs. They are defined with cpptraj masks or
VMD-like selections, and each is driven to increase, to decrease or to
reach a target value. Any number can be combined, some supervising the
run, some choosing which states seed new windows, some only recorded.
Typical uses include binding and unbinding of small molecules, peptides
or ions; association of proteins, nucleic acids or both; domain and
loop motions; base-pair opening; lipid or solvent access; and side-
chain or backbone rotations.

Windows run inside one OpenMM process with no restarts between them.
Walkers run concurrently on several GPUs, on one node or across nodes,
with MPI.

Only the selection of windows is supervised, not the dynamics, so
SuMD trajectories describe pathways, not rates or free energies. Use
states saved by a run to seed unbiased simulations for those.

## Install

```bash
conda env create -f environment.yml
conda activate sumd-openmm
pip install -e .
sumd-openmm --help
```

An existing OpenMM environment can install the Python requirements with
`pip install -e .`. Install `mpi4py` against the MPI library used to launch
parallel runs. VMD-like selections require `MDAnalysis`.

## Quick start

Create `run.inp` alongside a topology and coordinates. The selections in
this example use zero-based atom indices and must be replaced with indices
from your topology:

```ini
force_field = AMBER
topology_file = system.parm7
coordinate_file = system.rst7
platform = CUDA
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
sumd-openmm run.inp
sumd-inspect sumd_run
```

The dry run prints each selection and its initial value. All distances use
angstroms; angles use degrees; times use picoseconds.

| Type | Required selections | Value |
| --- | --- | --- |
| `distance` | `a, b` | Distance between centres of geometry |
| `distance_axis` | `a, b`, axis | Signed or unsigned component along an axis |
| `mindist` | `a, b` | Smallest atom-pair distance |
| `angle` | `a, b, c` | Three-atom angle |
| `dihedral` | `a, b, c, d` | Four-atom torsion |
| `rmsd` | `a`, reference | Shape RMSD after fitting |
| `rmsd_displacement` | `a`, `fit`, reference | RMSD of `a` after fitting `fit` |

Supervised metrics require `direction = decrease | increase | toward` and
`target`. The `toward` direction also requires `tolerance`. Increasing
and decreasing torsions may use `target_delta` in place of `target`.
Other metrics may use `role = stratify` with `bins`, or `role = monitor`.
See [configuration](docs/CONFIG_REFERENCE.md), [algorithm](docs/ALGORITHM.md),
[outputs](docs/OUTPUTS.md), [parallel use](docs/PARALLEL.md),
[validation](docs/VALIDATION.md), and [setup](docs/SETUP_LAYER.md).
The [examples](examples) show different processes and selection languages.

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
