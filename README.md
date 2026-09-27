# sumd-openmm

Supervised molecular dynamics (SuMD) and multiple-walker supervised
molecular dynamics (mwSuMD) with OpenMM.

## Methods

SuMD (Sabbadin and Moro, 2014) samples rare events such as ligand binding
without adding any force to the system. The simulation advances in short
windows of unbiased molecular dynamics. After each window, a collective
variable sampled along the window is fitted to a straight line. If it
moves in the requested direction, the next window continues from the end
of this one; otherwise the window is discarded and repeated from the same
state with new velocities.

mwSuMD (Deganutti et al., 2025) runs a batch of walkers from the same state
and always continues from the best one, keeping its coordinates and
velocities for the next batch. The best walker is the one with the most
favourable slope, SMscore (one metric) or DMscore (two metrics).

sumd-openmm implements both methods as published, and adds:

- any number of collective variables of eight types (distances, distances
  along an axis, minimum distances, atomic contacts, angles, dihedrals,
  RMSD and fitted RMSD), each driven to increase, to decrease or towards a
  value, with cpptraj masks or VMD-like selections;
- combined supervision of several metrics, staged supervision that moves
  from one metric to the next, and metrics that are only recorded;
- stratified seeding, which restarts from states spread along additional
  metrics instead of a single chain;
- any system OpenMM reads from Amber, CHARMM, GROMACS or OpenMM XML files;
- optional equilibration of a CHARMM-GUI Amber or GROMACS system in
  OpenMM, with its staged restraints, before SuMD;
- walkers on several GPUs, on one node or across nodes, with MPI, on
  NVIDIA (CUDA) or AMD (HIP) GPUs;
- a test mode that checks the installation and a short run of the real
  system, and an audit of trajectories and saved states after a run.

SuMD accepts or rejects windows but never changes the dynamics, so its
trajectories describe pathways, not rates or free energies. Saved states
can seed unbiased simulations for those.

## Install

```bash
conda create -n sumd-openmm -c conda-forge python=3.11 openmm parmed numpy scipy pip git
conda activate sumd-openmm
python -m pip install "sumd-openmm @ git+https://github.com/MidhunkMadhu/sumd-openmm.git"
sumd-openmm --test
```

To run walkers on several GPUs with MPI, install the package with
mpi4py compiled against the cluster's MPI, in one command on a login
node:

```bash
MPICC=mpicc pip install --no-binary mpi4py \
    "sumd-openmm[mpi] @ git+https://github.com/MidhunkMadhu/sumd-openmm.git"
```

On HPE Cray systems (Dardel, LUMI) use `MPICC="cc -shared"`.
`--no-binary mpi4py` makes pip compile mpi4py instead of using a prebuilt
copy linked to another MPI.

For a local checkout, `conda env create -f environment.yml` followed by
`python -m pip install -e .`. VMD-like selections need `MDAnalysis`. AMD
GPUs need a ROCm 6 runtime; [parallel use](docs/PARALLEL.md) gives a
generic GPU job script and tested versions for Dardel.

`sumd-openmm --test` lists the OpenMM platforms available and the MPI
library mpi4py uses, checks that the platforms compute the same forces,
and runs two SuMD cycles of a built-in system. [Troubleshooting](docs/TROUBLESHOOTING.md) covers common
installation and cluster problems: ROCm versions, mpi4py, conda in batch
jobs and others.

## Quick start

Create `run.inp` next to an equilibrated topology and coordinates. The
selections here are placeholders:

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
metric_1_a = :LIG&!@H=
metric_1_b = :45,112,290@CA
metric_1_role = supervise
metric_1_direction = decrease
metric_1_target = 4
```

```bash
sumd-openmm run.inp --dry-run    # selections and initial values
sumd-openmm run.inp --test       # two short cycles, audited, with speed
sumd-openmm run.inp
sumd-inspect sumd_run
```

To start from a CHARMM-GUI system instead, replace the first three lines
with `equilibration = charmm-gui` and `charmm_gui_dir = <folder>`; its
minimization and restrained equilibration stages run first, each logged
as it starts and ends ([equilibration](docs/EQUILIBRATION.md)).

For several walkers, set `walkers` and `walker_score`
(`slope`, `smscore` or `dmscore`), and `parallel = mpi` to run them on
separate GPUs.

Distances are in ångström, angles in degrees and times in picoseconds.

## Collective variables

| Type | Selections | Value |
| --- | --- | --- |
| `distance` | `a, b` | Distance between centres of geometry |
| `distance_axis` | `a, b`, axis | Component along an axis, signed or not |
| `mindist` | `a, b` | Smallest atom-pair distance |
| `contacts` | `a, b`, cutoff | Atom pairs closer than `cutoff` (4 Å) |
| `angle` | `a, b, c` | Three-atom angle |
| `dihedral` | `a, b, c, d` | Four-atom torsion |
| `rmsd` | `a`, reference | RMSD after fitting `a` itself |
| `rmsd_displacement` | `a`, `fit`, reference | RMSD of `a` after fitting `fit` |

A supervised metric needs `direction = decrease | increase | toward` and
`target`; `toward` also needs `tolerance`, and torsions may use
`target_delta`. Other metrics take `role = stratify` with `bins`, or
`role = monitor`. RMSD metrics can select reference atoms with separate
masks (`reference_a`, `reference_fit`) when the reference is numbered
differently.

## Selections

`selection_syntax = cpptraj` uses ParmEd Amber masks; `vmd` uses
MDAnalysis selections. The prefixes `cpptraj:`, `vmd:`, `indices:` and
`file:` choose the syntax of a single selection.

| Meaning | cpptraj | VMD-like |
| --- | --- | --- |
| Residue in file order | `:45` | `resnum 45` if topology numbering agrees |
| Named atom | `:45@CA` | `resid 45 and name CA` |
| Heavy atoms of a residue | `:LIG&!@H=` | `resname LIG and not name "H.*"` |

A multi-atom selection stands for its unweighted centre of geometry,
except in `mindist` and `contacts`, which use every atom pair. The dry run
reports the atoms each selection resolves to.

## Documentation

[Configuration](docs/CONFIG_REFERENCE.md) ·
[algorithm](docs/ALGORITHM.md) ·
[equilibration](docs/EQUILIBRATION.md) ·
[outputs](docs/OUTPUTS.md) ·
[parallel use and GPUs](docs/PARALLEL.md) ·
[validation](docs/VALIDATION.md) ·
[setup](docs/SETUP_LAYER.md) ·
[troubleshooting](docs/TROUBLESHOOTING.md) ·
[examples](examples)

[`examples/ligand_binding_amber`](examples/ligand_binding_amber) is a
complete ligand-binding system with an RMSD-supervised input, annotated
with the equivalent supervisedmdamber keys.

## References

- Sabbadin D, Moro S. *J Chem Inf Model* **54**, 372-376 (2014).
  DOI: [10.1021/ci400766b](https://doi.org/10.1021/ci400766b).
- Cuzzolin A et al. *J Chem Inf Model* **56**, 687-705 (2016).
  DOI: [10.1021/acs.jcim.5b00702](https://doi.org/10.1021/acs.jcim.5b00702).
- Deganutti G et al. *eLife* **13**, RP96513 (2025).
  DOI: [10.7554/eLife.96513](https://doi.org/10.7554/eLife.96513).
