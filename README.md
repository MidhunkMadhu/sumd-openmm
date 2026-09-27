# sumd-openmm — Supervised MD on OpenMM

A SuMD driver for ligand recognition in TAAR1, built on the MD_openmm
production setup, which is bundled in the package. It reproduces the Amber
SuMD reference (`supervisedmdamber`) in `single` mode and adds a
two-variable mode in which the pocket-2 gate aperture is used for **seed
bookkeeping, never as a second acceptance filter**. It runs serially on one
GPU or with MPI, one walker per GPU. The design brief is
`docs/SUMD_OPENMM_PROMPT.txt`, and §2 of that file explains the scientific
reason for the two-variable design.

> SuMD windows are ordinary unbiased dynamics; only the *selection* of windows
> is biased. That breaks detailed balance, so SuMD segments carry no valid
> kinetics or thermodynamics. They generate pathways and seeds. Every rate
> must come from unbiased runs seeded from SuMD states.

## Install

```bash
git clone git@github.com:<you>/sumd-openmm.git
cd sumd-openmm
conda env create -f environment.yml      # openmm, parmed, numpy (+ mpi4py, pytest)
conda activate sumd-openmm
pip install -e .                         # installs the sumd-openmm and sumd-inspect commands
pytest                                   # ~70 tests, a few seconds, no GPU needed
```

The package is self-contained. The MD_openmm code it uses is bundled in
`src/sumd_openmm/md_openmm/`, so there is no separate MD_openmm checkout to
point at. OpenMM is installed by conda rather than pip so the CUDA build
matches the machine. On a cluster, build `mpi4py` against the cluster MPI
(see the MPI section below).

## Running it

```bash
cd /path/to/system                   # run from the system directory, as for production
sumd-openmm sumd.inp --dry-run       # resolve masks, build the system, print initial CVs
sumd-openmm sumd.inp                 # run
touch sumd_run/STOP                  # optional: stop cleanly after the current cycle
sumd-inspect sumd_run                # audit the finished run
```

`python -m sumd_openmm sumd.inp` is equivalent to `sumd-openmm sumd.inp`.
Start from `examples/taar1_ral_sumd.inp`, which contains every key with
comments.

## Repository layout

| Path | Role | Needs OpenMM |
|---|---|---|
| `src/sumd_openmm/cli.py` | the `sumd-openmm` command: setup, the SuMD cycle loop (rank 0), outputs | yes |
| `src/sumd_openmm/engine.py` | the only code touching an OpenMM Context: windows, state save/restore, packets | yes |
| `src/sumd_openmm/executors.py` | where walkers run: serial, or MPI across ranks | yes (mpi4py for MPI) |
| `src/sumd_openmm/parallel.py` | MPI layout: GPU per rank, walker → rank distribution | no |
| `src/sumd_openmm/omm_setup.py` | builds the `Simulation` through the bundled MD_openmm code | yes |
| `src/sumd_openmm/md_openmm/` | **bundled MD_openmm helpers** plus `PROVENANCE.txt` (source commit, checksums) | yes |
| `src/sumd_openmm/cv.py` | Kabsch, minimum image (triclinic), CV1 (rmsd / distance / mindist), CV2, trend statistic | no |
| `src/sumd_openmm/supervision.py` | accept/reject rule, joint score, SMscore / DMscore | no |
| `src/sumd_openmm/seeding.py` | stratified pool: cells, frontier, least-visited parent choice | no |
| `src/sumd_openmm/selection.py` | Amber masks → index arrays (ParmEd `AmberMask`), residue-identity checks | no (ParmEd) |
| `src/sumd_openmm/config.py` | the SuMD keys added to the `.inp` | no |
| `src/sumd_openmm/dcdtools.py` | byte-level concatenation and reading of OpenMM DCDs (no topology needed) | no |
| `src/sumd_openmm/inspect_run.py` | the `sumd-inspect` command: integrity, acceptance, per-rank load, gate strata | no |
| `tools/vendor_md_openmm.py` | re-syncs the bundled helpers from an MD_openmm checkout | no |
| `examples/taar1_ral_sumd.inp` | TAAR1 8JLP + ralmitaront example, every key commented | |
| `tests/` | unit tests, fake-engine loop tests (serial, MPI, 3-rank mpirun), cpptraj parity | no |
| `docs/METHODS_NOTE.md` | methods template to fill per system | |
| `docs/SUMD_OPENMM_PROMPT.txt` | the design brief | |

## The bundled MD_openmm helpers

A SuMD run and a production run are built by the same code, so the
SuMD-seeded data can be read next to the existing production simulations.
`src/sumd_openmm/md_openmm/` holds:

- `production_helpers.py`: every import and function definition of MD_openmm's
  `openmm_production.py`, verbatim (`read_key_value_file`, `build_inputs`,
  `get_genvel`, `load_coordinate_file_auto`, `write_amber_restart`, … 31 in
  all);
- `omm_barostat.py`, `omm_rewrap.py`, `omm_vfswitch.py`, `omm_restraints.py`,
  `omm_readparams.py`: verbatim, with their CHARMM-GUI headers;
- `PROVENANCE.txt`: the MD_openmm commit they came from and the SHA-256 of
  each source file. The commit is also written into every run's log and
  `run_summary.json`.

The only edit is mechanical: imports between these files are made
package-relative. `openmm_production.py`'s top-level run code can't be
imported (it starts a simulation), so `omm_setup.build_simulation()` mirrors
it section by section. The one deliberate difference is that the Langevin
integrator gets an explicit seed, so a run is reproducible from
`random_seed`.

**After changing MD_openmm**, re-sync and test:

```bash
python tools/vendor_md_openmm.py /path/to/MD_openmm
MD_OPENMM_DIR=/path/to/MD_openmm pytest tests/test_md_openmm_helpers.py
```

The second command confirms that the bundled copy matches the checkout byte
for byte. If you change `openmm_production.py`'s top-level setup sequence
itself, update `build_simulation()` in `omm_setup.py` to match. Wrapping that
sequence in a `build_simulation(cfg)` function in MD_openmm would remove the
mirrored copy entirely.

## The algorithm

```
parent <- initial state (node 0)
for cycle in 1..max_cycles:
    [stratified] parent <- least-visited frontier cell of the pool
    for each walker (default 1):
        continue in place if parent is the state just produced (velocities kept, as Amber irest=1)
        else restore parent State (positions, velocities, box, time, params)
             and, with retry_velocities = reassign, draw new velocities with a fresh logged seed
        run window_ps, sampling CV1/CV2 every cv_sample_ps (frame at t=δ … τ, no t=0)
    pick the best walker; apply the rule
    accept   -> new node: DCD kept, State saved as XML; [single] parent <- node
    reject   -> DCD deleted; retries(parent) += 1
                past max_retries_per_parent: accept_best | step_back   (stratified: retire parent)
    converged -> stop
final trajectory = accepted windows on the path root -> converged (or best) node
```

**Rule** (identical to `supervisedmdamber.py`, evaluated on the window just run):

| condition | outcome |
|---|---|
| CV1_last < colvarcut and slope ≤ 0 | converged |
| slope ≤ 0, or (slope > 0 and CV1_last < colvarcut) | accept |
| otherwise | reject |

The slope is an OLS fit over all samples of the window (A/ps). SE(b) and R²
are logged too. `require_significant_slope = yes` tightens the progress
clause to b < 0 and |b| > 2·SE(b). `slope_points = 5` reproduces the
reference's 5-point, frame-index estimator exactly.

## CVs

**CV1, `rmsd`** reproduces cpptraj's `autoimage; rms :FIT` then `rms :LIG nofit`:

1. image the ligand next to the receptor (true minimum image, triclinic-safe);
2. Kabsch on receptor CA, `R = V·diag(1,1,d)·Uᵀ` with `d = sign det(VUᵀ)`;
3. transform the ligand with that fit, `l' = R(l − p̄) + q̄`;
4. `CV1 = sqrt(mean ||l' − l_ref||²)` with no refit.

This measures the ligand's displacement from its bound pose in the receptor
frame, not a shape RMSD. `distance` is the COM–COM distance (classic SuMD);
`mindist` is the minimum atom–atom distance. All use minimum-image vectors.

**CV2** is the minimum heavy-atom distance between the F154(4.56) and
T197(5.45) side chains. It uses the same definition as `md/extract_observables.py`:
`not name N CA C O and not name H*`, which is `:N&!@N,CA,C,O,H=` as an Amber mask.
`cv2_expect_a/b` make the run abort if the masks don't land on PHE/THR, so a
numbering difference can't go unnoticed.

Masks are full cpptraj masks resolved once by ParmEd (`:N` = N-th residue in
file order). Note that the Amber reference prefixed `:` itself. Topology and
reference atoms are paired in order and their names checked pairwise. No
MDAnalysis, MDTraj or cpptraj is called inside the loop; per-frame work is numpy.

## Supervision modes

| `supervision` | acceptance on | CV2 role | default |
|---|---|---|---|
| `single` | CV1 | measured and logged only | **yes**; the validation reference |
| `stratified` | CV1, same rule | chooses the parent: pool cells `(CV1 band, gate stratum)`, least-visited cell on the frontier | recommended for production |
| `joint` | `s = w1·z1 + w2·z2` (gate opening counts as progress) | inside the acceptance test | no; comparison only |

In `stratified` mode, the frontier is every cell whose band ≤ best band +
`frontier_bands`, and the parent comes from `argmin n_c` with random
tie-breaking. Gate-closed states near the frontier get windows on purpose,
so the waiting time survives into the seeds. Visit counts stay attached to
evicted and retired states, so a cell never looks fresh again after eviction.
A parent that exhausts `max_retries_per_parent` is retired from the pool. If
it was the last live state, the least-bad attempt is committed instead
(logged), so an unlucky start cannot end the run.

`joint` mode (and `walker_score = dmscore`) logs this warning, verbatim, at
start-up and writes it into `run_summary.json`:

> joint mode applies an AND-style filter that discards windows in which the
> ligand waits outside a closed gate. Since the gate is open only 0.8-3 % of
> the time in the apo receptor, kon estimates derived from joint-mode seeds
> are expected to be biased fast. Use for comparison only.

## Multiple walkers (mwSuMD-style)

With `walkers = N`, each cycle runs N windows from the same parent and the
best one is kept. In serial mode they run one after another on one GPU. With
`parallel = mpi` they run at the same time, one per GPU (next section). Batch scoring follows Deganutti
et al., *eLife* 96513 (2025):

- `walker_score = slope` picks the lowest slope, then applies the rule above.
- `walker_score = smscore` uses `sqrt(CV1_last · mean CV1)` and always
  extends the best walker, lowest score wins.
- `walker_score = dmscore` uses `100·[−(CV1_last/⟨CV1⟩ − 1) + (CV2_last/⟨CV2⟩ − 1)]`,
  normalised by the batch mean. It selects on the gate, so it carries the
  warning above.

`stratified + dmscore` is refused as contradictory, and score modes with
`walkers = 1` are refused because they would accept every window.
`retry_velocities = keep` gives mwSuMD's no-reassignment behaviour: walkers
then diverge only through the Langevin noise stream. The driver warns if two
attempts from the same parent come out identical.

## Parallel runs (MPI)

`parallel = mpi` runs the walkers of each cycle at the same time, one MPI rank
per GPU. A batch of 4 walkers then costs one window of wall-clock time instead
of four. Every rank builds its own `Simulation` through MD_openmm from the same
`.inp`. Rank 0 runs the SuMD logic and also runs a walker, so no GPU sits idle.

| key | values | meaning |
|---|---|---|
| `parallel` | `serial` (default) \| `mpi` | |
| `mpi_mode` | `multi_node` (default) | one rank per node, one GPU per node. Aborts if two ranks land on one node, since they would share a GPU |
| | `multi_gpu` | several ranks per node, GPU chosen by the **node-local** rank. Also covers several nodes with several GPUs each |
| `gpu_devices` | `auto` (default) | `multi_node`: OpenMM's default GPU. `multi_gpu`: local rank k → GPU k, or GPU 0 when the scheduler already binds one GPU per rank (`CUDA_VISIBLE_DEVICES` holds one id) |
| | `0,1,2,3` | `multi_gpu`: local rank k → k-th listed GPU. `multi_node`: first entry on every node |
| `walkers` | `auto` | one walker per rank (1 in serial mode) |

Example Slurm scripts (adjust account, partition and module names to your cluster):

```bash
# 4 nodes x 1 GPU                                  (.inp: parallel = mpi, mpi_mode = multi_node)
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
srun sumd-openmm sumd.inp

# 1 node x 4 GPUs                                  (.inp: parallel = mpi, mpi_mode = multi_gpu)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
srun sumd-openmm sumd.inp

# 2 nodes x 4 GPUs = 8 walkers: as above with --nodes=2, still mpi_mode = multi_gpu
```

Always start with a dry run under the same launcher. It builds every rank's
system, checks the layout, and prints the rank → host → GPU table without
simulating:

```bash
srun sumd-openmm sumd.inp --dry-run
```

How it works:

- **Walker placement.** Walkers go round-robin over ranks. Walker 0 goes to
  the rank whose Context already holds the parent, so it continues in place,
  exactly as in serial mode.
- **Moving states.** A rank that needs the parent gets it as a numpy packet
  (positions, velocities, box, time, parameters: ~5 MB for 100k atoms)
  broadcast from a rank that holds it. If no rank holds it, rank 0 reads the
  node's XML first. Ranks cache recent nodes, so retries from the same parent
  don't re-send it.
- **Saving results.** The rank that ran the accepted walker writes its State
  XML. Every rank writes its own window DCD into `output_dir`, which must
  therefore be on a filesystem shared by all nodes. This is checked at
  start-up with a probe file.
- **Seeds.** Rank 0 draws each rank's integrator seed and each walker's
  velocity seed, so an MPI run is reproducible from `random_seed` for a
  given number of ranks.
- **Logs.** Rank 0 prints the run log; other ranks log to
  `output_dir/ranks/rank_NNN.log`. `windows.jsonl` records the rank, host and
  GPU of every window.
- **Failures.** If any rank fails (bad layout, missing shared filesystem, CUDA
  error), the whole job aborts instead of hanging until the wall-time limit.
  The reason goes to stderr and to `output_dir/FATAL_rank_NNN.txt`.

`mpi4py` must be built against the cluster's MPI, e.g. `module load` the
cluster MPI, then `pip install --no-binary mpi4py mpi4py` in the sumd-openmm env
(and remove the conda `mpi4py` from `environment.yml` before creating it).
If `walkers` is smaller than the number of ranks, the driver warns that ranks
will sit idle. If it isn't a multiple of the rank count, some ranks run an
extra window per cycle while the rest wait.

## Outputs (`output_dir/`)

| File | Content |
|---|---|
| `windows.jsonl` | one row per window: cycle, walker, parent, child, verdict, reason, action, seed, start mode, restore ΔE, full CV1/CV2 series, slope, SE, R², score, band, stratum, chosen cell and its visit count |
| `nodes.csv` | the seeding tree: every committed state with its parent |
| `states/node_NNNNNN.xml` | OpenMM State of every node. **Usable directly as `coordinate_file` in `openmm_production.py`** (with `continuation = yes, genvel = no`, or `genvel = yes`) to launch the seeded unbiased runs |
| `windows/accepted/cCCCCC_wW.dcd` | every committed window |
| `sumd_traj.dcd` | accepted windows along the final path, in order |
| `final_path.txt`, `final_state.xml`, `final_state.rst7` | final path and state (the rst7 is written by MD_openmm's `write_amber_restart`) |
| `run_summary.json` | resolved config, indices, residue identities, seeds, software, platform/precision, wall time, stop reason |
| `progress.log` | human-readable log |

Disk: each node XML is about 150 bytes/atom (~15 MB for 100k atoms).

## Differences from the Amber reference (deliberate)

1. **Seeds.** Every restart that reassigns velocities uses a seed drawn from
   `random_seed` and logs it. For the record, `sumd.mdin` sets no `ig`.
   Amber's default `ig = -1` seeds from the wall clock, so reference retries
   do diverge, but they are neither logged nor reproducible. The reference
   never reassigns velocities (irest=1, ntx=5); `retry_velocities = keep`
   reproduces that.
2. **Retry limit.** `max_retries_per_parent` with `accept_best` or
   `step_back`. The reference can spin on one parent until MAXSTEP.
3. **Slope.** All samples by default; the 5-point estimator is available
   through `slope_points = 5`.
4. **Imaging.** A true minimum image. See the cpptraj finding below; this is
   the one place where the two implementations can give different CV values.
5. **Final trajectory.** Built from the parent pointers of the tree, rather
   than by deduplicating on the first mdout line.

## Validation (§11 of the brief)

Results are from the development machine: OpenMM 8.6.1 (CPU platform), ParmEd
4.3.1, CPPTRAJ V7.6.2. The test system is the reference's own `testfolder`
(67 216 atoms, GPCR in POPC, ligand `P0G` at `:314`).

**11.1 Unit tests.** `pytest` runs 73 tests. 64 need neither OpenMM, mpi4py nor a GPU; the other 9 run when OpenMM, mpi4py/mpirun, cpptraj, a test system or an MD_openmm checkout are available:

- Kabsch returns det = +1 on a mirrored input.
- CV1 is invariant to rigid motion of the whole system.
- CV1 is unchanged across all 26 neighbouring images, in an orthorhombic box
  and a strongly triclinic one; a control confirms it is nonsense without imaging.
- The minimum image matches brute force in a triclinic cell.
- The slope estimator matches `scipy.stats.linregress` (slope, intercept, SE, R²).
- The 5-point mode reproduces `extract_and_fit.py` verbatim.
- The decision rule matches the literal `supervisedmdamber.py` expressions on
  2 000 random cases.
- Pool, frontier, retirement and eviction semantics behave as specified.
- Config refusals work (GaMD; contradictory mode combinations).

**11.1 cpptraj parity.** `tests/test_cpptraj_parity.py` runs the reference's
exact `fit.cppin` / `calc_rmsd.cppin` pair. With the ligand in the same image
as the receptor (original frame, rigid translation, 20° rotation, ligand
displaced 3 Å and 6 Å), the maximum |ΔCV1| over the 5 frames is
**4.8 × 10⁻⁵ Å** (target 10⁻³ Å).

**Finding: cpptraj `autoimage` is not a minimum image.** Moving the ligand by
one lattice vector changes the cpptraj result:

| shift | cpptraj CV1 (Å) | cv.py CV1 (Å) |
|---|---|---|
| none, +a, −b, +c | 32.246 | 32.246 |
| −a | 67.398 | 32.246 |
| +b | 81.167 | 32.246 |
| −c | 87.360 | 32.246 |

In three of six directions cpptraj leaves the ligand in a far image, so the
reference CV reads 35 to 55 Å too high. `sumd.mdin` uses `iwrap = 1`, so
pmemd does wrap a bulk ligand across the box. In the Amber runs, then, a
window in which the ligand crosses a box face in one of those directions
likely got an arbitrary accept/reject. That is worth checking in existing
Amber SuMD logs as jumps in `prod_cvval_*.dat`. This was shown on synthetic
frames with this cpptraj version only.

**11.2 Smoke runs.** These used real OpenMM dynamics through the MD_openmm
path on the test system (CPU platform, NPT with the membrane barostat,
0.1 ps windows of 10 samples). The cutoff was set out of reach so accept,
reject and retry exhaustion all occur. `sumd-inspect` output:

| | single | stratified | 3 walkers, smscore | single, step_back, keep, 5-pt |
|---|---|---|---|---|
| cycles / windows simulated | 8 / 8 | 10 / 10 | 3 / 9 | 8 / 8 |
| windows committed / discarded | 5 / 3 | 4 / 6 | 3 / 6 | 4 / 4 |
| rule acceptance per cycle | 0.375 | 0.40 | 1.0 (by design) | 0.50 |
| retry-limit events | 2 (accept_best) | 1 (parent retired) | 0 | 2 (step_back) |
| max \|ΔEpot\| after restore (kJ/mol) | 0.017 | 0.013 | 0.016 | 0.011 |
| final traj = path windows, frame-identical | yes (30 fr) | yes (30 fr) | yes (10 fr) | yes (20 fr) |
| max DCD-last-frame vs saved State (Å) | 6.5e-6 | 6.4e-6 | 6.4e-6 | 6.4e-6 |

So rejected windows never reach the final trajectory, and every committed
DCD belongs to its saved state. Restored states reproduce the parent energy
to integrator/float noise, including box vectors under the barostat.
`retry_velocities = keep` produced no identical retries. Re-running with the
same `random_seed` reproduced the same accept/reject sequence and CVs to the
third decimal (CPU floating-point ordering accounts for the rest). The
initial CV1 from the driver, 32.2465 Å, equals cpptraj's value for the same
frame.

Throughput on this laptop CPU was ~8 s per 0.1 ps window at 67k atoms,
which is useless for production and only meant as a functional test.

**MPI runs.** Open MPI 5.0.11 + mpi4py, 3 ranks on one machine (CPU platform,
so the ranks share cores; on a cluster each rank has its own GPU). All with
real OpenMM dynamics through MD_openmm:

| | 3 walkers, smscore | stratified, 3 walkers | 5 walkers on 3 ranks |
|---|---|---|---|
| windows per rank | 4 / 4 / 4 | 6 / 6 / 6 | 6 / 5 / 4 (warned) |
| walkers ran concurrently | yes: each ~32 s, batch ~35 s | yes | yes |
| max \|ΔEpot\| after restore, incl. packets from other ranks (kJ/mol) | 0.025 | 0.020 | 0.019 |
| final traj = path windows, frame-identical | yes | yes | yes |
| max DCD-last-frame vs saved State (Å) | 6.2e-6 | 6.3e-6 | 6.1e-6 |

- MPI overhead was ~3 s per cycle here (state broadcast, restores, result
  gather, node XML write).
- The same `random_seed` and rank count reproduced the same cycles.
- Wrong layouts abort within 2 s with an explicit message. Tested: three ranks
  on one host under `multi_node`, and three ranks with `gpu_devices = 0,1`.
- `tests/test_parallel.py` also runs the whole loop on 3 real MPI ranks with
  a fake engine (rejections, retries, stratified pool, every rank used), so
  the MPI logic is covered without GPUs.

**Not tested on real GPUs or across real nodes.** Device assignment by
`DeviceIndex` and cross-node broadcasting follow standard OpenMM/MPI
behaviour, but the first cluster run should be a `--dry-run`, then a few
short cycles, then `sumd-inspect`.

**11.3 Parity against the Amber driver** and **11.4 stratified vs single on a
production-length run** have not been done. Both need GPU time on a real
system: run `single` with `slope_points = 5` and `retry_velocities = keep`
next to `supervisedmdamber` on the same start. Then compare acceptance rate,
cycles to cutoff and the CV1 envelope across ≥ 5 seeds each; `sumd-inspect`
reports these, plus the gate-stratum distribution of retained states for 11.4.

## Not implemented / open

- **SuGaMD.** `method = GaMD` is refused. OpenMM has no native GaMD; it
  needs a GaMD integrator or plugin and should be scoped separately rather
  than run as plain cMD under a GaMD label.
- **Resuming** an interrupted run. All states and the tree are on disk, but
  there is no resume entry point yet.
- **Independent entries are still needed.** MPI parallelises the walkers of
  *one* SuMD entry. §12.2 of the brief asks for 10 to 20 independent entries
  (different `random_seed` / starting pose); run those as separate jobs.
- The MSM, committor and water analyses (§12–13) are downstream and not part
  of this module.
