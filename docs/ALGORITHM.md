# Algorithm

Each MPI rank builds its own OpenMM Simulation. Each cycle starts from a
committed State. An accepted first walker continues in the same Context,
retaining velocities. Other starts restore position, velocity, box, time
and context parameters from the parent. With the default
`retry_velocities = auto`, a single-walker retry gets new velocities from
a logged seed, as in SuMD, while every walker of an mwSuMD batch keeps the
selected walker's velocities; the walkers then diverge through the
Langevin thermostat's random forces. `reassign` and `keep` force either
behaviour.

Every `cv_sample_ps`, all metrics are evaluated from one frame. For a
decreasing quantity the progress slope is `p = -b`; for an increasing
quantity it is `p = b`, where `b` is the least-squares slope over the
window. A `toward` metric supervises `|x-target|`, with circular distance
for torsions. A torsion driven to increase or decrease is unwrapped from
the parent's saved branch. The target comparison is strict.

| Condition | Decision |
| --- | --- |
| Inside target and `p >= 0` | Converged |
| `p >= 0` or inside target | Accept |
| Otherwise | Reject |

When significance is requested, the progress acceptance clause additionally
needs `p > 2 SE(b)`; convergence retains the rule above. `slope_points=5`
fits five sampled frame indices. Otherwise all samples and times are used.

For `supervision=combined`, every supervised metric contributes
`w_k * sign_k * (x_k - mu_k) / sigma_k` to a score where smaller is
better. Missing means and standard deviations come from the first window.
All supervised metrics must satisfy their targets to converge.
`supervision=multistep` applies the rule in the order listed in `stages`;
a stage advances when its metric converges.

With several walkers, `slope` picks greatest progress. `smscore` uses
`sqrt(last * mean)` for one nonnegative metric and chooses the endpoint
consistent with its direction. `dmscore` sums signed percentage deviations
from each metric's batch mean; larger is better. With chain seeding every
batch continues from its best walker, and the run converges when the best
walker is inside the active target(s). `walker_acceptance = sumd` instead
applies the SuMD table above to the best walker of a `slope` batch, so a
batch without progress is repeated.

### Correspondence with mwSuMD

Deganutti et al., *eLife* 13, RP96513 (2025), Methods, "mwSuMD protocol":

| Paper | sumd-openmm |
| --- | --- |
| Batches of walkers seeded from one state; the best is extended by a new batch of the same size and duration | `walkers = N`, chain seeding |
| One walker per batch is always productive | Default `walker_acceptance = always` |
| Velocities are not reassigned when a walker is extended | Default `retry_velocities = auto` |
| One metric: slope, or SMscore = sqrt(X_last · mean X), lowest if decreasing, highest if increasing | `walker_score = slope` or `smscore` |
| Two metrics: DMscore = ((X'_last / mean X'_batch − 1) + (X''_last / mean X''_batch − 1)) · 100, a decreasing term multiplied by −1, highest selected | `walker_score = dmscore`; two or more metrics |
| Distances between centroids, RMSD after fitting on a stable part, number of atomic contacts | `distance`, `rmsd_displacement`, `contacts` |
| One walker per GPU | `parallel = mpi` |
| Stop when one supervised metric reaches a threshold | give the other supervised metrics a target that is always met, e.g. an RMSD target of 1000 with `direction = decrease` |
| Phases supervising different metrics | successive runs, each started from the previous `final_state.xml` as `coordinate_file`; or `supervision = multistep` when each phase has one metric |

DMscore is defined in the paper for two metrics; here it accepts any
number of supervised metrics.

With `seeding=stratified`, the pool assigns each saved state a cell:
the band of the first supervised progress quantity and a bin for each
stratification metric. It chooses a least-visited cell on the frontier,
then a least-visited parent in that cell. Visit counts survive eviction.
After the retry limit, a parent retires; if this empties the pool, the
least-bad attempted window is committed.

Each committed node is saved as an OpenMM State XML. Each window has its
own DCD; discarded windows are deleted unless requested. The final DCD
concatenates the path from the root to the final node: the converged
node, otherwise the latest node of the chain or, with stratified seeding,
the most advanced node. `run_summary.json` also names the most advanced
node as `best_node`.
Supervision biases which segments survive; use unbiased simulations from
saved states for kinetic or thermodynamic estimates.
