# Algorithm

Each MPI rank builds its own OpenMM Simulation. Each cycle starts from a
committed State. An accepted first walker continues in the same Context,
retaining velocities. Other starts restore position, velocity, box, time
and context parameters from the parent. Velocity reassignment uses a
logged seed unless `retry_velocities = keep`.

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
from each metric's batch mean; larger is better. Score modes select a
walker every cycle and consider the active target converged when it is
inside its threshold.

With `seeding=stratified`, the pool assigns each saved state a cell:
the band of the first supervised progress quantity and a bin for each
stratification metric. It chooses a least-visited cell on the frontier,
then a least-visited parent in that cell. Visit counts survive eviction.
After the retry limit, a parent retires; if this empties the pool, the
least-bad attempted window is committed.

Each committed node is saved as an OpenMM State XML. Each window has its
own DCD; discarded windows are deleted unless requested. The final DCD
concatenates the path from the root to the selected final node.
Supervision biases which segments survive; use unbiased simulations from
saved states for kinetic or thermodynamic estimates.
