# Outputs

| File | Contents |
| --- | --- |
| `run_summary.json` | Input, resolved metric table with initial values, seeds, setup, stop reason and final path |
| `progress.log` | Timestamped events, warnings and stage changes |
| `windows.jsonl` | One record per attempted walker window |
| `nodes.csv` | One row per saved state; one column per named metric |
| `states/node_N.xml` | Full OpenMM State at each committed endpoint |
| `windows/accepted/*.dcd` | Sampled frames of accepted windows |
| `windows/rejected/*.dcd` | Optional discarded windows |
| `sumd_traj.dcd` | Concatenated final path, when a path has DCD windows |
| `final_path.txt` | State and window sequence used in the final DCD |
| `final_state.xml` | Final OpenMM State |
| `final_state.rst7` | Final Amber restart |
| `ranks/rank_NNN.log` | Non-root MPI rank log |
| `FATAL_rank_NNN.txt` | MPI failure details if a rank aborts |

`windows.jsonl` records cycle, walker, rank, parent and child identifiers,
verdict and reason, seed and restore mode, potential-energy deviation after
restore, wall time, sample times, `metric_series` and `metric_final`
dictionaries keyed by name, continuous `unwrapped_series` for directional
torsions, stage, slope, standard error, R squared, score, and selected
stratification cell. Undefined numerical values are JSON null.

`nodes.csv` includes node and parent IDs, cycle and walker, all metric
values, band, cell, simulation time, step, potential energy, reason,
window DCD and State XML path. `sumd-inspect OUTPUT_DIR` checks frame
identity, State consistency, restore deviations and sampling counts.
