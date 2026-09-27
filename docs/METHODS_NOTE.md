# Methods note: SuMD on OpenMM (template)

Fill one copy per system before any SuMD-derived data is read alongside the
existing 57.6 µs unbiased set. Every field must match the production runs, or
the difference must be stated. **Fields marked `TO FILL` are system facts not
recorded anywhere in this module. Do not infer them.**

`run_summary.json` in each output directory records the resolved
configuration, atom indices, residue identities, seeds, software versions,
platform and precision automatically. Copy from it rather than retyping.

| Item | Value | Source |
|---|---|---|
| System / PDB | TO FILL (e.g. 8JLP + ralmitaront) | |
| Force field (protein / lipid / ligand) | TO FILL | CHARMM-GUI build |
| Ligand parameterisation | TO FILL (method, charges) | |
| Water model | TO FILL | CHARMM-GUI build |
| Lipid composition | TO FILL | CHARMM-GUI build |
| Ion concentration, species | TO FILL | CHARMM-GUI build |
| Protonation D103(3.32) | TO FILL | |
| Protonation D69(2.50) | TO FILL | |
| Na⁺ placed in allosteric site? | TO FILL | |
| Ligand starting position(s) | TO FILL (distance from site, orientation, per entry) | |
| Topology numbering offset | TO FILL (analysis systems: bio = topo + 18) | `--dry-run` residue report |
| Timestep | `dt` | .inp |
| Constraints | `cons` | .inp |
| Thermostat | Langevin, `temp` K, `fric_coeff` ps⁻¹ | .inp / MD_openmm |
| Barostat | `p_type`, `p_ref` bar, `p_XYMode`, `p_ZMode`, `p_tens`, every `p_freq` steps | .inp / omm_barostat.py |
| Electrostatics | `coulomb`, `ewald_Tol` | .inp |
| vdW | `vdw`, `r_on`–`r_off` nm (Force-switch applied by MD_openmm for CHARMM FF only) | .inp |
| Long-range LJ correction | `lj_lrc` | .inp |
| Platform / precision | `engine.platform`, `CudaPrecision` | run_summary.json |
| OpenMM / CUDA versions | `software`, `engine` | run_summary.json |
| SuMD mode, CVs, cutoffs | `sumd_config` | run_summary.json |
| Window length, CV sampling | `window_ps`, `cv_sample_ps` | run_summary.json |
| Independent SuMD entries | TO FILL (count, seeds) | |
| Aggregate SuMD time (accepted / simulated) | TO FILL | sum over nodes.csv / windows.jsonl |
| Seeded unbiased runs | TO FILL (count × length) | |

## Statements that must appear in any write-up

- SuMD windows are unbiased dynamics, but the selection of windows breaks
  detailed balance. No rate or free energy is computed from SuMD segments.
- Which supervision mode produced the seeds. If `joint` or `walker_score =
  dmscore`, include the joint-mode warning verbatim (see README).
- Whether the seed set covers both gate strata at every CV1 band, and the
  imbalance if it does not.
