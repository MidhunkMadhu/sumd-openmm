# Equilibration

By default (`equilibration = none`) sumd-openmm runs production only:
`coordinate_file` must already be equilibrated, by any protocol and
software. This keeps full control of equilibration with the user.

`equilibration = charmm-gui` first runs the equilibration protocol of a
CHARMM-GUI Amber or GROMACS input folder in OpenMM, then starts SuMD from
the final state:

```ini
equilibration = charmm-gui
charmm_gui_dir = charmm-gui-1234567890      # the download, or its amber/ or gromacs/ folder
charmm_gui_format = amber                   # amber | gromacs; auto if only one is present
equilibration_dir = equilibration
```

`force_field`, `topology_file` and `coordinate_file` then come from the
folder; do not set `coordinate_file`. The cutoff, force switching,
constraints and the thermostat and barostat of `step7_production` become
the production defaults; any MD key in the input file overrides them.

## What is read

| | Amber folder | GROMACS folder |
| --- | --- | --- |
| Topology, start and restraint reference | `step5_input.parm7`, `step5_input.rst7` | `topol.top` + `toppar/`, `step5_input.gro` |
| Stages | `step6.0_minimization.mdin`, `step6.N_equilibration.mdin` | the same names with `.mdp` |
| Minimization | `imin=1`, `maxcyc` | `integrator = steep`, `nsteps` |
| Dynamics | `nstlim`, `dt`, `temp0`, `gamma_ln`, `irest`/`tempi` | `nsteps`, `dt`, `ref_t`, `tau_t`, `gen_vel`/`gen_temp` |
| Pressure | `ntp` 1/2/3 (3 with `csurften`: membrane), `pres0` | `pcoupl`, `pcoupltype` (semiisotropic: membrane), `ref_p` |
| Positional restraints | `ntr=1` GROUP input (`RES`, `ATOM`, `FIND`/`SEARCH`) or `restraintmask`/`restraint_wt` | `[ position_restraints ]` under the stage's `define` |
| Dihedral restraints | `nmropt=1`: `dihe.restraint` with the README's per-stage `FC` | `[ dihedral_restraints ]` under `-DDIHRES` |
| Nonbonded | `cut`, `fswitch`, `vdwmeth` | `rvdw`, `rvdw_switch`, `vdw-modifier`, `DispCorr` |

Restraint energies keep each engine's definition: Amber `w |r - r0|^2`
and flat-bottom `&rst` torsions with linear walls; GROMACS
`1/2 k (r - r0)^2` per component and `1/2 k max(0, |phi - phi0| - dphi)^2`.
On the tested system the Amber positional (141.81 kcal/mol) and torsional
(13.55 kcal/mol) restraint energies equal sander's.

Differences from the original engines: Langevin dynamics replaces
GROMACS's v-rescale thermostat (friction `1/tau_t`); OpenMM's Monte Carlo
barostats replace Berendsen, C-rescale and Parrinello-Rahman coupling;
restraint references are not scaled with the box (GROMACS
`refcoord_scaling`); minimization stops at an RMS force of 10 kJ/mol/nm.

## Output and progress

`equilibration_dir` receives, per stage, `<stage>.log` (energies,
temperature, volume, density, speed), `<stage>.dcd` and `<stage>.xml`,
then `equilibrated.xml` and, for Amber, `equilibrated.rst7`.
`equilibration.log` and the run's `progress.log` record each stage as it
starts and ends:

```
stage: equilibration - CHARMM-GUI AMBER protocol, 7 stages, in .../equilibration
stage 2/7 started: step6.1_equilibration: NVT 310 K, 125 ps at dt 1 fs; restraints: Protein posres 5146 atoms k=4184; ...
stage 2/7 finished on CUDA in 0:03:10: Epot ... kJ/mol, T 309.6 K, box 81.53 81.53 110.82 A
stage: SuMD production, 8 walker(s), up to 500 cycles of 20 ps
```

A stage is complete once its `.xml` exists. Rerunning the same input skips
completed stages, so a job stopped by its time limit resumes, and once
`equilibrated.xml` exists later runs go straight to production (for
example a new SuMD run with other metrics). Delete `equilibration_dir` to
equilibrate again. `--dry-run` lists the stages without running them;
`--test` runs every stage for 50 steps in `<equilibration_dir>_test`
before the two test cycles. With MPI, rank 0 equilibrates while the other
ranks wait.
