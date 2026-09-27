# Simulation setup

`omm_setup.build_simulation` reads the same MD keys as MD_openmm. The
bundled `md_openmm/production_helpers.py` contains the imports and
function definitions of its production driver: input parsing and
coordinate loading. `omm_utils.py` provides the barostat, CHARMM force
switching, molecule rewrapping and CHARMM file readers.
`build_simulation` assembles the force field, nonbonded interactions,
optional barostat, Langevin integrator, platform, topology and initial
state. An explicit integrator seed makes retries reproducible. The
vendored provenance is recorded in `md_openmm/PROVENANCE.txt`.

To refresh the bundled helpers from an MD_openmm checkout, run
`python tools/vendor_md_openmm.py /path/to/MD_openmm`, inspect the
changes and rerun the tests. Do not edit the vendored helper files
individually.

`force_field=OPENMM_XML` instead deserializes `system_xml` as an OpenMM
System, loads topology and starting positions from PDB or PDBx, and builds
a Langevin integrator from `temp`, `fric_coeff` and `dt`. Optional
`coordinate_file` supplies a State XML, checkpoint, PDB or PDBx. The
System's barostat stays in place; this path never adds another barostat.
The topology's atom count must match the System's particle count.
