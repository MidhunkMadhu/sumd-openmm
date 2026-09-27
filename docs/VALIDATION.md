# Validation

Run `pytest` for geometry, selections, configuration, decisions, scoring,
pool behavior, DCD concatenation, state restoration and fake serial and MPI
window loops. Run `sumd-openmm --help`, `python -m build` and
`sumd-inspect OUTPUT_DIR` after installing the package.

An OpenMM XML smoke system with two particles and a harmonic bond was used
to exercise the installed command on CPU. In a three-frame accepted path,
`sumd-inspect` reported identical final DCD frames and path frames and
a maximum DCD versus State deviation of 4.35e-7 angstrom. These
measurements cover the XML path and the output pipeline, not a large
biomolecular system or cluster GPU performance.

For RMSD parity, fit and measure corresponding atom groups on identical
periodic images in cpptraj and compare values with the same sampling
coordinates. A target tolerance of 1e-3 angstrom is appropriate for this
cross-program comparison. cpptraj `autoimage` can place a free molecule
in an image other than the mathematically nearest one; this package
computes the nearest triclinic image by searching 27 neighbors after
fractional wrapping. Compare images before comparing RMSDs.
