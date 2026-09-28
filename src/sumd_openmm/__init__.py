"""
sumd_openmm - Supervised Molecular Dynamics (SuMD) on OpenMM.

Command line:  sumd-openmm run.inp [--dry-run | --test]   (or python -m sumd_openmm)
               sumd-openmm --test                       (check the OpenMM installation)
               sumd-inspect run_dir [...]

Importing the package does not import OpenMM; only building a simulation does.
"""

__version__ = "0.8.0"
