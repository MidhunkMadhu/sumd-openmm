"""
omm_setup.py

Build the OpenMM Simulation from the same .inp file a production run uses,
through the MD_openmm code bundled in sumd_openmm.md_openmm (see
md_openmm/PROVENANCE.txt; regenerate with tools/vendor_md_openmm.py).

md_openmm.production_helpers holds the imports and function definitions of
MD_openmm's openmm_production.py, verbatim:

    read_key_value_file, build_inputs, get_genvel, normalize_none, get_str,
    get_bool, get_int, detect_coordinate_file_type, load_coordinate_file_auto,
    write_amber_restart, and (via its imports) barostat, vfswitch,
    restraints, rewrap

openmm_production.py's top-level run code is not importable (it starts a
simulation), so build_simulation() below mirrors it section by section
(topology -> createSystem -> vdW / LRC / e14 edits -> barostat -> restraints
-> Langevin integrator -> platform -> Simulation -> coordinate_file ->
velocity policy -> rewrap -> reset).

The one deliberate difference: the Langevin integrator is given an explicit
random-number seed so a SuMD run is reproducible from random_seed. Every
other setting comes from the .inp through MD_openmm's parser.

If openmm_production.py is ever refactored into a build_simulation()
function, re-vendor it and replace build_simulation() here with a call to it.
"""

import sys


def load_production_helpers():
    """The bundled MD_openmm helpers (imports OpenMM)."""
    from .md_openmm import production_helpers
    return production_helpers


def build_simulation(prod, cfg, integrator_seed, log=print, device_index=None):
    """
    Mirror of the top-level setup in openmm_production.py.

    prod : module returned by load_production_helpers()
    cfg  : dict from prod.read_key_value_file(inp)

    device_index : GPU for this process (MPI runs), set as OpenMM's
                   DeviceIndex property on CUDA/OpenCL; None = OpenMM default.

    Returns a SimpleNamespace with simulation, system, integrator, inputs,
    fftype, platform_name, start_info, temperature_K, dt_ps and the file
    names, for the run summary.
    """
    from openmm import (LangevinIntegrator, NonbondedForce, CustomNonbondedForce,
                        Platform)
    from openmm.app import (AmberPrmtopFile, GromacsGroFile, GromacsTopFile,
                            Simulation, LJPME)
    from openmm.unit import kelvin, nanometers, picosecond, picoseconds

    # MD_openmm requires nstep; SuMD runs by window instead, so it is ignored.
    cfg = dict(cfg)
    cfg.setdefault("nstep", "0")

    inputs = prod.build_inputs(cfg)
    fftype = prod.get_str(cfg, "force_field", "AMBER").upper()

    platform_request = prod.get_str(cfg, "platform", "CUDA")
    cuda_precision = prod.get_str(cfg, "cuda_precision", "single")

    topfile = prod.get_str(cfg, "topology_file", required=True)
    crdfile = prod.get_str(cfg, "coordinate_file", required=True)
    toppar = prod.normalize_none(prod.get_str(cfg, "toppar_file", None))
    gmx_include = prod.get_str(cfg, "gmx_include", "toppar")

    genvel = prod.get_genvel(cfg)
    rewrap_coordinates = prod.get_bool(cfg, "rewrap_coordinates", True)
    reset_step_and_time = prod.get_bool(cfg, "reset_step_and_time", False)

    # ---------------------------------------------- topology
    if fftype == "CHARMM":
        if toppar is None:
            sys.exit("Error: CHARMM requires toppar_file")

        from .md_openmm.omm_readparams import read_top, read_params, read_crd, gen_box

        top = read_top(topfile, "CHARMM")
        params = read_params(toppar)

        if prod.detect_coordinate_file_type(crdfile) == "charmm_coordinate":
            try:
                top = gen_box(top, read_crd(crdfile, "CHARMM"))
            except Exception as exc:
                log("WARNING: could not apply CHARMM box from coordinate_file: %s" % exc)

    elif fftype == "AMBER":
        params = None
        top = AmberPrmtopFile(topfile)

    elif fftype == "GROMACS":
        params = None
        if prod.detect_coordinate_file_type(crdfile) == "gromacs_gro":
            gro = GromacsGroFile(crdfile)
            top = GromacsTopFile(topfile, periodicBoxVectors=gro.getPeriodicBoxVectors(),
                                 includeDir=gmx_include)
        else:
            top = GromacsTopFile(topfile, includeDir=gmx_include)

    else:
        sys.exit("Error: force_field must be CHARMM, AMBER, or GROMACS")

    topology = top.topology

    # ---------------------------------------------- system
    nboptions = dict(
        nonbondedMethod=inputs.coulomb,
        nonbondedCutoff=inputs.r_off * nanometers,
        constraints=inputs.cons,
        ewaldErrorTolerance=inputs.ewald_Tol,
    )

    if inputs.vdw == "Switch":
        nboptions["switchDistance"] = inputs.r_on * nanometers

    if inputs.vdw == "LJPME":
        nboptions["nonbondedMethod"] = LJPME

    system = top.createSystem(params, **nboptions) if fftype == "CHARMM" else top.createSystem(**nboptions)

    if fftype == "CHARMM" and inputs.vdw == "Force-switch":
        system = prod.vfswitch(system, top, inputs)

    if inputs.lj_lrc == "yes":
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                force.setUseDispersionCorrection(True)
            if isinstance(force, CustomNonbondedForce) and force.getNumTabulatedFunctions() != 1:
                force.setUseLongRangeCorrection(True)

    if inputs.e14scale != 1.0:
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumExceptions()):
                    a1, a2, chg, sig, eps = force.getExceptionParameters(i)
                    force.setExceptionParameters(i, a1, a2, chg * inputs.e14scale, sig, eps)
                break

    if inputs.pcouple == "yes":
        system = prod.barostat(system, inputs)

    if fftype == "CHARMM" and inputs.rest == "yes":
        if prod.detect_coordinate_file_type(crdfile) != "charmm_coordinate":
            sys.exit("CHARMM restraints require coordinate_file to be a CHARMM coordinate or PDB file.")
        from .md_openmm.omm_readparams import read_crd
        system = prod.restraints(system, read_crd(crdfile, "CHARMM"), inputs)

    if inputs.rest == "yes":
        log("WARNING: rest = yes. SuMD windows must be unbiased dynamics; "
            "restraints change the Hamiltonian.")

    # ---------------------------------------------- integrator and platform
    integrator = LangevinIntegrator(inputs.temp * kelvin, inputs.fric_coeff / picosecond,
                                    inputs.dt * picoseconds)
    integrator.setRandomNumberSeed(int(integrator_seed))

    enabled = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]

    if platform_request in enabled:
        platform_name = platform_request
    else:
        platform_name = next((p for p in ("CUDA", "OpenCL", "CPU") if p in enabled), None)
        if platform_name is None:
            sys.exit("Error: no usable OpenMM platform found.")
        log("Requested platform %s not available; falling back to %s." % (platform_request, platform_name))

    platform = Platform.getPlatformByName(platform_name)
    prop = dict(CudaPrecision=cuda_precision) if platform_name == "CUDA" else dict()

    if device_index is not None:
        if platform_name in ("CUDA", "OpenCL"):
            prop["DeviceIndex"] = str(device_index)
        else:
            log("NOTE: DeviceIndex %s ignored on the %s platform" % (device_index, platform_name))

    simulation = Simulation(topology, system, integrator, platform, prop)

    # ---------------------------------------------- coordinate_file, velocity policy
    start_info = prod.load_coordinate_file_auto(simulation=simulation, filename=crdfile,
                                                fftype=fftype, genvel=genvel)

    if genvel:
        log("Generating velocities from temperature: %s K" % inputs.temp)
        simulation.context.setVelocitiesToTemperature(inputs.temp * kelvin, int(integrator_seed))
    elif start_info["kind"] in ("openmm_xml_state", "openmm_checkpoint"):
        if not start_info["has_velocities"]:
            sys.exit("genvel = no was requested, but coordinate_file has no velocities:\n%s" % crdfile)
    else:
        prod.use_velocities_from_coordinate_object(simulation, start_info["coordinate_object"],
                                                   start_info["velocity_source"], required=True)

    if rewrap_coordinates:
        simulation = prod.rewrap(simulation)

    if reset_step_and_time:
        simulation.currentStep = 0
        simulation.context.setTime(0.0 * picoseconds)

    from types import SimpleNamespace

    return SimpleNamespace(
        simulation=simulation, system=system, integrator=integrator, inputs=inputs,
        fftype=fftype, platform_name=platform_name, platform_request=platform_request,
        cuda_precision=cuda_precision, start_info=start_info, topfile=topfile,
        crdfile=crdfile, genvel=genvel, rewrap_coordinates=rewrap_coordinates,
        temperature_K=float(inputs.temp), dt_ps=float(inputs.dt),
        has_barostat=(inputs.pcouple == "yes"), device_index=device_index,
    )
