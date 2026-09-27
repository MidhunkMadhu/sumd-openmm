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

Deliberate differences: the Langevin integrator is given an explicit
random-number seed so a SuMD run is reproducible from random_seed; the
platform may be auto or HIP, and a GPU request never falls back to the CPU;
and an explicit vdw = Force-switch is also applied to Amber and GROMACS
topologies (CHARMM force fields exported by CHARMM-GUI, like Amber's
fswitch). Every other setting comes from the .inp through MD_openmm's parser.

If openmm_production.py is ever refactored into a build_simulation()
function, re-vendor it and replace build_simulation() here with a call to it.
"""

import re
import sys

GPU_PLATFORMS = ("CUDA", "HIP", "OpenCL")
PLATFORM_ORDER = GPU_PLATFORMS + ("CPU", "Reference")


class PlatformError(RuntimeError):
    pass


def load_production_helpers():
    """The bundled MD_openmm helpers (imports OpenMM)."""
    from .md_openmm import production_helpers
    return production_helpers


def available_platforms():
    from openmm import Platform
    return [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]


def plugin_failures():
    from openmm import Platform
    try:
        return list(Platform.getPluginLoadFailures())
    except Exception:
        return []


def rocm_hint(failures):
    """How to load the HIP plugin when it needs another ROCm release, or None."""
    for failure in failures:
        found = re.search(r"OpenMM\w*HIP\.so: (lib(?:hiprtc|amdhip64)\.so\.(\d+))", failure)
        if found:
            return ("HIP needs %s, from ROCm %s: load a ROCm %s module (e.g. "
                    "module load rocm/%s.x) before running" % (found[1], found[2], found[2], found[2]))
    return None


def choose_platform(requested, enabled, log=print):
    """
    Platform name for `platform = requested`.

    auto picks the fastest available (CUDA, HIP, OpenCL, CPU). A GPU request
    falls back only to another GPU platform: on an AMD node platform = CUDA
    runs on HIP (or OpenCL), and a node without any GPU platform is an error
    rather than a silent CPU run.
    """
    requested = (requested or "auto").strip()
    if requested.lower() == "auto":
        name = next((p for p in PLATFORM_ORDER if p in enabled), None)
    elif requested in enabled:
        return requested
    elif requested in GPU_PLATFORMS:
        name = next((p for p in GPU_PLATFORMS if p in enabled), None)
    else:
        name = None
    hint = rocm_hint(plugin_failures()) if requested in ("HIP", "auto") else None
    if name is None:
        failures = plugin_failures()
        raise PlatformError(
            "OpenMM platform %s is not available; this OpenMM has %s.%s\n%s"
            % (requested, ", ".join(enabled) or "none",
               "\nPlugin load failures:\n  " + "\n  ".join(failures) if failures else "",
               hint or "NVIDIA GPUs need the CUDA platform and AMD GPUs the HIP platform of "
               "OpenMM, with the matching driver or ROCm runtime loaded. Run "
               "`sumd-openmm --test` on a compute node to check."))
    if requested.lower() != "auto":
        log("WARNING: platform %s is not available; using %s" % (requested, name))
    if hint and name != "HIP":
        log("NOTE: " + hint)
    return name


def precision(cfg):
    """GPU precision; cuda_precision is the older name and applies to every GPU platform."""
    return cfg.get("precision", cfg.get("cuda_precision", "single"))


def platform_properties(platform, precision=None, device_index=None, log=print):
    """Precision and DeviceIndex under the names this platform accepts."""
    names = set(platform.getPropertyNames())
    props = {}
    if precision and "Precision" in names:
        props["Precision"] = precision
    if device_index is not None:
        if "DeviceIndex" in names:
            props["DeviceIndex"] = str(device_index)
        else:
            log("NOTE: DeviceIndex %s ignored on the %s platform" % (device_index, platform.getName()))
    return props


def force_switch(prod, system, inputs):
    """
    MD_openmm's CHARMM force switch on an Amber or GROMACS System. It handles
    plain Lennard-Jones and the (a/r6)^2 - b/r6 NBFIX table these readers
    write; other custom LJ forms (12-6-4, combination rule 1) are refused.
    """
    from types import SimpleNamespace
    from openmm import CustomNonbondedForce, NonbondedForce

    customs = [f for f in system.getForces() if isinstance(f, CustomNonbondedForce)]
    if len(customs) > 1 or any(f.getNumTabulatedFunctions() != 2 or
                               not f.getEnergyFunction().startswith("(a/r6)^2-b/r6")
                               for f in customs):
        raise ValueError("vdw = Force-switch supports standard or NBFIX Lennard-Jones only; "
                         "this topology has %s" % ", ".join(f.getEnergyFunction() for f in customs))
    group = next(f.getForceGroup() for f in system.getForces() if isinstance(f, NonbondedForce))
    return prod.vfswitch(system, SimpleNamespace(NONBONDED_FORCE_GROUP=group), inputs)


def build_xml_simulation(prod, cfg, integrator_seed, log=print, device_index=None):
    """Build a Simulation from a serialized OpenMM System and a structure."""
    from types import SimpleNamespace
    from openmm import XmlSerializer, LangevinIntegrator, Platform, MonteCarloBarostat
    from openmm.app import PDBFile, PDBxFile, Simulation
    from openmm.unit import kelvin, picosecond, picoseconds

    path = prod.get_str(cfg, "system_xml", required=True)
    topfile = prod.get_str(cfg, "topology_file", required=True)
    crdfile = prod.get_str(cfg, "coordinate_file", None)
    with open(path) as fh:
        system = XmlSerializer.deserialize(fh.read())
    top = PDBxFile(topfile) if topfile.lower().endswith((".cif", ".pdbx")) else PDBFile(topfile)
    if system.getNumParticles() != top.topology.getNumAtoms():
        raise ValueError("system_xml particle count differs from topology_file atom count")
    temperature = float(cfg.get("temp", 300))
    friction = float(cfg.get("fric_coeff", 1))
    dt = float(cfg.get("dt", 0.002))
    integrator = LangevinIntegrator(temperature * kelvin, friction / picosecond, dt * picoseconds)
    integrator.setRandomNumberSeed(int(integrator_seed))
    requested = cfg.get("platform", "CUDA")
    name = choose_platform(requested, available_platforms(), log)
    platform = Platform.getPlatformByName(name)
    props = platform_properties(platform, precision(cfg), device_index, log)
    sim = Simulation(top.topology, system, integrator, platform, props)
    kind = "structure"
    if crdfile and crdfile.lower().endswith(".xml"):
        with open(crdfile) as fh:
            state = XmlSerializer.deserialize(fh.read())
        sim.context.setState(state)
        kind = "openmm_xml_state"
    elif crdfile and crdfile.lower().endswith(".chk"):
        with open(crdfile, "rb") as fh:
            sim.context.loadCheckpoint(fh.read())
        kind = "openmm_checkpoint"
    elif crdfile:
        coord = PDBxFile(crdfile) if crdfile.lower().endswith((".cif", ".pdbx")) else PDBFile(crdfile)
        sim.context.setPositions(coord.positions)
    else:
        sim.context.setPositions(top.positions)
    genvel = prod.get_genvel(cfg)
    if genvel or kind == "structure":
        sim.context.setVelocitiesToTemperature(temperature * kelvin, int(integrator_seed))
    if cfg.get("reset_step_and_time", "no").lower() in ("yes", "true"):
        sim.currentStep = 0
        sim.context.setTime(0 * picoseconds)
    return SimpleNamespace(
        simulation=sim, system=system, integrator=integrator, inputs=None,
        fftype="OPENMM_XML", platform_name=name, platform_request=requested,
        start_info=dict(kind=kind), topfile=topfile, crdfile=crdfile or topfile,
        genvel=genvel, rewrap_coordinates=False, temperature_K=temperature,
        dt_ps=dt, has_barostat=any(isinstance(f, MonteCarloBarostat)
                                   for f in system.getForces()), device_index=device_index)


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
    if cfg.get("force_field", "AMBER").upper() == "OPENMM_XML":
        return build_xml_simulation(prod, cfg, integrator_seed, log, device_index)

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
    cuda_precision = precision(cfg)

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
        kind = prod.detect_coordinate_file_type(crdfile)
        if kind == "gromacs_gro":
            gro = GromacsGroFile(crdfile)
            top = GromacsTopFile(topfile, periodicBoxVectors=gro.getPeriodicBoxVectors(),
                                 includeDir=gmx_include)
        elif kind == "openmm_xml_state":
            # e.g. the State saved after equilibration: PME needs its box here
            from openmm import XmlSerializer
            with open(crdfile) as fh:
                box = XmlSerializer.deserialize(fh.read()).getPeriodicBoxVectors()
            top = GromacsTopFile(topfile, periodicBoxVectors=box, includeDir=gmx_include)
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
    elif inputs.vdw == "Force-switch" and "vdw" in cfg:
        system = force_switch(prod, system, inputs)
        log("vdw = Force-switch: LJ force switched from %.3g to %.3g nm on the %s topology "
            "(Amber fswitch)" % (inputs.r_on, inputs.r_off, fftype))
    elif inputs.vdw == "Force-switch":
        log("NOTE: MD_openmm's default vdw = Force-switch applies only to CHARMM topologies; this "
            "%s system truncates LJ at %.3g nm. Write vdw = Force-switch to switch it, e.g. for "
            "CHARMM force fields in Amber or GROMACS format." % (fftype, inputs.r_off))

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
        if inputs.p_type not in ("isotropic", "membrane"):
            raise ValueError("p_type must be isotropic or membrane, not %s" % inputs.p_type)
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

    platform_name = choose_platform(platform_request, available_platforms(), log)
    platform = Platform.getPlatformByName(platform_name)
    prop = platform_properties(platform, cuda_precision, device_index, log)

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
