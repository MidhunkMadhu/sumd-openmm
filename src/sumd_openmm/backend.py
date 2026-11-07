"""Engine contract and portable state I/O. No MD engine is imported here.

Distances are nm, velocities nm/ps, time ps and energy kJ/mol. Native data
is opaque to the supervisor and travels with a state, including over MPI.
"""

import base64
import io
import json
import os
import xml.etree.ElementTree as ET
from typing import Protocol

import numpy as np

OUTPUT_SCHEMA_VERSION = 2


class MDBackend(Protocol):
    name: str
    dt_ps: float
    temperature_K: float
    platform_name: str
    crdfile: str

    def snapshot(self): ...
    def restore(self, snap, check=True): ...
    def packet(self, snap): ...
    def restore_packet(self, packet, check=True): ...
    def time_ps(self, snap): ...
    def reseed(self, seed): ...
    def current_cvs(self, evaluator): ...
    def molecules(self): ...
    def save_state(self, snap, path): ...
    def load_state(self, path, step, epot): ...
    def run_window(self, n_samples, steps_per_sample, evaluator, dcd_path, stride,
                   log_path=None, log_stride=0): ...


def write_state(path, packet, engine, native=b"", native_format=""):
    """Atomic, versioned state envelope; no pickle or engine dependency."""
    root = ET.Element("SuMDState", version="1", engine=engine,
                      step=str(packet["step"]), time_ps=repr(packet["time"]),
                      epot_kJmol=repr(packet["epot"]))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, pos=packet["pos"], vel=packet["vel"], box=packet["box"])
    ET.SubElement(root, "Arrays", encoding="base64-npz", units="nm,nm/ps").text = (
        base64.b64encode(buffer.getvalue()).decode("ascii"))
    ET.SubElement(root, "Parameters").text = json.dumps(packet.get("params", {}))
    ET.SubElement(root, "Native", format=native_format, encoding="base64").text = (
        base64.b64encode(native).decode("ascii"))
    tmp = str(path) + ".tmp"
    ET.ElementTree(root).write(tmp, encoding="utf-8", xml_declaration=True)
    os.replace(tmp, path)


def read_state(path):
    root = ET.parse(path).getroot()
    if root.tag != "SuMDState" or root.get("version") != "1":
        raise ValueError("%s is not a supported SuMD state" % path)
    with np.load(io.BytesIO(base64.b64decode(root.findtext("Arrays"))), allow_pickle=False) as data:
        packet = {key: np.array(data[key], dtype=float) for key in ("pos", "vel", "box")}
    if (packet["pos"].ndim != 2 or packet["pos"].shape[1] != 3
            or packet["vel"].shape != packet["pos"].shape or packet["box"].shape != (3, 3)
            or not all(np.all(np.isfinite(a)) for a in packet.values())):
        raise ValueError("%s: invalid state arrays" % path)
    packet.update(step=int(root.get("step")), time=float(root.get("time_ps")),
                  epot=float(root.get("epot_kJmol")),
                  params=json.loads(root.findtext("Parameters") or "{}"),
                  engine=root.get("engine"),
                  native=base64.b64decode(root.findtext("Native") or ""),
                  native_format=root.find("Native").get("format", ""))
    return packet


def is_portable_state(path):
    with open(path, "rb") as fh:
        return b"<SuMDState" in fh.read(256)


def lengths_angles(box):
    lengths = np.linalg.norm(box, axis=1)
    angles = [np.degrees(np.arccos(np.clip(np.dot(box[i], box[j]) /
              (lengths[i] * lengths[j]), -1, 1))) for i, j in ((1, 2), (0, 2), (0, 1))]
    return np.r_[lengths * 10, angles]


def wrapped_positions(positions, box, molecules):
    """Translate whole molecules into the cell using their geometric centers."""
    positions = np.array(positions, copy=True)
    inverse = np.linalg.inv(box)
    for group in molecules:
        indices = np.asarray(group, dtype=int)
        positions[indices] -= np.floor(positions[indices].mean(axis=0) @ inverse) @ box
    return positions


def write_restart(path, packet, periodic=True, molecules=None):
    """Common Amber ASCII interchange export, independent of the MD engine."""
    from parmed.amber import Rst7
    rst = Rst7(natom=len(packet["pos"]), title="SuMD accepted state")
    rst.time = packet["time"]
    positions = (wrapped_positions(packet["pos"], packet["box"], molecules)
                 if periodic and molecules is not None else packet["pos"])
    rst.coordinates = positions * 10
    rst.vels = packet["vel"] * 10
    if periodic:
        rst.box = lengths_angles(packet["box"])
    rst.write(path)
    return path


def create_backend(cfg, scfg, seed, outdir, rank=0, device=None, log=print):
    """Only the selected backend loads its engine libraries."""
    if scfg.engine == "gromacs":
        from .gromacs import GromacsEngine
        return GromacsEngine(cfg, scfg, seed, outdir, rank, device, log)
    if scfg.engine == "amber":
        from .amber import AmberEngine
        return AmberEngine(cfg, scfg, seed, outdir, rank, device, log)
    from .engine import Engine
    from .omm_setup import build_simulation, load_production_helpers
    omm = build_simulation(load_production_helpers(), cfg, seed, log=log,
                           device_index=device,
                           energy_terms=scfg.energy_terms and scfg.energy_log_stride > 0)
    return Engine(omm, log)
