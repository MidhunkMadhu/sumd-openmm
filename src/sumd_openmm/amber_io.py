"""Amber native restart bundles and mdout records; energies are kcal/mol here."""
import io
import re
import zipfile

import numpy as np


def box_vectors(box):
    from parmed import unit
    from parmed.geometry import box_lengths_and_angles_to_vectors
    if box is None:
        return np.zeros((3, 3))
    vectors = box_lengths_and_angles_to_vectors(*np.asarray(box).reshape(6))
    return np.asarray(vectors.value_in_unit(unit.angstrom), dtype=float) / 10


def read_restart(path):
    from parmed.amber import Rst7
    rst = Rst7.open(str(path))
    velocities = rst.velocities
    return dict(pos=np.asarray(rst.coordinates, float).reshape(-1, 3) / 10,
                vel=None if velocities is None else np.asarray(velocities, float).reshape(-1, 3) / 10,
                box=box_vectors(rst.box), time=float(rst.time or 0))


def state_bundle(restart, gamd=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('restart.rst7', restart)
        if gamd is not None:
            archive.writestr('gamd-restart.dat', gamd)
    return buffer.getvalue()


def unpack_bundle(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        if 'restart.rst7' not in names or names - {'restart.rst7', 'gamd-restart.dat'}:
            raise ValueError('unsupported Amber state bundle')
        return {name: archive.read(name) for name in names}


_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?'
_KEYS = (r'NSTEP|TIME\(PS\)|TEMP\(K\)|PRESS|Etot|EKtot|EPtot|BOND|ANGLE|DIHED|'
         r'1-4 NB|1-4 EEL|VDWAALS|EELEC|EHBOND|RESTRAINT|CMAP|UB|IMP|VOLUME|Density|EGB|ESURF')
_PAIR = re.compile(r'(' + _KEYS + r')\s*=\s*(' + _NUMBER + r')', re.IGNORECASE)


def read_mdout(path):
    """Instantaneous records only; exclude averages and RMS summaries."""
    rows, row = {}, None
    with open(path) as handle:
        for line in handle:
            if 'A V E R A G E S' in line or 'R M S  F L U C T U A T I O N S' in line:
                break
            matches = list(_PAIR.finditer(line))
            if any(m[1].upper() == 'NSTEP' for m in matches):
                if row is not None and 'EPTOT' in row:
                    rows[int(row['NSTEP'])] = row
                row = {}
            if row is not None:
                for match in matches:
                    row[match[1].upper()] = float(match[2].replace('D', 'E').replace('d', 'e'))
    if row is not None and 'EPTOT' in row:
        rows[int(row['NSTEP'])] = row
    return rows


def read_trajectory(path):
    from parmed.amber import NetCDFTraj
    trajectory = NetCDFTraj.open_old(str(path))
    try:
        coordinates = np.array(trajectory.coordinates, dtype=float, copy=True) / 10
        times = np.array(trajectory.time, dtype=float, copy=True)
        cells = trajectory.box
        boxes = None if cells is None else np.array([box_vectors(row) for row in cells])
    finally:
        trajectory.close()
    return coordinates, times, boxes
