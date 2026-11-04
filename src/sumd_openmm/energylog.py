"""
energylog.py

Energy logs: one CSV per window, written next to its DCD, and the continuous
log of an accepted path, joined from them as sumd_traj.dcd is joined from the
window DCDs. OpenMM is not needed here.
"""

import csv
import os

import numpy as np

COLUMNS = ["Step", "Time_ps", "Epot_kJmol", "Ekin_kJmol", "Etot_kJmol", "Temp_K",
           "Volume_nm3", "Density_gmL"]
LABELS = ["Cycle", "AcceptedStep", "Walker"]
FINAL_LOG = "sumd_energy.log"
CHECK_LOG = "sumd_energy_log.csv"


def log_path(dcd_path):
    """The energy log that belongs to a window DCD."""
    return os.path.splitext(dcd_path)[0] + ".log"


def _header(path):
    with open(path, newline="") as fh:
        return next(csv.reader(fh), [])


def join_logs(windows, out_path):
    """
    Write the rows of `windows`, in order, into one CSV. Returns the row count.

    windows: (log path, cycle, accepted step, walker) per window. A window
    without a log is left out, and with no log at all no file is written.
    Windows with other columns leave the columns they lack empty.
    """
    present = [w for w in windows if os.path.exists(w[0])]
    if not present:
        return 0

    columns = []
    for window in present:
        columns += [name for name in _header(window[0]) if name not in columns]

    rows = 0
    with open(out_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns[:2] + LABELS + columns[2:], restval="")
        writer.writeheader()
        for path, cycle, step, walker in present:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    row.update(Cycle=cycle, AcceptedStep=step, Walker=walker)
                    writer.writerow(row)
                    rows += 1
    return rows


def summarise(path):
    """Mean and standard deviation of every measured column of a joined log."""
    names = _header(path)
    data = np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2)
    skip = {"Step", "Time_ps", *LABELS}
    return {name: dict(mean=round(float(np.mean(column)), 4), std=round(float(np.std(column)), 4))
            for name, column in ((n, data[:, i][np.isfinite(data[:, i])]) for i, n in enumerate(names))
            if name not in skip and column.size}
