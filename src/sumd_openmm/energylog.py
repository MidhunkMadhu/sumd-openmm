"""
energylog.py

Energy logs. Each window writes its numbers to a CSV, and the CSVs of an
accepted path are joined as sumd_traj.dcd is joined from the window DCDs. A
CSV is rendered into an Amber-style mdout, the .log of a window or of a path.
OpenMM is not needed here.
"""

import csv
import datetime
import os
from collections import OrderedDict

import numpy as np

COLUMNS = ["Step", "Time_ps", "Epot_kJmol", "Ekin_kJmol", "Etot_kJmol", "Temp_K", "Pressure_bar",
           "Volume_nm3", "Density_gmL", "Wall_s"]
LABELS = ["Cycle", "AcceptedStep", "Walker"]
ENERGY_TERMS = ("E_bond", "E_angle", "E_dihedral", "E_vdw", "E_elec_direct", "E_nonbonded",
                "E_elec_recip", "E_nonbonded_recip", "E_other")
FINAL_LOG = "sumd_energy.log"
FINAL_DATA = "sumd_energy.csv"
CHECK_DATA = "sumd_energy_log.csv"

KJ_PER_KCAL = 4.184


def format_row(step, time_ps, epot, ekin, temperature, pressure, volume, density, wall,
               terms, metrics):
    """One energy-CSV row in the OpenMM backend's number formats; None is blank."""
    def f(fmt, value):
        return "" if value is None or value == "" else fmt % float(value)
    fields = ["%d" % int(step), f("%.4f", time_ps), f("%.4f", epot), f("%.4f", ekin),
              f("%.4f", None if epot is None or ekin is None else float(epot) + float(ekin)),
              f("%.3f", temperature), f("%.3f", pressure), f("%.5f", volume), f("%.5f", density),
              f("%.3f", wall)]
    fields += [f("%.4f", terms.get(name)) for name in ENERGY_TERMS]
    fields += ["%.4f" % value for value in metrics]
    return ",".join(fields) + "\n"
TERM_NAMES = ("BOND", "ANGLE", "DIHED", "EELEC", "NONBND", "RECIP", "OTHER", "VDWAALS")
SECTION = "-" * 80
RECORD_END = " " + "-" * 78


def log_path(dcd_path):
    """The mdout of a window DCD."""
    return os.path.splitext(dcd_path)[0] + ".log"


def data_path(dcd_path):
    """The energy CSV of a window DCD. Windows of version 0.9.0 keep it under the .log name."""
    path = os.path.splitext(dcd_path)[0] + ".csv"
    old = log_path(dcd_path)
    return old if os.path.exists(old) and not os.path.exists(path) else path


def window_files(dcd_path):
    """The energy files of a window DCD that exist."""
    return [p for p in {data_path(dcd_path), log_path(dcd_path)} if os.path.exists(p)]


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
    skip = {"Step", "Time_ps", "Wall_s", *LABELS}
    return {name: dict(mean=round(float(np.mean(column)), 4), std=round(float(np.std(column)), 4))
            for name, column in ((n, data[:, i][np.isfinite(data[:, i])]) for i, n in enumerate(names))
            if name not in skip and column.size}


def _cell(name, value, pad=6):
    return "%-*s =%*.4f" % (pad, name, 15 - max(0, len(name) - pad), value)


def _row(pairs):
    return " " + "  ".join(_cell(name, value, pad) for (name, value), pad in zip(pairs, (6, 7, 10)))


def _record(row, metrics):
    """The values of one CSV row in Amber's names and units. A value the row lacks is left out."""
    def number(key, scale=1.0):
        return float(row[key]) * scale if row.get(key) not in (None, "") else None

    def energy(key):
        return number(key, 1 / KJ_PER_KCAL)

    direct, recip = energy("E_elec_direct"), energy("E_elec_recip")
    values = OrderedDict([
        ("NSTEP", number("Step")), ("TIME(PS)", number("Time_ps")), ("TEMP(K)", number("Temp_K")),
        ("PRESS", number("Pressure_bar")),
        ("Etot", energy("Etot_kJmol")), ("EKtot", energy("Ekin_kJmol")), ("EPtot", energy("Epot_kJmol")),
        ("BOND", energy("E_bond")), ("ANGLE", energy("E_angle")), ("DIHED", energy("E_dihedral")),
        ("EELEC", None if direct is None else direct + (recip or 0.0)),
        ("NONBND", energy("E_nonbonded")),
        ("RECIP", energy("E_nonbonded_recip") if direct is not None or recip is None else recip),
        ("OTHER", energy("E_other")), ("VDWAALS", energy("E_vdw")),
        ("VOLUME", number("Volume_nm3", 1000.0)), ("Density", number("Density_gmL")),
    ])
    values.update((name, number(name)) for name in metrics)
    return OrderedDict((key, value) for key, value in values.items() if value is not None)


def _block(values, metrics):
    """One record, laid out as Amber prints it."""
    lines = [" NSTEP =%9d   TIME(PS) =%12.3f  TEMP(K) =%9.2f  PRESS =%8.1f"
             % (values["NSTEP"], values["TIME(PS)"], values.get("TEMP(K)", 0.0), values.get("PRESS", 0.0))]
    fields = [(name, values[name]) for name in ("Etot", "EKtot", "EPtot") if name in values]
    fields += [(name, values[name]) for name in TERM_NAMES if name in values]
    lines += [_row(fields[i:i + 3]) for i in range(0, len(fields), 3)]
    lines += [" " * 52 + _cell(name, values[name], 10) for name in ("VOLUME", "Density") if name in values]
    lines += ["| %-24s %14.4f" % (name, values[name]) for name in metrics if name in values]
    return lines + [RECORD_END, "", ""]


def _statistics(records, function):
    """`function` of each value over `records`, with the step and time of the last one."""
    result = OrderedDict()
    for key in records[-1]:
        series = [record[key] for record in records if key in record]
        result[key] = float(function(series))
    result.update(NSTEP=records[-1]["NSTEP"], **{"TIME(PS)": records[-1]["TIME(PS)"]})
    return result


def _title(number, text):
    return [SECTION, "   %d.  %s" % (number, text), SECTION, ""]


def _preamble(meta, steps, ntpr, out_path):
    now = datetime.datetime.now()
    lines = ["", " " * 10 + "-" * 55, " " * 10 + "sumd-openmm %s" % meta.get("version", ""),
             " " * 10 + "-" * 55, "", "| Run on %s" % now.strftime("%m/%d/%Y at %H:%M:%S"), "",
             "File Assignments:", "| MDOUT: %s" % os.path.basename(out_path)]
    if meta.get("topology"):
        lines.append("|  PARM: %s" % meta["topology"])
    if meta.get("coordinates"):
        lines.append("|INPCRD: %s" % meta["coordinates"])
    lines.append("")
    if meta.get("input"):
        lines += ["", " Here is the input file:", ""] + meta["input"].rstrip().splitlines() + [""]

    lines += _title(1, "RESOURCE   USE: ")
    for key, label in (("platform", "Platform"), ("host", "Host")):
        if meta.get(key):
            lines.append("| %s: %s" % (label, meta[key]))
    lines.append("")

    lines += _title(2, "CONTROL  DATA  FOR  THE  RUN")
    lines += ["General flags:", "     imin    =%8d, irest   =%8d" % (0, 1), "",
              "Molecular dynamics:", "     nstlim  =%8d, ntpr    =%8d" % (steps, ntpr)]
    if meta.get("dt_ps"):
        lines.append("     dt      =%10.5f" % meta["dt_ps"])
    if meta.get("temperature_K"):
        lines.append("     temp0   =%10.5f" % meta["temperature_K"])
    lines.append("")

    return lines + _title(3, "ATOMIC COORDINATES AND VELOCITIES") + _title(4, "RESULTS")


def _timings(wall, steps, dt_ps):
    lines = _title(5, "TIMINGS")
    if wall:
        lines += ["| Final Performance Info:", "| " + "-" * 53, "| Average timings for all steps:",
                  "|     Elapsed(s) =%11.2f Per Step(ms) =%11.2f" % (wall, 1000 * wall / steps)]
        if dt_ps:
            per_ns = wall / (steps * dt_ps / 1000)
            lines.append("|         ns/day =%11.2f   seconds/ns =%11.2f" % (86400 / per_ns, per_ns))
        lines.append("| " + "-" * 53)
    return lines


def render(csv_path, out_path, meta=None, window=None):
    """
    Write the rows of an energy CSV as an Amber-style mdout. Returns the record count.

    meta: what the header reports (version, input, platform, host, topology,
    coordinates, dt_ps, temperature_K) and ntpr, the MD steps per record.
    window: the labels of a CSV that has no label columns, such as dict(Cycle=3, Walker=0).
    """
    meta = meta or {}
    with open(csv_path, newline="") as fh:
        rows = [{**(window or {}), **row} for row in csv.DictReader(fh)]
    if not rows:
        return 0

    known = {*COLUMNS, *LABELS, *ENERGY_TERMS}
    metrics = [name for name in rows[0] if name not in known]
    ntpr = int(meta.get("ntpr") or 1)
    steps = ntpr * len(rows)

    lines = _preamble(meta, steps, ntpr, out_path)
    records, walls, current = [], OrderedDict(), None
    for row in rows:
        label = "  ".join("%s %s" % (key, row[key]) for key in LABELS if row.get(key) not in (None, ""))
        if label and label != current:
            lines += ["| " + label, ""]
        current = label
        record = _record(row, metrics)
        records.append(record)
        lines += _block(record, metrics)
        if row.get("Wall_s"):
            walls[label] = float(row["Wall_s"])

    for title, function in (("A V E R A G E S   O V E R %6d R E C O R D S" % len(records), np.mean),
                            ("R M S  F L U C T U A T I O N S", np.std)):
        lines += ["      " + title, "", ""] + _block(_statistics(records, function), metrics)
    lines += _timings(sum(walls.values()), steps, meta.get("dt_ps"))

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(records)
