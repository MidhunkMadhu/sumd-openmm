"""Small TRR/XVG readers and a lossless restart TRR writer.

TRR is the documented GROMACS XDR format. Only coordinates, velocities,
forces and boxes are supported; obsolete embedded topology blocks fail.
"""

import re
import struct

import numpy as np


def _read(fh, size):
    value = fh.read(size)
    if len(value) != size:
        raise ValueError("truncated TRR file")
    return value


def read_trr(path):
    with open(path, "rb") as fh:
        while True:
            magic = fh.read(4)
            if not magic:
                return
            if len(magic) != 4 or struct.unpack(">i", magic)[0] != 1993:
                raise ValueError("invalid TRR magic")
            # GROMACS strings have a buffer-size integer before the XDR string.
            capacity, length = struct.unpack(">2i", _read(fh, 8))
            if not 0 < length <= capacity <= 256:
                raise ValueError("invalid TRR version")
            version = _read(fh, (length + 3) // 4 * 4)[:length]
            if version.rstrip(b"\0") != b"GMX_trn_file":
                raise ValueError("unsupported TRR version")
            sizes = struct.unpack(">13i", _read(fh, 52))
            ir, energy, box_n, vir, pressure, top, sym, x_n, v_n, f_n, n, step, _ = sizes
            if any((ir, energy, top, sym)) or n <= 0 or min(sizes[:10]) < 0:
                raise ValueError("unsupported TRR blocks")
            precision = box_n // 9 if box_n else next(s // (n * 3) for s in (x_n, v_n, f_n) if s)
            if precision not in (4, 8):
                raise ValueError("invalid TRR precision")
            dtype = ">f%d" % precision
            time, _ = np.frombuffer(_read(fh, 2 * precision), dtype=dtype)
            frame = dict(step=step, time=float(time))
            for name, size in (("box", box_n), ("virial", vir), ("pressure", pressure),
                               ("pos", x_n), ("vel", v_n), ("forces", f_n)):
                if size:
                    expected = 9 * precision if name in ("box", "virial", "pressure") else 3 * n * precision
                    if size != expected:
                        raise ValueError("invalid TRR block size")
                    frame[name] = np.frombuffer(_read(fh, size), dtype=dtype).astype(float).reshape(-1, 3)
            yield frame


def write_trr(path, packet):
    """One double-precision frame; avoid GRO's coordinate/velocity rounding."""
    n = len(packet["pos"])
    version = b"GMX_trn_file"
    with open(path, "wb") as fh:
        fh.write(struct.pack(">3i", 1993, len(version) + 1, len(version)))
        fh.write(version + b"\0" * (-len(version) % 4))
        fh.write(struct.pack(">13i", 0, 0, 72, 0, 0, 0, 0, n * 24, n * 24, 0,
                             n, packet["step"], 0))
        fh.write(struct.pack(">2d", packet["time"], 0))
        for name in ("box", "pos", "vel"):
            fh.write(np.asarray(packet[name], dtype=">f8").tobytes())


def read_xvg(path):
    names, rows = {}, []
    with open(path) as fh:
        for line in fh:
            match = re.match(r'@\s+s(\d+)\s+legend\s+"([^"]+)"', line)
            if match:
                names[int(match[1])] = match[2]
            elif line.strip() and not line.lstrip().startswith(("#", "@")):
                rows.append([float(x) for x in line.split()])
    data = np.asarray(rows, float)
    if data.ndim != 2 or data.shape[1] != len(names) + 1:
        raise ValueError("invalid energy XVG: %s" % path)
    return data[:, 0], {names[i]: data[:, i + 1] for i in sorted(names)}
