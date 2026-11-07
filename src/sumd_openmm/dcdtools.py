"""
dcdtools.py

Concatenate DCD files written by openmm.app.DCDFile, byte for byte, without
reading coordinates. Used to assemble the final supervised trajectory from
the accepted per-window DCDs along the root -> final-node path, and the
thinned copy in check/ after every AcceptedStep.

OpenMM DCD layout (openmm/app/dcdfile.py): a 276-byte header whose frame
count sits at byte 8 and last step at byte 20, then fixed-size frames
(optional 56-byte unit-cell record, then X, Y, Z records of 4*N+8 bytes).
"""

import os
import struct

HEADER_BYTES = 276


class DCDWriter:
    """Write the common CHARMM DCD layout without importing an MD engine.

    Input coordinates and row box vectors are in nm. Frame data is float32
    Angstroms, with cell angles stored as cosines, matching the reader below.
    """

    def __init__(self, fh, natoms, dt_ps, first_step, interval, periodic=True):
        self.fh, self.n, self.first, self.interval = fh, natoms, first_step, interval
        self.periodic, self.count = periodic, 0
        header = bytearray(HEADER_BYTES)
        struct.pack_into("<i4s", header, 0, 84, b"CORD")
        struct.pack_into("<ii", header, 12, first_step, interval)
        struct.pack_into("<f", header, 44, dt_ps / 0.04888821)
        struct.pack_into("<i", header, 48, int(periodic))
        struct.pack_into("<5i", header, 84, 24, 84, 164, 2, 0)
        header[100:180] = b"Created by SuMD".ljust(80)
        header[180:260] = b"Coordinates in Angstroms".ljust(80)
        struct.pack_into("<4i", header, 260, 164, 4, natoms, 4)
        fh.write(header)

    def write(self, positions, box=None):
        import numpy as np
        positions = np.asarray(positions)
        if positions.shape != (self.n, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("invalid DCD coordinates")
        self.count += 1
        self.fh.seek(8)
        self.fh.write(struct.pack("<i", self.count))
        self.fh.seek(20)
        self.fh.write(struct.pack("<i", self.first + (self.count - 1) * self.interval))
        self.fh.seek(0, os.SEEK_END)
        if self.periodic:
            box = np.asarray(box)
            lengths = np.linalg.norm(box, axis=1)
            cosines = [np.dot(box[i], box[j]) / (lengths[i] * lengths[j])
                       for i, j in ((0, 1), (0, 2), (1, 2))]
            self.fh.write(struct.pack("<i6di", 48, lengths[0] * 10, cosines[0],
                                      lengths[1] * 10, cosines[1], cosines[2], lengths[2] * 10, 48))
        size = struct.pack("<i", 4 * self.n)
        for axis in range(3):
            self.fh.write(size)
            self.fh.write(np.asarray(positions[:, axis] * 10, dtype="<f4").tobytes())
            self.fh.write(size)


def read_header(path):
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)

    if len(head) < HEADER_BYTES or struct.unpack("<i", head[:4])[0] != 84 or head[4:8] != b"CORD":
        raise ValueError("%s is not an OpenMM-style DCD" % path)

    nframes = struct.unpack("<i", head[8:12])[0]
    first_step, interval = struct.unpack("<ii", head[12:20])
    box_flag = struct.unpack("<i", head[48:52])[0]
    natoms = struct.unpack("<i", head[268:272])[0]

    frame_bytes = (56 if box_flag else 0) + 3 * (4 * natoms + 8)
    size = os.path.getsize(path)

    if size != HEADER_BYTES + nframes * frame_bytes:
        raise ValueError(
            "%s: size %d does not match header (%d frames x %d bytes + %d)"
            % (path, size, nframes, frame_bytes, HEADER_BYTES)
        )

    return dict(head=head, nframes=nframes, natoms=natoms, box_flag=box_flag,
                frame_bytes=frame_bytes, first_step=first_step, interval=interval)


def concat_dcds(paths, out_path, stride=1):
    """
    Write the frames of `paths`, in order, into one DCD. Returns frame count.

    With stride N, only every Nth frame is written, counted across the joined
    trajectory: the result equals the stride = 1 output sliced [::N].
    """
    if not paths:
        raise ValueError("no DCD files to concatenate")
    if stride < 1:
        raise ValueError("stride must be positive")

    headers = [read_header(p) for p in paths]
    h0 = headers[0]

    for p, h in zip(paths, headers):
        if (h["natoms"], h["box_flag"]) != (h0["natoms"], h0["box_flag"]):
            raise ValueError("%s: atom count / box flag differs from %s" % (p, paths[0]))

    total = sum(h["nframes"] for h in headers)
    kept = (total + stride - 1) // stride
    interval = h0["interval"] * stride
    head = bytearray(h0["head"])
    head[8:12] = struct.pack("<i", kept)
    head[16:20] = struct.pack("<i", interval)
    head[20:24] = struct.pack("<i", h0["first_step"] + (kept - 1) * interval)

    with open(out_path, "wb") as out:
        out.write(head)

        index = 0
        for p, h in zip(paths, headers):
            with open(p, "rb") as f:
                if stride == 1:
                    f.seek(HEADER_BYTES)
                    while True:
                        chunk = f.read(1 << 24)
                        if not chunk:
                            break
                        out.write(chunk)
                else:
                    for k in range(-index % stride, h["nframes"], stride):
                        f.seek(HEADER_BYTES + k * h["frame_bytes"])
                        out.write(f.read(h["frame_bytes"]))
            index += h["nframes"]

    return kept


def read_frames(path):
    """
    Coordinates of every frame, (nframes, natoms, 3) float32 in A, read with a
    memmap. No topology needed (CHARMM-GUI Chamber parm7 files, which
    MDAnalysis cannot read, are therefore not a problem).
    """
    import numpy as np

    h = read_header(path)
    n = h["natoms"]
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER_BYTES,
                    shape=(h["nframes"], h["frame_bytes"]))
    xyz = np.empty((h["nframes"], n, 3), dtype=np.float32)
    start = 56 if h["box_flag"] else 0

    for k in range(3):
        off = start + k * (4 * n + 8) + 4
        xyz[:, :, k] = raw[:, off:off + 4 * n].copy().view("<f4").reshape(h["nframes"], n)

    return xyz
