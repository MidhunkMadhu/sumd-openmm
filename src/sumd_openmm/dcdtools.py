"""
dcdtools.py

Concatenate DCD files written by openmm.app.DCDFile, byte for byte, without
reading coordinates. Used to assemble the final supervised trajectory from
the accepted per-window DCDs along the root -> final-node path.

OpenMM DCD layout (openmm/app/dcdfile.py): a 276-byte header whose frame
count sits at byte 8 and last step at byte 20, then fixed-size frames
(optional 56-byte unit-cell record, then X, Y, Z records of 4*N+8 bytes).
"""

import os
import struct

HEADER_BYTES = 276


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


def concat_dcds(paths, out_path):
    """Write the frames of `paths`, in order, into one DCD. Returns frame count."""
    if not paths:
        raise ValueError("no DCD files to concatenate")

    headers = [read_header(p) for p in paths]
    h0 = headers[0]

    for p, h in zip(paths, headers):
        if (h["natoms"], h["box_flag"]) != (h0["natoms"], h0["box_flag"]):
            raise ValueError("%s: atom count / box flag differs from %s" % (p, paths[0]))

    total = sum(h["nframes"] for h in headers)
    head = bytearray(h0["head"])
    head[8:12] = struct.pack("<i", total)
    head[20:24] = struct.pack("<i", h0["first_step"] + (total - 1) * h0["interval"])

    with open(out_path, "wb") as out:
        out.write(head)

        for p in paths:
            with open(p, "rb") as f:
                f.seek(HEADER_BYTES)
                while True:
                    chunk = f.read(1 << 24)
                    if not chunk:
                        break
                    out.write(chunk)

    return total


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
