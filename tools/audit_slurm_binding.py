#!/usr/bin/env python3
"""Collect and validate disjoint CPU core sets for eight GPU walkers."""
import json
import os
from pathlib import Path
import sys

from mpi4py import MPI
from sumd_openmm.gromacs import thread_count
from sumd_openmm.parallel import visible_devices

comm = MPI.COMM_WORLD
affinity = sorted(os.sched_getaffinity(0))
cores = set()
for cpu in affinity:
    topology = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
    cores.add((int((topology / 'physical_package_id').read_text()),
               int((topology / 'core_id').read_text())))
record = dict(rank=comm.rank, cpu_affinity=affinity, physical_cores=sorted(cores),
              threads=thread_count({}, os.environ), visible_devices=visible_devices(os.environ),
              gpu_masks={k: os.environ.get(k) for k in
                         ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')})
rows = comm.gather(record, root=0)
if comm.rank == 0:
    seen = set()
    for row in rows:
        assigned = set(map(tuple, row['physical_cores']))
        assert len(assigned) == 8 and row['threads'] == 8, row
        assert not seen.intersection(assigned), rows
        assert row['visible_devices'] is not None and len(row['visible_devices']) == 1, row
        seen.update(assigned)
    assert len(rows) == 8 and len(seen) == 64
    Path(sys.argv[1]).write_text(json.dumps(dict(passed=True, ranks=rows), indent=2) + '\n')
    print('Binding audit passed: 64 disjoint physical cores, 8 cores and one visible GPU per rank.')
