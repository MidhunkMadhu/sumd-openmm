"""Compare the public output contract and deterministic cross-engine results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sumd_openmm.backend import read_state
from sumd_openmm.dcdtools import read_frames
from sumd_openmm.inspect_run import inspect


def compare(a, b):
    a, b = Path(a), Path(b)
    files = lambda p: {str(x.relative_to(p)) for x in p.rglob('*') if x.is_file()
                       and 'engine_work' not in x.relative_to(p).parts}
    report = {'same_public_files': files(a) == files(b),
              'files_only_left': sorted(files(a)-files(b)), 'files_only_right': sorted(files(b)-files(a)),
              'same_csv_columns': True, 'same_json_keys': True}
    for name in sorted(files(a) & files(b)):
        if name.endswith('.csv'):
            with open(a/name) as x, open(b/name) as y:
                if next(csv.reader(x)) != next(csv.reader(y)):
                    report['same_csv_columns'] = False
        if name.endswith('.json'):
            x, y = json.loads((a/name).read_text()), json.loads((b/name).read_text())
            if x.keys() != y.keys():
                report['same_json_keys'] = False
    x, y = read_frames(a/'sumd_traj.dcd'), read_frames(b/'sumd_traj.dcd')
    p, q = read_state(a/'final_state.xml'), read_state(b/'final_state.xml')
    report.update(coordinate_shape=list(x.shape), bit_identical_coordinates=bool(np.array_equal(x,y)),
                  max_coordinate_difference_A=float(np.max(abs(x-y))),
                  final_potential_difference_kJmol=float(abs(p['epot']-q['epot'])),
                  audits=[inspect(str(a)), inspect(str(b))])
    report['passed'] = (report['same_public_files'] and report['same_csv_columns']
                        and report['same_json_keys'] and report['max_coordinate_difference_A'] < .001
                        and report['final_potential_difference_kJmol'] < .001
                        and all(r['final_traj_identical_to_path_windows']
                                and r['max_dcd_vs_state_mismatch_A'] < 1e-4 for r in report['audits']))
    return report


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('left'); ap.add_argument('right'); ap.add_argument('--output')
    args=ap.parse_args()
    result=compare(args.left,args.right)
    text=json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text+'\n')
    raise SystemExit(0 if result['passed'] else 1)
