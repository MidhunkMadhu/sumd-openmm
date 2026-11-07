"""Generate a 64-atom periodic Lennard-Jones validation system."""
from pathlib import Path
import numpy as np


def generate(folder):
    folder = Path(folder)
    rng = np.random.default_rng(733)
    positions = np.array(list(np.ndindex(4, 4, 4)), float) * .65 + .35
    positions += rng.uniform(-.025, .025, positions.shape)
    velocities = rng.normal(0, .03, positions.shape)
    velocities -= velocities.mean(axis=0)
    lines = ['64-atom LJ validation', str(len(positions))]
    for i, (p, v) in enumerate(zip(positions, velocities), 1):
        lines.append(f'{i:5d}{"ARG":<5}{"AR":>5}{i:5d}' + ''.join(f'{x:8.3f}' for x in p)
                     + ''.join(f'{x:8.4f}' for x in v))
    lines.append('   2.80000   2.80000   2.80000')
    (folder / 'argon.gro').write_text('\n'.join(lines) + '\n')
    (folder / 'argon.top').write_text('''[ defaults ]
1 2 yes 1.0 1.0
[ atomtypes ]
AR 18 39.948 0.0 A 0.34 0.997
[ moleculetype ]
ARG 0
[ atoms ]
1 AR 1 ARG AR 1 0.0 39.948
[ system ]
64 argon atoms
[ molecules ]
ARG 64
''')


if __name__ == '__main__':
    generate(Path(__file__).parent)
