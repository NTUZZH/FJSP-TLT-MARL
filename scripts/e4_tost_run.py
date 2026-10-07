"""E4 driver: paired TOST (ours vs single-agent) per cell from eval .npy files.

Pairing is by instance order (both evals iterate the same sorted stems).
With one training seed the seed-averaging step of the pre-specified
protocol is trivial; rerun after seeds 302/303 to produce the final table.

Usage: python scripts/e4_tost_run.py --ours 10x25+ppvct-mixed+m1-bcb-s301 \
           --single 10x25+ppvct-mixed+single-joint-s301 [--cells all]
Writes reports/e4_tost.md (table + JSON block).
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

import numpy as np

from stats_tost import paired_tost, format_row

ap = argparse.ArgumentParser()
ap.add_argument('--ours', required=True)
ap.add_argument('--single', required=True)
ap.add_argument('--cells', default='all')
args = ap.parse_args()


def cells_list():
    if args.cells != 'all':
        return args.cells.split(',')
    out = []
    for d in sorted(glob.glob('test_results/PPVCT/*')):
        cell = d.split('/')[-1]
        if (glob.glob(f'{d}/Result_greedy+{args.ours}_{cell}.npy') and
                glob.glob(f'{d}/Result_greedy+{args.single}_{cell}.npy')):
            out.append(cell)
    return out


rows = {}
lines = ['# E4 paired TOST: ours vs single-agent (per cell)\n',
         f'- ours: `{args.ours}`  single: `{args.single}`\n',
         '- protocol: stats_tost.py (pre-specified 2026-07-11); '
         'seeds averaged per instance (currently s301 only)\n\n']
for cell in cells_list():
    ours = np.load(f'test_results/PPVCT/{cell}/Result_greedy+{args.ours}_{cell}.npy')[:, 0]
    single = np.load(f'test_results/PPVCT/{cell}/Result_greedy+{args.single}_{cell}.npy')[:, 0]
    r = paired_tost(ours, single)
    rows[cell] = r
    line = f'{cell}: {format_row(r)}'
    print(line, flush=True)
    lines.append(f'- {line}\n')

n_eq = sum(1 for r in rows.values() if r['verdict'] == 'EQUIVALENT')
n_ob = sum(1 for r in rows.values() if r['verdict'] == 'OURS BETTER')
n_sb = sum(1 for r in rows.values() if r['verdict'] == 'SINGLE BETTER')
n_in = sum(1 for r in rows.values() if r['verdict'] == 'INCONCLUSIVE')
summary = (f'SUMMARY: {len(rows)} cells -> EQUIVALENT {n_eq}, OURS BETTER '
           f'{n_ob}, SINGLE BETTER {n_sb}, INCONCLUSIVE {n_in}')
print(summary, flush=True)
lines.append(f'\n{summary}\n\n```json\n' + json.dumps(rows, indent=1) + '\n```\n')
os.makedirs('reports', exist_ok=True)
with open('reports/e4_tost.md', 'w') as f:
    f.writelines(lines)
print('wrote reports/e4_tost.md')
