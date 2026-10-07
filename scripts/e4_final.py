"""E4/G3 FINAL: 3-seed paired TOST, headline MARL (guide) vs single-agent.

Per-instance makespans averaged over training seeds for each arm (the
pre-specified nuisance-factor treatment), then stats_tost.paired_tost per
cell. G3 verdict: parity (EQUIVALENT) in the majority of cells.

Usage: python scripts/e4_final.py [--seeds 301,302,303]
Writes reports/e4_tost_final.md.
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
ap.add_argument('--seeds', default='301,302,303')
ap.add_argument('--ours_prefix', default='m1-bcb-guide')
ap.add_argument('--single_prefix', default='single-joint')
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(',')]


def seed_mean(prefix, cell):
    return np.mean([np.load(f'test_results/PPVCT/{cell}/Result_greedy+10x25+'
                            f'ppvct-mixed+{prefix}-s{s}_{cell}.npy')[:, 0]
                    for s in SEEDS], axis=0)


cells = []
for d in sorted(glob.glob('test_results/PPVCT/*')):
    cell = d.split('/')[-1]
    ok = all(glob.glob(f'{d}/Result_greedy+10x25+ppvct-mixed+{p}-s{s}_{cell}.npy')
             for p in (args.ours_prefix, args.single_prefix) for s in SEEDS)
    if ok:
        cells.append(cell)

rows, lines = {}, [f'# E4 FINAL paired TOST ({len(SEEDS)} seeds): '
                   f'{args.ours_prefix} vs {args.single_prefix}\n\n']
for cell in cells:
    r = paired_tost(seed_mean(args.ours_prefix, cell),
                    seed_mean(args.single_prefix, cell))
    rows[cell] = r
    line = f'{cell}: {format_row(r)}'
    print(line, flush=True)
    lines.append(f'- {line}\n')

counts = {}
for r in rows.values():
    counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
n_eq = counts.get('EQUIVALENT', 0) + counts.get('OURS BETTER', 0)
g3 = 'PASS' if n_eq > len(rows) / 2 else 'FAIL'
summary = (f'SUMMARY: {len(rows)} cells -> ' +
           ', '.join(f'{k} {v}' for k, v in sorted(counts.items())) +
           f'  ||  G3 (parity-or-better in majority): {g3}')
print(summary)
lines.append(f'\n{summary}\n\n```json\n' + json.dumps(rows, indent=1) + '\n```\n')
os.makedirs('reports', exist_ok=True)
with open('reports/e4_tost_final.md', 'w') as f:
    f.writelines(lines)
print('wrote reports/e4_tost_final.md')
