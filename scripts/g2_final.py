"""GATE G2 FINAL CALL (3 seeds), per the pre-registered seed-extension rule.

Unit of replication: the test instance. Per-instance makespans are averaged
over the training seeds for EACH arm (nuisance factor, mirroring the TOST
protocol), then the pre-registered paired one-sided Wilcoxon (m1 < joint)
with Holm correction across the four gate cells is applied. PASS requires
all four significant, exactly as in attempt 1.

Usage: python scripts/g2_final.py [--seeds 301,302,303]
Writes notes/gate_G2_final.md.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, '.')

import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', default='301,302,303')
ap.add_argument('--alpha', type=float, default=0.05)
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(',')]
CELLS = ['v1+t0.3', 'v2+t0.3', 'v1+t0.6', 'v2+t0.6']


def seed_mean(prefix, cell):
    arrs = [np.load(f'test_results/PPVCT/{cell}/Result_greedy+10x25+'
                    f'ppvct-mixed+{prefix}-s{s}_{cell}.npy')[:, 0]
            for s in SEEDS]
    return np.mean(arrs, axis=0)


rows = []
for cell in CELLS:
    m1 = seed_mean('m1-bcb', cell)
    jt = seed_mean('joint-v1', cell)
    p = float(wilcoxon(m1, jt, alternative='less').pvalue)
    w = int((m1 < jt - 1e-6).sum())
    l = int((m1 > jt + 1e-6).sum())
    rows.append(dict(cell=cell, n=len(m1), m1_mean=float(m1.mean()),
                     joint_mean=float(jt.mean()),
                     improv_pct=float((jt.mean() - m1.mean()) / jt.mean() * 100),
                     wtl=f'{w}/{len(m1)-w-l}/{l}', p_raw=p))

order = np.argsort([r['p_raw'] for r in rows])
m = len(rows)
holm = {}
running_max = 0.0
for rank, idx in enumerate(order):
    adj = min(1.0, (m - rank) * rows[idx]['p_raw'])
    running_max = max(running_max, adj)
    holm[idx] = running_max
for i, r in enumerate(rows):
    r['p_holm'] = holm[i]
    r['significant'] = bool(holm[i] <= args.alpha)

verdict = 'PASS' if all(r['significant'] for r in rows) else 'FAIL'
lines = [f'# Gate G2 FINAL ({len(SEEDS)} seeds: {SEEDS})\n',
         '- per-instance seed-averaged makespans; paired one-sided Wilcoxon '
         '(m1 < joint); Holm across 4 cells; PASS requires all 4\n\n',
         '| cell | n | m1(3-seed) | joint(3-seed) | improv% | W/T/L | p_raw | p_holm | sig |\n',
         '|---|---|---|---|---|---|---|---|---|\n']
for r in rows:
    lines.append(f"| {r['cell']} | {r['n']} | {r['m1_mean']:.2f} | "
                 f"{r['joint_mean']:.2f} | {r['improv_pct']:+.2f}% | {r['wtl']} | "
                 f"{r['p_raw']:.2e} | {r['p_holm']:.2e} | "
                 f"{'YES' if r['significant'] else 'no'} |\n")
    print(lines[-1].strip())
lines.append(f'\n## FINAL VERDICT: {verdict}\n\n```json\n'
             + json.dumps(rows, indent=1) + '\n```\n')
os.makedirs('notes', exist_ok=True)
with open('notes/gate_G2_final.md', 'w') as f:
    f.writelines(lines)
print(f'FINAL VERDICT: {verdict}')
print('wrote notes/gate_G2_final.md')
