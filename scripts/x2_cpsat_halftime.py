"""Harvest the 1800-s CP-SAT incumbents for the production cells of Table II.

The solver rows already carry their full anytime trajectory [(t, ub, lb),...],
so the half-budget column is read off the existing 3600-s runs; nothing is
re-solved. Per instance the incumbent at 1800 s is the last logged ub at
t <= 1800; the warm-start seed lands before any logged point, so every
instance has one. Sanity checks per row: the trajectory's final ub must
equal the ledger's ub field, and the 1800-s incumbent can never be below it.

Prints the macro lines with provenance comments and the paired comparison
against the deployed size-mixture policy on the same instances. Nothing is
written; paste the lines into macros.tex after reading them.

Usage: python scripts/x2_cpsat_halftime.py
"""

import json
import sys

import numpy as np
from scipy.stats import wilcoxon

CELLS = {
    'FiftyTSix': '50x25+ppvct-mixed+v2+t0.6',
    'FiftyTTen': '50x25+ppvct-mixed+v2+t1.0',
    'Eighty':    '80x25+ppvct-mixed+v3+t1.0',
}
MIX = 'mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix'
SEEDS = [301, 302, 303]
CUT = 1800.0
TOL = 1e-6


def main():
    for tag, cell in CELLS.items():
        rows = {}
        for line in open(f'results/scaleup/cpsat_b/{cell}.jsonl'):
            if line.strip():
                r = json.loads(line)
                assert r['instance'] not in rows, f'duplicate {r["instance"]}'
                rows[r['instance']] = r
        names = sorted(rows)
        assert len(names) == 30, (cell, len(names))

        half, full = [], []
        for n in names:
            r = rows[n]
            traj = r['anytime']
            assert abs(traj[-1][1] - r['ub']) < TOL, (cell, n, 'final ub drift')
            pre = [ub for t, ub, _ in traj if t <= CUT]
            assert pre, (cell, n, 'no pre-1800 point')
            assert pre[-1] >= r['ub'] - TOL, (cell, n, 'incumbent regressed')
            half.append(pre[-1])
            full.append(r['ub'])
        half, full = np.array(half), np.array(full)

        pol = np.mean([[json.load(open(
            f'results/scaleup/policy/{MIX}-s{s}_{cell}.json'))['rows'][n]['ms']
            for n in names] for s in SEEDS], axis=0)
        d = pol - half
        w = (int((d < -TOL).sum()), int((np.abs(d) <= TOL).sum()),
             int((d > TOL).sum()))
        p = wilcoxon(pol, half).pvalue
        gap = 100 * np.mean((pol - half) / half)
        print(f'{cell}: 1800 s {half.mean():.1f}+-{half.std(ddof=1):.1f}  '
              f'3600 s {full.mean():.1f}+-{full.std(ddof=1):.1f}  '
              f'policy {pol.mean():.1f}')
        print(f'  policy vs 1800-s incumbent: {gap:+.2f}%, '
              f'W/T/L {"/".join(map(str, w))}, Wilcoxon p={p:.1e}')
        print(f'\\newcommand{{\\ScCpHalf{tag}}}{{{half.mean():.1f}$\\pm$'
              f'{half.std(ddof=1):.1f}}}'
              f'  % 1800-s incumbent from the 3600-s anytime trajectory,'
              f' results/scaleup/cpsat_b/{cell}.jsonl')
    return 0


if __name__ == '__main__':
    sys.exit(main())
