"""Harvest the 80-module CP-SAT cell and print the macro lines it fills.

The cell is filled only if all 30 instances produced a full-protocol
incumbent, which was pre-registered before the campaign started: dropping the
instances a solver could not finish would bias the cell mean towards the easy
instances and flatter the comparison. Every row is therefore checked for the
protocol it claims (four workers, a 3600 s budget, the strengthened model)
before any number is computed.

Prints the macro definitions with their provenance comments, and the paired
comparison against the deployed size-mixture policy on the same instances.
Nothing is written; paste the lines into macros.tex after reading them.

Usage: python scripts/x2_harvest_cpsat80.py
"""

import json
import sys

import numpy as np
from scipy.stats import wilcoxon

CELL = '80x25+ppvct-mixed+v3+t1.0'
LEDGER = f'results/scaleup/cpsat_b/{CELL}.jsonl'
MIX = 'mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix'
SEEDS = [301, 302, 303]
TOL = 1e-6
NEED = 30


def main():
    rows = {}
    for line in open(LEDGER):
        if line.strip():
            r = json.loads(line)
            assert r['instance'] not in rows, f'duplicate {r["instance"]}'
            rows[r['instance']] = r
    print(f'ledger rows: {len(rows)} unique of {NEED} required')
    if len(rows) < NEED:
        print('PRE-REGISTERED FILL RULE: the cell stays "--" until all 30 land')
        return 1

    bad = [n for n, r in rows.items()
           if r['n_workers'] != 4 or r['time_limit_s'] != 3600
           or not r.get('strengthened') or r['ub'] is None
           or r['walltime'] < 3595]
    print(f'rows failing the protocol check: {bad or "none"}')
    if bad:
        return 1

    names = sorted(rows)
    ub = np.array([rows[n]['ub'] for n in names])
    lb = np.array([rows[n]['lb'] for n in names])
    pol = np.mean([[json.load(open(
        f'results/scaleup/policy/{MIX}-s{s}_{CELL}.json'))['rows'][n]['ms']
        for n in names] for s in SEEDS], axis=0)

    d = pol - ub
    w = (int((d < -TOL).sum()), int((np.abs(d) <= TOL).sum()),
         int((d > TOL).sum()))
    p = wilcoxon(pol, ub).pvalue
    gap = 100 * np.mean((pol - ub) / ub)
    print(f'\nCP-SAT  {ub.mean():.1f} +- {ub.std(ddof=1):.1f} '
          f'(proven bound {lb.mean():.1f}, so its own gap is '
          f'{100 * np.mean((ub - lb) / ub):.1f}%)')
    print(f'policy  {pol.mean():.1f} +- {pol.std(ddof=1):.1f}')
    print(f'policy against the solver: {gap:+.2f}%, W/T/L {"/".join(map(str, w))}, '
          f'Wilcoxon p={p:.2e}')

    print('\n--- macro lines (sample standard deviation, the paper convention) ---')
    print(f'\\newcommand{{\\ScCpEighty}}{{{ub.mean():.1f}$\\pm${ub.std(ddof=1):.1f}}}'
          f'  % anytime CP-SAT, 3600 s wall x 4 workers, strengthened, PDR '
          f'warm start, n=30, results/scaleup/cpsat_b/{CELL}.jsonl')
    print(f'\\newcommand{{\\ScMixDCpEighty}}{{{abs(gap):.1f}}}'
          f'  % mixture policy vs that solver incumbent on the same 30 '
          f'instances, W/T/L {"/".join(map(str, w))}, Wilcoxon p={p:.1e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
