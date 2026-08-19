"""Recovery from a mid-execution machine breakdown, per re-planning budget.

Reads the ledgers written by scripts/x2_disruption.py and produces the table
the manuscript needs: for each cell, each baseline generator and each wall
budget, the recovered makespan of every arm, the paired comparison of the
policy against each rival, and the wall clock each arm actually consumed.

The comparison that matters is per budget, because the whole claim is about
what a method can deliver inside the time a plant has. An arm that overruns
its budget is reported with its realized wall time next to its makespan
rather than silently credited with the nominal budget.

Usage: python scripts/x2_disruption_report.py [--dir results/disruption]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

TOL = 1e-6
INSTANT = ('right_shift', 'pdr', 'policy', 'policy_sample')


def wtl(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return (int((d < -TOL).sum()), int((np.abs(d) <= TOL).sum()),
            int((d > TOL).sum()))


def pval(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(np.abs(a - b) <= TOL):
        return float('nan')
    return float(wilcoxon(a, b).pvalue)


def load(d):
    """records[(cell, baseline, budget)][method][instance] = list over seeds."""
    recs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    walls = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(f'{d}/*.jsonl')):
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            assert r['validated'], f'unvalidated record in {p}: {r["key"]}'
            b = r['budget_s']
            key = (r['cell'], r['baseline_method'], b)
            recs[key][r['method']][r['instance']].append(r['makespan'])
            walls[key][r['method']].append(r['wall_s'])
    return recs, walls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/disruption')
    a = ap.parse_args()
    recs, walls = load(a.dir)
    if not recs:
        print(f'no records under {a.dir}')
        return

    # arms that consume no budget are valid comparators at every budget, so
    # they are folded into each budget block rather than shown on their own
    free = {(c, b): recs[(c, b, None)] for (c, b, bd) in recs if bd is None}
    freewall = {(c, b): walls[(c, b, None)] for (c, b, bd) in walls
                if bd is None}
    budgets = sorted({k for k in recs if k[2] is not None},
                     key=lambda k: (k[0], k[1], k[2]))
    for key in budgets:
        cell, base, budget = key
        arms = dict(recs[key])
        arms.update(free.get((cell, base), {}))
        wl = dict(walls[key])
        wl.update(freewall.get((cell, base), {}))
        names = sorted(set.intersection(*(set(v) for v in arms.values())))
        if not names:
            continue
        mean = {m: np.array([np.mean(arms[m][n]) for n in names]) for m in arms}
        print(f'\n{cell} | baseline plan: {base} | budget: {budget:g} s '
              f'| n={len(names)}')
        print(f'  {"arm":14s} {"makespan":>10s} {"wall s":>8s} '
              f'{"vs policy":>12s} {"p":>9s}')
        ref = mean.get('policy')
        for m in sorted(mean):
            w = np.mean(wl[m])
            if ref is None or m == 'policy':
                cmp_s, p_s = '', ''
            else:
                d = 100 * np.mean((ref - mean[m]) / mean[m])
                cmp_s = f'{d:+.2f}% ' + '/'.join(map(str, wtl(ref, mean[m])))
                p_s = f'{pval(ref, mean[m]):.1e}'
            print(f'  {m:14s} {mean[m].mean():10.1f} {w:8.2f} '
                  f'{cmp_s:>12s} {p_s:>9s}')
    print('\nvs policy: policy against that arm, mean of per-instance relative '
          'differences (negative favours the policy) and win/tie/loss over '
          'instances. Makespans are per-instance means over training seeds.')


if __name__ == '__main__':
    main()
