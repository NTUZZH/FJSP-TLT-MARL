"""Merged heads against factored heads as instance size grows.

Both arms are trained at 10x25 on the same regime mixture, with the same
budget and the same three seeds; the only difference is that the control
merges the machine and vehicle heads into one. Evaluating both zero-shot on
the larger cells therefore isolates the head structure, which the primary
grid cannot do because every cell there is the training size.

Reads results/scaleup/policy/{arm}-s{seed}_{cell}.json for both arms and the
dispatching-rule reference for each cell, and reports per-cell means, the
paired difference with its Wilcoxon p-value, and win/tie/loss.

Usage: python scripts/x2_single_vs_marl_scale.py
"""

import json
import os

import numpy as np
from scipy.stats import wilcoxon

MARL = '10x25+ppvct-mixed+m1-bcb-guide'
SINGLE = '10x25+ppvct-mixed+single-joint'
SEEDS = [301, 302, 303]
TOL = 1e-6
CELLS = [('10 (trained)', None),
         ('20, V1, 0.6', '20x25+ppvct-mixed+v1+t0.6'),
         ('20, V1, 1.0', '20x25+ppvct-mixed+v1+t1.0'),
         ('30, V1, 0.6', '30x25+ppvct-mixed+v1+t0.6'),
         ('30, V2, 1.0', '30x25+ppvct-mixed+v2+t1.0'),
         ('50, V2, 0.6', '50x25+ppvct-mixed+v2+t0.6'),
         ('50, V2, 1.0', '50x25+ppvct-mixed+v2+t1.0'),
         ('80, V3, 1.0', '80x25+ppvct-mixed+v3+t1.0')]


def arm(a, cell, names):
    out = []
    for s in SEEDS:
        p = f'results/scaleup/policy/{a}-s{s}_{cell}.json'
        if not os.path.exists(p):
            return None
        rows = json.load(open(p))['rows']
        assert all(rows[n]['validated'] for n in names), f'{p}: unvalidated'
        out.append([rows[n]['ms'] for n in names])
    return np.mean(out, axis=0), out


def main():
    print(f'{"cell":13s} {"n":>3s} {"best PDR":>9s} {"single":>9s} '
          f'{"factored":>9s} {"delta%":>8s} {"W/T/L":>9s} {"p":>9s}')
    print('-' * 78)
    for label, cell in CELLS:
        if cell is None:
            continue
        pdr = json.load(open(f'results/scaleup/pdr/{cell}.json'))
        names = sorted(pdr)
        best = min((np.array([pdr[n][q] for n in names]).mean(), q)
                   for q in next(iter(pdr.values())))[1]
        ref = np.array([pdr[n][best] for n in names])
        a_s, b_s = arm(SINGLE, cell, names), arm(MARL, cell, names)
        if a_s is None or b_s is None:
            print(f'{label:13s} pending')
            continue
        sing, marl = a_s[0], b_s[0]
        d = 100 * np.mean((marl - sing) / sing)
        diff = marl - sing
        w = (int((diff < -TOL).sum()), int((np.abs(diff) <= TOL).sum()),
             int((diff > TOL).sum()))
        p = (float('nan') if np.all(np.abs(diff) <= TOL)
             else wilcoxon(marl, sing).pvalue)
        print(f'{label:13s} {len(names):3d} {ref.mean():9.1f} '
              f'{sing.mean():9.1f} {marl.mean():9.1f} {d:+8.2f} '
              f'{"/".join(map(str, w)):>9s} {p:9.2e}')
    print('\ndelta%: factored heads against merged heads, mean of per-instance '
          'relative differences; negative favors the factored (multi-agent) '
          'arm. W/T/L counts instances. Both arms are 3-seed per-instance '
          'means, zero-shot on every row (trained at 10x25).')


if __name__ == '__main__':
    main()
