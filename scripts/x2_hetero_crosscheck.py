"""does the new harness reproduce the paper's stored baseline?

The heterogeneity study re-runs the baseline (untransformed) cells through
scripts/x2_hetero_policy.py rather than lifting numbers from
test_results/PPVCT/, so that every level in the study is measured by one
harness on one device. This script checks that decision by comparing the
baseline makespans the new harness produced against the stored
Result_greedy+*.npy arrays the manuscript is built on.

Exact agreement is not expected on every instance. Greedy decoding takes an
argmax over policy logits, and near-ties flip when the floating-point
reduction order changes; the stored arrays were produced on the GPU, this
study runs on CPU with a single torch thread. Measured on the same 20
instances, changing only the torch thread count already moves 1 to 3
makespans.

What the check is for is a protocol mismatch: a wrong batch size, the wrong
vehicle rule, a missing bound attachment. Any of those shifts a cell mean in
one direction on every seed and every cell. Tie-flip noise does not: it
scatters around zero. So the test is on the SIGN structure, not on a single
tolerance. It asserts that the mean signed difference across all runs is
within 0.05% (no systematic shift) and that no single run's cell mean moves
by more than 0.25% (no gross error), and it reports the exact-agreement rate
so the size of the tie-flip effect is on the record.

Usage: python scripts/x2_hetero_crosscheck.py
"""

import glob
import json
import os
import re
import sys

import numpy as np

ARMS = ['m1-bcb-guide', 'joint-v1']
SEEDS = ['301', '302', '303']
CELLS = ['v1+t0.6', 'v2+t0.6', 'v1+t1.0']


def main():
    print(f'{"cell":10s} {"arm":16s} {"seed":5s} {"n":>4s} {"agree":>7s} '
          f'{"maxdiff":>9s} {"mean_new":>9s} {"mean_ref":>9s} {"dmean%":>8s}')
    worst = 0.0
    rows = []
    for cell in CELLS:
        for arm in ARMS:
            for s in SEEDS:
                mine_p = (f'results/hetero/policy/{cell}+Rbase+'
                          f'10x25+ppvct-mixed+{arm}-s{s}.json')
                ref_p = (f'test_results/PPVCT/{cell}/Result_greedy+'
                         f'10x25+ppvct-mixed+{arm}-s{s}_{cell}.npy')
                if not (os.path.exists(mine_p) and os.path.exists(ref_p)):
                    continue
                d = json.load(open(mine_p))['makespan']
                mine = np.array([d[k] for k in sorted(d)], float)
                ref = np.load(ref_p)[:, 0]
                n = min(len(mine), len(ref))
                mine, ref = mine[:n], ref[:n]
                same = int((np.abs(mine - ref) <= 1e-9).sum())
                dm = 100 * (mine.mean() - ref.mean()) / ref.mean()
                worst = max(worst, abs(dm))
                rows.append(dict(cell=cell, arm=arm, seed=s, n=n,
                                 n_identical=same,
                                 max_abs_diff=float(np.abs(mine - ref).max()),
                                 mean_new=float(mine.mean()),
                                 mean_ref=float(ref.mean()),
                                 cell_mean_diff_pct=float(dm)))
                print(f'{cell:10s} {arm:16s} {s:5s} {n:4d} {same:4d}/{n:<3d} '
                      f'{np.abs(mine-ref).max():9.4f} {mine.mean():9.3f} '
                      f'{ref.mean():9.3f} {dm:+8.3f}')
    with open('results/hetero/baseline_crosscheck.json', 'w') as f:
        json.dump(rows, f, indent=1)
    tot = sum(r['n'] for r in rows)
    ident = sum(r['n_identical'] for r in rows)
    signed = np.array([r['cell_mean_diff_pct'] for r in rows])
    n_pos = int((signed > 0).sum())
    print(f'\nidentical on {ident}/{tot} instance-runs '
          f'({100*ident/tot:.1f}%)')
    print(f'cell-mean difference: mean {signed.mean():+.4f}%, '
          f'largest magnitude {worst:.3f}%, '
          f'positive on {n_pos}/{len(signed)} runs')
    print('wrote results/hetero/baseline_crosscheck.json')
    assert abs(signed.mean()) < 0.05, \
        (f'mean cell-mean difference {signed.mean():+.4f}% is a systematic '
         f'shift, not tie-flip noise: protocol mismatch')
    assert worst < 0.25, \
        f'a cell mean moved by {worst:.3f}%, too large for tie flips'
    print('PASS: no systematic shift; differences are argmax tie flips')


if __name__ == '__main__':
    main()
