"""Per-cell table for the strengthened CP-SAT references (notes/cpsat_v2_full).

For every cell with a results/cpsat_v2/{dataset}.json file, reports mean UB,
mean LB, mean (UB-LB)/LB, count proved optimal, and the B(s0)-vs-proven-LB
comparison under both the v1 and the strengthened model, flagging cells whose
conclusion changes sign.

Usage: python scripts/p2_cpsat_v2_table.py [--md]
"""
import glob
import json
import os
import sys

import numpy as np

MD = '--md' in sys.argv

ORDER = ['v1+t0.1', 'v1+t0.3', 'v1+t0.6', 'v1+t1.0',
         'v2+t0.1', 'v2+t0.3', 'v2+t0.6', 'v2+t1.0',
         'v3+t0.1', 'v3+t0.3', 'v3+t0.6', 'v3+t1.0',
         'v4+t0.6', 'v4+t1.0',
         '15x25+ppvct-mixed+v2+t0.6', '15x25+ppvct-mixed+v2+t1.0']


def dataset_of(cell):
    return cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'


def load(cell):
    ds = dataset_of(cell)
    p = f'results/cpsat_v2/{ds}.json'
    if not os.path.exists(p):
        return None
    new = json.load(open(p))
    old = {}
    with open(f'or_solution/PPVCT/{cell}.jsonl') as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                old[r['instance']] = r
    b0 = json.load(open(f'results/certificate/{ds}.json'))
    names = sorted(set(new) & set(old) & set(b0))
    if not names:
        return None
    return dict(
        cell=cell, n=len(names), names=names,
        nub=np.array([new[k]['ub'] for k in names], float),
        nlb=np.array([new[k]['lb'] for k in names], float),
        oub=np.array([old[k]['ms'] for k in names], float),
        olb=np.array([old[k]['lb'] for k in names], float),
        b0=np.array([b0[k] for k in names], float),
        seed=np.array([new[k]['pdr_seed'] for k in names], float),
        nopt=sum(1 for k in names if new[k]['status'] == 'OPTIMAL'),
        oopt=sum(1 for k in names if old[k]['status'] == 'OPTIMAL'),
        complete=len(new) >= 100,
    )


rows = []
for cell in ORDER:
    d = load(cell)
    if d is None:
        rows.append((cell, None))
        continue
    d['old_win'] = int((d['b0'] > d['olb'] + 1e-9).sum())
    d['new_win'] = int((d['b0'] > d['nlb'] + 1e-9).sum())
    d['ratio'] = float((d['b0'] / d['nlb']).mean())
    d['old_ratio'] = float((d['b0'] / d['olb']).mean())
    d['ogap'] = float(((d['oub'] - d['olb']) / d['olb']).mean() * 100)
    d['ngap'] = float(((d['nub'] - d['nlb']) / d['nlb']).mean() * 100)
    d['lb_lift'] = float((d['nlb'] / d['olb'] - 1).mean() * 100)
    d['ub_move'] = float((d['nub'] / d['oub'] - 1).mean() * 100)
    d['sound'] = int((d['nlb'] <= d['oub'] + 1e-9).sum())
    d['admis'] = int((d['b0'] <= d['nub'] + 1e-9).sum())
    # conclusion flips when B(s0) went from beating the proven LB on a
    # majority of instances to beating it on a minority (or vice versa)
    d['flip'] = (d['old_win'] * 2 > d['n']) != (d['new_win'] * 2 > d['n'])
    rows.append((cell, d))

if MD:
    print('| cell | n | UB v1 | UB v2 | LB v1 | LB v2 | gap v1 | gap v2 | '
          'n_opt v1/v2 | B0>LB v1 | B0>LB v2 | mean B0/LB v2 | flips |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for cell, d in rows:
        if d is None:
            print(f'| {cell} | - | pending | | | | | | | | | | |')
            continue
        print(f'| {cell} | {d["n"]} | {d["oub"].mean():.2f} | '
              f'{d["nub"].mean():.2f} | {d["olb"].mean():.2f} | '
              f'{d["nlb"].mean():.2f} | {d["ogap"]:.1f}% | {d["ngap"]:.1f}% | '
              f'{d["oopt"]}/{d["nopt"]} | {d["old_win"]}/{d["n"]} | '
              f'{d["new_win"]}/{d["n"]} | {d["ratio"]:.4f} | '
              f'{"YES" if d["flip"] else "no"} |')
else:
    for cell, d in rows:
        if d is None:
            print(f'{cell:32s} pending')
            continue
        print(f'{cell:32s} n={d["n"]:3d} '
              f'UB {d["oub"].mean():8.3f}->{d["nub"].mean():8.3f} '
              f'({d["ub_move"]:+6.2f}%)  '
              f'LB {d["olb"].mean():8.3f}->{d["nlb"].mean():8.3f} '
              f'({d["lb_lift"]:+6.2f}%)  gap {d["ogap"]:6.2f}%->{d["ngap"]:6.2f}%'
              f'  opt {d["oopt"]:3d}->{d["nopt"]:3d}'
              f'  B0>LB {d["old_win"]:3d}->{d["new_win"]:3d}'
              f'  B0/LB {d["old_ratio"]:.4f}->{d["ratio"]:.4f}'
              f'  sound {d["sound"]}/{d["n"]} admis {d["admis"]}/{d["n"]}'
              f'{"  <<< FLIPS" if d["flip"] else ""}'
              f'{"" if d["complete"] else "  (partial)"}')
