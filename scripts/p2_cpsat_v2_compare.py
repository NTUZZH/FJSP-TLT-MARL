"""Compare the strengthened CP-SAT references (results/cpsat_v2) against the
v1 references (or_solution/PPVCT) and the analytic root bound B(s0)
(results/certificate). Usage: python scripts/p2_cpsat_v2_compare.py v1+t0.6
"""
import json
import os
import sys

import numpy as np

cells = sys.argv[1:] or ['v1+t0.6']

for cell in cells:
    dirname = cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'
    new = json.load(open(f'results/cpsat_v2/{dirname}.json'))
    old = {}
    with open(f'or_solution/PPVCT/{cell}.jsonl') as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                old[r['instance']] = r
    b0 = json.load(open(f'results/certificate/{dirname}.json'))

    names = sorted(set(new) & set(old) & set(b0))
    nub = np.array([new[n]['ub'] for n in names], float)
    nlb = np.array([new[n]['lb'] for n in names], float)
    oub = np.array([old[n]['ms'] for n in names], float)
    olb = np.array([old[n]['lb'] for n in names], float)
    bb = np.array([b0[n] for n in names], float)
    seed = np.array([new[n]['pdr_seed'] for n in names], float)
    stat = [new[n]['status'] for n in names]

    print(f'== {cell}  n={len(names)} ==')
    print(f'  status(new): '
          f'{ {s: stat.count(s) for s in sorted(set(stat))} }')
    print(f'  UB   old {oub.mean():.3f}  new {nub.mean():.3f}  '
          f'delta {nub.mean()-oub.mean():+.3f} '
          f'({(nub/oub-1).mean()*100:+.3f}%)  '
          f'new better on {(nub < oub - 1e-9).sum()}, worse on '
          f'{(nub > oub + 1e-9).sum()}, equal {(abs(nub-oub)<=1e-9).sum()}')
    print(f'  LB   old {olb.mean():.3f}  new {nlb.mean():.3f}  '
          f'lift {nlb.mean()-olb.mean():+.3f} '
          f'({(nlb/olb-1).mean()*100:+.3f}%)  '
          f'raised on {(nlb > olb + 1e-9).sum()}, lowered on '
          f'{(nlb < olb - 1e-9).sum()}')
    print(f'  gap (UB-LB)/LB  old {((oub-olb)/olb).mean()*100:.2f}%  '
          f'new {((nub-nlb)/nlb).mean()*100:.2f}%')
    print(f'  B(s0) mean {bb.mean():.3f}')
    print(f'  B(s0) >  old LB on {(bb > olb + 1e-9).sum()}/{len(names)}   '
          f'B(s0) >  new LB on {(bb > nlb + 1e-9).sum()}/{len(names)}')
    print(f'  B(s0) >= new LB on {(bb >= nlb - 1e-9).sum()}/{len(names)}   '
          f'mean(B0 - newLB) {(bb-nlb).mean():+.3f}  '
          f'mean(B0/newLB-1) {(bb/nlb-1).mean()*100:+.2f}%')
    print(f'  B(s0) <= new UB on {(bb <= nub + 1e-9).sum()}/{len(names)} '
          f'(admissibility check)')
    # soundness: the old UB is a certified feasible makespan, so a sound LB
    # from the strengthened model can never exceed it.
    bad = [(n, nlb[i], oub[i]) for i, n in enumerate(names)
           if nlb[i] > oub[i] + 1e-9]
    print(f'  SOUNDNESS new LB <= old UB on {len(names)-len(bad)}/{len(names)}'
          + (f'  VIOLATIONS: {bad[:5]}' if bad else '  (ok)'))
    bad2 = [(n, nlb[i], nub[i]) for i, n in enumerate(names)
            if nlb[i] > nub[i] + 1e-9]
    print(f'  SOUNDNESS new LB <= new UB on {len(names)-len(bad2)}/{len(names)}'
          + (f'  VIOLATIONS: {bad2[:5]}' if bad2 else '  (ok)'))
    print(f'  PDR seed mean {seed.mean():.3f}; new UB vs seed '
          f'{(nub/seed-1).mean()*100:+.2f}%')
    worse = [n for n in names if bb[names.index(n)] < nlb[names.index(n)] - 1e-9]
    if worse:
        print(f'  instances where new LB beats B(s0): {worse[:20]}')
