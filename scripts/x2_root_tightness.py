"""How tight is the analytic root bound where the optimum is known?

Admissibility says B(s0) <= C*.  It says nothing about how far below C* the
bound sits, and a bound that is valid but loose certifies nothing useful.
On every instance the strengthened CP-SAT reference closed (status OPTIMAL),
the optimum is known, so the ratio

    B(s0) / OPT

is measurable.  Closed instances concentrate in the cells the solver can
close (lighter travel, larger fleets); tightness on the hard cells cannot be
measured because no optimum is known there, and the printed coverage makes
that explicit.

Inputs : results/certificate/{cell}.json  (B(s0) per instance)
         results/cpsat_v2/{cell}.jsonl    (status/ub per instance)
Usage  : python scripts/x2_root_tightness.py
"""
import glob
import json
import os
import statistics as st

ratios, bycell = [], {}
for path in sorted(glob.glob('results/cpsat_v2/*.jsonl')):
    cell = os.path.basename(path)[:-6]
    if cell.endswith('_nofleet') or cell.endswith('_noenergy'):
        continue          # ablation runs, not references
    cert = os.path.join('results/certificate', cell + '.json')
    if not os.path.exists(cert):
        continue
    B = json.load(open(cert))
    for line in open(path):
        r = json.loads(line)
        if r['status'] != 'OPTIMAL':
            continue
        b = B.get(r['instance'])
        if b is None or r['ub'] <= 0:
            continue
        ratios.append(b / r['ub'])
        bycell.setdefault(cell, []).append(b / r['ub'])

assert ratios, 'no closed instances found'
assert max(ratios) <= 1 + 1e-9, 'root bound above a proven optimum'
exact = [x for x in ratios if x >= 0.9999]
print('closed instances      : %d in %d of 16 cells' % (len(ratios), len(bycell)))
print('B(s0)/OPT mean        : %.4f' % st.mean(ratios))
print('B(s0) == OPT exactly  : %d (%.1f%%)' % (len(exact),
                                               100 * len(exact) / len(ratios)))
print('worst ratio           : %.4f' % min(ratios))
for c, v in sorted(bycell.items()):
    print('  %-34s n=%3d mean=%.4f min=%.4f' % (c, len(v), st.mean(v), min(v)))
