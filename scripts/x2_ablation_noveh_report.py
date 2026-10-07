"""Fleet-term ablation: BOLT (m1-bcb-guide) against the same arm trained and
evaluated without B_veh (m1-bcb-guide-noveh), with the test fixed before the
runs (supplement).

Primary cells: the two fleet-bound cells of Fig. 3(b), v1+t0.6 (training) and
v1+t1.0 (zero-shot). Paired one-sided Wilcoxon, BOLT shorter than the
ablation, on per-instance 3-seed means, Holm over the two cells, alpha 0.05.
Every other grid cell is descriptive (mean relative difference and W/T/L).
Both arms' greedy rollouts come from test_results/PPVCT/{cell}/ (GPU
evaluations for both, so no device mixing).

Prints the verdict, a per-seed table, and macro lines with provenance.
Usage: python scripts/x2_ablation_noveh_report.py
"""
import glob
import os
import re

import numpy as np
from scipy.stats import wilcoxon

FULL = '10x25+ppvct-mixed+m1-bcb-guide'
NOVEH = '10x25+ppvct-mixed+m1-bcb-guide-noveh'
PRIMARY = ['v1+t0.6', 'v1+t1.0']
CELLS = ['v1+t0.1', 'v1+t0.3', 'v1+t0.6', 'v1+t1.0',
         'v2+t0.1', 'v2+t0.3', 'v2+t0.6', 'v2+t1.0',
         'v3+t0.1', 'v3+t0.3', 'v3+t0.6', 'v3+t1.0']
TOL = 1e-6


def seeds(arm, cell):
    out = {}
    for p in sorted(glob.glob(f'test_results/PPVCT/{cell}/Result_greedy+{arm}-s*_{cell}.npy')):
        m = re.search(r'-s(\d+)_', os.path.basename(p))
        if m and os.path.basename(p) == f'Result_greedy+{arm}-s{m.group(1)}_{cell}.npy':
            out[int(m.group(1))] = np.load(p)[:, 0]
    return out


def main():
    rows, pvals = [], {}
    for cell in CELLS:
        f, n = seeds(FULL, cell), seeds(NOVEH, cell)
        if sorted(f) != [301, 302, 303] or sorted(n) != [301, 302, 303]:
            print(f'{cell}: incomplete (full seeds {sorted(f)}, no-veh seeds {sorted(n)})')
            continue
        fm, nm = np.mean([f[s] for s in f], 0), np.mean([n[s] for s in n], 0)
        assert fm.shape == nm.shape
        rel = 100.0 * (nm.mean() - fm.mean()) / fm.mean()     # + means removing B_veh hurts
        d = fm - nm
        w, l = int((d < -TOL).sum()), int((d > TOL).sum())
        p = wilcoxon(fm, nm, alternative='less').pvalue if np.any(np.abs(d) > TOL) else 1.0
        pvals[cell] = p
        seedrel = [100.0 * (n[s].mean() - f[s].mean()) / f[s].mean() for s in (301, 302, 303)]
        rows.append((cell, fm.mean(), nm.mean(), rel, f'{w}/{len(d) - w - l}/{l}', p, seedrel))
    print(f'{"cell":9s} {"BOLT":>8s} {"no B_veh":>9s} {"rel%":>7s} {"W/T/L":>9s} {"p(1-sided)":>11s}  per-seed rel%')
    for cell, a, b, rel, wtl, p, sr in rows:
        tag = '  PRIMARY' if cell in PRIMARY else ''
        print(f'{cell:9s} {a:8.1f} {b:9.1f} {rel:+7.2f} {wtl:>9s} {p:11.2e}  '
              + ' '.join(f'{x:+.2f}' for x in sr) + tag)
    prim = [c for c in PRIMARY if c in pvals]
    if len(prim) == 2:
        order = sorted(prim, key=lambda c: pvals[c])
        holm, running = {}, 0.0
        for k, c in enumerate(order):
            running = max(running, min(1.0, (2 - k) * pvals[c]))
            holm[c] = running
        verdict = all(holm[c] < 0.05 for c in prim)
        print('\nPRE-SPECIFIED TEST (Holm over the two fleet-bound cells): '
              + ', '.join(f'{c} p_holm={holm[c]:.2e}' for c in prim)
              + f' -> {"PASS" if verdict else "FAIL"}')
        others = [r for r in rows if r[0] not in PRIMARY]
        if others:
            lo, hi = min(r[3] for r in others), max(r[3] for r in others)
            print(f'chain-bound and remaining cells: relative difference {lo:+.2f} to {hi:+.2f}%')
        R = {r[0]: r for r in rows}
        rest = [r for r in rows if r[0] not in PRIMARY + ['v1+t0.3']]
        print('\n% ---- fleet-term ablation (scripts/x2_ablation_noveh_report.py) ----')
        print(f'\\newcommand{{\\AblVehDVOneTSix}}{{{R["v1+t0.6"][3]:.1f}}}  % (no-veh - BOLT)/BOLT in %, v1+t0.6, p_holm={holm["v1+t0.6"]:.1e}')
        print(f'\\newcommand{{\\AblVehDVOneTTen}}{{{R["v1+t1.0"][3]:.1f}}}  % same, v1+t1.0 (zero-shot), p_holm={holm["v1+t1.0"]:.1e}')
        print(f'\\newcommand{{\\AblVehWtlVOneTSix}}{{{R["v1+t0.6"][4]}}}  % per-instance sign counts, v1+t0.6')
        print(f'\\newcommand{{\\AblVehWtlVOneTTen}}{{{R["v1+t1.0"][4]}}}  % per-instance sign counts, v1+t1.0')
        print(f'\\newcommand{{\\AblVehDVOneTThree}}{{{R["v1+t0.3"][3]:.1f}}}  % descriptive, v1+t0.3')
        print(f'\\newcommand{{\\AblVehRestMax}}{{{max(abs(r[3]) for r in rest):.1f}}}  % largest |rel| over the nine cells with |V|>1 or tau/p=0.1')
        pmax = max(holm.values())
        print(f'\\newcommand{{\\AblVehPHolmMax}}{{10^{{{int(np.ceil(np.log10(pmax)))}}}}}  % upper bound on the larger Holm-adjusted p of the two primary cells ({pmax:.1e})')
        # ladder row on the four gate cells, same format as fill_macros.ms()
        for cell, suff in [('v1+t0.3', 'VOneTThree'), ('v2+t0.3', 'VTwoTThree'),
                           ('v1+t0.6', 'VOneTSix'), ('v2+t0.6', 'VTwoTSix')]:
            n = seeds(NOVEH, cell)
            arr = np.mean([n[s] for s in sorted(n)], 0)
            print(f'\\newcommand{{\\LadNoVeh{suff}}}{{{arr.mean():.1f}$\\pm${arr.std(ddof=1):.1f}}}'
                  f'  % m1-bcb-guide-noveh, seeds {sorted(n)}, n={len(arr)}')
        print('\n% ---- supplement table rows: cell & rel% & W/T/L & per-seed rel% ----')
        for cell, a, b, rel, wtl, p, sr in rows:
            v, t = cell.split('+')
            fmt = lambda x: f'{x:+.1f}'.replace('-0.0', '0.0').replace('+0.0', '0.0')
            print(f'${v[1]}$ & ${t[1:]}$ & ${fmt(rel)}$ & {wtl} & '
                  + ', '.join(f'${fmt(x)}$' for x in sr) + ' \\\\')

main()
