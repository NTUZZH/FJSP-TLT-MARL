"""Recovery from a machine breakdown: the arms that spend no search budget.

A machine fails partway through execution and the residual has to be
re-planned. This reports the three arms a plant has without a solver: keep
every decision and delay what the breakdown invalidated (right shift),
re-dispatch the residual with the nine rule pairs and take the best, and
continue the learned policy's rollout from the disrupted state.

Both baseline generators are reported, because they ask different questions.
Disrupting a dispatching schedule is the neutral case; disrupting the
policy's own schedule is what a plant running this policy actually faces.

The arms that spend a wall-clock budget are not here. They need an
uncontended machine, since a search starved of CPU returns a worse schedule
and that bias favours the policy.

Emits the manuscript macros so nothing is transcribed by hand.

Usage: python scripts/x2_disruption_phase1.py
"""

import glob
import json
import sys
from collections import defaultdict

sys.argv = [sys.argv[0]]
import numpy as np
from scipy.stats import wilcoxon

TOL = 1e-6
TAGS = {('50x25+ppvct-mixed+v2+t1.0', 'policy'): 'FiftyOwn',
        ('50x25+ppvct-mixed+v2+t1.0', 'pdr'): 'FiftyPdr',
        ('80x25+ppvct-mixed+v3+t1.0', 'policy'): 'EightyOwn',
        ('80x25+ppvct-mixed+v3+t1.0', 'pdr'): 'EightyPdr'}


def compare(ours, rival):
    d = np.asarray(ours) - np.asarray(rival)
    w = int((d < -TOL).sum())
    t = int((np.abs(d) <= TOL).sum())
    l = int((d > TOL).sum())
    gap = 100 * (np.mean(ours) - np.mean(rival)) / np.mean(rival)
    p = float('nan') if t == len(d) else wilcoxon(ours, rival).pvalue
    return gap, f'{w}/{t}/{l}', p


macros, worst_rs, worst_pdr, walls = [], [], [], []
for f in sorted(glob.glob('results/disruption/*.jsonl')):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    cell = rows[0]['cell']
    for base in ('policy', 'pdr'):
        sub = [r for r in rows if r['baseline_method'] == base]
        if not sub:
            continue
        acc = defaultdict(lambda: defaultdict(list))
        for r in sub:
            acc[r['method']][r['instance']].append(r['makespan'])
        insts = sorted(acc['policy'])
        val = {m: np.array([np.mean(acc[m][i]) for i in insts]) for m in acc}
        wall = np.mean([r['wall_s'] for r in sub if r['method'] == 'policy'])
        walls.append((cell, wall))
        tag = TAGS[(cell, base)]
        print(f'\n{cell}, disrupted plan = {base}  (policy re-plan {wall:.1f}s)')
        for m in ('right_shift', 'pdr', 'policy'):
            print(f'   {m:12s} {val[m].mean():8.1f}')
        for rival, short in (('right_shift', 'Rs'), ('pdr', 'Pdr')):
            gap, wtl, p = compare(val['policy'], val[rival])
            print(f'   vs {rival:12s} {gap:+6.2f}%  W/T/L {wtl:8s} p={p:.2e}')
            macros.append(f'\\newcommand{{\\Dis{short}{tag}}}{{{abs(gap):.1f}}}')
            macros.append(f'\\newcommand{{\\Dis{short}Wtl{tag}}}{{{wtl}}}')
            (worst_rs if rival == 'right_shift' else worst_pdr).append(abs(gap))

print('\nrange over the four settings: '
      f'right shift {min(worst_rs):.1f}-{max(worst_rs):.1f}%, '
      f'dispatching {min(worst_pdr):.1f}-{max(worst_pdr):.1f}%')
print('\nmacros:')
for m in macros:
    print('  ' + m)
print(f'  \\newcommand{{\\DisRsLow}}{{{min(worst_rs):.1f}}}')
print(f'  \\newcommand{{\\DisRsHigh}}{{{max(worst_rs):.1f}}}')
print(f'  \\newcommand{{\\DisPdrLow}}{{{min(worst_pdr):.1f}}}')
print(f'  \\newcommand{{\\DisPdrHigh}}{{{max(worst_pdr):.1f}}}')
# LaTeX macro names take letters only, so the size is spelled out.
WORD = {'50': 'Fifty', '80': 'Eighty'}
for cell, w in walls[::2]:
    print(f'  \\newcommand{{\\DisWall{WORD[cell.split("x")[0]]}}}{{{w:.0f}}}')
