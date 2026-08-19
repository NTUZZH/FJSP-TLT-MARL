"""Merged heads against factored heads, both trained on the size mixture.

The decisive control for the paper's architecture choice. Both arms train on
the same 10/15/20-module mixture, the same fleet and travel grids, the same
2000 updates and the same three seeds, and both carry the certified price
channel; they differ only in whether the machine and vehicle decisions come
from one merged head or from two per-class heads. Everything is evaluated on
the three production cells, zero-shot in size.

The verdict follows the rule registered in notes/decisions.md on 2026-08-17,
before the third seed finished training: per-instance three-seed means, the
paired +-2% equivalence test of scripts/stats_tost.py, and the difference
test reported beside it but never used as parity evidence.

Both arms must be evaluated on the same device: a greedy rollout takes an
argmax over logits, and CPU and CUDA resolve near-ties differently, which
moves a cell mean by up to about 1%. These files are the GPU evaluations.

Usage: python scripts/x2_single_vs_marl_mix.py
"""

import json
import os
import sys

sys.argv = [sys.argv[0]]
import numpy as np

sys.path.insert(0, '.')
from scripts.stats_tost import paired_tost

MARL = 'mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix'
SINGLE = 'mix10-15-20x25+ppvct-mixed+single-joint-mix'
SEEDS = [301, 302, 303]
CELLS = [('50, V2, 0.6', '50x25+ppvct-mixed+v2+t0.6'),
         ('50, V2, 1.0', '50x25+ppvct-mixed+v2+t1.0'),
         ('80, V3, 1.0', '80x25+ppvct-mixed+v3+t1.0')]
SAMPLE_CELL = '80x25+ppvct-mixed+v3+t1.0'


def greedy(arm, cell):
    per_seed = []
    for s in SEEDS:
        p = f'results/scaleup/policy/{arm}-s{s}_{cell}.json'
        if not os.path.exists(p):
            return None
        rows = json.load(open(p))['rows']
        keys = sorted(rows)
        assert all(rows[k]['validated'] for k in keys), f'unvalidated: {p}'
        per_seed.append([rows[k]['ms'] for k in keys])
    return np.mean(np.array(per_seed, float), axis=0)


def sampled(arm, cell):
    per_seed = []
    for s in SEEDS:
        p = f'results/sample_decode/{arm}-s{s}_{cell}_N64.json'
        if not os.path.exists(p):
            return None
        rows = json.load(open(p))['rows']
        per_seed.append([rows[k] for k in sorted(rows)])
    return np.mean(np.array(per_seed, float), axis=0)


def report(label, ours, single):
    if ours is None or single is None:
        print(f'  {label:14s} not evaluated on both arms')
        return None
    r = paired_tost(ours, single)
    print(f'  {label:14s} factored {r["ours_mean"]:7.1f}  merged {r["single_mean"]:7.1f}  '
          f'{r["mean_rel_diff_pct"]:+6.2f}%  '
          f'90% CI [{r["ci90_lo_pct"]:+.2f}, {r["ci90_hi_pct"]:+.2f}]  '
          f'W/T/L {r["wtl"]:8s} -> {r["verdict"]}')
    return r


TAGS = {'50, V2, 0.6': 'FiftyTSix', '50, V2, 1.0': 'FiftyTTen',
        '80, V3, 1.0': 'Eighty', '80, sampled': 'EightySamp'}
MACRO_ROWS = []


def macro_lines():
    """The table's numbers as macros, so scripts/x2_verify_tab_main.py can
    recompute them and the supplement cannot drift away from the artifacts."""
    out = []
    for label, r in MACRO_ROWS:
        t = TAGS[label]
        out.append(f"\\newcommand{{\\MixSaFac{t}}}{{{r['ours_mean']:.1f}}}")
        out.append(f"\\newcommand{{\\MixSaMer{t}}}{{{r['single_mean']:.1f}}}")
        out.append(f"\\newcommand{{\\MixSaRel{t}}}{{{r['mean_rel_diff_pct']:+.2f}}}")
        out.append(f"\\newcommand{{\\MixSaCi{t}}}"
                   f"{{{r['ci90_lo_pct']:+.2f}, {r['ci90_hi_pct']:+.2f}}}")
    return out


print('Greedy decode, three-seed per-instance means, n=30 per cell')
verdicts = []
for label, cell in CELLS:
    r = report(label, greedy(MARL, cell), greedy(SINGLE, cell))
    verdicts.append(r)
    if r:
        MACRO_ROWS.append((label, r))

print('\nBest-of-64 sampled decode')
rs = report('80, V3, 1.0', sampled(MARL, SAMPLE_CELL), sampled(SINGLE, SAMPLE_CELL))
if rs:
    MACRO_ROWS.append(('80, sampled', rs))

good = [v for v in verdicts if v]
print('\nPre-registered rule (notes/decisions.md, 2026-08-17):')
if all(v['verdict'] == 'EQUIVALENT' for v in good):
    print('  (a) equivalence holds in every cell -> keep the parity sentence.')
elif any(v['verdict'] == 'SINGLE BETTER' for v in good):
    cells = [c for (c, _), v in zip(CELLS, verdicts) if v and v['verdict'] == 'SINGLE BETTER']
    print(f'  (b) the merged head is better beyond the margin in: {", ".join(cells)}')
    print('      -> state that loss with its size and margin, justify the factored')
    print('         form on deployment alone, and REMOVE the parity claim rather')
    print('         than narrowing it to the cells that pass.')
else:
    print('  (a/b) neither: at least one cell is INCONCLUSIVE at this sample size.')
    print('      -> report the interval, claim neither parity nor a loss.')


print('\nmacros for the supplement table:')
for line in macro_lines():
    print('  ' + line)
