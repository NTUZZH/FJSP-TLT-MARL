"""Cross-size win/loss ledger: every arm against every baseline at 30, 50 and
80 modules, computed from the artifact files only.

This is a reading aid for the author, not a manuscript source. It recomputes
what Table III and Fig. S-3 report, plus the 30-module cells that live in the
supplement, so the whole picture can be read in one place.

Conventions match scripts/x2_scale_report_b.py:
  * a policy's value on an instance is the mean over training seeds 301-303;
  * the best fixed dispatching pair is the pair with the lowest cell mean
    (one rule pair chosen per cell, then applied to every instance);
  * best-of-9 dispatching is the per-instance minimum over the nine pairs;
  * ties at 1e-6; paired Wilcoxon signed-rank, two-sided;
  * sample std (ddof=1), the paper-wide convention.

Solver budgets differ by size and are printed with the row: 300 s at 30
modules (results/scaleup/cpsat), 3600 s at 50 and 80 (results/scaleup/cpsat_b).

Usage: python scripts/x2_winloss_ledger.py
"""

import json
import os
import sys

sys.argv = [sys.argv[0]]
import numpy as np
from scipy.stats import wilcoxon

TOL = 1e-6
SEEDS = [301, 302, 303]
ANCHOR = '10x25+ppvct-mixed+m1-bcb-guide'
MIXTURE = 'mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix'

CELLS = [
    ('30x25+ppvct-mixed+v1+t0.6', 30, 1, 0.6, 'cpsat', 300),
    ('30x25+ppvct-mixed+v2+t1.0', 30, 2, 1.0, 'cpsat', 300),
    ('50x25+ppvct-mixed+v2+t0.6', 50, 2, 0.6, 'cpsat_b', 3600),
    ('50x25+ppvct-mixed+v2+t1.0', 50, 2, 1.0, 'cpsat_b', 3600),
    ('80x25+ppvct-mixed+v3+t1.0', 80, 3, 1.0, 'cpsat_b', 3600),
]


def instances(d):
    return sorted(k for k in d if k.startswith('instance_'))


def load_pdr(cell):
    """Return (best fixed pair name, its per-instance vector, best-of-9)."""
    d = json.load(open(f'results/scaleup/pdr/{cell}.json'))
    keys = instances(d)
    pairs = sorted(d[keys[0]])
    means = {p: np.mean([d[k][p] for k in keys]) for p in pairs}
    best = min(means, key=means.get)
    fixed = np.array([d[k][best] for k in keys], float)
    boN = np.array([min(d[k][p] for p in pairs) for k in keys], float)
    return best, fixed, boN


def load_ga(cell):
    d = json.load(open(f'results/scaleup/ga/{cell}.json'))
    keys = instances(d)
    ms = np.array([d[k]['ga'] for k in keys], float)
    cpu = np.array([d[k]['cpu'] for k in keys], float)
    return ms, cpu


def load_cpsat(cell, folder):
    p = f'results/scaleup/{folder}/{cell}.json'
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    keys = instances(d)
    if len(keys) < 30:
        return None, None
    ub = np.array([d[k]['ub'] for k in keys], float)
    gap = np.array([(d[k]['ub'] - d[k]['lb']) / d[k]['ub'] for k in keys], float)
    return ub, gap


def load_policy(arm, cell):
    per_seed = []
    for s in SEEDS:
        p = f'results/scaleup/policy/{arm}-s{s}_{cell}.json'
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        rows = d['rows']
        keys = instances(rows)
        assert all(rows[k]['validated'] for k in keys), f'unvalidated schedule in {p}'
        per_seed.append([rows[k]['ms'] for k in keys])
    return np.mean(np.array(per_seed, float), axis=0)


def load_sample(arm, cell):
    per_seed = []
    for s in SEEDS:
        p = f'results/sample_decode/{arm}-s{s}_{cell}_N64.json'
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        rows = d['rows']
        per_seed.append([rows[k] for k in instances(rows)])
    return np.mean(np.array(per_seed, float), axis=0)


def compare(ours, theirs):
    """Ours minus theirs, from our point of view: negative gap = we are ahead."""
    d = np.asarray(ours, float) - np.asarray(theirs, float)
    w = int((d < -TOL).sum())
    t = int((np.abs(d) <= TOL).sum())
    l = int((d > TOL).sum())
    gap = 100.0 * (ours.mean() - theirs.mean()) / theirs.mean()
    p = float('nan') if t == len(d) else wilcoxon(ours, theirs).pvalue
    return gap, f'{w}/{t}/{l}', p


def line(label, gap, wtl, p):
    verdict = 'WIN ' if gap < 0 else 'LOSS'
    ps = '   n/a' if np.isnan(p) else f'{p:7.1e}'
    print(f'    {verdict} vs {label:<22s} {gap:+6.2f}%  W/T/L {wtl:<8s} p={ps}')


for cell, n, v, tau, folder, budget in CELLS:
    print(f'\n=== {n} modules, |V|={v}, tau/p={tau}  ({cell}) ===')
    pair, pdr_fixed, pdr_bo9 = load_pdr(cell)
    ga, ga_cpu = load_ga(cell)
    cp, cp_gap = load_cpsat(cell, folder)
    arms = [('anchor policy (10-module)', load_policy(ANCHOR, cell)),
            ('mixture policy', load_policy(MIXTURE, cell)),
            ('mixture + best-of-64', load_sample(MIXTURE, cell))]
    print(f'  best fixed pair = {pair}: {pdr_fixed.mean():.1f}+-{pdr_fixed.std(ddof=1):.1f}'
          f' | best-of-9 {pdr_bo9.mean():.1f}'
          f' | GA-60s {ga.mean():.1f}+-{ga.std(ddof=1):.1f} ({ga_cpu.mean():.0f} CPU-s all-in)'
          + (f' | CP-SAT-{budget}s {cp.mean():.1f}+-{cp.std(ddof=1):.1f}'
             f' (own proven gap {100*cp_gap.mean():.0f}%)' if cp is not None
             else f' | CP-SAT-{budget}s absent'))
    for name, ours in arms:
        if ours is None:
            print(f'  {name}: not evaluated on this cell')
            continue
        print(f'  {name}: {ours.mean():.1f}+-{ours.std(ddof=1):.1f}')
        line(f'best pair ({pair})', *compare(ours, pdr_fixed))
        line('best-of-9 dispatching', *compare(ours, pdr_bo9))
        line('GA 60 CPU-s', *compare(ours, ga))
        if cp is not None:
            line(f'CP-SAT {budget} s', *compare(ours, cp))
