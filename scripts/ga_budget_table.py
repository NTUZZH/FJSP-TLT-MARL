"""Build reports/ga_budget_sweep.md: GA search budget vs the bound-guided policy.

Question: at what per-instance search budget does the PDR-seeded GA overtake
the policy's schedule quality?

Rows per cell: budget 1/5/15 s from results/ga_v2_budget/, 60 s copied from
results/ga_v2/, plus a Best-PDR reference row (per-instance best of the nine
PDR pairs, results/pdr/) and the policy's own compute per instance.

The policy's per-instance makespans are the 3-seed means that
scripts/fill_macros.py uses for every published number: this script imports
that module and calls its seed_mean(), so there is one code path, not two.
W/T/L follows fill_macros' convention: wins for the POLICY, tie at 1e-6.

Run: python scripts/ga_budget_table.py   (from repo root)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')
import fill_macros as fm          # noqa: E402  (path set above)

CELLS = ['v1+t0.6', 'v1+t1.0', 'v2+t0.6', 'v2+t1.0']
BUDGETS = [0.25, 1, 5, 15, 60]
OUT = 'reports/ga_budget_sweep.md'

# Certified end-to-end policy compute per instance
# (\PolSecVOneTSix / \PolSecCpuVOneTSix: 335 decisions x 3.0 ms GPU, x 4.4 ms
# on four pinned CPU cores).
POL_GPU_S, POL_CPU_S = 1.0, 1.5


def ga_rows(cell, budget):
    """Per-instance GA makespans + audit fields for one cell/budget."""
    p = (f'results/ga_v2/10x25+ppvct-mixed+{cell}.json' if budget == 60 else
         f'results/ga_v2_budget/10x25+ppvct-mixed+{cell}+b{budget}.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if len(d) < 100:
        return None
    ks = sorted(d)
    return dict(
        path=p,
        ga=np.array([d[k]['ga'] for k in ks]),
        seed=np.array([d[k]['seed'] for k in ks]),
        improved=int(sum(d[k]['improved_on_seed'] for k in ks)),
        gens=float(np.mean([d[k]['gens'] for k in ks])),
        cpu=float(np.mean([d[k]['cpu'] for k in ks])),
        wall=float(np.mean([d[k]['wall'] for k in ks])),
        n=len(ks))


def wtl(ours, base):
    """W/T/L of the policy against `base`, fill_macros' convention."""
    w = int((ours < base - 1e-6).sum())
    t = int((np.abs(ours - base) <= 1e-6).sum())
    return w, t, len(ours) - w - t


def best_pdr(cell):
    """Per-instance best of the nine PDR pairs (not the best-mean pair)."""
    p = f'results/pdr/10x25+ppvct-mixed+{cell}.json'
    d = json.load(open(p))
    return np.array([min(d[k].values()) for k in sorted(d)])


def main():
    out = ['# GA search budget vs the bound-guided policy',
           '',
           'At what per-instance search budget does the PDR-seeded GA (v2, fixed',
           'vehicle encoding) overtake the bound-guided policy on schedule',
           'quality? Four cells of the 10x25 PPVCT grid, all 100 test instances,',
           'population 100 and nine PDR seed chromosomes at every budget: only',
           '`--budget` changes.',
           '',
           '**Budgets are worker CPU-seconds of search** (`time.process_time`),',
           'not wall-clock, so a contended box buys the same amount of search.',
           'The measured `cpu` column exceeds the budget by a fixed setup cost of',
           'about 1.5--2 s per instance: nine PDR rollouts, the 100 initial',
           'population decodes, and the independent validator re-check, none of',
           'which the budget clock covers. Read each budget as *search bought on',
           'top of a PDR start*, and the `cpu` column as the true price.',
           '',
           'Policy column: `m1-bcb-guide`, greedy rollout, per-instance mean over',
           'seeds 301/302/303, loaded through `scripts/fill_macros.py`',
           '(`seed_mean`), the same arrays behind every published macro. W/T/L is',
           "the policy's, paired per instance, tie at 1e-6.",
           '',
           'Best-PDR is the per-instance minimum over the nine dispatching-rule',
           'pairs in `results/pdr/`, which is exactly the GA\'s own `seed` field',
           '(verified identical), so it is also the GA\'s starting point.',
           '',
           f'Policy compute per instance: {POL_GPU_S:.1f} s on GPU / '
           f'{POL_CPU_S:.1f} s on four pinned CPU cores (335 decisions at 3.0 /',
           '4.4 ms per decision). Put both on the',
           f'same scale and the policy costs about {4 * POL_CPU_S:.0f} CPU-core-'
           'seconds when run without a GPU, while the GA rows below are',
           'single-core seconds: the 1 s row spends roughly 2.6 core-seconds all',
           'in, less than half the policy\'s CPU-only cost.',
           '',
           'The 1, 5 and 15 s rows were the planned sweep. The GA already led at',
           '1 s in all four cells, so a 0.25 s probe was added to bracket the',
           'crossover from below; the floor of the whole family is the Best-PDR',
           'row, which is what the GA returns when no search time is bought.',
           '',
           'Sources: `results/ga_v2_budget/*+b{0.25,1,5,15}.json` (this sweep),',
           '`results/ga_v2/*.json` (60 s), `results/pdr/*.json`,',
           '`test_results/PPVCT/<cell>/`. Produced by',
           '`scripts/ga_budget_table.py`; runs by `scripts/run_ga_budget_sweep.sh`.',
           '']

    summary = []
    for cell in CELLS:
        ours, k = fm.seed_mean(cell, fm.HEADLINE)
        assert ours is not None and k == 3, f'{cell}: policy seeds k={k}'
        bp = best_pdr(cell)

        out += [f'## {cell}', '',
                '| per-instance budget | mean makespan | improved on seed | '
                'policy W/T/L vs this | mean gens | mean cpu (s) |',
                '|---|---|---|---|---|---|']
        w, t, l = wtl(ours, bp)
        out.append(f'| Best-PDR (9 pairs, per instance) | {bp.mean():.1f} | '
                   f'-- | {w}/{t}/{l} | -- | -- |')

        cross = None
        for b in BUDGETS:
            r = ga_rows(cell, b)
            if r is None:
                out.append(f'| GA {b:g} s | (missing) | | | | |')
                continue
            assert np.allclose(r['seed'], bp), f'{cell} b{b}: seed != best-PDR'
            w, t, l = wtl(ours, r['ga'])
            out.append(f'| GA {b:g} s | {r["ga"].mean():.1f} | '
                       f'{r["improved"]}/{r["n"]} | {w}/{t}/{l} | '
                       f'{r["gens"]:.1f} | {r["cpu"]:.1f} |')
            if cross is None and r['ga'].mean() < ours.mean():
                cross = (b, w, t, l, r['ga'].mean())
            summary.append((cell, b, r['ga'].mean(), w, t, l))
        out.append(f'| **policy (m1-bcb-guide, 3-seed mean)** | '
                   f'**{ours.mean():.1f}** | -- | -- | -- | '
                   f'{POL_GPU_S:.1f} GPU / {POL_CPU_S:.1f} CPU |')
        out.append('')
        if cross:
            b, w, t, l, m = cross
            if b == BUDGETS[0]:
                bw, bt, bl = wtl(ours, bp)
                out.append(
                    f'Crossover: **between a pure PDR start and {b:g} CPU-s**. '
                    f'With no search at all the GA sits at Best-PDR ({bp.mean():.1f}), '
                    f'which the policy beats ({bw}/{bt}/{bl}); {b:g} CPU-s of search '
                    f'already takes the GA to {m:.1f} against the policy\'s '
                    f'{ours.mean():.1f}, and the policy holds only {w} of 100 '
                    f'instances ({w}/{t}/{l}).')
            else:
                out.append(
                    f'Crossover: the GA passes the policy in mean makespan at '
                    f'**{b:g} CPU-s** per instance ({m:.1f} vs {ours.mean():.1f}), '
                    f'and the policy holds only {w} of 100 instances '
                    f'({w}/{t}/{l}).')
        else:
            out.append(f'Crossover: the GA does not reach the policy '
                       f'({ours.mean():.1f}) at any budget tested.')
        out.append('')

    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    open(OUT, 'w').write('\n'.join(out) + '\n')
    print(f'wrote {OUT}')
    for row in summary:
        print(row)


if __name__ == '__main__':
    sys.exit(main())
