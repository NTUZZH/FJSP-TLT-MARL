"""assemble the heterogeneity results into tables and statistics.

Reads results/hetero/{policy,pdr,bound}/ plus results/hetero/verification.json
and writes results/hetero/summary.json and a Markdown block appended to
reports/hetero_summary.md.

Aggregation follows the paper's own convention (scripts/fill_macros.py): a
policy arm's per-instance makespan is the mean over its three training
seeds, and instance-level statistics are computed on those seed-means. The
design is paired throughout: instance i at level R is the same base instance
as instance i at the baseline, so every comparison across arms and across
levels is a paired one and uses the Wilcoxon signed-rank test.

Two dispatching references, as in scripts/x2_winloss_ledger.py: the best
fixed pair (the pair with the lowest cell mean at that level) and best-of-9
(the per-instance minimum over the nine pairs).

C_best for the tightness ratio B(s0)/C_best is the per-instance minimum over
every arm evaluated here: all six policy runs and all nine dispatching pairs.

Usage: python scripts/x2_hetero_report.py [--append]
"""

import argparse
import json
import os
import sys

cli = argparse.ArgumentParser()
cli.add_argument('--cells', type=str, default='v1+t0.6,v2+t0.6,v1+t1.0')
cli.add_argument('--levels', type=str, default='base,2,5,10,20')
cli.add_argument('--seeds', type=str, default='301,302,303')
cli.add_argument('--append', action='store_true',
                 help='append the Markdown block to reports/hetero_summary.md')
A = cli.parse_args()
sys.argv = [sys.argv[0]]

import numpy as np
from scipy.stats import wilcoxon

GUIDE = 'm1-bcb-guide'
SHARED = 'joint-v1'
PDIR = 'results/hetero/policy'


def policy_path(cell, level, arm, seed):
    return f'{PDIR}/{cell}+R{level}+10x25+ppvct-mixed+{arm}-s{seed}.json'


def load_policy(cell, level, arm, seeds):
    """(names, [n_seeds, n] makespans, list of price summaries) or None."""
    names, mats, prices = None, [], []
    for s in seeds:
        p = policy_path(cell, level, arm, s)
        if not os.path.exists(p):
            return None
        d = json.load(open(p))
        keys = sorted(d['makespan'])
        if names is None:
            names = keys
        assert keys == names, f'instance sets differ across seeds at {p}'
        mats.append([d['makespan'][k] for k in keys])
        prices.append(d.get('price'))
    return names, np.array(mats, float), prices


def load_pdr(cell, level, names):
    p = f'results/hetero/pdr/{cell}+R{level}.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p))['makespan']
    pairs = sorted(next(iter(d.values())).keys())
    mat = np.array([[d[n][pr] for pr in pairs] for n in names], float)
    means = mat.mean(axis=0)
    bi = int(np.argmin(means))
    return dict(pairs=pairs, mat=mat, best_pair=pairs[bi],
                best_fixed=mat[:, bi], best_of_9=mat.min(axis=1))


def load_bound(cell, level, names):
    p = f'results/hetero/bound/{cell}+R{level}.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    per = d['per_instance']
    return dict(B=np.array([per[n]['B'] for n in names], float),
                active_counts=d['active_counts'],
                mean_chain=d['mean_chain'], mean_mch=d['mean_mch'],
                mean_veh=d['mean_veh'])


def paired(a, b, label):
    """a vs b: a lower is better. Returns dict with mean %, p, W/T/L."""
    d = a - b
    mean_pct = 100.0 * (b.mean() - a.mean()) / b.mean()
    w = int((a < b - 1e-9).sum())
    t = int((np.abs(d) <= 1e-9).sum())
    lo = len(a) - w - t
    if t == len(a):
        p_two, p_less = 1.0, 1.0
    else:
        p_two = float(wilcoxon(a, b).pvalue)
        p_less = float(wilcoxon(a, b, alternative='less').pvalue)
    return dict(comparison=label, n=len(a), mean_improvement_pct=mean_pct,
                p_two_sided=p_two, p_one_sided_a_lower=p_less,
                wtl=f'{w}/{t}/{lo}')


def main():
    cells = A.cells.split(',')
    levels = A.levels.split(',')
    seeds = A.seeds.split(',')
    ver = {(r['cell'], str(r['R']) if r['R'] != 1 else 'base'): r
           for r in json.load(open('results/hetero/verification.json'))}

    rows = []
    for cell in cells:
        for level in levels:
            g = load_policy(cell, level, GUIDE, seeds)
            j = load_policy(cell, level, SHARED, seeds)
            if g is None or j is None:
                print(f'-- {cell} R={level}: incomplete '
                      f'(guide={g is not None}, shared={j is not None})')
                continue
            names, gm, gp = g
            _, jm, _ = j
            gsm, jsm = gm.mean(axis=0), jm.mean(axis=0)     # seed-means
            pdr = load_pdr(cell, level, names)
            bnd = load_bound(cell, level, names)

            arms = [gm[k] for k in range(gm.shape[0])] + \
                   [jm[k] for k in range(jm.shape[0])]
            if pdr is not None:
                arms += [pdr['mat'][:, k] for k in range(pdr['mat'].shape[1])]
            c_best = np.min(np.stack(arms), axis=0)

            row = dict(cell=cell, level=level, n=len(names))
            row['guide'] = dict(
                mean=float(gsm.mean()), std=float(gsm.std(ddof=1)),
                per_seed_mean=[float(x) for x in gm.mean(axis=1)])
            row['shared'] = dict(
                mean=float(jsm.mean()), std=float(jsm.std(ddof=1)),
                per_seed_mean=[float(x) for x in jm.mean(axis=1)])
            row['guide_vs_shared'] = paired(gsm, jsm, 'guide < shared-reward')
            # per-instance relative advantage, kept so the trend across R can
            # be tested on the same instances rather than on cell means
            row['_rel_adv'] = ((jsm - gsm) / jsm).tolist()
            if pdr is not None:
                row['pdr'] = dict(
                    best_pair=pdr['best_pair'],
                    best_fixed_mean=float(pdr['best_fixed'].mean()),
                    best_fixed_std=float(pdr['best_fixed'].std(ddof=1)),
                    best_of_9_mean=float(pdr['best_of_9'].mean()),
                    pair_means={p: float(pdr['mat'][:, i].mean())
                                for i, p in enumerate(pdr['pairs'])})
                row['guide_vs_best_pdr'] = paired(
                    gsm, pdr['best_fixed'], f'guide < best fixed pair '
                                            f'({pdr["best_pair"]})')
                row['guide_vs_best_of_9'] = paired(
                    gsm, pdr['best_of_9'], 'guide < best-of-9 dispatching')
            if bnd is not None:
                tight = bnd['B'] / c_best
                row['bound'] = dict(
                    mean_B=float(bnd['B'].mean()),
                    mean_chain=bnd['mean_chain'], mean_mch=bnd['mean_mch'],
                    mean_veh=bnd['mean_veh'],
                    active_counts=bnd['active_counts'],
                    tightness_mean=float(tight.mean()),
                    tightness_median=float(np.median(tight)),
                    tightness_min=float(tight.min()),
                    tightness_max=float(tight.max()),
                    c_best_mean=float(c_best.mean()))
                assert tight.max() <= 1.0 + 1e-9, \
                    f'{cell} R={level}: root bound above a feasible makespan'
            # price channel: pooled over the three guide seeds
            pr = [x for x in gp if x]
            if pr:
                nc = sum(x['n_candidates'] for x in pr)
                npz = sum(x['n_positive'] for x in pr)
                row['price'] = dict(
                    frac_positive=npz / nc,
                    frac_events_with_any_positive=float(np.mean(
                        [x['frac_events_with_any_positive'] for x in pr])),
                    mean_positive_hours=float(np.mean(
                        [x['mean_positive_hours'] for x in pr])),
                    median_positive_hours=float(np.mean(
                        [x['median_positive_hours'] for x in pr])),
                    mean_positive_scaled=float(np.mean(
                        [x['mean_positive_scaled'] for x in pr])),
                    inv_slope=float(np.mean([x['inv_slope'] for x in pr])))
            v = ver.get((cell, level))
            if v:
                row['realized_R'] = dict(
                    mean=v['realized_R_mean'], median=v['realized_R_median'],
                    p95=v['realized_R_p95'],
                    drift_mean=v['drift_mean'], drift_max=v['drift_max'])
            rows.append(row)

    # trend of each headline metric across R, within a cell (paired vs baseline)
    trends = []
    for cell in cells:
        by = {r['level']: r for r in rows if r['cell'] == cell}
        if 'base' not in by:
            continue
        b = by['base']
        for level in levels:
            if level == 'base' or level not in by:
                continue
            r = by[level]
            # does the advantage itself move with R? Paired on the same base
            # instances: per-instance relative advantage at level R against
            # the same instance's relative advantage at the baseline.
            aR = np.array(r['_rel_adv'])
            a0 = np.array(b['_rel_adv'])
            adv_shift_p = (1.0 if np.allclose(aR, a0)
                           else float(wilcoxon(aR, a0).pvalue))
            trends.append(dict(
                cell=cell, level=level,
                rel_adv_mean_pct=100 * float(aR.mean()),
                rel_adv_mean_pct_base=100 * float(a0.mean()),
                adv_shift_vs_base_pp=100 * float(aR.mean() - a0.mean()),
                adv_shift_p=adv_shift_p,
                guide_makespan_vs_base_pct=100 * (r['guide']['mean'] -
                                                  b['guide']['mean']) / b['guide']['mean'],
                shared_makespan_vs_base_pct=100 * (r['shared']['mean'] -
                                                   b['shared']['mean']) / b['shared']['mean'],
                guide_advantage_pct=r['guide_vs_shared']['mean_improvement_pct'],
                guide_advantage_pct_base=b['guide_vs_shared']['mean_improvement_pct'],
                tightness=r.get('bound', {}).get('tightness_mean'),
                tightness_base=b.get('bound', {}).get('tightness_mean'),
                frac_positive=r.get('price', {}).get('frac_positive'),
                frac_positive_base=b.get('price', {}).get('frac_positive')))

    os.makedirs('results/hetero', exist_ok=True)
    for r in rows:
        r.pop('_rel_adv', None)
    with open('results/hetero/summary.json', 'w') as f:
        json.dump(dict(rows=rows, trends=trends), f, indent=1)
    print('wrote results/hetero/summary.json')

    md = render(rows, cells, levels) + render_trends(trends, cells, levels)
    print(md)
    if A.append:
        os.makedirs('reports', exist_ok=True)
        with open('reports/hetero_summary.md', 'a') as f:
            f.write(md)
        print('appended to reports/hetero_summary.md')


def pv(p):
    return '<1e-4' if p < 1e-4 else f'{p:.3g}'


def lv(level):
    """Level label. R=100 is the stress test, not a clean mean-preserving
    contrast (the one-hour floor truncates its small draws), so it is marked
    everywhere it appears."""
    return '100 (stress)' if level == '100' else level


def render(rows, cells, levels):
    L = ['\n\n## Results (zero-shot; makespans in hours)\n']
    L.append('Arms: bound-guided = `m1-bcb-guide` seeds 301/302/303; '
             'shared-reward = `joint-v1` seeds 301/302/303. Per-instance '
             'makespans are averaged over the three seeds before the paired '
             'test, as elsewhere in this project. "Best pair" is the '
             'dispatching pair with the lowest cell mean at that level; '
             '"best-of-9" is the per-instance minimum over the nine pairs. '
             'n = 100 paired instances per row.\n')
    for cell in cells:
        sub = [r for r in rows if r['cell'] == cell]
        if not sub:
            continue
        L.append(f'\n### Cell {cell}\n')
        L.append('| R | realized max/min (mean / median) | bound-guided | '
                 'shared-reward | guide vs shared (%, p, W/T/L) | best pair | '
                 'guide vs best pair (%, p) | best-of-9 | B(s0) | '
                 'B(s0)/C_best | price frac>0 | mean positive price (h) |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for level in levels:
            r = next((x for x in sub if x['level'] == level), None)
            if r is None:
                continue
            rr = r.get('realized_R', {})
            gs = r['guide_vs_shared']
            bd = r.get('bound', {})
            pd = r.get('pdr', {})
            gp = r.get('guide_vs_best_pdr', {})
            pc = r.get('price', {})
            lvl = 'base' if level == 'base' else lv(level)
            L.append(
                f'| {lvl} | {rr.get("mean", float("nan")):.2f} / '
                f'{rr.get("median", float("nan")):.2f} | '
                f'{r["guide"]["mean"]:.2f}+-{r["guide"]["std"]:.2f} | '
                f'{r["shared"]["mean"]:.2f}+-{r["shared"]["std"]:.2f} | '
                f'{gs["mean_improvement_pct"]:+.2f}%, '
                f'{pv(gs["p_two_sided"])}, {gs["wtl"]} | '
                f'{pd.get("best_pair", "-")} '
                f'{pd.get("best_fixed_mean", float("nan")):.2f} | '
                f'{gp.get("mean_improvement_pct", float("nan")):+.2f}%, '
                f'{pv(gp.get("p_two_sided", 1.0))} | '
                f'{pd.get("best_of_9_mean", float("nan")):.2f} | '
                f'{bd.get("mean_B", float("nan")):.2f} | '
                f'{bd.get("tightness_mean", float("nan")):.4f} | '
                f'{pc.get("frac_positive", float("nan")):.3f} | '
                f'{pc.get("mean_positive_hours", float("nan")):.2f} |')
        L.append('')
        L.append('Root bound components and the term that attains the max:\n')
        L.append('| R | B_chain | B_mch | B_veh | active term (of 100) |')
        L.append('|---|---|---|---|---|')
        for level in levels:
            r = next((x for x in sub if x['level'] == level), None)
            if r is None or 'bound' not in r:
                continue
            bd = r['bound']
            act = ' '.join(f'{k}={v}' for k, v in bd['active_counts'].items()
                           if v)
            L.append(f'| {lv(level)} | {bd["mean_chain"]:.2f} | '
                     f'{bd["mean_mch"]:.2f} | {bd["mean_veh"]:.2f} | {act} |')
    return '\n'.join(L) + '\n'


def render_trends(trends, cells, levels):
    L = ['\n## Trend across R\n',
         'The advantage column is the per-instance relative advantage of the '
         'bound-guided arm over the shared-reward arm, (shared - guide) / '
         'shared, averaged over the 100 instances. The shift column compares '
         'that quantity at level R with the same instances\' value at the '
         'baseline (paired Wilcoxon), so it tests whether heterogeneity '
         'changes the advantage rather than whether the advantage exists.\n']
    L.append('| cell | R | guide makespan vs base | shared makespan vs base | '
             'advantage | shift vs base (pp, p) | B(s0)/C_best | price frac>0 |')
    L.append('|---|---|---|---|---|---|---|---|')
    for cell in cells:
        for level in levels:
            t = next((x for x in trends
                      if x['cell'] == cell and x['level'] == level), None)
            if t is None:
                continue
            L.append(
                f'| {cell} | {lv(level)} | '
                f'{t["guide_makespan_vs_base_pct"]:+.2f}% | '
                f'{t["shared_makespan_vs_base_pct"]:+.2f}% | '
                f'{t["rel_adv_mean_pct"]:+.2f}% | '
                f'{t["adv_shift_vs_base_pp"]:+.2f}pp, {pv(t["adv_shift_p"])} | '
                f'{t["tightness"]:.4f} (base {t["tightness_base"]:.4f}) | '
                f'{t["frac_positive"]:.3f} (base {t["frac_positive_base"]:.3f}) |')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
