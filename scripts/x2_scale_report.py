"""Phase A scale-up: every number the analysis note quotes, computed from the
artifact files. Nothing here is estimated and nothing is written outside
results/scaleup/.

Arms read:
  results/scaleup/pdr/{cell}.json          nine dispatching-rule pairs
  results/scaleup/ga/{cell}.json           PDR-seeded GA, 60 CPU-s
  results/scaleup/ga_budget/{b}s/{cell}.json   the same GA at 0.25/1/5 CPU-s
  results/scaleup/policy/{arm}-s{seed}_{cell}.json   zero-shot policy rollouts
  results/scaleup/certificate/{cell}.json  root bound B(s0)
  results/scaleup/cpsat/{cell}.jsonl       strengthened CP-SAT, 300 s

Conventions match scripts/x2_scale_table.py and scripts/fill_macros.py:
policy value on an instance is the mean over training seeds 301/302/303;
best fixed PDR pair is the pair with the lowest cell mean; best-of-9 is the
per-instance minimum; ties at 1e-6.

The 10-module reference block is read from the manuscript's own read-only
result files (results/pdr/, test_results/PPVCT/) so the degradation
comparison is like-for-like.

Usage: python scripts/x2_scale_report.py [--md]
"""

import glob
import json
import os
import re
import sys

WANT_MD = '--md' in sys.argv
sys.argv = [sys.argv[0]]
import numpy as np
from scipy.stats import wilcoxon

CELLS = ['20x25+ppvct-mixed+v1+t0.6', '20x25+ppvct-mixed+v1+t1.0',
         '30x25+ppvct-mixed+v1+t0.6', '30x25+ppvct-mixed+v2+t1.0']
ARM = '10x25+ppvct-mixed+m1-bcb-guide'
SEEDS = [301, 302, 303]
BUDGETS = ['0.25', '1', '5']
SWEEP_CELLS = ['20x25+ppvct-mixed+v1+t1.0', '30x25+ppvct-mixed+v1+t0.6']
TOL = 1e-6


def wtl(a, b):
    """W/T/L of a against b; a win is a strictly lower makespan for a."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return (int((d < -TOL).sum()), int((np.abs(d) <= TOL).sum()),
            int((d > TOL).sum()))


def pval(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(np.abs(a - b) <= TOL):
        return float('nan')
    return float(wilcoxon(a, b).pvalue)


def load_cell(cell):
    d = {}
    pdr = json.load(open(f'results/scaleup/pdr/{cell}.json'))
    names = sorted(pdr)
    d['names'] = names
    d['n'] = len(names)
    pairs = sorted(next(iter(pdr.values())))
    arr = {q: np.array([pdr[n][q] for n in names]) for q in pairs}
    d['best_pair'] = min(arr, key=lambda q: arr[q].mean())
    d['pdr_fixed'] = arr[d['best_pair']]
    d['pdr_b9'] = np.array([min(pdr[n].values()) for n in names])

    ga = json.load(open(f'results/scaleup/ga/{cell}.json'))
    d['ga'] = np.array([ga[n]['ga'] for n in names])
    d['ga_improved'] = int(sum(ga[n]['improved_on_seed'] for n in names))
    d['ga_cpu'] = float(np.mean([ga[n]['cpu'] for n in names]))
    d['ga_gens'] = float(np.mean([ga[n]['gens'] for n in names]))
    # the GA's own seed column must be the best-of-9 PDR value
    d['ga_seed_matches_b9'] = bool(np.allclose(
        [ga[n]['seed'] for n in names], d['pdr_b9'], atol=1e-6))

    seed_ms = {}
    for s in SEEDS:
        p = f'results/scaleup/policy/{ARM}-s{s}_{cell}.json'
        rows = json.load(open(p))['rows']
        assert all(rows[n]['validated'] for n in names), f'{p}: unvalidated row'
        seed_ms[s] = np.array([rows[n]['ms'] for n in names])
    d['seed_ms'] = seed_ms
    d['pol'] = np.mean([seed_ms[s] for s in SEEDS], axis=0)
    d['dec'] = float(np.mean([
        json.load(open(f'results/scaleup/policy/{ARM}-s{s}_{cell}.json'))
        ['rows'][n]['decisions'] for s in SEEDS for n in names]))

    b0p = f'results/scaleup/certificate/{cell}.json'
    if os.path.exists(b0p):
        B0 = json.load(open(b0p))
        d['b0'] = np.array([B0[n] for n in names])
    else:
        d['b0'] = None

    cp = f'results/scaleup/cpsat/{cell}.jsonl'
    d['cpsat'] = None
    if os.path.exists(cp):
        recs = {}
        for l in open(cp):
            if l.strip():
                r = json.loads(l)
                recs[r['instance']] = r
        if recs:
            d['cpsat'] = recs
            d['cp_all'] = [n for n in names if n in recs]
            # a 300 s solve that proves nothing returns status UNKNOWN with no
            # UB and no LB; those instances carry no comparable number, so they
            # are counted and excluded rather than imputed
            d['cp_names'] = [n for n in d['cp_all']
                             if recs[n].get('ub') is not None]
            d['cp_unknown'] = [n for n in d['cp_all'] if n not in d['cp_names']]
            if not d['cp_names']:
                d['cpsat'] = None
    return d


def budget_rows(cell, pol):
    out = []
    for b in BUDGETS:
        p = f'results/scaleup/ga_budget/{b}s/{cell}.json'
        if not os.path.exists(p):
            continue
        g = json.load(open(p))
        names = sorted(g)
        a = np.array([g[n]['ga'] for n in names])
        out.append(dict(budget=b, mean=float(a.mean()),
                        improved=int(sum(g[n]['improved_on_seed'] for n in names)),
                        n=len(names),
                        gens=float(np.mean([g[n]['gens'] for n in names])),
                        cpu=float(np.mean([g[n]['cpu'] for n in names])),
                        wtl=wtl(pol, a), arr=a))
    return out


# ---------------------------------------------------------------- 10 modules
def ten_module(cell_suffix, ds=None, tdir=None):
    """Best fixed pair, best-of-9 and 3-seed policy mean on the matching
    10x25 cell, from the manuscript's own read-only artifacts. Passing ds and
    tdir points the same computation at the published 15x25 transfer cells."""
    ds = ds or f'10x25+ppvct-mixed+{cell_suffix}'
    cell_suffix = tdir or cell_suffix
    if not os.path.exists(f'results/pdr/{ds}.json'):
        return None
    pdr = json.load(open(f'results/pdr/{ds}.json'))
    names = sorted(pdr)
    pairs = sorted(next(iter(pdr.values())))
    arr = {q: np.array([pdr[n][q] for n in names]) for q in pairs}
    bp = min(arr, key=lambda q: arr[q].mean())
    b9 = np.array([min(pdr[n].values()) for n in names])
    seeds = {}
    for p in sorted(glob.glob(
            f'test_results/PPVCT/{cell_suffix}/'
            f'Result_greedy+{ARM}-s*_{cell_suffix}.npy')):
        m = re.search(r'-s(\d+)_', os.path.basename(p))
        if m:
            seeds[int(m.group(1))] = np.load(p)[:, 0]
    if not seeds:
        return None
    pol = np.mean(list(seeds.values()), axis=0)
    out = dict(n=len(names), best_pair=bp, pdr_fixed=arr[bp], pdr_b9=b9,
               pol=pol, seeds=sorted(seeds), ga=None, ga_budget={})
    gp = f'results/ga_v2/{ds}.json'
    if os.path.exists(gp):
        g = json.load(open(gp))
        out['ga'] = np.array([g[n]['ga'] for n in names])
    for b in BUDGETS:
        bp_ = f'results/ga_v2_budget/{ds}+b{b}.json'
        if os.path.exists(bp_):
            g = json.load(open(bp_))
            out['ga_budget'][b] = np.array([g[n]['ga'] for n in names])
    return out


def main():
    data = {c: load_cell(c) for c in CELLS}

    print('=' * 100)
    print('A. PER-CELL MEANS (n = number of test instances)')
    print('=' * 100)
    hdr = ('| cell | n | best fixed PDR pair | mean | best-of-9 PDR | GA 60 CPU-s '
           '| policy s301 | s302 | s303 | policy 3-seed | CP-SAT UB | CP-SAT LB | B(s0) |')
    print(hdr)
    print('|' + '---|' * 13)
    for c in CELLS:
        d = data[c]
        cp_ub = cp_lb = 'pending'
        if d['cpsat']:
            nm = d['cp_names']
            cp_ub = (f"{np.mean([d['cpsat'][n]['ub'] for n in nm]):.1f} "
                     f"(solved {len(nm)}/{len(d['cp_all'])})")
            cp_lb = f"{np.mean([d['cpsat'][n]['lb'] for n in nm]):.1f}"
        b0 = f"{d['b0'].mean():.1f}" if d['b0'] is not None else 'pending'
        print(f"| {c} | {d['n']} | {d['best_pair']} | {d['pdr_fixed'].mean():.1f} "
              f"| {d['pdr_b9'].mean():.1f} | {d['ga'].mean():.1f} "
              f"| {d['seed_ms'][301].mean():.1f} | {d['seed_ms'][302].mean():.1f} "
              f"| {d['seed_ms'][303].mean():.1f} | {d['pol'].mean():.1f} "
              f"| {cp_ub} | {cp_lb} | {b0} |")

    print()
    print('Policy rollout cost (CONTENDED, batched forward pass; not a certified latency):')
    for c in CELLS:
        d = data[c]
        lat = [json.load(open(f'results/scaleup/policy/{ARM}-s{s}_{c}.json'))
               ['rows'][n]['lat_batch_s'] for s in SEEDS for n in d['names']]
        print(f"  {c}: mean decisions/instance {d['dec']:.0f}, "
              f"mean batched forward {1000 * np.mean(lat):.1f} ms "
              f"(batch {json.load(open(f'results/scaleup/policy/{ARM}-s301_{c}.json'))['batch']}, "
              f"device {json.load(open(f'results/scaleup/policy/{ARM}-s301_{c}.json'))['device']})")

    print()
    print('=' * 100)
    print('B. W/T/L, per seed and pooled (3-seed mean). Win = lower makespan for the first arm.')
    print('=' * 100)
    for c in CELLS:
        d = data[c]
        print(f'\n--- {c} ---')
        refs = [('GA-60s', d['ga']), ('best fixed PDR', d['pdr_fixed']),
                ('best-of-9 PDR', d['pdr_b9'])]
        if d['cpsat']:
            nm = d['cp_names']
            idx = [d['names'].index(n) for n in nm]
            refs.append(('CP-SAT-300s UB',
                         np.array([d['cpsat'][n]['ub'] for n in nm])))
        else:
            idx = None
        for label, ref in refs:
            if label == 'CP-SAT-300s UB':
                for s in SEEDS:
                    w, t, l = wtl(d['seed_ms'][s][idx], ref)
                    print(f'  policy s{s} vs {label}: {w}/{t}/{l}')
                w, t, l = wtl(d['pol'][idx], ref)
                print(f'  policy 3-seed-mean vs {label}: {w}/{t}/{l} '
                      f'(p={pval(d["pol"][idx], ref):.2e}) '
                      f'gap {np.mean((d["pol"][idx] - ref) / ref):+.2%}')
                w, t, l = wtl(d['ga'][idx], ref)
                print(f'  GA-60s vs {label}: {w}/{t}/{l} '
                      f'(p={pval(d["ga"][idx], ref):.2e}) '
                      f'gap {np.mean((d["ga"][idx] - ref) / ref):+.2%}')
            else:
                for s in SEEDS:
                    w, t, l = wtl(d['seed_ms'][s], ref)
                    print(f'  policy s{s} vs {label}: {w}/{t}/{l}')
                w, t, l = wtl(d['pol'], ref)
                print(f'  policy 3-seed-mean vs {label}: {w}/{t}/{l} '
                      f'(p={pval(d["pol"], ref):.2e}) '
                      f'gap {np.mean((d["pol"] - ref) / ref):+.2%}')

    print()
    print('=' * 100)
    print('C. GA BUDGET SWEEP (policy W/T/L against the GA at each budget)')
    print('=' * 100)
    for c in SWEEP_CELLS:
        d = data[c]
        print(f'\n--- {c} (policy 3-seed mean {d["pol"].mean():.1f}) ---')
        w, t, l = wtl(d['pol'], d['pdr_b9'])
        print(f'  best-of-9 PDR (no search) {d["pdr_b9"].mean():8.1f} '
              f'| policy W/T/L {w}/{t}/{l}')
        for r in budget_rows(c, d['pol']):
            w, t, l = r['wtl']
            print(f"  GA {r['budget']:>5} CPU-s        {r['mean']:8.1f} "
                  f"| policy W/T/L {w}/{t}/{l} | improved {r['improved']}/{r['n']} "
                  f"| gens {r['gens']:.1f} | cpu {r['cpu']:.1f}s")
        w, t, l = wtl(d['pol'], d['ga'])
        print(f"  GA    60 CPU-s        {d['ga'].mean():8.1f} "
              f"| policy W/T/L {w}/{t}/{l} | improved {d['ga_improved']}/{d['n']} "
              f"| gens {d['ga_gens']:.1f} | cpu {d['ga_cpu']:.1f}s")
        # the same ladder at the trained size, for the crossover comparison
        t10 = ten_module(c.split('ppvct-mixed+')[1])
        if t10:
            print(f"  [10 modules, n={t10['n']}, policy {t10['pol'].mean():.1f}] "
                  f"best-of-9 {t10['pdr_b9'].mean():.1f} "
                  f"W/T/L {'/'.join(map(str, wtl(t10['pol'], t10['pdr_b9'])))}"
                  + ''.join(
                      f" | GA {b}s {v.mean():.1f} "
                      f"W/T/L {'/'.join(map(str, wtl(t10['pol'], v)))}"
                      for b, v in t10['ga_budget'].items())
                  + (f" | GA 60s {t10['ga'].mean():.1f} "
                     f"W/T/L {'/'.join(map(str, wtl(t10['pol'], t10['ga'])))}"
                     if t10['ga'] is not None else ''))

    print()
    print('=' * 100)
    print('D. CERTIFICATE (C_policy - B0)/B0, policy = 3-seed mean')
    print('=' * 100)
    for c in CELLS:
        d = data[c]
        if d['b0'] is None:
            print(f'{c}: certificate pending')
            continue
        cert = (d['pol'] - d['b0']) / d['b0']
        gcert = (d['ga'] - d['b0']) / d['b0']
        line = (f"{c}: B0 {d['b0'].mean():.1f} | policy cert {cert.mean():.2%} "
                f"[{cert.min():.2%}, {cert.max():.2%}] | GA-60s cert {gcert.mean():.2%}")
        if d['cpsat']:
            nm = d['cp_names']
            idx = [d['names'].index(n) for n in nm]
            ub = np.array([d['cpsat'][n]['ub'] for n in nm])
            lb = np.array([d['cpsat'][n]['lb'] for n in nm])
            line += (f" | CP-SAT cert {np.mean((ub - d['b0'][idx]) / d['b0'][idx]):.2%}"
                     f" | LB/B0 {np.mean(lb / d['b0'][idx]):.3f}"
                     f" | CP-SAT own gap (UB-LB)/LB {np.mean((ub - lb) / lb):.2%}")
        print(line)

    print()
    print('=' * 100)
    print('E. ZERO-SHOT DEGRADATION: 10 modules (trained size) vs 20/30 modules')
    print('=' * 100)
    print('| cell suffix | size | n | policy vs best fixed PDR | policy vs best-of-9 PDR | policy vs GA-60s |')
    print('|' + '---|' * 6)
    for c in CELLS:
        d = data[c]
        suffix = c.split('ppvct-mixed+')[1]
        size = c.split('x25')[0]
        t10 = ten_module(suffix)
        if t10:
            df = np.mean((t10['pol'] - t10['pdr_fixed']) / t10['pdr_fixed'])
            d9 = np.mean((t10['pol'] - t10['pdr_b9']) / t10['pdr_b9'])
            gcol = '-'
            if t10['ga'] is not None:
                gcol = (f"{np.mean((t10['pol'] - t10['ga']) / t10['ga']):+.2%} "
                        f"W/T/L {wtl(t10['pol'], t10['ga'])}")
            print(f"| {suffix} | 10 (trained) | {t10['n']} | {df:+.2%} "
                  f"W/T/L {wtl(t10['pol'], t10['pdr_fixed'])} "
                  f"| {d9:+.2%} W/T/L {wtl(t10['pol'], t10['pdr_b9'])} | {gcol} |")
        if suffix == 'v2+t1.0':   # published 15-module transfer cell, same seeds
            t15 = ten_module(suffix, ds='15x25+ppvct-mixed+v2+t1.0',
                             tdir='15x25+ppvct-mixed+v2+t1.0')
            if t15:
                df = np.mean((t15['pol'] - t15['pdr_fixed']) / t15['pdr_fixed'])
                d9 = np.mean((t15['pol'] - t15['pdr_b9']) / t15['pdr_b9'])
                print(f"| {suffix} | 15 (published transfer) | {t15['n']} "
                      f"| {df:+.2%} W/T/L {wtl(t15['pol'], t15['pdr_fixed'])} "
                      f"| {d9:+.2%} W/T/L {wtl(t15['pol'], t15['pdr_b9'])} | - |")
        df = np.mean((d['pol'] - d['pdr_fixed']) / d['pdr_fixed'])
        d9 = np.mean((d['pol'] - d['pdr_b9']) / d['pdr_b9'])
        dg = np.mean((d['pol'] - d['ga']) / d['ga'])
        print(f"| {suffix} | {size} | {d['n']} | {df:+.2%} "
              f"W/T/L {wtl(d['pol'], d['pdr_fixed'])} "
              f"| {d9:+.2%} W/T/L {wtl(d['pol'], d['pdr_b9'])} "
              f"| {dg:+.2%} W/T/L {wtl(d['pol'], d['ga'])} |")

    print()
    print('=' * 100)
    print('F. SEED SPREAD (is seed 302 anomalous?)')
    print('=' * 100)
    for c in CELLS:
        d = data[c]
        means = {s: float(d['seed_ms'][s].mean()) for s in SEEDS}
        base = np.mean(list(means.values()))
        print(f'{c}: ' + ', '.join(
            f's{s} {v:.1f} ({(v - base) / base:+.2%})' for s, v in means.items())
            + f' | sd across seeds {np.std(list(means.values()), ddof=1):.2f}'
            + f' | s302 beats s301 on {wtl(d["seed_ms"][302], d["seed_ms"][301])}'
            + f', beats s303 on {wtl(d["seed_ms"][302], d["seed_ms"][303])}')
    # the same three seeds on the trained size, for reference
    for c in CELLS:
        suffix = c.split('ppvct-mixed+')[1]
        arrs = {}
        for p in sorted(glob.glob(f'test_results/PPVCT/{suffix}/'
                                  f'Result_greedy+{ARM}-s*_{suffix}.npy')):
            m = re.search(r'-s(\d+)_', os.path.basename(p))
            if m:
                arrs[int(m.group(1))] = np.load(p)[:, 0].mean()
        if arrs:
            base = np.mean(list(arrs.values()))
            print(f'  [10 modules] {suffix}: ' + ', '.join(
                f's{s} {v:.1f} ({(v - base) / base:+.2%})'
                for s, v in sorted(arrs.items())))

    print()
    print('=' * 100)
    print('G. INTEGRITY CHECKS')
    print('=' * 100)
    for c in CELLS:
        d = data[c]
        ok = d['ga_seed_matches_b9']
        cpn = len(d['cp_names']) if d['cpsat'] else 0
        cpa = len(d.get('cp_all', []))
        print(f"{c}: n={d['n']} | GA seed column == best-of-9 PDR: {ok} "
              f"| policy rows validated: True | CP-SAT attempted {cpa}/{d['n']}, "
              f"returned a solution on {cpn}")
        if d['cpsat']:
            bad = [n for n in d['cp_names']
                   if d['cpsat'][n]['lb'] > d['cpsat'][n]['ub'] + 1e-6]
            st = {}
            for n in d['cp_all']:
                st[d['cpsat'][n]['status']] = st.get(d['cpsat'][n]['status'], 0) + 1
            wt = [d['cpsat'][n]['walltime'] for n in d['cp_all']]
            print(f'   CP-SAT status {st} | LB>UB on {len(bad)} | '
                  f'wall {np.mean(wt):.1f}s [{min(wt):.1f}, {max(wt):.1f}]')
            # UB must never beat the root bound, and never be worse than its seed
            v1 = [n for n in d['cp_names']
                  if d['b0'] is not None
                  and d['b0'][d['names'].index(n)] > d['cpsat'][n]['ub'] + 1e-6]
            v2 = [n for n in d['cp_names']
                  if d['cpsat'][n]['ub'] > d['cpsat'][n]['pdr_seed'] + 1e-6]
            print(f'   B0 above CP-SAT UB on {len(v1)} | UB worse than its warm '
                  f'start on {len(v2)}')


if __name__ == '__main__':
    main()
