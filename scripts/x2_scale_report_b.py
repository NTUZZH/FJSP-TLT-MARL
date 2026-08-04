"""Phase B scale-up (50 and 80 modules): every number the analysis note quotes,
computed from the artifact files. Nothing is estimated, nothing is imputed, and
nothing is written outside results/scaleup/.

Arms read:
  results/scaleup/pdr/{cell}.json              nine dispatching-rule pairs
  results/scaleup/ga/{cell}.json               PDR-seeded GA, 60 CPU-s
  results/scaleup/ga_long/600s/{cell}.json     the same GA, 600 CPU-s
  results/scaleup/policy/{arm}-s{seed}_{cell}.json  zero-shot policy rollouts
  results/scaleup/certificate/{cell}.json      root bound B(s0)
  results/scaleup/cpsat_b/{cell}.jsonl         anytime CP-SAT (gated; may be absent)

Conventions match scripts/x2_scale_report.py: the policy value on an instance
is the mean over training seeds 301/302/303, the best fixed PDR pair is the
pair with the lowest cell mean, best-of-9 is the per-instance minimum, ties at
1e-6.

Usage: python scripts/x2_scale_report_b.py
"""

import glob
import json
import os
import re
import sys

sys.argv = [sys.argv[0]]
import numpy as np
from scipy.stats import wilcoxon

CELLS = ['50x25+ppvct-mixed+v2+t0.6', '50x25+ppvct-mixed+v2+t1.0',
         '80x25+ppvct-mixed+v3+t1.0']
PHASE_A = ['20x25+ppvct-mixed+v1+t0.6', '20x25+ppvct-mixed+v1+t1.0',
           '30x25+ppvct-mixed+v1+t0.6', '30x25+ppvct-mixed+v2+t1.0']
ARM = '10x25+ppvct-mixed+m1-bcb-guide'
SEEDS = [301, 302, 303]
CKPTS = [0.0, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0]
TOL = 1e-6


def wtl(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return (int((d < -TOL).sum()), int((np.abs(d) <= TOL).sum()),
            int((d > TOL).sum()))


def pval(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(np.abs(a - b) <= TOL):
        return float('nan')
    return float(wilcoxon(a, b).pvalue)


def at_time(trace, t):
    """Incumbent makespan after t CPU seconds of search (step function)."""
    v = trace[0][1]
    for ts, ms in trace:
        if ts <= t + 1e-9:
            v = ms
        else:
            break
    return float(v)


def load_ga(path, names):
    if not os.path.exists(path):
        return None
    g = json.load(open(path))
    out = dict(ms=np.array([g[n]['ga'] for n in names]),
               seed=np.array([g[n]['seed'] for n in names]),
               gens=np.array([g[n]['gens'] for n in names]),
               cpu=np.array([g[n]['cpu'] for n in names]),
               wall=np.array([g[n]['wall'] for n in names]),
               improved=int(sum(g[n]['improved_on_seed'] for n in names)))
    for k in ('startup_cpu', 'search_cpu', 'pdr_seed_cpu', 'pop_init_cpu'):
        if k in g[names[0]]:
            out[k] = np.array([g[n][k] for n in names])
    if 'anytime' in g[names[0]]:
        out['trace'] = [g[n]['anytime'] for n in names]
    return out


def load_cell(cell):
    d = {'cell': cell}
    pdr = json.load(open(f'results/scaleup/pdr/{cell}.json'))
    names = sorted(pdr)
    d['names'], d['n'] = names, len(names)
    pairs = sorted(next(iter(pdr.values())))
    arr = {q: np.array([pdr[n][q] for n in names]) for q in pairs}
    d['pair_means'] = {q: float(v.mean()) for q, v in arr.items()}
    d['best_pair'] = min(arr, key=lambda q: arr[q].mean())
    d['pdr_fixed'] = arr[d['best_pair']]
    d['pdr_b9'] = np.array([min(pdr[n].values()) for n in names])
    d['b9_wins'] = int(sum(abs(arr[d['best_pair']][i] - d['pdr_b9'][i]) <= TOL
                           for i in range(len(names))))

    d['ga60'] = load_ga(f'results/scaleup/ga/{cell}.json', names)
    d['ga600'] = load_ga(f'results/scaleup/ga_long/600s/{cell}.json', names)

    seed_ms, dec, lat = {}, [], []
    for s in SEEDS:
        p = f'results/scaleup/policy/{ARM}-s{s}_{cell}.json'
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        rows = j['rows']
        assert all(rows[n]['validated'] for n in names), f'{p}: unvalidated row'
        seed_ms[s] = np.array([rows[n]['ms'] for n in names])
        dec += [rows[n]['decisions'] for n in names]
        lat += [rows[n]['lat_batch_s'] for n in names]
        d['batch'] = j['batch']
        d['device'] = j['device']
    d['seed_ms'] = seed_ms
    d['pol'] = (np.mean([seed_ms[s] for s in sorted(seed_ms)], axis=0)
                if seed_ms else None)
    d['dec'] = float(np.mean(dec)) if dec else None
    d['lat'] = float(np.mean(lat)) if lat else None

    b0p = f'results/scaleup/certificate/{cell}.json'
    d['b0'] = (np.array([json.load(open(b0p))[n] for n in names])
               if os.path.exists(b0p) else None)

    d['cpsat'] = None
    cp = f'results/scaleup/cpsat_b/{cell}.jsonl'
    if os.path.exists(cp):
        recs = {}
        for line in open(cp):
            if line.strip():
                r = json.loads(line)
                recs[r['instance']] = r
        if recs:
            d['cpsat'] = recs
    return d


def phase_a_row(cell):
    """Best-of-9 and 3-seed policy on a Phase A cell, for the size trend."""
    p = f'results/scaleup/pdr/{cell}.json'
    if not os.path.exists(p):
        return None
    pdr = json.load(open(p))
    names = sorted(pdr)
    pairs = sorted(next(iter(pdr.values())))
    arr = {q: np.array([pdr[n][q] for n in names]) for q in pairs}
    bp = min(arr, key=lambda q: arr[q].mean())
    b9 = np.array([min(pdr[n].values()) for n in names])
    seeds = {}
    for s in SEEDS:
        pp = f'results/scaleup/policy/{ARM}-s{s}_{cell}.json'
        if os.path.exists(pp):
            rows = json.load(open(pp))['rows']
            seeds[s] = np.array([rows[n]['ms'] for n in names])
    if not seeds:
        return None
    pol = np.mean([seeds[s] for s in sorted(seeds)], axis=0)
    ga = load_ga(f'results/scaleup/ga/{cell}.json', names)
    return dict(n=len(names), pdr_fixed=arr[bp], pdr_b9=b9, pol=pol,
                ga=ga['ms'] if ga else None)


def ten_module(suffix):
    """The matching 10x25 cell from the manuscript's read-only artifacts."""
    ds = f'10x25+ppvct-mixed+{suffix}'
    if not os.path.exists(f'results/pdr/{ds}.json'):
        return None
    pdr = json.load(open(f'results/pdr/{ds}.json'))
    names = sorted(pdr)
    pairs = sorted(next(iter(pdr.values())))
    arr = {q: np.array([pdr[n][q] for n in names]) for q in pairs}
    bp = min(arr, key=lambda q: arr[q].mean())
    b9 = np.array([min(pdr[n].values()) for n in names])
    seeds = {}
    for p in sorted(glob.glob(f'test_results/PPVCT/{suffix}/'
                              f'Result_greedy+{ARM}-s*_{suffix}.npy')):
        m = re.search(r'-s(\d+)_', os.path.basename(p))
        if m:
            seeds[int(m.group(1))] = np.load(p)[:, 0]
    if not seeds:
        return None
    return dict(n=len(names), pdr_fixed=arr[bp], pdr_b9=b9,
                pol=np.mean(list(seeds.values()), axis=0))


def main():
    data = {c: load_cell(c) for c in CELLS}

    print('=' * 108)
    print('A. PER-CELL MEANS (n instances per cell)')
    print('=' * 108)
    print('| cell | n | ops | best fixed PDR pair | mean | best-of-9 PDR | '
          'GA 60 CPU-s | GA 600 CPU-s | s301 | s302 | s303 | policy 3-seed | '
          'CP-SAT | B(s0) |')
    print('|' + '---|' * 14)
    for c in CELLS:
        d = data[c]
        dm = json.load(open(f'data/PPVCT/{c}/test/dataset_meta.json'))
        g6 = f"{d['ga60']['ms'].mean():.1f}" if d['ga60'] else 'pending'
        g60 = f"{d['ga600']['ms'].mean():.1f}" if d['ga600'] else 'not run'
        cp = 'pending (gated)'
        if d['cpsat']:
            ok = [r for r in d['cpsat'].values() if r.get('ub') is not None]
            cp = (f"{np.mean([r['ub'] for r in ok]):.1f} "
                  f"({len(ok)}/{len(d['cpsat'])} solved)" if ok else
                  f"0/{len(d['cpsat'])} solved")
        b0 = f"{d['b0'].mean():.1f}" if d['b0'] is not None else 'pending'
        ss = ([f"{d['seed_ms'][s].mean():.1f}" if s in d['seed_ms'] else 'pending'
               for s in SEEDS])
        pol = f"{d['pol'].mean():.1f}" if d['pol'] is not None else 'pending'
        print(f"| {c} | {d['n']} | {dm['ops_mean']:.0f} | {d['best_pair']} "
              f"| {d['pdr_fixed'].mean():.1f} | {d['pdr_b9'].mean():.1f} "
              f"| {g6} | {g60} | {ss[0]} | {ss[1]} | {ss[2]} | {pol} "
              f"| {cp} | {b0} |")

    print()
    print('Dispatching-rule detail (does one fixed pair win every instance?):')
    for c in CELLS:
        d = data[c]
        srt = sorted(d['pair_means'].items(), key=lambda kv: kv[1])
        print(f"  {c}: best {srt[0][0]} {srt[0][1]:.1f}, runner-up "
              f"{srt[1][0]} {srt[1][1]:.1f}, worst {srt[-1][0]} {srt[-1][1]:.1f}"
              f" | best fixed pair is the per-instance winner on "
              f"{d['b9_wins']}/{d['n']}")

    print()
    print('=' * 108)
    print('B. GA COST: metered budget vs the honest all-in price')
    print('=' * 108)
    print('| cell | budget | mean gens | PDR seeding CPU | initial pop CPU | '
          'startup CPU | search CPU | all-in CPU | wall | improved on seed |')
    print('|' + '---|' * 10)
    for c in CELLS:
        for lbl, key in (('60 CPU-s', 'ga60'), ('600 CPU-s', 'ga600')):
            g = data[c][key]
            if not g:
                continue
            f = lambda k: (f"{g[k].mean():.1f}" if k in g else '-')
            print(f"| {c} | {lbl} | {g['gens'].mean():.1f} "
                  f"| {f('pdr_seed_cpu')} | {f('pop_init_cpu')} "
                  f"| {f('startup_cpu')} | {f('search_cpu')} "
                  f"| {g['cpu'].mean():.1f} | {g['wall'].mean():.1f} "
                  f"| {g['improved']}/{data[c]['n']} |")

    print()
    print('=' * 108)
    print('C. GA ANYTIME TRACE: incumbent after t CPU-s of search, and the '
          'policy W/T/L against it')
    print('=' * 108)
    for c in CELLS:
        d = data[c]
        for lbl, key in (('60 CPU-s', 'ga60'), ('600 CPU-s', 'ga600')):
            g = d[key]
            if not g or 'trace' not in g:
                continue
            print(f"\n--- {c}, GA budget {lbl} "
                  f"(policy 3-seed {d['pol'].mean():.1f}, "
                  f"best-of-9 PDR {d['pdr_b9'].mean():.1f}) ---")
            print('| t (CPU-s of search) | GA incumbent | gens done | '
                  'policy W/T/L vs GA@t | improved on seed |')
            print('|' + '---|' * 5)
            budget = 60.0 if key == 'ga60' else 600.0
            for t in [x for x in CKPTS if x <= budget]:
                cur = np.array([at_time(tr, t) for tr in g['trace']])
                gd = np.mean([sum(1 for ts, _ in tr if ts <= t + 1e-9) - 1
                              for tr in g['trace']])
                imp = int((cur < g['seed'] - TOL).sum())
                w, tt, l = wtl(d['pol'], cur) if d['pol'] is not None else (0, 0, 0)
                print(f"| {t:g} | {cur.mean():.1f} | {gd:.1f} | "
                      f"{w}/{tt}/{l} | {imp}/{d['n']} |")
            fin = np.array([tr[-1][1] for tr in g['trace']])
            assert np.allclose(fin, g['ms'], atol=1e-6), \
                f'{c} {lbl}: anytime tail disagrees with the recorded GA result'
            first = [next((ts for ts, ms in tr if ms < tr[0][1] - TOL), None)
                     for tr in g['trace']]
            got = [x for x in first if x is not None]
            print(f"  first improvement over the PDR seed: on {len(got)}/{d['n']} "
                  f"instances, mean {np.mean(got):.1f} CPU-s "
                  f"[{min(got):.1f}, {max(got):.1f}]" if got else
                  '  no instance improved on its PDR seed')

    print()
    print('=' * 108)
    print('D. W/T/L, per seed and pooled. Win = strictly lower makespan for the '
          'first arm.')
    print('=' * 108)
    for c in CELLS:
        d = data[c]
        if d['pol'] is None:
            print(f'\n--- {c} --- policy pending')
            continue
        print(f'\n--- {c} ---')
        refs = [('best fixed PDR (%s)' % d['best_pair'], d['pdr_fixed']),
                ('best-of-9 PDR', d['pdr_b9'])]
        if d['ga60']:
            refs.append(('GA-60 CPU-s', d['ga60']['ms']))
        if d['ga600']:
            refs.append(('GA-600 CPU-s', d['ga600']['ms']))
        for label, ref in refs:
            for s in SEEDS:
                if s in d['seed_ms']:
                    w, t, l = wtl(d['seed_ms'][s], ref)
                    print(f'  policy s{s} vs {label}: {w}/{t}/{l}')
            w, t, l = wtl(d['pol'], ref)
            print(f'  policy 3-seed-mean vs {label}: {w}/{t}/{l} '
                  f'(p={pval(d["pol"], ref):.2e}) '
                  f'gap {np.mean((d["pol"] - ref) / ref):+.2%}')
        if d['ga60']:
            w, t, l = wtl(d['ga60']['ms'], d['pdr_b9'])
            print(f'  GA-60 vs best-of-9 PDR: {w}/{t}/{l} '
                  f'gap {np.mean((d["ga60"]["ms"] - d["pdr_b9"]) / d["pdr_b9"]):+.2%}')

    print()
    print('=' * 108)
    print('E. CERTIFICATE (C - B0)/B0 against the root bound')
    print('=' * 108)
    for c in CELLS:
        d = data[c]
        if d['b0'] is None:
            print(f'{c}: certificate pending')
            continue
        line = f"{c}: B0 {d['b0'].mean():.1f}"
        if d['pol'] is not None:
            cert = (d['pol'] - d['b0']) / d['b0']
            line += (f" | policy {cert.mean():.2%} "
                     f"[{cert.min():.2%}, {cert.max():.2%}]")
        for lbl, key in (('GA-60', 'ga60'), ('GA-600', 'ga600')):
            if d[key]:
                cg = (d[key]['ms'] - d['b0']) / d['b0']
                line += f" | {lbl} {cg.mean():.2%}"
        cb9 = (d['pdr_b9'] - d['b0']) / d['b0']
        line += f" | best-of-9 PDR {cb9.mean():.2%}"
        if d['cpsat']:
            ok = [n for n, r in d['cpsat'].items() if r.get('ub') is not None]
            if ok:
                idx = [d['names'].index(n) for n in ok]
                ub = np.array([d['cpsat'][n]['ub'] for n in ok])
                line += (f" | CP-SAT UB "
                         f"{np.mean((ub - d['b0'][idx]) / d['b0'][idx]):.2%}")
        else:
            line += ' | CP-SAT pending (gated)'
        print(line)

    print()
    print('=' * 108)
    print('F. POLICY ROLLOUT COST on CPU (CONTENDED: the box also ran a CP-SAT '
          'queue and a GPU trainer, so these are upper bounds, not certified '
          'latencies)')
    print('=' * 108)
    for c in CELLS:
        d = data[c]
        if d['lat'] is None:
            print(f'{c}: policy pending')
            continue
        per_dec = d['lat'] / d['batch']
        print(f"  {c}: {d['dec']:.0f} decisions/instance | batched forward "
              f"{1000 * d['lat']:.1f} ms at batch {d['batch']} on {d['device']} "
              f"| amortized {1000 * per_dec:.2f} ms/decision "
              f"| implied network time {d['dec'] * per_dec:.1f} s/instance")
        for p in sorted(glob.glob(
                f'results/scaleup/policy/{ARM}-s*_{c}+b1lat.json')):
            j = json.load(open(p))
            r = list(j['rows'].values())
            print(f"    batch-1 probe ({os.path.basename(p)}): "
                  f"{np.mean([x['lat_batch_s'] for x in r]) * 1000:.2f} "
                  f"ms/decision over {len(r)} instances, "
                  f"{np.mean([x['decisions'] for x in r]):.0f} decisions each")

    print()
    print('=' * 108)
    print('G. SIZE TREND: the policy margin over dispatching rules, 10 to 80 '
          'modules')
    print('=' * 108)
    print('| cell suffix | modules | n | policy vs best fixed PDR | '
          'policy vs best-of-9 PDR | policy vs GA-60 |')
    print('|' + '---|' * 6)
    order = [('v1+t0.6', [('10', None), ('20', PHASE_A[0]), ('30', PHASE_A[2])]),
             ('v1+t1.0', [('10', None), ('20', PHASE_A[1])]),
             ('v2+t0.6', [('10', None), ('50', CELLS[0])]),
             ('v2+t1.0', [('10', None), ('30', PHASE_A[3]), ('50', CELLS[1])]),
             ('v3+t1.0', [('10', None), ('80', CELLS[2])])]
    for suffix, rows in order:
        for size, cell in rows:
            if cell is None:
                r = ten_module(suffix)
                if r is None:
                    continue
                r = dict(r, ga=None)
                tag = '10 (trained)'
            else:
                r = data[cell] if cell in data else phase_a_row(cell)
                if cell in data:
                    r = dict(n=r['n'], pdr_fixed=r['pdr_fixed'],
                             pdr_b9=r['pdr_b9'], pol=r['pol'],
                             ga=r['ga60']['ms'] if r['ga60'] else None)
                if r is None or r['pol'] is None:
                    continue
                tag = size
            gcol = '-'
            if r.get('ga') is not None:
                gcol = (f"{np.mean((r['pol'] - r['ga']) / r['ga']):+.2%} "
                        f"{'/'.join(map(str, wtl(r['pol'], r['ga'])))}")
            print(f"| {suffix} | {tag} | {r['n']} "
                  f"| {np.mean((r['pol'] - r['pdr_fixed']) / r['pdr_fixed']):+.2%} "
                  f"{'/'.join(map(str, wtl(r['pol'], r['pdr_fixed'])))} "
                  f"| {np.mean((r['pol'] - r['pdr_b9']) / r['pdr_b9']):+.2%} "
                  f"{'/'.join(map(str, wtl(r['pol'], r['pdr_b9'])))} | {gcol} |")

    print()
    print('=' * 108)
    print('H. SEED SPREAD')
    print('=' * 108)
    for c in CELLS:
        d = data[c]
        if not d['seed_ms']:
            continue
        means = {s: float(v.mean()) for s, v in d['seed_ms'].items()}
        base = np.mean(list(means.values()))
        line = f'{c}: ' + ', '.join(
            f's{s} {v:.1f} ({(v - base) / base:+.2%})'
            for s, v in sorted(means.items()))
        if len(means) > 1:
            line += f' | sd across seeds {np.std(list(means.values()), ddof=1):.2f}'
        print(line)
        for s in sorted(d['seed_ms']):
            print(f'   s{s} vs best fixed PDR: '
                  f"{'/'.join(map(str, wtl(d['seed_ms'][s], d['pdr_fixed'])))}")

    print()
    print('=' * 108)
    print('I. INTEGRITY CHECKS')
    print('=' * 108)
    for c in CELLS:
        d = data[c]
        msgs = [f"n={d['n']}"]
        for key, lbl in (('ga60', 'GA-60'), ('ga600', 'GA-600')):
            g = d[key]
            if g:
                msgs.append(f'{lbl} seed column == best-of-9 PDR: '
                            f'{bool(np.allclose(g["seed"], d["pdr_b9"], atol=1e-6))}')
                msgs.append(f'{lbl} never worse than its seed: '
                            f'{bool((g["ms"] <= g["seed"] + TOL).all())}')
        if d['pol'] is not None:
            msgs.append(f'policy rows validated: True ({len(d["seed_ms"])} seeds)')
        if d['b0'] is not None:
            arms = [d['pdr_b9']]
            if d['ga60']:
                arms.append(d['ga60']['ms'])
            if d['ga600']:
                arms.append(d['ga600']['ms'])
            if d['pol'] is not None:
                arms += list(d['seed_ms'].values())
            bad = sum(int((d['b0'] > a + 1e-6).sum()) for a in arms)
            msgs.append(f'B0 above a feasible makespan on {bad} of '
                        f'{sum(len(a) for a in arms)} checks')
        print(f'{c}: ' + ' | '.join(msgs))
    print()
    print('CP-SAT anytime arm: '
          + ('present' if any(data[c]['cpsat'] for c in CELLS)
             else 'PENDING (gated behind the main reference queue and the '
                  'Phase A 300 s scale batch)'))


if __name__ == '__main__':
    main()
