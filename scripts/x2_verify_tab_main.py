"""Recompute the Table I and Table III cells from the artifacts and diff them
against macros.tex.

Guards the failure mode that matters most before submission: a printed number
that no longer matches the file it came from, or that was computed from the
wrong arm. Reads the macro definitions out of macros.tex, recomputes each one
from the path named in its provenance comment, and reports any disagreement.

Conventions (stated in the table caption and notes, verified here):
  ours / single-agent : 3-seed per-instance mean of the greedy rollouts
  best PDR pair       : the pair with the lowest cell mean
  spread              : sample standard deviation (ddof=1) over instances
  Gap%                : (mean(ours) - mean(CP-SAT UB)) / mean(CP-SAT UB)
  W/T/L               : ours vs the best PDR pair, per instance, tol 1e-6

Usage: python scripts/x2_verify_tab_main.py
Exit status 0 if every cell reproduces, 1 otherwise.
"""

import json
import os
import re
import sys

import numpy as np

MACROS = 'paper/main_manuscript_bundle/macros.tex'
ARM = '10x25+ppvct-mixed+m1-bcb-guide'
SINGLE = '10x25+ppvct-mixed+single-joint'
SEEDS = [301, 302, 303]
TOL = 1e-6
ROMAN = {'One': 1, 'Two': 2, 'Three': 3}
TAU = {'TOne': '0.1', 'TThree': '0.3', 'TSix': '0.6', 'TTen': '1.0'}


def parse_macros():
    txt = open(MACROS).read()
    out = {}
    for m in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', txt):
        out[m.group(1)] = m.group(2)
    return out


def num_pm(s):
    """'229.9$\\pm$9.1' -> (229.9, 9.1); '+7.0\\%' -> (7.0, None)."""
    s = s.replace('\\%', '').replace('$\\pm$', ' ')
    p = s.split()
    return (float(p[0]), float(p[1]) if len(p) > 1 else None)


def seed_stack(arm, suffix):
    a = []
    for s in SEEDS:
        p = (f'test_results/PPVCT/{suffix}/'
             f'Result_greedy+{arm}-s{s}_{suffix}.npy')
        if not os.path.exists(p):
            return None
        a.append(np.load(p)[:, 0])
    return np.mean(a, axis=0)


def cell(vnum, tkey):
    """Every recomputed quantity for one regime cell."""
    suffix = f'v{vnum}+t{TAU[tkey]}'
    ds = f'10x25+ppvct-mixed+{suffix}'
    pdr = json.load(open(f'results/pdr/{ds}.json'))
    names = sorted(pdr)
    pairs = sorted(next(iter(pdr.values())))
    arr = {q: np.array([pdr[n][q] for n in names]) for q in pairs}
    best = min(arr, key=lambda q: arr[q].mean())
    d = {'pdr': arr[best], 'best_pair': best, 'n': len(names)}

    ga = json.load(open(f'results/ga_v2/{ds}.json'))
    d['ga'] = np.array([ga[n]['ga'] for n in names])

    ub = {}
    for line in open(f'results/cpsat_v2/{ds}.jsonl'):
        if line.strip():
            r = json.loads(line)
            ub[r['instance']] = r['ub']
    d['cp'] = np.array([ub[n] for n in names])

    d['ours'] = seed_stack(ARM, suffix)
    d['single'] = seed_stack(SINGLE, suffix)
    return d


def scale_cells(mac, bad):
    """Table III (production batches) and the sampled-decoding headline.

    These twelve macros were hand-written from report output rather than
    generated, and once carried population standard deviations while every
    other table used the sample standard deviation. Checking them here keeps
    the paper on one convention.
    """
    mix = 'mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix'
    cells = {'50x25+ppvct-mixed+v2+t0.6': 'FiftyTSix',
             '50x25+ppvct-mixed+v2+t1.0': 'FiftyTTen',
             '80x25+ppvct-mixed+v3+t1.0': 'Eighty'}
    n = 0
    for cl, tag in cells.items():
        pdr = json.load(open(f'results/scaleup/pdr/{cl}.json'))
        names = sorted(pdr)
        pairs = sorted(next(iter(pdr.values())))
        arr = {q: np.array([pdr[n_][q] for n_ in names]) for q in pairs}
        ga = json.load(open(f'results/scaleup/ga/{cl}.json'))
        series = {
            'Pdr': arr[min(arr, key=lambda q: arr[q].mean())],
            'Ga': np.array([ga[n_]['ga'] for n_ in names]),
            'Mix': np.mean([[json.load(open(
                f'results/scaleup/policy/{mix}-s{s}_{cl}.json'))['rows'][n_]['ms']
                for n_ in names] for s in SEEDS], axis=0)}
        cp, cph = {}, {}
        if os.path.exists(f'results/scaleup/cpsat_b/{cl}.jsonl'):
            for line in open(f'results/scaleup/cpsat_b/{cl}.jsonl'):
                if line.strip():
                    r = json.loads(line)
                    cp[r['instance']] = r['ub']
                    # half-budget incumbent: last logged ub at t <= 1800 s
                    # (same rule as scripts/x2_cpsat_halftime.py)
                    cph[r['instance']] = [u for t, u, _ in r['anytime']
                                          if t <= 1800][-1]
        if len(cp) == len(names):
            series['Cp'] = np.array([cp[n_] for n_ in names])
            series['CpHalf'] = np.array([cph[n_] for n_ in names])
        elif cp:
            print(f'  Sc Cp{tag}: {len(cp)}/{len(names)} solver rows, '
                  f'cell not filled yet (pre-specified rule)')
        for k, a in series.items():
            name = f'Sc{k}{tag}'
            if name not in mac:
                continue
            n += 1
            got = f'{a.mean():.1f}+-{a.std(ddof=1):.1f}'
            gm, gs = num_pm(mac[name])
            want = f'{gm:.1f}+-{gs:.1f}'
            if got != want:
                bad.append((name, want, got))
            print(f'  {name:18s} {want:>16s} {got:>16s}'
                  f'{"" if got == want else "  <-- MISMATCH"}')
    for cl, name in (('50x25+ppvct-mixed+v2+t1.0', 'SampMixFifty'),
                     ('50x25+ppvct-mixed+v2+t0.6', 'SampMixFiftyTSix')):
        if name not in mac:
            continue
        rows = [json.load(open(f'results/sample_decode/{mix}-s{s}_{cl}_N64.json'))['rows']
                for s in SEEDS]
        a = np.mean([[r[k] for k in sorted(r)] for r in rows], axis=0)
        n += 1
        got = f'{a.mean():.1f}+-{a.std(ddof=1):.1f}'
        gm, gs = num_pm(mac[name])
        want = f'{gm:.1f}+-{gs:.1f}'
        if got != want:
            bad.append((name, want, got))
        print(f'  {name:18s} {want:>16s} {got:>16s}'
              f'{"" if got == want else "  <-- MISMATCH"}')
    return n



def mixsa_cells(mac, bad):
    """Table S-XI: the size-mixture single-agent control at production scale.

    These sixteen macros back the paper's answer to "why not one merged
    head", so they must be recomputed from the artifacts rather than trusted.
    Both arms are read from their GPU evaluations: a greedy rollout takes an
    argmax over logits, and processor and accelerator kernels resolve
    near-ties differently, so a device-mixed comparison would be wrong by
    about a percent.
    """
    sys.path.insert(0, '.')
    from scripts.stats_tost import paired_tost
    marl = 'mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix'
    single = 'mix10-15-20x25+ppvct-mixed+single-joint-mix'
    seeds = [301, 302, 303]
    rows = {'FiftyTSix': ('50x25+ppvct-mixed+v2+t0.6', 'greedy'),
            'FiftyTTen': ('50x25+ppvct-mixed+v2+t1.0', 'greedy'),
            'Eighty': ('80x25+ppvct-mixed+v3+t1.0', 'greedy'),
            'EightySamp': ('80x25+ppvct-mixed+v3+t1.0', 'sampled'),
            'FiftyLow': ('50x25+ppvct-mixed+v1+t0.3', 'greedy'),
            'EightyLow': ('80x25+ppvct-mixed+v1+t0.3', 'greedy')}
    # the one-vehicle short-travel cells have their own results folder
    pdir = {'50x25+ppvct-mixed+v1+t0.3': 'results/shorttravel/policy',
            '80x25+ppvct-mixed+v1+t0.3': 'results/shorttravel/policy'}

    def arm(name, cl, how):
        out = []
        for sd in seeds:
            if how == 'greedy':
                f = f'{pdir.get(cl, "results/scaleup/policy")}/{name}-s{sd}_{cl}.json'
                if not os.path.exists(f):
                    return None
                r = json.load(open(f))['rows']
                out.append([r[k]['ms'] for k in sorted(r)])
            else:
                f = f'results/sample_decode/{name}-s{sd}_{cl}_N64.json'
                if not os.path.exists(f):
                    return None
                r = json.load(open(f))['rows']
                out.append([r[k] for k in sorted(r)])
        return np.mean(np.array(out, float), axis=0)

    n = 0
    for tag, (cl, how) in rows.items():
        a, b = arm(marl, cl, how), arm(single, cl, how)
        if a is None or b is None:
            print(f'MixSa{tag}: artifacts missing, skipped')
            continue
        r = paired_tost(a, b)
        for key, val in (('Fac', f"{r['ours_mean']:.1f}"),
                         ('Mer', f"{r['single_mean']:.1f}"),
                         ('Rel', f"{r['mean_rel_diff_pct']:+.2f}"),
                         ('Ci', f"{r['ci90_lo_pct']:+.2f}, {r['ci90_hi_pct']:+.2f}")):
            name = f'MixSa{key}{tag}'
            if name not in mac:
                continue
            n += 1
            print(f'{"S-XIII":10s} {name:26s} {mac[name]:>22s} {val:>22s}')
            if mac[name].strip() != val:
                bad.append((name, mac[name], val))
    return n


def main():
    mac = parse_macros()
    bad, checked = [], 0
    print(f'{"cell":10s} {"quantity":9s} {"macro":>18s} {"recomputed":>18s}')
    print('-' * 60)
    for vname, vnum in ROMAN.items():
        for tkey in TAU:
            key = f'V{vname}{tkey}'
            if f'qOurs{key}' not in mac:
                continue
            d = cell(vnum, tkey)
            if d['ours'] is None or d['single'] is None:
                print(f'v{vnum}+t{TAU[tkey]}: policy artifacts missing, skipped')
                continue
            gap = 100 * (d['ours'].mean() - d['cp'].mean()) / d['cp'].mean()
            diff = d['ours'] - d['pdr']
            wtl = (f"{int((diff < -TOL).sum())}/{int((abs(diff) <= TOL).sum())}"
                   f"/{int((diff > TOL).sum())}")
            items = [('Pdr', f"{d['pdr'].mean():.1f}", f"{d['pdr'].std(ddof=1):.1f}"),
                     ('Ga', f"{d['ga'].mean():.1f}", f"{d['ga'].std(ddof=1):.1f}"),
                     ('Cp', f"{d['cp'].mean():.1f}", f"{d['cp'].std(ddof=1):.1f}"),
                     ('Ours', f"{d['ours'].mean():.1f}", f"{d['ours'].std(ddof=1):.1f}"),
                     ('Sa', f"{d['single'].mean():.1f}",
                      f"{d['single'].std(ddof=1):.1f}")]
            for tag, mu, sd in items:
                name = f'q{tag}{key}'
                if name not in mac:
                    continue
                checked += 1
                gm, gs = num_pm(mac[name])
                got = f'{mu}+-{sd}'
                want = f'{gm:.1f}+-{gs:.1f}'
                if got != want:
                    bad.append((name, want, got))
                print(f'v{vnum}+t{TAU[tkey]:4s} {tag:9s} {want:>18s} '
                      f'{got:>18s} {"" if got == want else "  <-- MISMATCH"}')
            for name, want, got in ((f'qGap{key}', mac.get(f'qGap{key}'),
                                     f'{gap:+.1f}'),
                                    (f'qWtl{key}', mac.get(f'qWtl{key}'), wtl)):
                if want is None:
                    continue
                checked += 1
                w = want.replace('\\%', '')
                if w != got:
                    bad.append((name, w, got))
                print(f'v{vnum}+t{TAU[tkey]:4s} {name[1:5]:9s} {w:>18s} '
                      f'{got:>18s} {"" if w == got else "  <-- MISMATCH"}')
    print('-' * 60)
    print('Table III (production batches) and sampled decoding:')
    checked += scale_cells(mac, bad)
    checked += mixsa_cells(mac, bad)
    print('-' * 60)
    print(f'{checked} macros checked, {len(bad)} mismatched')
    for name, want, got in bad:
        print(f'  {name}: macro {want}, artifacts give {got}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
