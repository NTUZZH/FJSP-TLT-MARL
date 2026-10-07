"""Production-batch macros (50- and 80-module cells), computed from artifacts.

Every macro of macros.tex that depends on the 50- and 80-module production
cells is recomputed here from the result files (nothing typed by hand), with
the manuscript's conventions: makespans in hours as mean $\\pm$ sample standard
deviation (ddof=1) over instances; a learned arm's value on an instance is the
mean over training seeds 301/302/303; win/tie/loss with ties at 1e-6; paired
Wilcoxon signed-rank; relative differences in percent of the arm named in the
provenance comment. Macros whose source is not on disk yet (the gated solver
arm, the sampled decode) are reported as pending and left untouched.

Usage:
  python scripts/x2_production_macros.py            # print the macro lines
  python scripts/x2_production_macros.py --write    # also patch macros.tex in place
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import wilcoxon

WRITE = '--write' in sys.argv
MACROS = 'paper/main_manuscript_bundle/macros.tex'
MIX = 'mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix'
MIXSA = 'mix10-15-20x25+ppvct-mixed+single-joint-mix'
TEN = '10x25+ppvct-mixed+m1-bcb-guide'
SEEDS = (301, 302, 303)
CELLS = {'FiftyTSix': '50x25+ppvct-mixed+v2+t0.6',
         'FiftyTTen': '50x25+ppvct-mixed+v2+t1.0',
         'Eighty': '80x25+ppvct-mixed+v3+t1.0'}
TOL = 1e-6
out, pending = [], []


def pm(x):
    x = np.asarray(x, float)
    return f'{x.mean():.1f}$\\pm${x.std(ddof=1):.1f}'


def wtl(a, b):
    d = np.asarray(a) - np.asarray(b)
    w, l = int((d < -TOL).sum()), int((d > TOL).sum())
    return f'{w}/{len(d) - w - l}/{l}'


def pval(a, b):
    p = wilcoxon(np.asarray(a), np.asarray(b)).pvalue
    m, e = f'{p:.1e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}', p


def rel(a, b):
    """(a - b) / b in percent, positive when a is above b."""
    return 100.0 * (np.mean(a) - np.mean(b)) / np.mean(b)


def macro(name, value, note):
    out.append((name, value, note))


def n_of(cell):
    return json.load(open(f'data/PPVCT/{cell}/test/dataset_meta.json'))['n_instances']


def policy(arm, cell, tag=''):
    ms = []
    for s in SEEDS:
        p = f'results/scaleup/policy/{arm}-s{s}_{cell}{tag}.json'
        if not os.path.exists(p):
            return None, None
        rows = json.load(open(p))['rows']
        ms.append({k: v['ms'] for k, v in rows.items()})
    inst = sorted(ms[0])
    return inst, np.array([[m[i] for i in inst] for m in ms]).mean(0)


def sampled(arm, cell):
    ms, walls = [], []
    for s in SEEDS:
        p = f'results/sample_decode/{arm}-s{s}_{cell}_N64.json'
        if not os.path.exists(p):
            return None, None, None
        d = json.load(open(p))
        ms.append(d['rows'])
        walls.append(d['meta']['wall_per_instance_s'])
    inst = sorted(ms[0])
    # wall-clock: the seed-301 rerun on an exclusive GPU with four pinned cores
    # when it exists (it reproduces the seed-301 rows exactly); the three-seed
    # runs above shared the machine and their walls spread by up to 1.6x
    ex = f'results/sample_decode/{arm}-s301_{cell}_N64+excl.json'
    if os.path.exists(ex):
        d = json.load(open(ex))
        assert d['rows'] == ms[0], f'{ex} does not reproduce the seed-301 rows'
        return inst, np.array([[m[i] for i in inst] for m in ms]).mean(0), float(d['meta']['wall_per_instance_s'])
    return inst, np.array([[m[i] for i in inst] for m in ms]).mean(0), float(np.mean(walls))


def cpsat(cell):
    p = f'results/scaleup/cpsat_b/{cell}.jsonl'
    if not os.path.exists(p):
        return None
    rows = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            rows[r['instance']] = r
    if len(rows) < n_of(cell):
        return None  # fill rule: every instance or the cell stays open
    return rows


def half_incumbent(r, t=1800.0):
    ub = r['ub']
    traj = r.get('anytime') or []
    pts = [x for x in traj if x[0] <= t]
    return pts[-1][1] if pts else r.get('horizon', ub)


def main():
    pdr_p, cert_rel = {}, {}
    for tag, cell in CELLS.items():
        n = n_of(cell)
        pdr = json.load(open(f'results/scaleup/pdr/{cell}.json'))
        inst = sorted(pdr)
        pairs = sorted({k for i in inst for k in pdr[i]})
        best_pair = min(pairs, key=lambda k: np.mean([pdr[i][k] for i in inst]))
        best = np.array([pdr[i][best_pair] for i in inst])
        ga = json.load(open(f'results/scaleup/ga/{cell}.json'))
        gav = np.array([ga[i]['ga'] for i in inst])
        ii, mix = policy(MIX, cell)
        assert ii == inst, (tag, 'policy instances differ from PDR instances')
        b0 = json.load(open(f'results/scaleup/certificate/{cell}.json'))
        b0v = np.array([b0[i]['B0'] if isinstance(b0[i], dict) else b0[i] for i in inst])
        src = f'n={n}, fixed base seeds (seed0 {json.load(open(f"data/PPVCT/{cell}/test/dataset_meta.json"))["seed0"]})'
        macro(f'ScPdr{tag}', pm(best), f'best fixed pair {best_pair}, {src}')
        macro(f'ScGa{tag}', pm(gav), f'PDR-seeded GA, 60 CPU-s of search, {src}')
        macro(f'ScMix{tag}', pm(mix), f'size-mixture policy, greedy, per-instance 3-seed means, GPU, {src}')
        macro(f'ScMixDPdr{tag}', f'{-rel(mix, best):.1f}', f'(PDR - mixture)/PDR in %, {src}')
        ps, p = pval(mix, best)
        pdr_p[tag] = p
        macro(f'ScMixWtl{tag}', wtl(mix, best), f'mixture vs best fixed pair, Wilcoxon p={p:.1e}, {src}')
        cert_rel[tag] = 100.0 * np.mean((mix - b0v) / b0v)
        if tag == 'FiftyTSix':
            macro('ScGaGapMixHigh', f'{rel(mix, gav):.1f}', '3-seed mixture gap to GA-60 in % of GA, 50x25 v2+t0.6')
        if tag == 'FiftyTTen':
            macro('ScGaGapMixLow', f'{rel(mix, gav):.1f}', '3-seed mixture gap to GA-60 in % of GA, 50x25 v2+t1.0')
            macro('ScGaStartFifty', f'{np.mean([ga[i]["startup_cpu"] for i in inst]):.0f}',
                  'GA seeding+init CPU-s per instance, anytime-panel cell v2+t1.0')
            dec = [json.load(open(f'results/scaleup/policy/{MIX}-s301_{cell}.json'))['rows'][i]['decisions'] for i in inst]
            macro('ScDecFifty', f'{np.mean(dec):.0f}', 'decisions per 50-module episode (mean), mixture s301')
        if tag == 'Eighty':
            macro('ScGaGapMixEighty', f'{rel(mix, gav):.1f}', '3-seed mixture gap to GA-60 in % of GA, 80x25')
            macro('ScGaStartEighty', f'{np.mean([ga[i]["startup_cpu"] for i in inst]):.0f}',
                  'GA seeding+init CPU-s per instance at 80 modules')
        # 10-module policy gap to GA at 50 (both 50-cells pooled), ScGaGapTenFifty
        if tag == 'FiftyTTen':
            gaps = []
            for c in (CELLS['FiftyTSix'], CELLS['FiftyTTen']):
                _, ten = policy(TEN, c)
                g = json.load(open(f'results/scaleup/ga/{c}.json'))
                gaps.append(rel(ten, [g[i]['ga'] for i in sorted(g)]))
            macro('ScGaGapTenFifty', f'{np.mean(gaps):.0f}',
                  f'10-module policy gap to GA-60 at 50 modules (+{gaps[0]:.1f}/+{gaps[1]:.1f} over the two cells)')
        # GA-600 (50-module cells)
        gl = f'results/scaleup/ga_long/600s/{cell}.json'
        if os.path.exists(gl):
            glv = np.array([json.load(open(gl))[i]['ga'] for i in inst])
            macro(f'GaLong{tag}', pm(glv), f'PDR-seeded GA, 600 CPU-s of search, {src}')
        # sampled decode
        si, smp, wall = sampled(MIX, cell)
        if smp is None:
            pending.append(f'Samp*{tag} (sampled decode not on disk)')
        else:
            assert si == inst
            if tag == 'FiftyTSix':
                macro('SampMixFiftyTSix', pm(smp), f'best-of-64 mixture, per-instance 3-seed means, {src}')
                macro('SampMixDGaFiftyTSix', f'{-rel(smp, gav):.1f}', f'(GA60 - sampled)/GA60 in %, W/T/L {wtl(smp, gav)}, p={pval(smp, gav)[1]:.1e}')
            if tag == 'FiftyTTen':
                macro('SampMixFifty', pm(smp), f'best-of-64 mixture, per-instance 3-seed means, {src}')
                macro('SampMixDGa', f'{-rel(smp, gav):.1f}', '(GA60 - sampled)/GA60 in %, 50x25 v2+t1.0')
                macro('SampMixWtlGa', wtl(smp, gav), 'sampled vs GA-60, 50x25 v2+t1.0')
                macro('SampMixPvalGa', pval(smp, gav)[0], 'Wilcoxon, sampled vs GA-60, 50x25 v2+t1.0')
                macro('SampMixDGreedy', f'{-rel(smp, mix):.1f}', '(greedy - sampled)/greedy in %, 50x25 v2+t1.0')
                macro('SampWallFifty', f'{10 * round(wall / 10):.0f}', f'wall-s per instance, batched GPU, exclusive GPU and four pinned cores ({wall:.0f} s measured)')
            if tag == 'Eighty':
                macro('SampMixEighty', pm(smp), f'best-of-64 mixture, per-instance 3-seed means, {src}')
                macro('SampMixDGaEighty', f'{-rel(smp, gav):.1f}', '(GA60 - sampled)/GA60 in %, 80x25')
                macro('SampMixWtlEighty', wtl(smp, gav), f'sampled vs GA-60, 80x25, p={pval(smp, gav)[1]:.1e}')
                macro('SampWallEighty', f'{10 * round(wall / 10):.0f}', f'wall-s per instance, batched GPU, exclusive GPU and four pinned cores ({wall:.0f} s measured)')
            if os.path.exists(gl):
                macro(f'GaLongD{tag}', f'{rel(smp, glv):.1f}', f'(sampled - GA600)/GA600 in %, {src}')
                macro(f'GaLongWtl{tag}', wtl(smp, glv), f'sampled vs GA-600, p={pval(smp, glv)[1]:.1e}')
        # solver
        cp = cpsat(cell)
        if cp is None:
            pending.append(f'ScCp{tag}, ScCpHalf{tag} and the solver comparisons (ledger incomplete)')
        else:
            ub = np.array([cp[i]['ub'] for i in inst])
            lb = np.array([cp[i]['lb'] for i in inst])
            hb = np.array([half_incumbent(cp[i]) for i in inst])
            macro(f'ScCp{tag}', pm(ub), f'warm-started strengthened CP-SAT, 3600 s x 4 workers, {src}')
            macro(f'ScCpHalf{tag}', pm(hb), 'incumbent at 1800 s read off the same anytime trajectories')
            if tag == 'FiftyTTen':
                macro('ScCpNoMatchDef', f'{rel(ub, mix):.1f}', '(CP3600 - mixture)/mixture in %, hardest 50-cell v2+t1.0')
                macro('ScMixWtlCpFifty', wtl(mix, ub), 'mixture vs CP-SAT 3600-s incumbent, hardest 50-cell')
                macro('ScMixCpPval', pval(mix, ub)[0], 'Wilcoxon, mixture vs CP-SAT 3600 s, hardest 50-cell')
            if tag == 'Eighty':
                macro('ScMixDCpEighty', f'{-rel(mix, ub):.1f}', '(CP3600 - mixture)/CP3600 in %, 80x25')
                macro('ScMixWtlCpEighty', wtl(mix, ub), 'mixture vs CP-SAT 3600 s, 80x25')
                macro('ScMixPvalCpEighty', pval(mix, ub)[0], 'Wilcoxon, mixture vs CP-SAT 3600 s, 80x25')
                macro('ScCpOwnGapEighty', f'{100 * np.mean((ub - lb) / ub):.0f}', "solver's own proven gap (ub-lb)/ub in %, 80x25")
                macro('ScMixDCpHalfEighty', f'{-rel(mix, hb):.1f}', '(CP1800 - mixture)/CP1800 in %, 80x25')
                if smp is not None:
                    macro('SampMixDCpEighty', f'{-rel(smp, ub):.1f}', f'(CP3600 - sampled)/CP3600 in %, 80x25, W/T/L {wtl(smp, ub)}')
            if tag == 'FiftyTSix':
                # wall-s at which the CP-SAT mean incumbent first reaches the mixture mean
                target = mix.mean()
                grid = np.arange(0, 3601, 1)
                curve = []
                for t in grid:
                    curve.append(np.mean([half_incumbent(cp[i], t) for i in inst]))
                hit = next((int(t) for t, v in zip(grid, curve) if v <= target), None)
                macro('ScCpCrossMild', str(hit) if hit is not None else '--',
                      'wall-s at which the CP-SAT mean incumbent first reaches the 3-seed mixture mean, milder 50-cell')
    macro('ScMixPdrPvalMax', f'{max(pdr_p.values()):.1e}'.replace('e-0', '\\times10^{-').replace('e-', '\\times10^{-') + '}',
          'largest of the 3 vs-PDR Wilcoxon p-values')
    lo, hi = min(cert_rel.values()), max(cert_rel.values())
    macro('CertFiftyLow', f'{lo:.0f}', f'mean (C-B0)/B0 in %, mixture greedy C, min over the three production cells ({", ".join(f"{k}={v:.1f}" for k, v in cert_rel.items())})')
    macro('CertFiftyHigh', f'{hi:.0f}', 'same, max over the three production cells')
    # ---- print and patch
    print('% ---- production-batch macros (scripts/x2_production_macros.py) ----')
    for name, value, note in out:
        print(f'\\newcommand{{\\{name}}}{{{value}}}  % {note}')
    if pending:
        print('\nPENDING (macros left untouched):')
        for p in pending:
            print('  ', p)
    if WRITE:
        s = open(MACROS).read()
        missing = []
        for name, value, note in out:
            pat = re.compile(r'^\\newcommand\{\\' + name + r'\}\{[^\n]*$', re.M)
            new = f'\\newcommand{{\\{name}}}{{{value}}}  % {note}'
            if pat.search(s):
                s = pat.sub(lambda m: new, s, count=1)
            else:
                missing.append(name)
        open(MACROS, 'w').write(s)
        print(f'\npatched {MACROS}: {len(out) - len(missing)} macros rewritten; not found: {missing}')


main()
