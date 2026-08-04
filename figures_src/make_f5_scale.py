"""Render F5 (scale-up: zero-shot size boundary + anytime comparison) and
S2 (GA compute-budget sweep against the policy) for the manuscript.

Every plotted number is read from a result file; nothing is hand-typed. The
script is idempotent and re-runnable as data lands: policy seeds are globbed
(so a new mixture seed is pooled automatically), the CP-SAT anytime ledger is
used with whatever rows exist, and cells with insufficient data are skipped
with a printed note.

Fonts: Liberation Serif (Times-metric); house palette: pastel fills, medium
lines, black text only; no titles inside the figure.

Run:  python figures_src/make_f5_scale.py [--only f5|s2]   (from repo root)
Outputs: paper/figures/f5_scale.{pdf,png}
         paper/figures/s2_scale_budget.{pdf,png}
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ style
# house palette (paper-figures skill), identical to figures_src/make_f3_f4.py
BLUE_F, ROSE_F, AMBER_F, GREY_F = '#a8c6e3', '#eda9b0', '#f3cf8f', '#c3cbd3'
BLUE_M, ROSE_M = '#4f81ad', '#c25b6a'
AMBER_M, GREY_M = '#b8860b', '#6b7480'
# medium shades of one blue family for the per-cell lines of S2
BLUE_SHADES = ['#22405c', '#3d6b96', '#5f93bd', '#89b4d8']

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Liberation Serif',
    'mathtext.it': 'Liberation Serif:italic',
    'mathtext.bf': 'Liberation Serif:bold',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'axes.linewidth': 0.6, 'axes.edgecolor': 'black',
    'pdf.fonttype': 42,
})

TEN = '10x25+ppvct-mixed+m1-bcb-guide'          # trained on 10 modules
MIX = 'mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix'   # trained on 10/15/20

# size-10 / size-15 cells of the main grid whose (fleet, travel) settings also
# appear in the scale-up ladder; filtered below to those both policies ran.
MAIN_CANDIDATES = {
    10: ['v1+t0.6', 'v1+t1.0', 'v2+t0.6', 'v2+t1.0', 'v3+t1.0'],
    15: ['15x25+ppvct-mixed+v2+t0.6', '15x25+ppvct-mixed+v2+t1.0'],
}
BUDGETS = [0.25, 1.0, 5.0, 15.0, 60.0, 600.0]   # nominal GA search budgets
LOG = []                                        # audit trail, printed at end


def note(s):
    LOG.append(s)
    print(s)


# ------------------------------------------------------------------ loaders
def scale_cells():
    """{module count: [cell, ...]} discovered from the scale-up PDR sweep."""
    out = {}
    for p in sorted(glob.glob('results/scaleup/pdr/*.json')):
        cell = os.path.basename(p)[:-5]
        out.setdefault(int(cell.split('x')[0]), []).append(cell)
    return out


def pdr_best(cell, scale):
    """(per-instance makespans of the best FIXED pair, pair name, n)."""
    p = (f'results/scaleup/pdr/{cell}.json' if scale else
         f'results/pdr/{cell if cell.startswith("15x25") else f"10x25+ppvct-mixed+{cell}"}.json')
    d = json.load(open(p))
    inst = sorted(d)
    pairs = sorted(next(iter(d.values())))
    arr = {q: np.array([d[i][q] for i in inst]) for q in pairs}
    best = min(arr, key=lambda q: arr[q].mean())
    return dict(zip(inst, arr[best])), best, len(inst)


def policy_seeds(model_stem, cell, scale):
    """{seed: {instance: makespan}} for every seed found on disk."""
    out = {}
    if scale:
        for p in sorted(glob.glob(f'results/scaleup/policy/{model_stem}-s*_{cell}.json')):
            if p.endswith('+b1lat.json'):
                continue
            m = re.search(r'-s(\d+)_', os.path.basename(p))
            rows = json.load(open(p))['rows']
            out[int(m.group(1))] = {i: rows[i]['ms'] for i in rows}
    else:
        for p in sorted(glob.glob(
                f'test_results/PPVCT/{cell}/Result_greedy+{model_stem}-s*_{cell}.npy')):
            m = re.search(r'-s(\d+)_', os.path.basename(p))
            a = np.load(p)[:, 0]
            out[int(m.group(1))] = {f'instance_{k:03d}': v for k, v in enumerate(a)}
    return out


def seed_mean(seeds, inst):
    """Per-instance mean over seeds, restricted to `inst`."""
    return np.mean([[d[i] for i in inst] for d in seeds.values()], axis=0)


def policy_schedule_time(model_stem, seed, cell):
    """End-to-end seconds to produce a schedule: decisions x measured latency.

    Prefers the single-instance (batch 1) latency file when one exists; else
    uses the batched file, in which one forward pass advances the whole batch,
    so decisions x latency is the time to produce that batch of schedules.
    """
    for suffix in ('+b1lat', ''):
        p = f'results/scaleup/policy/{model_stem}-s{seed}_{cell}{suffix}.json'
        if os.path.exists(p):
            d = json.load(open(p))
            r = d['rows']
            t = float(np.mean([r[i]['decisions'] * r[i]['lat_batch_s'] for i in r]))
            return t, d['device'], d['batch'], len(r), bool(d.get('latency_contended'))
    return None


def ga_traces(path, inst=None):
    """[(cost_cpu_s, best_makespan, end_cpu_s)] per instance.

    Cost includes the GA startup (PDR seeding plus population initialisation),
    which is paid before the search can return anything; end_cpu_s is where the
    run actually stopped, so no curve is drawn past it.
    """
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    keys = sorted(d) if inst is None else [i for i in inst if i in d]
    out = []
    for i in keys:
        if 'anytime' not in d[i]:
            return None
        st = d[i]['startup_cpu']
        out.append((np.array([st + a[0] for a in d[i]['anytime']]),
                    np.array([a[1] for a in d[i]['anytime']]),
                    st + d[i].get('search_cpu', d[i]['cpu'] - st)))
    return out


def step_mean(traces, grid):
    """Mean over instances of the step functions, NaN before all have started."""
    y = np.full(len(grid), np.nan)
    for k, g in enumerate(grid):
        vals = []
        for tr in traces:
            ts, vs = tr[0], tr[1]
            j = np.searchsorted(ts, g, side='right') - 1
            if j < 0:
                vals = None
                break
            vals.append(vs[j])
        if vals:
            y[k] = np.mean(vals)
    return y


def cpsat_ledger(cell):
    """{instance: row} from the anytime CP-SAT ledger (last row per instance)."""
    p = f'results/scaleup/cpsat_b/{cell}.jsonl'
    if not os.path.exists(p):
        return {}
    seen = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            if r.get('anytime'):
                seen[r['instance']] = r
    return seen


# ------------------------------------------------------------------ F5
def build_f5():
    fig = plt.figure(figsize=(3.5, 2.18))
    axa = fig.add_axes([0.175, 0.175, 0.805, 0.79])
    figb = plt.figure(figsize=(3.5, 2.40))
    axb = figb.add_axes([0.175, 0.165, 0.805, 0.80])

    # ---------------- (a) zero-shot size boundary
    sc = scale_cells()
    plan = [(s, c, False) for s, c in sorted(MAIN_CANDIDATES.items())]
    plan += [(s, c, True) for s, c in sorted(sc.items())]

    series = {'ten': [], 'mix': []}     # (size, per-cell margins, seed margins)
    for size, cells, scale in plan:
        keep = []
        for c in cells:
            t = policy_seeds(TEN, c, scale)
            m = policy_seeds(MIX, c, scale)
            if t and m:
                keep.append((c, t, m))
            elif t or m:
                note(f'  (a) skip {size}/{c}: only '
                     f'{"10-only" if t else "mixture"} evaluated')
        if not keep:
            continue
        per = {'ten': [], 'mix': []}
        seedmarg = {'ten': {}, 'mix': {}}
        for c, t, m in keep:
            pb, bname, n = pdr_best(c, scale)
            inst = sorted(set(pb) & set(next(iter(t.values()))))
            base = np.mean([pb[i] for i in inst])
            for tag, seeds in (('ten', t), ('mix', m)):
                mm = seed_mean(seeds, inst)
                per[tag].append(100 * (mm.mean() - base) / base)
                for s, d in seeds.items():
                    v = np.mean([d[i] for i in inst])
                    seedmarg[tag].setdefault(s, []).append(
                        100 * (v - base) / base)
                note(f'  (a) size {size:2d} {c:32s} n={len(inst)} '
                     f'bestPDR={base:8.2f} ({bname}) {tag}={mm.mean():8.2f} '
                     f'margin={per[tag][-1]:+6.2f}% seeds={sorted(seeds)}')
        for tag in ('ten', 'mix'):
            sm = [np.mean(v) for v in seedmarg[tag].values()]
            series[tag].append((size, np.array(per[tag]), np.array(sm),
                                len(seedmarg[tag])))

    style = {'ten': (ROSE_M, ROSE_F, 'o', '-', 'trained on 10 modules'),
             'mix': (BLUE_M, BLUE_F, 's', '--', 'trained on 10, 15 and 20 modules')}
    for tag in ('ten', 'mix'):
        col, fill, mk, ls, lab = style[tag]
        xs = [d[0] for d in series[tag]]
        ys = [d[1].mean() for d in series[tag]]
        nseed = max(d[3] for d in series[tag])
        if tag == 'ten' and nseed > 1:            # seed range band
            lo = [d[2].min() for d in series[tag]]
            hi = [d[2].max() for d in series[tag]]
            axa.fill_between(xs, lo, hi, color=fill, alpha=0.45, lw=0,
                             zorder=1)
            note('  (a) 10-only seed band (per-seed size means): ' +
                 ', '.join(f'{x}:[{a:+.2f},{b:+.2f}]'
                           for x, a, b in zip(xs, lo, hi)))
        for x, cellvals in [(d[0], d[1]) for d in series[tag]]:
            if len(cellvals) > 1:
                axa.plot([x] * len(cellvals), cellvals, mk, ms=2.6,
                         mfc='none', mec=col, mew=0.7, ls='none', zorder=3)
        axa.plot(xs, ys, ls, color=col, lw=1.3, marker=mk, ms=4,
                 mfc=col, mec=col, zorder=4,
                 label=f'{lab} ({nseed} seed{"s" if nseed > 1 else ""})')
        note(f'  (a) {tag}: ' + ', '.join(f'{x}:{y:+.2f}%'
                                          for x, y in zip(xs, ys)))

    axa.axhline(0, color=GREY_M, lw=0.8, ls='--', zorder=2)
    axa.set_xscale('log')
    ticks = sorted({d[0] for d in series['ten']})
    axa.set_xticks(ticks)
    axa.set_xticklabels([str(t) for t in ticks])
    axa.minorticks_off()
    axa.set_xlim(min(ticks) * 0.88, max(ticks) * 1.14)
    axa.set_ylim(-15.4, 8.6)
    axa.text(min(ticks) * 0.91, 0.5, 'best fixed dispatching pair',
             fontsize=6.5, va='bottom', ha='left', color='black')
    axa.set_xlabel('modules per instance')
    axa.set_ylabel('makespan vs best fixed pair (%)\n(below 0: policy better)')
    axa.legend(frameon=False, loc='lower left', fontsize=6.5,
               labelspacing=0.25, handlelength=2.0, borderaxespad=0.15)
    axa.spines[['top', 'right']].set_visible(False)
    axa.tick_params(width=0.6, length=2.5)

    # ---------------- (b) anytime comparison on one 50-module cell
    cands = [c for c in sc.get(50, [])][::-1]     # prefer the t1.0 cell
    cellb, led = None, {}
    for c in cands:
        L = cpsat_ledger(c)
        if len(L) > len(led):
            cellb, led = c, L
    if not led:
        note('  (b) no CP-SAT anytime rows on disk yet; panel (b) skipped')
        return fig, None, None
    inst = sorted(led)
    note(f'  (b) cell {cellb}; CP-SAT ledger rows n={len(inst)}: '
         f'{inst[0]}..{inst[-1]}')

    pb, bname, npdr = pdr_best(cellb, True)
    base = np.mean([pb[i] for i in inst])
    note(f'  (b) best fixed pair {bname} = {base:.2f} on these n={len(inst)} '
         f'(full n={npdr}: {np.mean(list(pb.values())):.2f})')

    xlo, xhi = 0.2, 4000.0
    grid = np.logspace(np.log10(xlo), np.log10(xhi), 700)

    # GA: prefer the long (600 s) run, which contains the short run's trace
    for gp in (f'results/scaleup/ga_long/600s/{cellb}.json',
               f'results/scaleup/ga/{cellb}.json'):
        tr = ga_traces(gp, inst)
        if tr:
            break
    if tr:
        y = step_mean(tr, grid)
        t0 = max(t[0][0] for t in tr)
        tend = min(t[2] for t in tr)          # no curve past the actual run
        ok = (~np.isnan(y)) & (grid <= tend)
        axb.step(grid[ok], y[ok], where='post', color=AMBER_M, lw=1.2,
                 zorder=4, label='genetic algorithm (CPU-s)')
        note(f'  (b) GA from {gp}, n={len(tr)}, curve {t0:.1f} CPU-s '
             f'({y[ok][0]:.2f}) to {grid[ok][-1]:.0f} CPU-s '
             f'({y[ok][-1]:.2f}); runs stop at {tend:.0f} CPU-s')

    # CP-SAT: incumbent upper bound over solver wall time (+ warm-start cost)
    ctr = [(np.array([led[i]['warmstart_cpu_s'] + a[0] for a in led[i]['anytime']]),
            np.array([a[1] for a in led[i]['anytime']])) for i in inst]
    yc = step_mean(ctr, grid)
    ok = ~np.isnan(yc)
    tmax = min(led[i]['warmstart_cpu_s'] + led[i]['walltime'] for i in inst)
    ok &= grid <= tmax
    axb.step(grid[ok], yc[ok], where='post', color=GREY_M, lw=1.2, zorder=4,
             label='CP-SAT incumbent (wall, 4 workers)')
    note(f'  (b) CP-SAT n={len(ctr)}, curve starts '
         f'{max(t[0][0] for t in ctr):.1f} s at {yc[ok][0]:.2f} (warm start), '
         f'ends {grid[ok][-1]:.0f} s at {yc[ok][-1]:.2f}; '
         f'final UB mean {np.mean([led[i]["ub"] for i in inst]):.2f}')

    axb.plot([xlo, xhi], [base, base], ls='--', color=GREY_M, lw=0.9, zorder=3,
             label='best fixed dispatching pair')
    for tag, stem, col in (('ten', TEN, ROSE_M), ('mix', MIX, BLUE_M)):
        seeds = policy_seeds(stem, cellb, True)
        if not seeds:
            continue
        v = float(seed_mean(seeds, inst).mean())
        lat = policy_schedule_time(stem, min(seeds), cellb)
        t = lat[0] if lat else xlo
        axb.plot([t, xhi], [v, v], ls='-' if tag == 'ten' else '-.',
                 color=col, lw=1.3, zorder=5,
                 label=('policy trained on 10 modules' if tag == 'ten'
                        else 'policy trained on 10, 15, 20 modules'))
        axb.plot([t], [v], marker='o' if tag == 'ten' else 's', ms=3.4,
                 color=col, zorder=6)
        note(f'  (b) {tag} policy seeds={sorted(seeds)} mean={v:.2f} '
             f'schedule time={t:.2f} s from {lat[1]} batch={lat[2]} '
             f'(n={lat[3]} instances, contended={lat[4]})')
        for who, yy in (('GA', y if tr else None), ('CP-SAT', yc)):
            if yy is None:
                continue
            m = (~np.isnan(yy)) & (yy <= v)
            note(f'  (b) {who} reaches the {tag} policy level ({v:.2f}) at '
                 + (f'{grid[m][0]:.0f} s' if m.any() else 'no point in budget'))

    axb.set_xscale('log')
    axb.set_xlim(xlo, xhi)
    axb.set_xlabel('compute per instance (s, log scale)')
    axb.set_ylabel('mean makespan')
    axb.legend(frameon=False, loc='lower left', fontsize=7.0,
               labelspacing=0.22, handlelength=2.0, borderaxespad=0.15)
    axb.spines[['top', 'right']].set_visible(False)
    axb.tick_params(width=0.6, length=2.5)

    for ax, tag in ((axa, '(a)'), (axb, '(b)')):
        continue  # single-panel figures carry no tag
        ax.text(-0.175, 1.025, tag, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='left')
    return fig, figb, cellb


# ------------------------------------------------------------------ S2
def build_s2():
    sc = scale_cells()
    fig = plt.figure(figsize=(3.5, 2.65))
    ax = fig.add_axes([0.165, 0.16, 0.815, 0.82])

    lines = []
    for size in sorted(sc):
        if size not in (20, 30, 50):
            continue
        for cell in sc[size]:
            pts = {}
            tr = (ga_traces(f'results/scaleup/ga_long/600s/{cell}.json') or
                  ga_traces(f'results/scaleup/ga/{cell}.json'))
            if tr:                      # reconstruct any budget from the trace
                span = min(t[2] - t[0][0] for t in tr)
                for b in BUDGETS:
                    if b > span:
                        continue        # budget beyond the recorded trace
                    vals, costs = [], []
                    for ts, vs, _ in tr:
                        j = 0
                        while j + 1 < len(ts) and ts[j] - ts[0] < b:
                            j += 1
                        vals.append(vs[j])
                        costs.append(ts[j])
                    key = round(float(np.mean(costs)), 1)
                    if key not in pts:   # keep the cheapest budget that
                        pts[key] = (float(np.mean(vals)), b)   # reaches it
                for b, p in [(60.0, f'results/scaleup/ga/{cell}.json'),
                             (600.0, f'results/scaleup/ga_long/600s/{cell}.json')]:
                    if os.path.exists(p) and b <= span:
                        d = json.load(open(p))
                        hit = [v[0] for v in pts.values() if v[1] == b]
                        if hit:
                            note(f'  (s2) {cell} b={b:g}: reconstructed '
                                 f'{hit[0]:.2f} vs on-disk run '
                                 f'{np.mean([d[i]["ga"] for i in d]):.2f} '
                                 f'(cpu {np.mean([d[i]["cpu"] for i in d]):.1f})')
            else:                       # separate fixed-budget runs on disk
                for b, p in [(0.25, f'results/scaleup/ga_budget/0.25s/{cell}.json'),
                             (1.0, f'results/scaleup/ga_budget/1s/{cell}.json'),
                             (5.0, f'results/scaleup/ga_budget/5s/{cell}.json'),
                             (60.0, f'results/scaleup/ga/{cell}.json')]:
                    if not os.path.exists(p):
                        continue
                    d = json.load(open(p))
                    ms = float(np.mean([d[i]['ga'] for i in d]))
                    cpu = float(np.mean([d[i]['cpu'] for i in d]))
                    hit = [k for k, v in pts.items() if abs(v[0] - ms) < 1e-6]
                    if hit:             # same output as a cheaper budget
                        continue
                    pts[round(cpu, 1)] = (ms, b)
            if len(pts) < 2:
                note(f'  (s2) skip {cell}: {len(pts)} budget point(s) on disk')
                continue
            seeds = policy_seeds(TEN, cell, True)
            inst = sorted(next(iter(seeds.values())))
            ref = float(seed_mean(seeds, inst).mean())
            mixs = policy_seeds(MIX, cell, True)
            mixr = (float(seed_mean(mixs, inst).mean()) / ref) if mixs else None
            xs = sorted(pts)
            lines.append(dict(size=size, cell=cell, xs=xs,
                              ys=[pts[x][0] / ref for x in xs],
                              bs=[pts[x][1] for x in xs], ref=ref, mix=mixr))
            note(f'  (s2) {cell}: 10-only policy mean={ref:.2f} (n={len(inst)}, '
                 f'seeds={sorted(seeds)}), mixture ratio='
                 + (f'{mixr:.4f}' if mixr else 'n/a'))
            for x in xs:
                note(f'        budget {pts[x][1]:6.2f} CPU-s search -> '
                     f'{pts[x][0]:9.2f} makespan at {x:7.1f} CPU-s total '
                     f'(ratio {pts[x][0]/ref:.4f})')

    mk = ['o', 's', '^', 'D', 'v']
    for k, L in enumerate(lines):
        v, t = re.search(r'\+v(\d)\+t([\d.]+)$', L['cell']).groups()
        lab = (f"{L['size']} modules, $|V|{{=}}{v}$, "
               r'$\bar\tau/\bar p{=}' + t + '$')
        ax.plot(L['xs'], L['ys'], '-', color=BLUE_SHADES[k % len(BLUE_SHADES)],
                lw=1.2, marker=mk[k % len(mk)], ms=3.6, label=lab, zorder=4)

    ax.axhline(1.0, color=GREY_M, lw=0.9, ls='--', zorder=2)
    mixes = [L['mix'] for L in lines if L['size'] == 50 and L['mix']]
    if mixes:
        ax.axhline(float(np.mean(mixes)), color=BLUE_M, lw=0.9, ls=':', zorder=2)
        note(f'  (s2) mixture-policy reference line at '
             f'{np.mean(mixes):.4f} (50-module cells: '
             f'{", ".join(f"{m:.4f}" for m in mixes)})')

    # what the policy itself costs, in the same CPU-seconds as the GA
    big = [L['cell'] for L in lines if L['size'] == 50]
    lat = policy_schedule_time(TEN, 301, big[0]) if big else None
    ax.set_xscale('log')
    ax.set_xlabel('genetic-algorithm compute per instance (CPU-s, log scale)')
    ax.set_ylabel('mean makespan relative to the\npolicy trained on 10 modules')
    lo = min(min(L['ys']) for L in lines)
    ax.set_ylim(lo - 0.055, 1.10)
    ax.set_xlim(4, 1600)
    ax.text(4.4, 0.9955, 'policy trained on 10 modules', fontsize=6.5,
            va='top', ha='left')
    if mixes:
        ax.text(4.4, float(np.mean(mixes)) + 0.005,
                'policy trained on 10, 15 and 20 modules', fontsize=6.5,
                va='bottom', ha='left')
    if lat:
        ax.axvline(lat[0], color=GREY_M, lw=0.8, ls='-.', zorder=2)
        ax.text(lat[0] * 1.12, lo - 0.048,
                f'policy: one 50-module\nschedule, {lat[0]:.1f} CPU-s',
                fontsize=6.5, va='bottom', ha='left')
        note(f'  (s2) policy cost line at {lat[0]:.2f} CPU-s ({big[0]}, '
             f'{lat[1]} batch={lat[2]}, n={lat[3]} instances, '
             f'contended={lat[4]})')
    ax.legend(frameon=False, loc='upper center', ncol=2, fontsize=6.5,
              labelspacing=0.25, columnspacing=1.1, handlelength=1.8,
              borderaxespad=0.05)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=0.6, length=2.5)
    return fig


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['f5', 's2'])
    a = ap.parse_args()
    os.makedirs('paper/figures', exist_ok=True)
    if a.only in (None, 'f5'):
        note('F5 (paper/figures/f5_scale.pdf)')
        fig, figb, cellb = build_f5()
        fig.savefig('paper/figures/f5_scale.pdf')
        fig.savefig('paper/figures/f5_scale.png', dpi=300)
        if figb is not None:
            figb.savefig('paper/figures/s4_anytime.pdf')
            figb.savefig('paper/figures/s4_anytime.png', dpi=300)
            plt.close(figb)
        plt.close(fig)
    if a.only in (None, 's2'):
        note('S2 (paper/figures/s2_scale_budget.pdf)')
        fig = build_s2()
        fig.savefig('paper/figures/s2_scale_budget.pdf')
        fig.savefig('paper/figures/s2_scale_budget.png', dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
