"""Render S5 (root-bound tightness by regime) for the supplement.

The bound of Theorem 1 is admissible, so B(s_0) <= C* <= C for every
feasible schedule C. Admissibility says nothing about how far below the
optimum the bound sits, and a valid but loose bound certifies nothing
useful. This figure measures the ratio on stored results:

    tightness = B(s_0) / C_best,     C_best = the best makespan any
                                     evaluated method attained on that
                                     instance (per instance, then averaged
                                     over the cell)

against the three regime variables the paper varies:

  (a) minimum loaded fleet utilization, one point per primary-grid cell;
  (b) station scarcity at 50 modules (25/15/9 stations, |V|=2, tau/p=0.1),
      where the machine component of the bound binds on every instance;
  (c) processing-time heterogeneity, three cells x five levels.

Every plotted number is read from a result file; nothing is hand-typed.
No solver and no policy is run: the script only reads and aggregates.

Arms entering C_best, per panel (all of them for that study):
  (a) every policy checkpoint evaluated on the cell (per seed), the nine
      dispatching pairs, the 60-CPU-s genetic algorithm, and the
      strengthened 300-s CP-SAT incumbent;
  (b) the nine dispatching pairs, the 60-CPU-s genetic algorithm, and the
      three mixture-trained policy seeds (no CP-SAT reference exists for
      these cells);
  (c) as recorded by scripts/x2_hetero_report.py: the three bound-guided
      and three shared-reward policy seeds and the nine dispatching pairs.

Fonts: TeX Gyre Termes (the manuscript face); house palette; black text
only; no titles inside the figure.

Run:  python figures_src/make_s5_tightness.py     (from repo root)
Outputs: paper/figures/s5_tightness.{pdf,png}
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x2_style import register_fonts, FAMILY

# ------------------------------------------------------------------ style
# house palette (paper-figures skill), identical to figures_src/make_f5_scale.py
BLUE_F, ROSE_F, AMBER_F, GREY_F = '#a8c6e3', '#eda9b0', '#f3cf8f', '#c3cbd3'
BLUE_M, ROSE_M = '#4f81ad', '#c25b6a'
AMBER_M, GREY_M = '#b8860b', '#6b7480'

register_fonts()
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [FAMILY],
    'mathtext.fontset': 'custom',
    'mathtext.rm': FAMILY,
    'mathtext.it': FAMILY + ':italic',
    'mathtext.bf': FAMILY + ':bold',
    'mathtext.sf': FAMILY,
    'mathtext.tt': FAMILY,
    'mathtext.cal': FAMILY + ':italic',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'axes.linewidth': 0.6, 'axes.edgecolor': 'black',
    'pdf.fonttype': 42,
})

GRID = [f'v{v}+t{t}' for v in (1, 2, 3) for t in ('0.1', '0.3', '0.6', '1.0')]
MCH = [('50x25+ppvct-mixed+v2+t0.1', 25),
       ('50x15+ppvct-mixed+v2+t0.1', 15),
       ('50x9+ppvct-mixed+v2+t0.1', 9)]
HETERO = [('v1+t0.6', 1, '0.6'), ('v2+t0.6', 2, '0.6'), ('v1+t1.0', 1, '1.0')]

NOTES = []


def note(s):
    NOTES.append(s)
    print(s, flush=True)


def cell_label(v, t):
    return r'$|V|{=}%d$, $\bar\tau/\bar p{=}%s$' % (v, t)


def tightness(B, arms, what):
    """Per-instance B(s0)/C_best, with the admissibility self-check."""
    S = np.stack(arms)
    cb = S.min(axis=0)
    t = B / cb
    assert t.max() <= 1.0 + 1e-9, f'{what}: root bound above a feasible makespan'
    return t, cb


# ------------------------------------------------------------- panel (a)
def panel_a():
    util = json.load(open('results/fleet_util.json'))
    out = []
    for cell in GRID:
        stem = f'10x25+ppvct-mixed+{cell}'
        b0 = json.load(open(f'results/certificate/{stem}.json'))
        names = sorted(b0)
        B = np.array([b0[n] for n in names])

        arms = []
        for p in sorted(glob.glob(f'test_results/PPVCT/{cell}/Result_*.npy')):
            a = np.load(p)[:, 0]
            if len(a) == len(names):
                arms.append(a)
        n_pol = len(arms)
        pdr = json.load(open(f'results/pdr/{stem}.json'))
        for q in sorted(next(iter(pdr.values()))):
            arms.append(np.array([pdr[n][q] for n in names]))
        ga = json.load(open(f'results/ga_v2/{stem}.json'))
        arms.append(np.array([ga[n]['ga'] for n in names]))
        cp = {}
        for ln in open(f'results/cpsat_v2/{stem}.jsonl'):
            if ln.strip():
                r = json.loads(ln)
                if r['ub'] is not None:
                    cp[r['instance']] = r['ub']
        assert len(cp) == len(names), f'{cell}: partial CP-SAT ledger'
        arms.append(np.array([cp[n] for n in names]))

        t, cb = tightness(B, arms, cell)
        u = 100 * util[stem]['util_mean']
        out.append((u, float(t.mean())))
        note(f'  (a) {cell:9s} n={len(names)} |V|={util[stem]["n_veh"]} '
             f'u={u:5.1f}%  B0={B.mean():8.2f}  C_best={cb.mean():8.2f}  '
             f'tightness={t.mean():.4f} (sd {t.std(ddof=1):.4f}) '
             f'[{n_pol} policy arms + 9 dispatching pairs + GA + CP-SAT]')
    return out


# ------------------------------------------------------------- panel (b)
def panel_b():
    out = []
    for cell, M in MCH:
        b0 = json.load(open(f'results/scaleup/certificate/{cell}.json'))
        names = sorted(b0)
        B = np.array([b0[n] for n in names])
        arms = []
        pdr = json.load(open(f'results/scaleup/pdr/{cell}.json'))
        for q in sorted(next(iter(pdr.values()))):
            arms.append(np.array([pdr[n][q] for n in names]))
        ga = json.load(open(f'results/scaleup/ga/{cell}.json'))
        arms.append(np.array([ga[n]['ga'] for n in names]))
        seeds = sorted(glob.glob(
            f'results/scaleup/policy/*guide-mix-s30*_{cell}.json'))
        for p in seeds:
            rows = json.load(open(p))['rows']
            arms.append(np.array([rows[n]['ms'] for n in names]))
        t, cb = tightness(B, arms, cell)
        out.append((M, float(t.mean())))
        note(f'  (b) {M:2d} stations  n={len(names)}  B0={B.mean():9.2f}  '
             f'C_best={cb.mean():9.2f}  tightness={t.mean():.4f} '
             f'(sd {t.std(ddof=1):.4f})  C_best/B0={float((cb / B).mean()):.3f} '
             f'[9 dispatching pairs + GA + {len(seeds)} policy seeds]')
    return out


# ------------------------------------------------------------- panel (c)
def panel_c():
    rows = json.load(open('results/hetero/summary.json'))['rows']
    out = []
    for cell, v, t in HETERO:
        pts = [(r['realized_R']['mean'], r['bound']['tightness_mean'],
                r['level'], r['n'])
               for r in rows if r['cell'] == cell and 'bound' in r]
        pts.sort()
        if not pts:
            note(f'  (c) {cell}: no rows on disk, series skipped')
            continue
        out.append((cell, v, t, pts))
        note(f'  (c) {cell:9s} levels ' +
             ', '.join(f'{p[2]} (R={p[0]:.2f}, n={p[3]}): {p[1]:.4f}'
                       for p in pts))
    return out


# ------------------------------------------------------------------ main
def main():
    note('S5 (paper/figures/s5_tightness.pdf)')
    A, Bp, C = panel_a(), panel_b(), panel_c()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 4.70))
    fig.subplots_adjust(left=0.175, right=0.975, top=0.960, bottom=0.072,
                        hspace=0.55)

    # (a) fleet utilization
    ax1.plot([p[0] for p in A], [p[1] for p in A], 'o', ms=4.2,
             mfc=BLUE_F, mec=BLUE_M, mew=0.8, ls='none')
    ax1.set_xlabel('minimum loaded utilization (%)')
    ax1.set_xlim(2, 81)
    ax1.set_ylim(0.66, 1.01)
    ax1.set_yticks([0.7, 0.8, 0.9, 1.0])

    # (b) station scarcity
    xs = np.arange(len(Bp))
    ax2.plot(xs, [p[1] for p in Bp], '-o', color=BLUE_M, lw=1.2, ms=4.2,
             mfc=BLUE_F, mec=BLUE_M, mew=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(p[0]) for p in Bp])
    ax2.set_xlim(-0.35, len(Bp) - 0.65)
    ax2.set_xlabel('stations')
    ax2.set_ylim(0.50, 0.92)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])

    # (c) processing-time heterogeneity
    style = [(BLUE_M, BLUE_F, 'o', '-'), (AMBER_M, AMBER_F, 's', '--'),
             (ROSE_M, ROSE_F, '^', '-.')]
    for (cell, v, t, pts), (cm, cf, mk, ls) in zip(C, style):
        ax3.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=cm,
                 lw=1.1, marker=mk, ms=3.8, mfc=cf, mec=cm, mew=0.8,
                 label=cell_label(v, t))
    ax3.set_xlabel('realized within-operation max/min ratio')
    ax3.set_xlim(0.7, 8.3)
    ax3.set_ylim(0.685, 0.845)
    ax3.set_yticks([0.70, 0.75, 0.80])
    ax3.legend(frameon=False, loc='upper right', labelspacing=0.25,
               handlelength=1.9, borderaxespad=0.15)

    for ax, tag in ((ax1, '(a)'), (ax2, '(b)'), (ax3, '(c)')):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(width=0.6, length=2.5)
        ax.text(-0.145, 1.04, tag, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='left')

    fig.supylabel('root-bound tightness $B(s_0)/C$', fontsize=8, x=0.020)

    os.makedirs('paper/figures', exist_ok=True)
    fig.savefig('paper/figures/s5_tightness.pdf')
    fig.savefig('paper/figures/s5_tightness.png', dpi=300)
    plt.close(fig)
    note('wrote paper/figures/s5_tightness.{pdf,png}')


if __name__ == '__main__':
    sys.exit(main())
