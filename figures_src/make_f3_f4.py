"""Render F3 (credit-and-guidance) and F4 (coupling-regret map) for the
manuscript. Every plotted number is read from result files; nothing is
hand-typed. Fonts: Liberation Serif (Times-metric); house palette: pastel
fills, medium lines, black text only.

Run: python figures_src/make_f3_f4.py   (from repo root)
Outputs: paper/figures/f3_credit.pdf, paper/figures/f4_regret_map.pdf
"""

import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RNG = np.random.default_rng(20260711)

# house palette (paper-figures skill)
BLUE_F, ROSE_F, AMBER_F, GREY_F = '#a8c6e3', '#eda9b0', '#f3cf8f', '#c3cbd3'
BLUE_M, ROSE_M = '#4f81ad', '#c25b6a'
AMBER_M, GREY_M = '#b8860b', '#6b7480'

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x2_style import register_fonts, FAMILY
register_fonts()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [FAMILY],
    'mathtext.fontset': 'custom',
    'mathtext.rm': FAMILY,
    'mathtext.it': FAMILY + ':italic',
    'mathtext.bf': FAMILY + ':bold',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'text.color': 'black', 'axes.labelcolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'axes.linewidth': 0.6, 'axes.edgecolor': 'black',
    'pdf.fonttype': 42,
})

CELLS = [(v, t) for v in (1, 2, 3) for t in (0.1, 0.3, 0.6, 1.0)]
SEEDS = (301, 302, 303)


def seed_mean(cell, prefix, seeds=SEEDS):
    arrs = []
    for s in seeds:
        p = (f'test_results/PPVCT/{cell}/Result_greedy+10x25+ppvct-mixed+'
             f'{prefix}-s{s}_{cell}.npy')
        if os.path.exists(p):
            arrs.append(np.load(p)[:, 0])
    return np.mean(arrs, axis=0) if arrs else None


def vali_curve(prefix):
    """(updates, mean, lo, hi) of vali_score across available seeds."""
    per_seed = {}
    for s in SEEDS:
        p = f'train_log/PPVCT/train_10x25+ppvct-mixed+{prefix}-s{s}.log.jsonl'
        if not os.path.exists(p):
            continue
        us, vs = [], []
        for ln in open(p):
            r = json.loads(ln)
            if 'vali_score' in r:
                us.append(r['update'])
                vs.append(r['vali_score'])
        if us and max(us) >= 1900:      # completed runs only (2000 updates)
            per_seed[s] = dict(zip(us, vs))
    if not per_seed:
        return None
    common = sorted(set.intersection(*[set(d) for d in per_seed.values()]))
    m = np.array([[d[u] for u in common] for d in per_seed.values()])
    return np.array(common), m.mean(0), m.min(0), m.max(0)


# ---------------------------------------------------------------- F3
ARMS = [
    ('joint-v1', 'Shared reward', GREY_M, '-', 1.0),
    ('m1-bcb', '+ BCB credit (M1)', AMBER_M, '--', 1.0),
    ('m1-bcb-guide', '+ price channel (ours)', BLUE_M, '-', 1.6),
    ('coma-critic', 'COMA-style critic', GREY_M, ':', 1.0),
    ('m2-shaped', 'M2: credit in return', ROSE_M, '-.', 1.0),
]

figc, ax1 = plt.subplots(figsize=(3.5, 1.62))
figc.subplots_adjust(left=0.135, right=0.965, top=0.965, bottom=0.255)
figb, ax2 = plt.subplots(figsize=(3.5, 1.60))
figb.subplots_adjust(left=0.125, right=0.99, top=0.965, bottom=0.27)

clip_hi = 380.0
for prefix, label, color, ls, lw in ARMS:
    c = vali_curve(prefix)
    if c is None:
        continue
    u, mean, lo, hi = c
    ax1.plot(u, np.minimum(mean, clip_hi), ls, color=color, lw=lw,
             label=label)
    ax1.fill_between(u, np.minimum(lo, clip_hi), np.minimum(hi, clip_hi),
                     color=color, alpha=0.14, lw=0)
    if mean[-1] > clip_hi:
        ax1.annotate(f'M2 off scale\n(final {mean[-1]:.0f})',
                     xy=(u[np.argmax(np.minimum(mean, clip_hi) >= clip_hi)],
                         clip_hi),
                     xytext=(620, 362), fontsize=7, style='italic',
                     arrowprops=dict(arrowstyle='-', lw=0.6, color='black'))
ax1.set_xlabel('PPO update')
ax1.set_ylabel('validation makespan')
ax1.set_xlim(0, 2000)
ax1.legend(frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.60),
           fontsize=6.5, labelspacing=0.22, handlelength=1.6)
ax1.spines[['top', 'right']].set_visible(False)

# (b) guide vs credit-only, per-cell improvement with 95% bootstrap CI
tvals = (0.1, 0.3, 0.6, 1.0)
fleet_fill = {1: BLUE_F, 2: AMBER_F, 3: GREY_F}
fleet_edge = {1: BLUE_M, 2: AMBER_M, 3: GREY_M}
width = 0.26
for v in (1, 2, 3):
    xs, ys, err = [], [], []
    for j, t in enumerate(tvals):
        cell = f'v{v}+t{t}'
        g = seed_mean(cell, 'm1-bcb-guide')
        m = seed_mean(cell, 'm1-bcb')
        if g is None or m is None:
            continue
        d = 100 * (m - g) / m            # per-instance improvement %
        boot = np.array([d[RNG.integers(0, len(d), len(d))].mean()
                         for _ in range(2000)])
        xs.append(j + (v - 2) * width)
        ys.append(d.mean())
        err.append([d.mean() - np.percentile(boot, 2.5),
                    np.percentile(boot, 97.5) - d.mean()])
    ax2.bar(xs, ys, width, color=fleet_fill[v], edgecolor=fleet_edge[v],
            lw=0.8, label=f'$|V|={v}$',
            yerr=np.array(err).T, error_kw=dict(lw=0.7, capsize=2,
                                                ecolor='black'))
ax2.axhline(0, color=GREY_M, lw=0.6)
ax2.set_xticks(range(len(tvals)))
ax2.set_xticklabels(['0.1', '0.3', '0.6', '1.0 (zero-shot)'])
ax2.set_xlabel(r'travel intensity $\bar\tau/\bar p$')
ax2.set_ylabel('improvement (%)')
ax2.legend(frameon=False, loc='upper left')
ax2.spines[['top', 'right']].set_visible(False)

os.makedirs('paper/figures', exist_ok=True)
figb.savefig('paper/figures/f3_credit.pdf')
figb.savefig('paper/figures/f3_credit.png', dpi=300)
figc.savefig('paper/figures/s1_curves.pdf')
figc.savefig('paper/figures/s1_curves.png', dpi=300)
plt.close(figb); plt.close(figc)

# ---------------------------------------------------------------- F4
reg = np.full((3, 4), np.nan)
for i, v in enumerate((1, 2, 3)):
    for j, t in enumerate(tvals):
        cell = f'v{v}+t{t}'
        p = (f'test_results/PPVCT/{cell}/Result_greedy-NVF+10x25+ppvct-mixed+'
             f'e0b-uncontended-s301_{cell}.npy')
        ours = seed_mean(cell, 'm1-bcb-guide')
        if os.path.exists(p) and ours is not None:
            e0 = np.load(p)[:, 0]
            reg[i, j] = 100 * (e0.mean() - ours.mean()) / ours.mean()

# minimum loaded fleet utilization per cell, B_veh(s0)/UB = W_tr/(|V| C):
# a floor on the share of fleet time any schedule must spend carrying loads
# (scripts/x2_fleet_util.py -> results/fleet_util.json; empty legs excluded)
util = np.full((3, 4), np.nan)
_fu = json.load(open('results/fleet_util.json'))
for i, v in enumerate((1, 2, 3)):
    for j, t in enumerate(tvals):
        rec = _fu.get(f'10x25+ppvct-mixed+v{v}+t{t}')
        if rec and rec.get('util_mean') is not None:
            util[i, j] = 100 * rec['util_mean']

cmap = LinearSegmentedColormap.from_list(
    'regret', [BLUE_F, '#ffffff', ROSE_F])
norm = TwoSlopeNorm(vcenter=0.0, vmin=-2.0, vmax=np.nanmax(reg))

fig, ax = plt.subplots(figsize=(3.5, 1.62))
fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.235)
ax.imshow(reg, cmap=cmap, norm=norm, aspect='auto')
for i in range(3):
    for j in range(4):
        ax.text(j, i - 0.16, f'{reg[i, j]:+.1f}%', ha='center', va='center',
                fontsize=8, color='black')
        ax.text(j, i + 0.19, f'{util[i, j]:.0f}% loaded', ha='center',
                va='center', fontsize=6.2, color='black')
ax.set_xticks(range(4))
ax.set_xticklabels(['0.1', '0.3', '0.6', '1.0 (zero-shot)'])
ax.set_yticks(range(3))
ax.set_yticklabels(['1', '2', '3'])
ax.set_xlabel(r'travel intensity $\bar\tau/\bar p$')
ax.set_ylabel(r'fleet size $|V|$')
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
fig.savefig('paper/figures/f4_regret_map.pdf')
fig.savefig('paper/figures/f4_regret_map.png', dpi=300)
print('regret grid:\n', np.round(reg, 1))
print('wrote paper/figures/f3_credit.{pdf,png}, f4_regret_map.{pdf,png}')
