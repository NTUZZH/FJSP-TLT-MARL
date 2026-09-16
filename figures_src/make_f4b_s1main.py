"""Render two panels for the manuscript:

  paper/figures/f4b_components.pdf  active bound component per regime cell
                                    (panel (b) beside the coupling-regret map)
  paper/figures/s1_curves_main.pdf  training curves, three arms
                                    (main-text version of the supplement's
                                     five-arm figure)

Every plotted number is read from a result file or a training log; nothing is
hand-typed. Geometry, fonts and palette follow the sibling panels rendered by
figures_src/make_f3_f4.py, so the two halves of each figure align cell for
cell.

Run: python figures_src/make_f4b_s1main.py   (from repo root)
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x2_style import register_fonts, FAMILY
register_fonts()

# house palette (paper-figures skill): pastel fills, medium lines, black text
BLUE_F, ROSE_F, AMBER_F, GREY_F = '#a8c6e3', '#eda9b0', '#f3cf8f', '#c3cbd3'
BLUE_M, ROSE_M = '#4f81ad', '#c25b6a'
AMBER_M, GREY_M = '#b8860b', '#6b7480'

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

FLEETS = (1, 2, 3)
TAUS = (0.1, 0.3, 0.6, 1.0)
SEEDS = (301, 302, 303)
os.makedirs('paper/figures', exist_ok=True)


# ------------------------------------------------- F4(b) active component
# results/bound_terms_grid.json <- scripts/x2_bound_terms_grid.py
# One hue per bound component, taken from the house fill palette and lightened
# 25% toward white so the categorical map carries the same ink weight as the
# diverging regret map it sits beside. The component is also written into every
# cell, so the encoding survives greyscale printing.
FILL = {'chain': BLUE_F, 'mch': GREY_F, 'veh': AMBER_F}
WORD = {'chain': 'chain-bound', 'mch': 'machine-bound', 'veh': 'fleet-bound'}
TINT = 0.25

rows = {r['cell']: r for r in json.load(open('results/bound_terms_grid.json'))}
rgb = np.zeros((3, 4, 3))
label = [[None] * 4 for _ in range(3)]
for i, v in enumerate(FLEETS):
    for j, t in enumerate(TAUS):
        r = rows[f'10x25+ppvct-mixed+v{v}+t{t}']
        comp = r['dominant']
        base = np.array(matplotlib.colors.to_rgb(FILL[comp]))
        rgb[i, j] = base + (1.0 - base) * TINT
        label[i][j] = (WORD[comp], f'{100 * r["dominant_share"]:.0f}%')

fig, ax = plt.subplots(figsize=(3.5, 1.62))
fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.235)
ax.imshow(rgb, aspect='auto')
for i in range(3):
    for j in range(4):
        word, share = label[i][j]
        ax.text(j, i - 0.16, word, ha='center', va='center', fontsize=8,
                color='black')
        ax.text(j, i + 0.19, share, ha='center', va='center', fontsize=6.2,
                color='black')
ax.set_xticks(range(4))
ax.set_xticklabels(['0.1', '0.3', '0.6', '1.0 (zero-shot)'])
ax.set_yticks(range(3))
ax.set_yticklabels(['1', '2', '3'])
ax.set_xlabel(r'travel intensity $\bar\tau/\bar p$')
ax.set_ylabel(r'fleet size $|V|$')
for spine in ax.spines.values():
    spine.set_linewidth(0.6)
fig.savefig('paper/figures/f4b_components.pdf')
fig.savefig('paper/figures/f4b_components.png', dpi=300)
plt.close(fig)
for i, v in enumerate(FLEETS):
    print(f'|V|={v}: ' + '  '.join(f't{t}: {label[i][j][0]} {label[i][j][1]}'
                                   for j, t in enumerate(TAUS)))


# ------------------------------------------------- S1 main: three arms
def vali_curve(prefix):
    """(updates, mean, min, max) of vali_score across the completed seeds."""
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


# drawn control arms first, BOLT last so its line stays on top; the legend is
# reordered to the reading order of the caption
ARMS = [
    ('joint-v1', 'Shared reward', GREY_M, '-', 1.0),
    ('m1-bcb-guide-naive', '+ myopic price', ROSE_M, '--', 1.0),
    ('m1-bcb-guide', '+ certified price (BOLT)', BLUE_M, '-', 1.6),
]
LEGEND_ORDER = ['Shared reward', '+ certified price (BOLT)', '+ myopic price']

fig, ax = plt.subplots(figsize=(3.5, 1.62))
fig.subplots_adjust(left=0.135, right=0.965, top=0.965, bottom=0.255)
handles = {}
for prefix, lab, color, ls, lw in ARMS:
    c = vali_curve(prefix)
    if c is None:
        raise SystemExit(f'no completed seeds for {prefix}')
    u, mean, lo, hi = c
    handles[lab] = ax.plot(u, mean, ls, color=color, lw=lw, label=lab)[0]
    ax.fill_between(u, lo, hi, color=color, alpha=0.14, lw=0)
    print(f'{prefix:20s} n_updates={len(u)}  final mean={mean[-1]:.1f}  '
          f'range=[{lo[-1]:.1f}, {hi[-1]:.1f}]')
ax.set_xlabel('PPO update')
ax.set_ylabel('validation makespan')
ax.set_xlim(0, 2000)
ax.legend([handles[k] for k in LEGEND_ORDER], LEGEND_ORDER,
          frameon=False, loc='upper right', fontsize=6.8, labelspacing=0.24,
          handlelength=1.9, borderaxespad=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.savefig('paper/figures/s1_curves_main.pdf')
fig.savefig('paper/figures/s1_curves_main.png', dpi=300)
plt.close(fig)

print('wrote paper/figures/f4b_components.{pdf,png}, '
      's1_curves_main.{pdf,png}')
