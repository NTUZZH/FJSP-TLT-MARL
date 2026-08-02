"""E3 coupling-regret heatmap (proposal §8 E3, F4 second money figure).

Regret of a penalty-trained (uncontended, NVF-deployed) policy relative to
the explicit-coupling policy, across the (fleet |V|, travel intensity
tau/p) grid, in the TRUE scarce env:
    regret(cell) = (ms_penalty - ms_explicit) / ms_explicit   (per-instance
    paired mean).

Reads eval_ppvct .npy outputs (run gate_eval / eval_ppvct first for both
models on all 9 cells). Produces a publication PDF at the placed size with
per-cell mean regret % and significance stars.

Usage:
  python scripts/e3_regret_map.py \
    --explicit 10x25+ppvct-mixed+joint-v1-s301 \
    --penalty 10x25+ppvct-mixed+e0b-uncontended-s301 --penalty_veh_rule NVF
"""

import sys, os, argparse, glob
sys.argv, _argv = [sys.argv[0]], sys.argv[1:]

cli = argparse.ArgumentParser()
cli.add_argument('--explicit', required=True)
cli.add_argument('--penalty', required=True)
cli.add_argument('--explicit_veh_rule', default='policy')
cli.add_argument('--penalty_veh_rule', default='NVF')
cli.add_argument('--out', default='results/figures/f4_regret_map.pdf')
args = cli.parse_args(_argv)

import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FLEETS = [1, 2, 3]
RATIOS = [0.1, 0.3, 0.6, 1.0]


def load(model, veh_rule, cell):
    tag = 'greedy' if veh_rule == 'policy' else f'greedy-{veh_rule}'
    p = f'test_results/PPVCT/{cell}/Result_{tag}+{model}_{cell}.npy'
    if not os.path.exists(p):
        return None
    return np.load(p)[:, 0]


def main():
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    reg = np.full((len(FLEETS), len(RATIOS)), np.nan)
    star = [['' for _ in RATIOS] for _ in FLEETS]
    missing = []
    for i, V in enumerate(FLEETS):
        for j, r in enumerate(RATIOS):
            cell = f'v{V}+t{r}'
            e = load(args.explicit, args.explicit_veh_rule, cell)
            p = load(args.penalty, args.penalty_veh_rule, cell)
            if e is None or p is None:
                missing.append(cell); continue
            rr = (p - e) / e
            reg[i, j] = 100 * rr.mean()
            try:
                _, pv = wilcoxon(e, p, alternative='less')
            except ValueError:
                pv = 1.0
            star[i][j] = '***' if pv < 1e-3 else '**' if pv < 1e-2 else '*' if pv < 5e-2 else ''
    if missing:
        print('MISSING cells (run eval first):', missing)

    # publication figure: single column ~3.4in wide
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    vmax = np.nanmax(np.abs(reg)) if np.isfinite(reg).any() else 1
    im = ax.imshow(reg, cmap='RdYlGn_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(RATIOS))); ax.set_xticklabels([f'{r}' for r in RATIOS])
    ax.set_yticks(range(len(FLEETS))); ax.set_yticklabels([f'{v}' for v in FLEETS])
    ax.set_xlabel(r'travel intensity $\bar\tau/\bar p$', fontsize=8)
    ax.set_ylabel(r'fleet size $|V|$', fontsize=8)
    ax.tick_params(labelsize=7)
    for i in range(len(FLEETS)):
        for j in range(len(RATIOS)):
            if np.isfinite(reg[i, j]):
                ax.text(j, i, f'{reg[i,j]:+.1f}\n{star[i][j]}', ha='center',
                        va='center', fontsize=7,
                        color='black' if abs(reg[i, j]) < 0.6 * vmax else 'white')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('penalty-model regret (\\%)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}')
    print('regret matrix (rows=|V| 1/2/3, cols=tau/p 0.1/0.3/0.6):')
    print(np.array_str(reg, precision=2))


if __name__ == '__main__':
    main()
