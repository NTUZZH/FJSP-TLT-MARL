"""F3 credit-assignment money figure (proposal §8 E2).

Panel A: vali-score learning curves of the four arms (shared / M1-BCB /
COMA-critic / M2-shaped) from their training jsonls, x = update.
Panel B: final gate-region quality bars (scarce cells mean) once evals land.
Panel C: lazy-agent diagnostics (vehicle-busy Gini, idle rate) once
diagnostics land (results/diagnostics/*.json).

Panels render only from data that EXISTS; missing arms are listed in the
console (never fabricated). Publication size: full column pair 7.16in.

Usage: python scripts/f3_credit_figure.py [--out results/figures/f3_credit.pdf]
"""

import sys, os, json, glob, argparse
sys.argv, _argv = [sys.argv[0]], sys.argv[1:]

cli = argparse.ArgumentParser()
cli.add_argument('--out', default='results/figures/f3_credit.pdf')
args = cli.parse_args(_argv)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS = [
    ('shared', 'joint-v1-s301', 'tab:gray'),
    ('M1-BCB', 'm1-bcb-s301', 'tab:blue'),
    ('COMA', 'coma-critic-s301', 'tab:orange'),
    ('M2', 'm2-shaped-s301', 'tab:green'),
]


def curve(suffix):
    p = f'train_log/PPVCT/train_10x25+ppvct-mixed+{suffix}.log.jsonl'
    if not os.path.exists(p):
        return None
    xs, ys = [], []
    for l in open(p):
        r = json.loads(l)
        if 'vali_score' in r:
            xs.append(r['update']); ys.append(r['vali_score'])
    return (np.array(xs), np.array(ys)) if xs else None


def diagnostics(suffix):
    p = f'results/diagnostics/{suffix}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))

    missing = []
    for name, suf, color in ARMS:
        c = curve(suf)
        if c is None:
            missing.append(name); continue
        axes[0].plot(c[0], c[1], color=color, lw=1.0, label=name)
    axes[0].set_xlabel('PPO update', fontsize=7)
    axes[0].set_ylabel('validation score (mean cell makespan)', fontsize=7)
    axes[0].legend(fontsize=6, frameon=False)
    axes[0].set_title('(a) learning curves', fontsize=8)

    # Panel B: gate-region test means per arm (from test_results npy when run)
    labels, means, errs, cols = [], [], [], []
    for name, suf, color in ARMS:
        vals = []
        for cell in ('v1+t0.6', 'v2+t0.6'):
            p = f'test_results/PPVCT/{cell}/Result_greedy+10x25+ppvct-mixed+{suf}_{cell}.npy'
            if os.path.exists(p):
                vals.append(np.load(p)[:, 0])
        if vals:
            v = np.concatenate(vals)
            labels.append(name); means.append(v.mean())
            errs.append(v.std(ddof=1) / np.sqrt(len(v))); cols.append(color)
    if labels:
        axes[1].bar(labels, means, yerr=errs, color=cols, width=0.6)
        axes[1].set_ylabel('scarce-cell makespan', fontsize=7)
    axes[1].set_title('(b) scarce-regime quality', fontsize=8)

    # Panel C: diagnostics (vehicle Gini / idle) if present
    got = False
    for i, (name, suf, color) in enumerate(ARMS):
        d = diagnostics(suf)
        if d:
            got = True
            axes[2].scatter(d.get('veh_gini'), d.get('veh_idle_mean'),
                            color=color, label=name, s=18)
    if got:
        axes[2].set_xlabel('vehicle contribution Gini', fontsize=7)
        axes[2].set_ylabel('mean vehicle idle rate', fontsize=7)
        axes[2].legend(fontsize=6, frameon=False)
    axes[2].set_title('(c) lazy-agent diagnostics', fontsize=8)

    for ax in axes:
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}; missing arms: {missing or "none"}')


if __name__ == '__main__':
    main()
