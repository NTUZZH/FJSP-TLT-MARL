"""Gate evaluation: paired per-instance comparison of two checkpoints on a
set of cells, with Wilcoxon + Holm correction (proposal §9 gate protocol).

Reuses eval_ppvct greedy rollout. Both arms evaluated on the SAME instances
(paired). Writes notes/<gate>.md with honest numbers and the verdict.

Usage (G1):
  python -u scripts/gate_eval.py --gate G1 \
     --explicit 10x25+ppvct-mixed+joint-v1-s301 \
     --baseline 10x25+ppvct-mixed+e0b-uncontended-s301 --baseline_veh_rule NVF \
     --cells v1+t0.6,v2+t0.6 --alpha 0.05
"""

import os
import sys, os, json, argparse, glob

cli = argparse.ArgumentParser()
cli.add_argument('--gate', type=str, required=True)
cli.add_argument('--explicit', type=str, required=True)
cli.add_argument('--baseline', type=str, required=True)
cli.add_argument('--explicit_veh_rule', type=str, default='policy')
cli.add_argument('--baseline_veh_rule', type=str, default='NVF')
cli.add_argument('--cells', type=str, required=True)
cli.add_argument('--split', type=str, default='test')
cli.add_argument('--alpha', type=float, default=0.05)
cli.add_argument('--batch', type=int, default=20)
cli.add_argument('--direction', type=str, default='explicit_lower',
                 help='explicit_lower = explicit should have SMALLER makespan')
args_cli = cli.parse_args()
sys.argv = [sys.argv[0]]

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, '.')


def run_eval(model_name, veh_rule, cells, split, batch):
    """Invoke eval_ppvct.py as a subprocess (clean arch load per model),
    then read the .npy results it writes."""
    import subprocess
    tag = 'greedy' if veh_rule == 'policy' else f'greedy-{veh_rule}'
    cmd = ['python', '-u', 'scripts/eval_ppvct.py', '--model_name', model_name,
           '--cells', ','.join(cells), '--veh_rule', veh_rule,
           '--split', split, '--batch', str(batch)]
    env = dict(os.environ)
    print(f'  running: {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, check=True, env=env)
    out = {}
    for cell in cells:
        p = f'test_results/PPVCT/{cell}/Result_{tag}+{model_name}_{cell}.npy'
        arr = np.load(p)
        out[cell] = arr[:, 0]  # makespans in instance order
    return out


def main():
    cells = args_cli.cells.split(',')
    exp = run_eval(args_cli.explicit, args_cli.explicit_veh_rule, cells,
                   args_cli.split, args_cli.batch)
    bas = run_eval(args_cli.baseline, args_cli.baseline_veh_rule, cells,
                   args_cli.split, args_cli.batch)

    rows = []
    pvals = []
    for cell in cells:
        e, b = exp[cell], bas[cell]
        assert len(e) == len(b), f'{cell}: paired length mismatch'
        diff = b - e  # positive = explicit better (lower makespan)
        # one-sided Wilcoxon: explicit < baseline
        try:
            stat, p = wilcoxon(e, b, alternative='less')
        except ValueError:
            p = 1.0  # all-equal
        wins = int((e < b - 1e-9).sum())
        ties = int((np.abs(e - b) <= 1e-9).sum())
        losses = int((e > b + 1e-9).sum())
        rows.append(dict(cell=cell, n=len(e),
                         explicit_mean=float(e.mean()),
                         baseline_mean=float(b.mean()),
                         mean_improvement_pct=float(100 * diff.mean() / b.mean()),
                         wtl=f'{wins}/{ties}/{losses}',
                         p_raw=float(p)))
        pvals.append(p)

    # Holm correction across cells
    order = np.argsort(pvals)
    m = len(pvals)
    holm = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, adj)
        holm[idx] = running_max
    for i, row in enumerate(rows):
        row['p_holm'] = float(holm[i])
        row['significant'] = bool(holm[i] < args_cli.alpha and
                                  row['mean_improvement_pct'] > 0)

    all_sig = all(r['significant'] for r in rows)
    verdict = 'PASS' if all_sig else 'FAIL'

    md = [f'# Gate {args_cli.gate} evaluation ({args_cli.split})\n',
          f'- explicit arm: `{args_cli.explicit}` (veh={args_cli.explicit_veh_rule})',
          f'- baseline arm: `{args_cli.baseline}` (veh={args_cli.baseline_veh_rule})',
          f'- alpha={args_cli.alpha}, Holm-corrected across {len(cells)} cells',
          f'- paired one-sided Wilcoxon (explicit < baseline), by instance\n',
          '| cell | n | explicit | baseline | improv% | W/T/L | p_raw | p_holm | sig |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['cell']} | {r['n']} | {r['explicit_mean']:.2f} | "
                  f"{r['baseline_mean']:.2f} | {r['mean_improvement_pct']:+.2f}% | "
                  f"{r['wtl']} | {r['p_raw']:.2e} | {r['p_holm']:.2e} | "
                  f"{'YES' if r['significant'] else 'no'} |")
    md.append(f'\n## Verdict: {verdict}')
    md.append(f'(Gate requires explicit to win significantly on ALL '
              f'pre-specified cells.)\n')
    md.append('```json\n' + json.dumps(rows, indent=1) + '\n```')

    out_path = f'notes/gate_{args_cli.gate}.md'
    os.makedirs('notes', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(md))
    print('\n'.join(md))
    print(f'\nwrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
