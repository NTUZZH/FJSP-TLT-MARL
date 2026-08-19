"""Parse train_*.jsonl -> monitoring PNG + latest vali table (any model).

Usage: python scripts/monitor_training.py [model_name ...]
Defaults to every train_log/PPVCT/train_*.log.jsonl found.
Output: results/monitor/{model}.png + console summary.
"""

import sys, os, json, glob

sys.argv, argv = [sys.argv[0]], sys.argv[1:]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    os.makedirs('results/monitor', exist_ok=True)
    paths = [f'train_log/PPVCT/train_{m}.log.jsonl' for m in argv] if argv \
        else sorted(glob.glob('train_log/PPVCT/train_*.log.jsonl'))
    for path in paths:
        model = os.path.basename(path)[6:-10]
        recs = [json.loads(l) for l in open(path) if l.strip()]
        if not recs:
            continue
        cells = sorted({tuple(r['cell']) for r in recs})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for c in cells:
            xs = [r['update'] for r in recs if tuple(r['cell']) == c]
            ys = [r['train_ms'] for r in recs if tuple(r['cell']) == c]
            axes[0].plot(xs, ys, marker='.', lw=0.8, ms=2,
                         label=f'V{c[0]} t{c[1]}')
        axes[0].set_title(f'{model}: train makespan per cell')
        axes[0].set_xlabel('update'); axes[0].legend(fontsize=6)
        vx = [r['update'] for r in recs if 'vali_score' in r]
        vy = [r['vali_score'] for r in recs if 'vali_score' in r]
        axes[1].plot(vx, vy, marker='o', ms=3)
        axes[1].set_title('vali score (mean of cell means)')
        axes[1].set_xlabel('update')
        fig.tight_layout()
        out = f'results/monitor/{model}.png'
        fig.savefig(out, dpi=120); plt.close(fig)
        last_v = next((r for r in reversed(recs) if 'vali' in r), None)
        print(f'{model}: {len(recs)} updates, latest={recs[-1]["update"]}')
        if last_v:
            print(f'  vali@{last_v["update"]}: score={last_v["vali_score"]:.2f}')
            for k, v in sorted(last_v['vali'].items()):
                print(f'    {k}: {v:.1f}')
        print(f'  -> {out}')


if __name__ == '__main__':
    main()
