"""Queue gate for the size-mixture calibration smoke (Paper X2 arm e).

A 20-update smoke run is required before
the 2000-update mixture run, because the arm's 7-12 GPU-h budget rests on a
predicted 19-35 s/update at 20 modules that has never been measured. This
script turns that smoke into a queue gate: it runs as its own line between the
smoke and the full arm, and a non-zero exit stops the queue before 10 GPU-h are
spent on a configuration that is broken or four times slower than budgeted.

Three assertions:
  1. all three training sizes actually occurred (the round-robin schedule is
     live, not silently pinned to one size);
  2. every recorded loss and value loss is finite (no NaN/inf from the larger
     instances);
  3. no 20-module update took longer than --max_s wall seconds. Updates that
     carry a validation pass are excluded from the timing, since their wall
     delta also contains the full size-mixed validation sweep.

Usage (queue line):
  python -u scripts/x2_gate_mix_smoke.py
"""

import argparse
import json
import os
import re
import sys

cli = argparse.ArgumentParser()
cli.add_argument('--model_name', type=str,
                 default='mix10-15-20x25+ppvct-mixed+smoke-mix-s999')
cli.add_argument('--max_s', type=float, default=40.0,
                 help='wall-clock ceiling for one 20-module update; the design '
                      'bracket is 19-35 s uncontended (section 4.6)')
cli.add_argument('--check_size', type=int, default=20,
                 help='module count whose per-update time is gated')
cli.add_argument('--min_updates', type=int, default=20)
args = cli.parse_args()

LOG = f'train_log/PPVCT/train_{args.model_name}.log.jsonl'
CFG = f'train_log/PPVCT/config_{args.model_name}.json'


def die(msg):
    print(f'GATE FAIL [mix-smoke]: {msg}', file=sys.stderr)
    raise SystemExit(1)


def expected_sizes(model_name):
    m = re.match(r'^mix([0-9-]+)x\d+\+', model_name)
    if not m:
        die(f"cannot read the size mixture out of model name '{model_name}'; "
            f"expected a 'mix10-15-20x25+...' prefix")
    return [int(x) for x in m.group(1).split('-')]


def main():
    if not os.path.exists(LOG):
        die(f'{LOG} does not exist. The smoke line\n'
            f'  python -u scripts/p2_train_mappo.py --model_suffix smoke-mix '
            f'--credit m1 --guide --seed 999 --max_updates 20 '
            f'--size_mix 10,15,20 --vali_size_mix 10,20 --vali_every 20\n'
            f'must run and finish before this gate.')
    recs = []
    with open(LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if len(recs) < args.min_updates:
        die(f'{LOG} has {len(recs)} updates, expected at least '
            f'{args.min_updates}: the smoke did not finish.')

    sizes = expected_sizes(args.model_name)
    n_cells = 9
    if os.path.exists(CFG):
        snap = json.load(open(CFG))
        fg, rg = snap.get('fleet_grid'), snap.get('ratio_grid')
        if fg and rg:
            n_cells = len(fg) * len(rg)
        if snap.get('size_mix') and list(snap['size_mix']) != sizes:
            die(f"config snapshot size_mix {snap['size_mix']} does not match "
                f'the model name mixture {sizes}')

    def size_of(rec):
        # the trainer records n_modules on every size-mixed update; older logs
        # are covered by the same round-robin arithmetic the trainer uses
        if 'n_modules' in rec:
            return int(rec['n_modules'])
        return sizes[(int(rec['update']) // n_cells) % len(sizes)]

    # ---- 1. every training size occurred -------------------------------
    seen = sorted({size_of(r) for r in recs})
    if seen != sorted(sizes):
        die(f'training sizes seen = {seen}, expected {sorted(sizes)}. The '
            f'round-robin over --size_mix did not reach every size in '
            f'{len(recs)} updates.')

    # ---- 2. losses finite ----------------------------------------------
    import math
    bad = [r['update'] for r in recs
           if not (math.isfinite(r.get('loss', float('nan')))
                   and math.isfinite(r.get('vloss', float('nan')))
                   and math.isfinite(r.get('train_ms', float('nan'))))]
    if bad:
        die(f'non-finite loss/vloss/train_ms at updates {bad[:8]} '
            f'({len(bad)} of {len(recs)}).')

    # ---- 3. per-update wall time at the gated size ----------------------
    prev = 0.0
    rows = []
    for r in recs:
        d = float(r['wall']) - prev
        prev = float(r['wall'])
        rows.append((int(r['update']), size_of(r), d, 'vali' in r))
    per_size = {}
    for _, sz, d, has_v in rows:
        if not has_v:
            per_size.setdefault(sz, []).append(d)
    print(f'[mix-smoke] {len(recs)} updates, sizes {seen}, '
          f'{n_cells} regime cells')
    print(f'{"size":>6}{"n (no vali)":>13}{"min s":>10}{"median s":>11}{"max s":>10}')
    for sz in sorted(per_size):
        v = sorted(per_size[sz])
        med = v[len(v) // 2]
        print(f'{sz:>6}{len(v):>13}{v[0]:>10.1f}{med:>11.1f}{v[-1]:>10.1f}')
    skipped = [(u, sz, round(d, 1)) for u, sz, d, hv in rows if hv]
    if skipped:
        print(f'[mix-smoke] excluded from timing (validation pass bundled in): '
              f'{skipped}')

    gated = per_size.get(args.check_size, [])
    if not gated:
        die(f'no {args.check_size}-module update without a validation pass in '
            f'{LOG}; the timing assertion could not be evaluated. Lengthen the '
            f'smoke (--max_updates) or move --vali_every off the last update.')
    worst = max(gated)
    if worst > args.max_s:
        die(f'{args.check_size}-module updates ran at up to {worst:.1f} s '
            f'(ceiling {args.max_s:.0f} s; design bracket 19-35 s '
            f'uncontended). Either the box is contended or the arm is more '
            f'expensive than budgeted: do NOT start the 2000-update run.')
    print(f'[mix-smoke] {args.check_size}-module worst update {worst:.1f} s '
          f'<= {args.max_s:.0f} s ceiling')
    print('GATE PASS [mix-smoke]: sizes complete, losses finite, '
          f'{args.check_size}-module updates within budget.')


if __name__ == '__main__':
    main()
