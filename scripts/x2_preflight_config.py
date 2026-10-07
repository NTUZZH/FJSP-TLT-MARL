"""Pre-launch check of a training run against the run it will be compared to.

Two modes, both exit non-zero on any failure so a launcher can abort:

  dry   NEW is the record written by `p2_train_mappo.py --dry_run PATH`
        (resolved snapshot + CLI + parameter count + state-dict shapes).
        Checks: (i) the snapshot differs from the comparator's config_*.json
        only in the keys listed by --allow; (ii) every allowed key that does
        differ has the value given by --expect; (iii) the parameter count and
        every state-dict shape equal those of the comparator checkpoint;
        (iv) the CLI-only settings that the snapshot does not record
        (vali_subset, vali_every) equal --cli_expect.
        The dry run hides the GPU (CUDA_VISIBLE_DEVICES=''), so `device` reads
        'cpu' there; it is excused in this mode only and re-checked in `post`.
  post  NEW is the config_*.json the real run wrote in its first seconds.
        Same key diff with `device` NOT excused.

Usage:
  python scripts/x2_preflight_config.py dry  NEW.json COMPARATOR_CONFIG.json \
      --ckpt COMPARATOR.pth --allow bound_veh,model_name,model_suffix,seed_train \
      --expect bound_veh=0,model_name=NAME --cli_expect vali_subset=20,vali_every=20 \
      --threads 2
  python scripts/x2_preflight_config.py post NEW_CONFIG.json COMPARATOR_CONFIG.json \
      --allow ... --expect ...
"""
import argparse
import json
import sys

MISSING = '<absent>'


def parse_kv(s):
    out = {}
    for item in [x for x in s.split(',') if x.strip()]:
        k, v = item.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def norm(v):
    """Compare values as the snapshot writer stores them (json, default=str)."""
    return json.loads(json.dumps(v, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dry', 'post'])
    ap.add_argument('new')
    ap.add_argument('comparator')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--allow', default='')
    ap.add_argument('--expect', default='')
    ap.add_argument('--cli_expect', default='')
    ap.add_argument('--threads', type=int, default=0,
                    help='dry mode: required torch intra-op thread count')
    a = ap.parse_args()

    rec = json.load(open(a.new))
    new = rec['config'] if a.mode == 'dry' else rec
    old = json.load(open(a.comparator))
    allow = {k for k in a.allow.split(',') if k}
    if a.mode == 'dry':
        allow.add('device')       # GPU hidden in the dry run; checked in post
    expect = parse_kv(a.expect)
    fails = []

    print(f'--- config diff: {a.new} vs {a.comparator}')
    keys = sorted(set(new) | set(old))
    n_diff = 0
    for k in keys:
        nv, ov = norm(new.get(k, MISSING)), norm(old.get(k, MISSING))
        if nv == ov:
            continue
        n_diff += 1
        ok = k in allow
        tag = 'allowed' if ok else 'NOT ALLOWED'
        if k == 'device' and a.mode == 'dry':
            tag = 'excused in dry run (GPU hidden); strict in post'
        print(f'  {k}: comparator={ov!r} new={nv!r}  [{tag}]')
        if not ok:
            fails.append(f'unexpected difference in {k}')
        elif k in expect and str(nv) != expect[k]:
            fails.append(f'{k}={nv!r}, expected {expect[k]!r}')
    if n_diff == 0:
        print('  (identical)')
    for k, v in expect.items():
        if str(norm(new.get(k, MISSING))) != v:
            if not any(f.startswith(k) for f in fails):
                fails.append(f'{k}={new.get(k, MISSING)!r}, expected {v!r}')

    if a.mode == 'dry':
        cli_exp = parse_kv(a.cli_expect)
        for k, v in cli_exp.items():
            got = str(rec['cli'].get(k, MISSING))
            print(f'--- cli-only {k}={got} (expected {v})')
            if got != v:
                fails.append(f'cli {k}={got}, expected {v}')
        if a.ckpt:
            import torch
            sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
            ck_shapes = {k: list(v.shape) for k, v in sd.items()}
            ck_numel = sum(v.numel() for v in sd.values())
            new_numel = 0
            for s in rec['state_shapes'].values():
                n = 1
                for d in s:
                    n *= d
                new_numel += n
            print(f'--- parameters: new run {rec["n_params"]} (state dict '
                  f'{new_numel} values, {len(rec["state_shapes"])} tensors); '
                  f'comparator checkpoint {ck_numel} values, '
                  f'{len(ck_shapes)} tensors')
            if ck_shapes != rec['state_shapes']:
                diff = sorted(k for k in set(ck_shapes) | set(rec['state_shapes'])
                              if ck_shapes.get(k) != rec['state_shapes'].get(k))
                fails.append(f'state-dict shapes differ on {diff[:5]}')
            if ck_numel != new_numel:
                fails.append(f'parameter count {new_numel} != comparator {ck_numel}')
        print(f'--- torch threads in the dry run: {rec["torch_threads"]}')
        if a.threads and rec['torch_threads'] != a.threads:
            fails.append(f'torch threads {rec["torch_threads"]} != {a.threads}')

    if fails:
        print('PREFLIGHT FAIL: ' + '; '.join(fails))
        sys.exit(1)
    print('PREFLIGHT OK')


if __name__ == '__main__':
    main()
