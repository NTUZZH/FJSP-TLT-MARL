"""root bound and its active component on the heterogeneity cells.

Per instance at the root state s0: the three terms of Theorem 1
(B_chain, B_mch, B_veh), the bound B(s0) = max of the three, and which term
attains the maximum. Read straight off the instance and the environment's own
bound implementation, exactly as scripts/x2_bound_terms.py and
scripts/p8_certificate.py do; no solver and no policy.

Tightness B(s0)/C_best is assembled later by scripts/x2_hetero_report.py,
with C_best the per-instance minimum over every arm actually evaluated.

Writes results/hetero/bound/{cell}+R{level}.json.

Usage: python scripts/x2_hetero_bound.py --cells v1+t0.6 --levels base,2,5,10,20
"""

import argparse
import json
import os
import sys

cli = argparse.ArgumentParser()
cli.add_argument('--cells', type=str, default='v1+t0.6,v2+t0.6,v1+t1.0')
cli.add_argument('--levels', type=str, default='base,2,5,10,20')
cli.add_argument('--split', type=str, default='test')
cli.add_argument('--batch', type=int, default=20)
A = cli.parse_args()
sys.argv = [sys.argv[0]]

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import glob
import numpy as np

sys.path.insert(0, '.')
from params import configs
configs.device = 'cpu'
from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound

NAMES = ['chain', 'mch', 'veh']


def dataset_dir(cell, level, split):
    base = cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'
    if level == 'base':
        return f'data/PPVCT/{base}/{split}'
    return f'data/PPVCT_HET/{base}+hetR{level}/{split}'


def root_terms(stems):
    jls, pts, lags, opt, mct, lay = [], [], [], [], [], []
    for s in stems:
        jl, pt, meta = load_instance(s)
        jls.append(np.asarray(jl))
        pts.append(np.asarray(pt, dtype=float))
        lags.append(np.asarray(meta['time_lag'], dtype=float))
        opt.append(np.asarray(meta['op_type']))
        mct.append(np.asarray(meta['mch_type']))
        lay.append(meta['transport'])
    env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1], use_lag_features=True)
    env.set_initial_data(jls, pts, lags, opt, mct, lay)
    b = TransportBound(env, use_mch=True, use_veh=True)
    terms = np.stack([np.max(env.op_ct_lb, axis=1), b._b_mch(), b._b_veh()])
    return terms, b.value()


def main():
    os.makedirs('results/hetero/bound', exist_ok=True)
    for cell in A.cells.split(','):
        for level in A.levels.split(','):
            out_path = f'results/hetero/bound/{cell}+R{level}.json'
            if os.path.exists(out_path):
                print(f'skip {cell} R={level} (exists)', flush=True)
                continue
            ds = dataset_dir(cell, level, A.split)
            stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
            assert stems, f'no instances under {ds}'
            per = {}
            for i0 in range(0, len(stems), A.batch):
                chunk = stems[i0:i0 + A.batch]
                terms, B = root_terms(chunk)
                win = np.argmax(terms, axis=0)
                for e, s in enumerate(chunk):
                    per[os.path.basename(s)] = dict(
                        B=float(B[e]), chain=float(terms[0, e]),
                        mch=float(terms[1, e]), veh=float(terms[2, e]),
                        active=NAMES[int(win[e])])
            act = {n: sum(1 for v in per.values() if v['active'] == n)
                   for n in NAMES}
            out = dict(cell=cell, level=level, dataset=ds, n=len(per),
                       active_counts=act,
                       mean_B=float(np.mean([v['B'] for v in per.values()])),
                       mean_chain=float(np.mean([v['chain'] for v in per.values()])),
                       mean_mch=float(np.mean([v['mch'] for v in per.values()])),
                       mean_veh=float(np.mean([v['veh'] for v in per.values()])),
                       per_instance=per)
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=1)
            print(f'{cell} R={level:>4s}: n={out["n"]} B={out["mean_B"]:.2f} '
                  f'chain={out["mean_chain"]:.2f} mch={out["mean_mch"]:.2f} '
                  f'veh={out["mean_veh"]:.2f}  active '
                  + ' '.join(f'{k}={v}' for k, v in act.items() if v),
                  flush=True)


if __name__ == '__main__':
    main()
