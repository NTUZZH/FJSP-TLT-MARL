"""Minimum fleet utilization per regime cell (B1 regime variable).

At the root state veh_free == 0, so B_veh(s0) = W_tr / |V| where W_tr is the
total minimal LOADED move work the instance still forces.  For any feasible
schedule of makespan C the fleet's loaded busy fraction is at least
    W_tr / (|V| * C) = B_veh(s0) / C,
and since C* <= C for the best schedule we know, using the CP-SAT incumbent
gives a valid (conservative) lower bound on the loaded utilization any
schedule must reach.  Empty legs are excluded, so the true utilization is
higher.

Writes results/fleet_util.json and prints a per-cell table.
"""
import glob
import json
import os
import sys

sys.argv = [sys.argv[0]]
import numpy as np

sys.path.insert(0, '.')
from params import configs
configs.device = 'cpu'
from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound


def root_bveh(stems):
    """B_veh(s0) and |V| for a batch of same-shape instances."""
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
    return b._b_veh(), env.n_veh


def main():
    dirs = sorted(glob.glob('data/PPVCT/1[05]x25+ppvct-mixed+v*+t*/test'))
    out = {}
    for ds in dirs:
        cell = ds.split('/')[2]
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        if not stems:
            continue
        bveh, nveh = {}, None
        for i0 in range(0, len(stems), 20):
            chunk = stems[i0:i0 + 20]
            b, nv = root_bveh(chunk)
            nveh = int(nv)
            for s, v in zip(chunk, b):
                bveh[os.path.basename(s)] = float(v)
        # CP-SAT incumbent as the reference makespan
        cp = f'results/cpsat_v2/{cell}.jsonl'
        util = []
        if os.path.exists(cp):
            for ln in open(cp):
                r = json.loads(ln)
                b = bveh.get(r['instance'])
                if b is not None and r['ub'] > 0:
                    util.append(b / r['ub'])
        out[cell] = {
            'n_veh': nveh,
            'n': len(util),
            'W_tr_mean': float(np.mean(list(bveh.values())) * nveh),
            'util_mean': float(np.mean(util)) if util else None,
            'util_min': float(np.min(util)) if util else None,
            'util_max': float(np.max(util)) if util else None,
        }
        u = out[cell]
        print('%-34s |V|=%d  n=%3d  loaded-util mean=%.3f  [%.3f, %.3f]'
              % (cell, u['n_veh'], u['n'], u['util_mean'] or -1,
                 u['util_min'] or -1, u['util_max'] or -1), flush=True)
    with open('results/fleet_util.json', 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
