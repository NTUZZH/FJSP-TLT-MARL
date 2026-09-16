"""Which component of B = max(B_chain, B_mch, B_veh) is active at the root
state, over the whole primary test grid.

Same computation as scripts/x2_bound_terms.py, swept over the 12 primary-grid
cells (10 modules, 25 stations, fleet |V| in {1,2,3} x travel intensity
tau/p in {0.1,0.3,0.6,1.0}) at the full n=100 test instances per cell, so the
active-bottleneck panel of the manuscript figure is drawn cell-for-cell
against the coupling-regret map.

No solving and no policy: the terms are read off the instance and the env's
own bound implementation.

Writes results/bound_terms_grid.json and prints the table.

Usage:
  python scripts/x2_bound_terms_grid.py
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

NAMES = ['chain', 'mch', 'veh']
FLEETS = (1, 2, 3)
TAUS = (0.1, 0.3, 0.6, 1.0)
CELLS = [f'10x25+ppvct-mixed+v{v}+t{t}' for v in FLEETS for t in TAUS]
TIE_TOL = 1e-9


def cell_terms(cell, limit=None):
    """Root-state B_chain, B_mch, B_veh per instance, plus the winner."""
    stems = sorted(g[:-4] for g in
                   glob.glob(f'data/PPVCT/{cell}/test/instance_*.fjs'))
    if not stems:
        stems = sorted(g[:-4] for g in
                       glob.glob(f'data/PPVCT/{cell}/instance_*.fjs'))
    if not stems:
        return None
    if limit is not None:
        stems = stems[:limit]
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
    top = terms.max(axis=0)
    # strict winner per instance; a tie is any instance where two or more
    # components attain the max within TIE_TOL
    at_max = terms >= (top - TIE_TOL)
    n_at_max = at_max.sum(axis=0)
    tied = n_at_max > 1
    strict = np.argmax(terms, axis=0)
    active = {NAMES[i]: int(((strict == i) & ~tied).sum()) for i in range(3)}
    n = len(stems)
    dom = max(active, key=lambda k: active[k])
    return dict(cell=cell, n=n, n_machines=int(pts[0].shape[1]),
                n_vehicles=int(env.n_veh),
                chain=float(terms[0].mean()), mch=float(terms[1].mean()),
                veh=float(terms[2].mean()),
                active=active, ties=int(tied.sum()),
                dominant=dom, dominant_share=active[dom] / n)


def main():
    rows = []
    print(f'{"cell":32s} {"M":>3s} {"|V|":>3s} {"n":>4s} {"B_chain":>9s} '
          f'{"B_mch":>9s} {"B_veh":>9s}   active term (of n)')
    for cell in CELLS:
        r = cell_terms(cell)
        if r is None:
            print(f'{cell:32s}  (no instances)')
            continue
        rows.append(r)
        act = ' '.join(f'{k}={v}' for k, v in r['active'].items() if v)
        if r['ties']:
            act += f' tie={r["ties"]}'
        print(f'{r["cell"]:32s} {r["n_machines"]:3d} {r["n_vehicles"]:3d} '
              f'{r["n"]:4d} {r["chain"]:9.1f} {r["mch"]:9.1f} '
              f'{r["veh"]:9.1f}   {act}')
    os.makedirs('results', exist_ok=True)
    with open('results/bound_terms_grid.json', 'w') as f:
        json.dump(rows, f, indent=1)
    won = {n: sum(r['active'].get(n, 0) for r in rows) for n in NAMES}
    print('\ninstances where each term strictly wins the max: '
          + ', '.join(f'{k}={v}' for k, v in won.items())
          + f', ties={sum(r["ties"] for r in rows)}')
    print('wrote results/bound_terms_grid.json')


main()
