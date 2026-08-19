"""Which term of B = max(B_chain, B_mch, B_veh) is active, per regime cell.

The bound of Theorem 1 is a maximum of three relaxations, and the reward, the
credit and the action prices are all built on it. This script reports, for
every cell the paper uses, the three terms at the root state and which one
wins, so the claim that the max is a genuine max rather than one term wearing
three names is answered from the instances themselves.

No solving and no policy: the terms are read off the instance and the env's
own bound implementation, so the table is exact and costs seconds.

Reading the output. The reported grid is vehicle-bound wherever the fleet is
scarce and chain-bound where it is ample; the machine term never wins there,
which is why the machine-contention cells exist. Those cut the factory from
25 stations to 15 and 9 while holding the fleet and the travel intensity
inside the training mixture, and they are machine-bound throughout.

Writes results/bound_terms.json and prints the table.

Usage:
  python scripts/x2_bound_terms.py                 # every cell below
  python scripts/x2_bound_terms.py CELL [CELL ...] # named cells only
"""

import glob
import json
import os
import sys

_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]

import numpy as np

sys.path.insert(0, '.')
from params import configs
configs.device = 'cpu'
from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound

# The primary grid is sampled at its four corners (scarce and ample fleet,
# light and heavy travel), then the production sizes, then the
# machine-contention cells.
DEFAULT_CELLS = [
    '10x25+ppvct-mixed+v1+t0.6', '10x25+ppvct-mixed+v3+t0.6',
    '10x25+ppvct-mixed+v1+t1.0', '10x25+ppvct-mixed+v3+t1.0',
    '30x25+ppvct-mixed+v2+t1.0',
    '50x25+ppvct-mixed+v2+t0.6', '50x25+ppvct-mixed+v2+t1.0',
    '80x25+ppvct-mixed+v3+t1.0',
    '50x25+ppvct-mixed+v2+t0.1', '50x15+ppvct-mixed+v2+t0.1',
    '50x9+ppvct-mixed+v2+t0.1',
    '50x15+ppvct-mixed+v2+t1.0', '50x9+ppvct-mixed+v2+t1.0',
    '80x15+ppvct-mixed+v3+t1.0',
]
NAMES = ['chain', 'mch', 'veh']


def cell_terms(cell):
    """Root-state B_chain, B_mch, B_veh per instance, plus the winner."""
    stems = sorted(g[:-4] for g in
                   glob.glob(f'data/PPVCT/{cell}/test/instance_*.fjs'))
    if not stems:
        stems = sorted(g[:-4] for g in
                       glob.glob(f'data/PPVCT/{cell}/instance_*.fjs'))
    if not stems:
        return None
    stems = stems[:30]
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
    win = np.argmax(terms, axis=0)
    return dict(cell=cell, n=len(stems), n_machines=int(pts[0].shape[1]),
                n_vehicles=int(env.n_veh),
                chain=float(terms[0].mean()), mch=float(terms[1].mean()),
                veh=float(terms[2].mean()),
                active={NAMES[i]: int((win == i).sum()) for i in range(3)})


def main():
    rows = []
    print(f'{"cell":32s} {"M":>3s} {"|V|":>3s} {"B_chain":>9s} {"B_mch":>9s} '
          f'{"B_veh":>9s}   active term (of n)')
    for cell in (_ARGS or DEFAULT_CELLS):
        r = cell_terms(cell)
        if r is None:
            print(f'{cell:32s}  (no instances)')
            continue
        rows.append(r)
        act = ' '.join(f'{k}={v}' for k, v in r['active'].items() if v)
        print(f'{r["cell"]:32s} {r["n_machines"]:3d} {r["n_vehicles"]:3d} '
              f'{r["chain"]:9.1f} {r["mch"]:9.1f} {r["veh"]:9.1f}   {act}')
    os.makedirs('results', exist_ok=True)
    with open('results/bound_terms.json', 'w') as f:
        json.dump(rows, f, indent=1)
    won = {n: sum(r['active'].get(n, 0) for r in rows) for n in NAMES}
    print(f'\ninstances where each term wins the max: '
          + ', '.join(f'{k}={v}' for k, v in won.items()))
    print('wrote results/bound_terms.json')


main()
