"""Root certificates B(s0) for the scale-up pilot cells (Phase A).

B0 = B(s_root) needs no solver, and Theorem 1 admissibility at the root gives
B0 <= C* <= C for every feasible schedule C, so (C - B0) / B0 is a valid
optimality certificate for any schedule the pilot produces.

Same construction as scripts/p8_certificate.py (FJSPEnvTransport +
TransportBound with the machine and vehicle terms both on, read off
env.max_endTime at the root); it writes to results/scaleup/certificate/ so the
manuscript's results/certificate/ is untouched. As a self-check it verifies
admissibility against every arm evaluated in this pilot: B0 must not exceed
any PDR, GA, or policy makespan on the same instance.

Usage: python scripts/x2_scale_cert.py [--cells CELL[,CELL...]]
"""

import argparse
import glob
import json
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument('--cells', type=str, default='')
_ap.add_argument('--chunk', type=int, default=10)
_A = _ap.parse_args()

sys.argv = [sys.argv[0]]

import numpy as np

sys.path.insert(0, '.')
from params import configs
configs.device = 'cpu'
from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound

CELLS = ['20x25+ppvct-mixed+v1+t0.6', '20x25+ppvct-mixed+v1+t1.0',
         '30x25+ppvct-mixed+v1+t0.6', '30x25+ppvct-mixed+v2+t1.0',
         '50x25+ppvct-mixed+v2+t0.6', '50x25+ppvct-mixed+v2+t1.0',
         '80x25+ppvct-mixed+v3+t1.0']


def root_bounds(stems):
    jls, pts, lags, opt, mct, lay = [], [], [], [], [], []
    for s in stems:
        jl, pt, meta = load_instance(s)
        jls.append(np.asarray(jl)); pts.append(np.asarray(pt, dtype=float))
        lags.append(np.asarray(meta['time_lag'], dtype=float))
        opt.append(np.asarray(meta['op_type']))
        mct.append(np.asarray(meta['mch_type']))
        lay.append(meta['transport'])
    env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1], use_lag_features=True)
    env.set_initial_data(jls, pts, lags, opt, mct, lay)
    env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
    return env.max_endTime.copy()


def main():
    os.makedirs('results/scaleup/certificate', exist_ok=True)
    cells = [c for c in _A.cells.split(',') if c.strip()] or CELLS
    for cell in cells:
        stems = sorted(g[:-4] for g in
                       glob.glob(f'data/PPVCT/{cell}/test/instance_*.fjs'))
        assert stems, f'no instances in data/PPVCT/{cell}/test'
        out = {}
        for i0 in range(0, len(stems), _A.chunk):
            chunk = stems[i0:i0 + _A.chunk]
            for s, v in zip(chunk, root_bounds(chunk)):
                out[os.path.basename(s)] = float(v)
        path = f'results/scaleup/certificate/{cell}.json'
        with open(path + '.tmp', 'w') as f:
            json.dump(out, f, indent=0)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path + '.tmp', path)

        # admissibility against every arm evaluated on this cell: all nine PDR
        # pairs, every GA budget, and every policy seed. B0 must not exceed a
        # single feasible makespan.
        viol, n_checked = [], 0
        pdr_p = f'results/scaleup/pdr/{cell}.json'
        if os.path.exists(pdr_p):
            pdr = json.load(open(pdr_p))
            for n, d in pdr.items():
                for pair, ms in d.items():
                    n_checked += 1
                    if out[n] > ms + 1e-6:
                        viol.append((f'pdr:{pair}', n))
        for ga_p in ([f'results/scaleup/ga/{cell}.json']
                     + sorted(glob.glob(f'results/scaleup/ga_long/*/{cell}.json'))
                     + sorted(glob.glob(f'results/scaleup/ga_budget/*/{cell}.json'))):
            if not os.path.exists(ga_p):
                continue
            gaj = json.load(open(ga_p))
            tag = os.path.relpath(ga_p, 'results/scaleup')
            for n, d in gaj.items():
                n_checked += 1
                if out[n] > d['ga'] + 1e-6:
                    viol.append((tag, n))
        for pp in sorted(glob.glob(f'results/scaleup/policy/*_{cell}.json')):
            pol = json.load(open(pp))['rows']
            for n, d in pol.items():
                n_checked += 1
                if out[n] > d['ms'] + 1e-6:
                    viol.append((os.path.basename(pp), n))
        for cp in (f'results/scaleup/cpsat/{cell}.json',
                   f'results/scaleup/cpsat_b/{cell}.json'):
            if not os.path.exists(cp):
                continue
            for n, d in json.load(open(cp)).items():
                if d.get('ub') is None:
                    continue
                n_checked += 1
                if out[n] > d['ub'] + 1e-6:
                    viol.append((os.path.basename(cp), n))
        print(f'{cell}: n={len(out)} B0 mean={np.mean(list(out.values())):.1f} '
              f'checked {n_checked} makespans, admissibility '
              f'{"OK" if not viol else f"VIOLATED {viol[:3]}"}', flush=True)
        assert not viol, f'{cell}: B0 above a feasible makespan on {viol[:3]}'


if __name__ == '__main__':
    main()
