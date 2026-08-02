"""Solver-free optimality certificates (novelty item C).

B0 = B(s_root) is computable without any solver. For a schedule with
makespan C the shipped certificate is gap_cert = (C - B0) / B0, valid
because B0 <= C* <= C (Theorem 1 admissibility at the root).

This script computes B0 per instance per cell and, where CP-SAT references
exist, VERIFIES the certificate direction on real data:
    B0 <= CP-SAT UB   always (admissibility vs any feasible schedule),
    B0 <= CP-SAT LB'  not required (different relaxations), reported only.
Writes results/certificate/{cell}.json = {instance: B0}; summary printed.
The certified-gap table entries are assembled by fill_macros once policy
makespans land.

Usage: python scripts/p8_certificate.py [cells...]   (default: all test dirs)
"""

import glob
import json
import os
import sys

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]

import numpy as np

sys.path.insert(0, '.')
from params import configs
configs.device = 'cpu'
from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound


def root_bounds(stems):
    """B0 for a batch of same-shape instances (one env build)."""
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
    if ARGS:
        dirs = []
        for c in ARGS:
            d = c if 'x25+' in c else f'10x25+ppvct-mixed+{c}'
            dirs.append(f'data/PPVCT/{d}/test')
    else:
        dirs = sorted(glob.glob('data/PPVCT/1[05]x25+ppvct-mixed+v*+t*/test'))
    os.makedirs('results/certificate', exist_ok=True)
    for ds in dirs:
        cell = ds.split('/')[2]
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        if not stems:
            continue
        out = {}
        B = []
        for i0 in range(0, len(stems), 20):
            chunk = stems[i0:i0 + 20]
            b = root_bounds(chunk)
            B.extend(b.tolist())
            for s, v in zip(chunk, b):
                out[os.path.basename(s)] = float(v)
        with open(f'results/certificate/{cell}.json', 'w') as f:
            json.dump(out, f, indent=0)

        # verify against CP-SAT references when available
        short = cell.replace('10x25+ppvct-mixed+', '')
        ref_path = (f'or_solution/PPVCT/{short}.jsonl'
                    if os.path.exists(f'or_solution/PPVCT/{short}.jsonl')
                    else f'or_solution/PPVCT/{cell}.jsonl')
        line = f'{cell}: n={len(out)} B0 mean={np.mean(B):.1f}'
        if os.path.exists(ref_path):
            refs = {}
            with open(ref_path) as f:
                for ln in f:
                    r = json.loads(ln)
                    refs[r['instance']] = r
            viol = [n for n, b in out.items()
                    if n in refs and b > refs[n]['ms'] + 1e-6]
            both = [n for n in out if n in refs]
            gaps = [(refs[n]['ms'] - out[n]) / out[n] for n in both]
            lbs = [(out[n], refs[n].get('lb')) for n in both
                   if refs[n].get('lb') is not None]
            tighter = sum(1 for b, l in lbs if b > l + 1e-6)
            line += (f'  vs CP-SAT: cov {len(both)}, ADMISSIBILITY '
                     f'{"OK" if not viol else f"VIOLATED x{len(viol)}"}'
                     f', mean (UB-B0)/B0 {np.mean(gaps):.1%}'
                     f', B0 tighter than CP-SAT LB on {tighter}/{len(lbs)}')
            assert not viol, f'{cell}: B0 above CP-SAT UB on {viol[:3]}'
        print(line, flush=True)


if __name__ == '__main__':
    main()
