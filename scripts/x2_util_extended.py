"""Minimum loaded fleet utilization: scale cells and the Link benchmark.

Extends scripts/x2_fleet_util.py (primary grid) to the two places the
regime claim must transfer:
  (1) the scale-up cells (50 and 80 modules): the realistic batch sizes;
  (2) the external Link benchmark, per evaluated fleet size.

u = W_tr / (|V| * C): W_tr is the fleet-size-independent minimal loaded
work (root B_veh at |V|=1), C the best known mean makespan for the cell,
so u is the smallest share of fleet time any schedule can spend carrying
loads. Empty legs are excluded; both approximations push u down, so every
value is a floor.

Writes results/fleet_util_extended.json.
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


def wtr_batch(stems, n_veh_override=1):
    """W_tr per instance = root B_veh at |V|=1 (f_v=0, so B_veh=W_tr)."""
    jls, pts, lags, opt, mct, lay = [], [], [], [], [], []
    for s in stems:
        jl, pt, meta = load_instance(s)
        jls.append(np.asarray(jl))
        pts.append(np.asarray(pt, dtype=float))
        lags.append(np.asarray(meta['time_lag'], dtype=float))
        opt.append(np.asarray(meta['op_type']))
        mct.append(np.asarray(meta['mch_type']))
        lay.append(dict(meta['transport'], n_vehicles=n_veh_override))
    env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1], use_lag_features=True)
    env.set_initial_data(jls, pts, lags, opt, mct, lay)
    b = TransportBound(env, use_mch=True, use_veh=True)
    return b._b_veh()          # == W_tr at |V|=1, root


def mean_wtr(data_dir, cap=100):
    stems = sorted(g[:-4] for g in glob.glob(f'{data_dir}/instance_*.fjs'))[:cap]
    W = []
    for i0 in range(0, len(stems), 10):
        W.extend(wtr_batch(stems[i0:i0 + 10]).tolist())
    return float(np.mean(W)), len(W)


out = {}

# ---- scale cells: C = best known mean (GA where run, else mixture policy) ----
SCALE = {
    '50x25+ppvct-mixed+v2+t0.6': 2,
    '50x25+ppvct-mixed+v2+t1.0': 2,
    '80x25+ppvct-mixed+v3+t1.0': 3,
}
for cell, V in SCALE.items():
    W, n = mean_wtr(f'data/PPVCT/{cell}/test', cap=30)
    # C = best known mean over every arm that ran the cell. Formats differ
    # per arm, so each is parsed explicitly; sample-decode files are the
    # per-seed best-of-64 rows and are averaged per instance across seeds.
    best = [None, None]

    def offer(vals, name):
        if vals is not None and len(vals):
            c = float(np.mean(vals))
            if best[0] is None or c < best[0]:
                best[0], best[1] = c, name

    for p in (glob.glob(f'results/scaleup/ga/{cell}.json')
              + glob.glob(f'results/scaleup/ga_long/*/{cell}.json')):
        d = json.load(open(p))
        offer([r['ga'] for r in d.values()],
              f'ga:{os.path.relpath(p, "results/scaleup")}')
    for p in glob.glob(f'results/scaleup/pdr/{cell}.json'):
        d = json.load(open(p))
        pairs = sorted(next(iter(d.values())))
        offer(min(([d[k][q] for k in sorted(d)] for q in pairs),
                  key=np.mean), f'pdr:{os.path.basename(p)}')
    pol = [json.load(open(p))['rows']
           for p in glob.glob(f'results/scaleup/policy/*_{cell}.json')
           if '+b1lat' not in p]
    if pol:
        keys = sorted(pol[0])
        offer(np.mean([[r[k]['ms'] for k in keys] for r in pol
                       if sorted(r) == keys], axis=0), 'policy:3-seed-mean')
    groups = {}
    for p in glob.glob(f'results/sample_decode/*_{cell}_N64.json'):
        model = os.path.basename(p).split('-s3')[0]
        groups.setdefault(model, []).append(json.load(open(p))['rows'])
    for model, rows in groups.items():
        keys = sorted(rows[0])
        offer(np.mean([[r[k] for k in keys] for r in rows], axis=0),
              f'sample_decode:{model}:seed-mean')
    p_ = f'results/scaleup/cpsat_b/{cell}.jsonl'
    if os.path.exists(p_):
        offer([json.loads(l)['ub'] for l in open(p_) if l.strip()],
              'cpsat_b:ub')
    best_c, best_name = best
    u = W / (V * best_c) if best_c else None
    out[cell] = {'n': n, 'V': V, 'W_tr_mean': W, 'C_best_mean': best_c,
                 'C_source': best_name, 'util': u}
    print('%-30s |V|=%d W_tr=%8.1f C_best=%8.1f (%s) u=%.3f'
          % (cell, V, W, best_c or -1, best_name, u or -1), flush=True)

# ---- Link benchmark: C = best released/ours mean per fleet size ----
W_link, n_link = mean_wtr('data/LINK/15x10/test', cap=100)
score = json.load(open(
    '../external/replays_l1/15x10+link+link-m1-guide-s301/score_vs_released.json'))
link = {}
for vkey, rec in sorted(score.items(), key=lambda kv: int(kv[0][1:])):
    V = int(vkey[1:])
    cands = {'ours': rec['ours_mean']}
    for name, a in rec['anchors'].items():
        cands[name] = a['mean']
    best_name = min(cands, key=cands.get)
    C = cands[best_name]
    link[vkey] = {'V': V, 'C_best_mean': C, 'C_source': best_name,
                  'util': W_link / (V * C)}
    print('link %-4s |V|=%2d W_tr=%8.1f C_best=%8.1f (%s) u=%.3f'
          % (vkey, V, W_link, C, best_name, W_link / (V * C)), flush=True)
out['link'] = {'n': n_link, 'W_tr_mean': W_link, 'per_fleet': link}

with open('results/fleet_util_extended.json', 'w') as f:
    json.dump(out, f, indent=1)
print('wrote results/fleet_util_extended.json')
