"""External anchor L1: Link JSSPT instances in our environment.

Generator: clean-room port of the released `transport_random_uniform`
distribution, CERTIFIED distribution-equivalent against the authors' own
released heuristic results (24/24 deterministic combos pass a two-sample KS
test when run with THEIR schedule code; see
notes/external_link_semantics.md, gate of 2026-07-27).

Adapter (design in the same note): J jobs x (M real ops + 1 out-buf
delivery op); machines = M real + J per-job pseudo-machines at the out-buf
cell (their out-buf has no capacity limit; a dedicated pseudo-machine per
job makes our capacity-1 constraint vacuous). Cells 0..M-1 = machines,
M = in-buf, M+1 = out-buf; tau verbatim (asymmetric, non-metric is fine);
lag = 0; job_start_cell = in-buf (first op requires a transport).

Delivery-op duration: their final op has duration 0, but our op_pt matrix
encodes eligibility as pt > 0, so the pseudo-op gets pt = 1 on its dedicated
machine. The shift is uniform over jobs, hence EXACT to undo:
    makespan_link = makespan_ours - 1.
Report only the corrected number.

Self-test:  python scripts/link_external.py
Dataset:    python scripts/link_external.py --write_test data/LINK/15x10 \
                --n 100 --seed_base 910000
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, '.')

PSEUDO_PT = 1.0


def gen_link_instance(num_jobs, num_machines, seed):
    """The certified generator (do not change draw order: the KS gate
    certified exactly this sequence)."""
    random.seed(seed)
    jobs = []
    for _ in range(num_jobs):
        machines = list(range(num_machines))
        random.shuffle(machines)
        jobs.append([(m, random.randint(1, 100)) for m in machines])
    n = num_machines + 2
    tt = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        tt[i][i] = 0
    return jobs, tt      # jobs WITHOUT the out-buf op; travel incl. buffers


def link_to_env_inputs(jobs, travel, n_veh):
    """(jobs, travel) -> (job_length, op_pt, meta) in our save_instance
    format. See module docstring for the mapping."""
    J = len(jobs)
    M = len(travel) - 2
    in_buf, out_buf = M, M + 1
    n_mch = M + J
    ops_per_job = M + 1
    N = J * ops_per_job

    job_length = np.full(J, ops_per_job, dtype=int)
    # integer dtype: durations are integral and the .fjs text format parses
    # ints ("92.0" would break load_instance)
    op_pt = np.zeros((N, n_mch), dtype=int)
    op_type = np.zeros(N, dtype=int)
    mch_type = np.zeros(n_mch, dtype=int)
    mch_type[M:] = 1                                   # pseudo out-buf machines
    o = 0
    for j, job in enumerate(jobs):
        assert len(job) == M, 'expected a machine permutation per job'
        for (m, d) in job:
            op_pt[o, m] = int(d)
            o += 1
        op_pt[o, M + j] = int(PSEUDO_PT)               # delivery pseudo-op
        op_type[o] = 1
        o += 1

    station_cell = [m for m in range(M)] + [out_buf] * J
    # cell coordinates are a feature input only; lay the M+2 cells on a grid
    ncol = 4
    cell_xy = [[c // ncol, c % ncol] for c in range(M + 2)]
    tau = np.asarray(travel, dtype=float)
    real_pt = op_pt[op_pt > 0]
    meta = dict(
        op_type=op_type, mch_type=mch_type,
        time_lag=np.zeros(N),
        transport=dict(
            station_cell=station_cell, cell_xy=cell_xy,
            speed_const=1.0, tau_cells=tau.tolist(),
            n_vehicles=int(n_veh), veh_start_cell=int(in_buf),
            job_start_cell=int(in_buf),
            tau_over_p=float(tau[tau > 0].mean() / real_pt.mean())))
    return job_length, op_pt, meta


def write_dataset(out_dir, num_jobs, num_machines, n, seed_base, n_veh):
    """Instances are V-independent; n_veh recorded is a default the loader
    may override per evaluation cell."""
    from ppvc_instance_generator import save_instance
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        jobs, tt = gen_link_instance(num_jobs, num_machines, seed_base + i)
        jl, pt, meta = link_to_env_inputs(jobs, tt, n_veh)
        save_instance(f'{out_dir}/instance_{i:03d}', jl, pt, meta)
    with open(f'{out_dir}/PROVENANCE.json', 'w') as f:
        json.dump(dict(generator='link transport_random_uniform (certified port)',
                       num_jobs=num_jobs, num_machines=num_machines,
                       n=n, seed_base=seed_base,
                       gate='notes/external_link_semantics.md 2026-07-27',
                       makespan_correction=f'-{PSEUDO_PT} (delivery pseudo-op)'),
                  f, indent=1)
    print(f'wrote {n} instances -> {out_dir}')


def self_test():
    from params import configs
    configs.device = 'cpu'
    configs.fea_j_input_dim = 14
    configs.fea_m_input_dim = 10
    configs.n_op_types = 2
    configs.n_mch_types = 2
    from transport_marl.fjsp_env_transport import FJSPEnvTransport
    from transport_marl.bound import TransportBound
    from transport_marl.validator_t import validate_transport_schedule

    rng = np.random.default_rng(0)
    jls, pts, lags, opt, mct, lay = [], [], [], [], [], []
    V = 3
    for i in range(4):
        jobs, tt = gen_link_instance(15, 10, 910_000 + i)
        jl, pt, meta = link_to_env_inputs(jobs, tt, V)
        jls.append(jl); pts.append(pt); lags.append(meta['time_lag'])
        opt.append(meta['op_type']); mct.append(meta['mch_type'])
        lay.append(meta['transport'])
    env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1], use_lag_features=True)
    env.set_initial_data(jls, pts, lags, opt, mct, lay)
    env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
    B0 = env.max_endTime.copy()
    print('root bounds:', np.round(B0, 1))

    # random rollout
    while env.done().min() < 1:
        a = np.zeros(env.number_of_envs, dtype=int)
        for e in range(env.number_of_envs):
            if env.n_realized[e] >= env.number_of_ops:
                continue
            if env.event_type[e] == 0:
                legal = np.nonzero(~env.dynamic_pair_mask[e].reshape(-1))[0]
            else:
                legal = np.nonzero(env.task_active[e])[0]
            a[e] = rng.choice(legal)
        env.step(a)

    os.makedirs('../external/replays', exist_ok=True)
    for e in range(env.number_of_envs):
        rec = env.schedule_record(e)
        res = validate_transport_schedule(
            jls[e], pts[e], lags[e],
            np.array(lay[e]['station_cell']), np.array(lay[e]['tau_cells']),
            V, int(lay[e]['veh_start_cell']), rec,
            job_start_cell=int(lay[e]['job_start_cell']))
        ms_link = rec['makespan'] - PSEUDO_PT
        assert res['feasible'], f'env {e}: {res["violations"][:4]}'
        assert env.max_endTime[e] <= rec['makespan'] + 1e-6, 'bound above makespan'
        print(f'env {e}: feasible, makespan(link units) {ms_link:.1f}, '
              f'B0 {B0[e]:.1f} (gap {(ms_link - B0[e]) / B0[e]:+.1%})')
        # replay file: op commitment order (their decision = one (job, agv)
        # per op in global sequence), for the D1 cross-check in their code
        order = np.argsort(np.asarray(rec['op_start']))
        veh_of_op = {t['op']: t['veh'] for t in rec['transports']}
        seq = []
        for o in order:
            j = int(np.searchsorted(np.cumsum(jls[e]), o, side='right'))
            seq.append([j, int(veh_of_op.get(int(o), 0))])
        with open(f'../external/replays/replay_{e}.json', 'w') as f:
            json.dump(dict(seed=910_000 + e, num_jobs=15, num_machines=10,
                           n_veh=V, ours_makespan_link_units=float(ms_link),
                           sequence=seq), f)
    print('replays written -> ../external/replays/ (run '
          'external/replay_in_link.py for the D1 cross-check)')
    print('SELF-TEST PASS')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--write_test', type=str, default='')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed_base', type=int, default=910_000)
    ap.add_argument('--num_jobs', type=int, default=15)
    ap.add_argument('--num_machines', type=int, default=10)
    ap.add_argument('--n_veh', type=int, default=3)
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]
    if args.write_test:
        write_dataset(args.write_test, args.num_jobs, args.num_machines,
                      args.n, args.seed_base, args.n_veh)
    else:
        self_test()
