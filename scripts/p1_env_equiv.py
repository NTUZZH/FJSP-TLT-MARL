"""P1 acceptance: batched FJSPEnvTransport must agree with the reference
single-instance simulator (sim_single) under identical PDR policies.

E1  E=1 env vs sim: makespans EXACTLY equal on 10 instances x 4 rule pairs
    x 2 regimes (V=2 r=0.3, V=1 r=0.6); env schedules pass the independent
    validator.
E2  E=8 batch (8 instances stepped together) equals the 8 E=1 runs.
E3  tau==0 & ample fleet through the ENV == no-transport sim makespan.
E4  reward telescoping: sum of rewards == (init_quality - C_max) * inv_slope.
"""

import sys, glob
import numpy as np

sys.path.insert(0, '.')
from transport_marl.layout import build_transport_layout
from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from ppvc_instance_generator import load_instance


# ---- PDR drivers for the env (mirror pdr_pairs keys on TRUE quantities) ----

def env_mch_action(env, e, rule):
    cand_mask = ~env.dynamic_pair_mask[e]           # [J, M] available
    pairs = []
    jrw = None
    if rule == 'MWKR':
        jrw = {}
    for j, m in zip(*np.nonzero(cand_mask)):
        o = env.candidate[e, j]
        pt = env.true_op_pt[e, o, m]
        if rule == 'SPT':
            key = (pt, j, m)
        elif rule == 'MWKR':
            if j not in jrw:
                last = env.job_last_op_id[e, j]
                jrw[j] = float(env.true_op_min_pt[e, o:last + 1].sum())
            key = (-jrw[j], j, pt, m)
        else:  # FIFO
            key = (env.true_candidate_free_time[e, j], j, pt, m)
        pairs.append((key, j, m))
    _, j, m = min(pairs)
    return j * env.number_of_machines + m


def env_veh_action(env, e, rule):
    v = env.event_veh[e]
    loc = env.veh_cell[e, v]
    tasks = np.nonzero(env.task_active[e])[0]
    keys = []
    for j in tasks:
        frm, to = env.task_from[e, j], env.task_to[e, j]
        if rule == 'STT':
            key = (env.tau_cells[e, loc, frm] + env.tau_cells[e, frm, to], j)
        elif rule == 'NVF':
            key = (env.tau_cells[e, loc, frm], j)
        else:  # FIFO
            key = (env.task_release[e, j], j)
        keys.append((key, j))
    return min(keys)[1]


def run_env(env, mch_rule, veh_rule):
    """Roll all envs to completion; returns (makespans [E], reward_sums [E])."""
    E = env.number_of_envs
    rsum = np.zeros(E)
    guard = 0
    while env.done().min() < 1:
        actions = np.zeros(E, dtype=int)
        for e in range(E):
            if env.n_realized[e] >= env.number_of_ops:
                continue
            if env.event_type[e] == 0:
                actions[e] = env_mch_action(env, e, mch_rule)
            else:
                actions[e] = env_veh_action(env, e, veh_rule)
        _, r, _ = env.step(actions)
        rsum += r
        guard += 1
        if guard > 6 * env.number_of_ops + 1000:
            raise RuntimeError('env rollout did not terminate')
    return env.current_makespan.copy(), rsum


def build_pack(stems, ratio, n_veh):
    jls, pts, lags, opt, mct, lay = [], [], [], [], [], []
    for stem in stems:
        jl, pt, meta = load_instance(stem)
        layout = build_transport_layout(jl, pt, np.asarray(meta['mch_type']),
                                        ratio, n_veh)
        jls.append(np.asarray(jl)); pts.append(np.asarray(pt, dtype=float))
        lags.append(np.asarray(meta['time_lag'], dtype=float))
        opt.append(np.asarray(meta['op_type'])); mct.append(np.asarray(meta['mch_type']))
        lay.append(layout)
    return jls, pts, lags, opt, mct, lay


def main():
    stems = sorted(g[:-4] for g in glob.glob('data/PPVC/10x25+ppvc-mixed/instance_*.fjs'))[:10]
    fails = 0
    rule_pairs = [('SPT', 'STT'), ('MWKR', 'NVF'), ('FIFO', 'FIFO'), ('SPT', 'FIFO')]

    print('== E1: env(E=1) vs sim, 10 inst x 4 rules x 2 regimes ==')
    for ratio, nv in [(0.3, 2), (0.6, 1)]:
        for mn, vn in rule_pairs:
            bad = 0
            for stem in stems:
                jl, pt, meta = load_instance(stem)
                layout = build_transport_layout(jl, pt, np.asarray(meta['mch_type']), ratio, nv)
                sim = TransportSim(jl, pt, meta['time_lag'], layout['station_cell'],
                                   layout['tau_cells'], nv, layout['veh_start_cell'])
                ms_sim = run_pdr_pair(sim, mn, vn)
                jls, pts, lags, opt, mct, lay = build_pack([stem], ratio, nv)
                env = FJSPEnvTransport(len(jl), pt.shape[1], use_lag_features=True)
                env.set_initial_data(jls, pts, lags, opt, mct, lay)
                ms_env, _ = run_env(env, mn, vn)
                rec = env.schedule_record(0)
                res = validate_transport_schedule(jl, pt, meta['time_lag'],
                                                  np.array(layout['station_cell']),
                                                  np.array(layout['tau_cells']),
                                                  nv, layout['veh_start_cell'], rec)
                if abs(ms_sim - ms_env[0]) > 1e-9 or not res['feasible']:
                    bad += 1
                    print(f'  MISMATCH {stem} {mn}+{vn} r={ratio} V={nv}: '
                          f'sim={ms_sim:.4f} env={ms_env[0]:.4f} '
                          f'feasible={res["feasible"]} {res["violations"][:2]}')
            tag = 'OK ' if bad == 0 else 'FAIL'
            print(f'  {tag} r={ratio} V={nv} {mn}+{vn}: {10 - bad}/10 exact-equal & valid')
            fails += bad

    print('== E2: batched E=8 equals per-instance E=1 ==')
    jls, pts, lags, opt, mct, lay = build_pack(stems[:8], 0.3, 2)
    envB = FJSPEnvTransport(len(jls[0]), pts[0].shape[1], use_lag_features=True)
    envB.set_initial_data(jls, pts, lags, opt, mct, lay)
    msB, rsumB = run_env(envB, 'SPT', 'STT')
    ok = True
    for i, stem in enumerate(stems[:8]):
        jl, pt, meta = load_instance(stem)
        layout = build_transport_layout(jl, pt, np.asarray(meta['mch_type']), 0.3, 2)
        sim = TransportSim(jl, pt, meta['time_lag'], layout['station_cell'],
                           layout['tau_cells'], 2, layout['veh_start_cell'])
        ms1 = run_pdr_pair(sim, 'SPT', 'STT')
        if abs(msB[i] - ms1) > 1e-9:
            ok = False
            print(f'  MISMATCH inst {i}: batch={msB[i]:.4f} sim={ms1:.4f}')
    print(f'  {"OK " if ok else "FAIL"} batch == per-instance')
    fails += (not ok)

    print('== E3: env tau=0 & ample fleet == no-transport sim ==')
    jl, pt, meta = load_instance(stems[0])
    layout = build_transport_layout(jl, pt, np.asarray(meta['mch_type']), 0.3, len(jl))
    lay0 = dict(layout); lay0['tau_cells'] = (np.array(layout['tau_cells']) * 0).tolist()
    env0 = FJSPEnvTransport(len(jl), pt.shape[1], use_lag_features=True)
    env0.set_initial_data([np.asarray(jl)], [np.asarray(pt, dtype=float)],
                          [np.asarray(meta['time_lag'], dtype=float)],
                          [np.asarray(meta['op_type'])], [np.asarray(meta['mch_type'])],
                          [lay0])
    ms0, _ = run_env(env0, 'SPT', 'FIFO')
    sim1 = TransportSim(jl, pt, meta['time_lag'],
                        np.zeros_like(np.array(layout['station_cell'])),
                        np.array(layout['tau_cells']) * 0, len(jl), 0)
    ms1 = run_pdr_pair(sim1, 'SPT', 'FIFO')
    ok = abs(ms0[0] - ms1) < 1e-9
    print(f'  {"OK " if ok else "FAIL"} env tau0={ms0[0]:.4f} vs no-transport sim={ms1:.4f}')
    fails += (not ok)

    print('== E4: reward telescoping ==')
    jls, pts, lags, opt, mct, lay = build_pack(stems[:4], 0.6, 1)
    envT = FJSPEnvTransport(len(jls[0]), pts[0].shape[1], use_lag_features=True)
    envT.set_initial_data(jls, pts, lags, opt, mct, lay)
    init_q = envT.init_quality.copy()
    msT, rsum = run_env(envT, 'MWKR', 'STT')
    target = (init_q - msT) * envT.inv_slope
    ok = np.allclose(rsum, target, atol=1e-6)
    print(f'  {"OK " if ok else "FAIL"} sum(r) == (B(s0) - Cmax)*s '
          f'(max err {np.abs(rsum - target).max():.2e})')
    fails += (not ok)

    print(f'\n{"ALL PASS" if fails == 0 else f"{fails} FAILURES"}')
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
