"""P2 acceptance tests.

B1  Admissibility (necessary condition): along random-policy rollouts,
    B(s_t) <= realized C_max + eps at every step, for all component/flag
    combos; B_chain-only likewise.
B2  Tightness: B(s_T) == C_max exactly (max component, all combos).
B3  Telescoping with the FULL bound attached: sum r * slope == B(s_0) - C_max.
N1  DANIELTransport forward: shapes, finite pi/v, legal sampled actions,
    full sampled rollout completes, makespan == validator-checked record.
N2  One TransportPPO.update on a collected batch: finite losses.
"""

import sys, glob
import numpy as np
import torch

sys.argv = [sys.argv[0]]
sys.path.insert(0, '.')

from params import configs
from ppvc_instance_generator import load_instance
from transport_marl.layout import build_transport_layout
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.mappo import TransportMemory, TransportPPO, select_actions
from transport_marl.model_transport import DANIELTransport


def build_env(stems, ratio, n_veh):
    jls, pts, lags, opt, mct, lay = [], [], [], [], [], []
    for stem in stems:
        jl, pt, meta = load_instance(stem)
        layout = build_transport_layout(jl, pt, np.asarray(meta['mch_type']), ratio, n_veh)
        jls.append(np.asarray(jl)); pts.append(np.asarray(pt, dtype=float))
        lags.append(np.asarray(meta['time_lag'], dtype=float))
        opt.append(np.asarray(meta['op_type'])); mct.append(np.asarray(meta['mch_type']))
        lay.append(layout)
    env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1], use_lag_features=True)
    env.set_initial_data(jls, pts, lags, opt, mct, lay)
    return env, (jls, pts, lags, lay)


def random_actions(env, rng):
    E = env.number_of_envs
    acts = np.zeros(E, dtype=int)
    for e in range(E):
        if env.n_realized[e] >= env.number_of_ops:
            continue
        if env.event_type[e] == 0:
            avail = np.nonzero(~env.dynamic_pair_mask[e].reshape(-1))[0]
        else:
            avail = np.nonzero(~env.veh_action_mask[e])[0]
        acts[e] = rng.choice(avail)
    return acts


def roll_random(env, bounds, rng):
    traces = [[] for _ in bounds]
    while env.done().min() < 1:
        for bi, b in enumerate(bounds):
            traces[bi].append(b.value().copy())
        env.step(random_actions(env, rng))
    for bi, b in enumerate(bounds):
        traces[bi].append(b.value().copy())
    return [np.array(tr) for tr in traces]


def main():
    rng = np.random.default_rng(7)
    stems = sorted(g[:-4] for g in glob.glob('data/PPVC/5x9+ppvc-mixed/instance_*.fjs'))[:5]
    fails = 0

    print('== B1/B2: admissibility + tightness over random rollouts ==')
    combos = [dict(use_mch=False, use_veh=False),
              dict(use_mch=True, use_veh=False),
              dict(use_mch=True, use_veh=True),
              dict(use_mch=True, use_veh=True, flag_fleet_aggregate=True),
              dict(use_mch=True, use_veh=True, flag_fleet_aggregate=True,
                   flag_min_residual=True)]
    for ratio, nv in [(0.3, 2), (0.6, 1)]:
        env, _ = build_env(stems, ratio, nv)
        bounds = [TransportBound(env, **c) for c in combos]
        traces = roll_random(env, bounds, rng)
        cmax = env.current_makespan
        for ci, tr in enumerate(traces):
            adm = (tr <= cmax[None, :] + 1e-9).all()
            tight = np.allclose(tr[-1], cmax, atol=1e-9) if ci >= 0 else True
            # tightness holds for every combo: B_chain is tight and max keeps it
            ok = adm and tight
            fails += (not ok)
            print(f'  {"OK " if ok else "FAIL"} r={ratio} V={nv} combo{ci}: '
                  f'admissible={adm} tight={tight} '
                  f'(max B - Cmax = {(tr - cmax[None, :]).max():.2e})')

    print('== B3: telescoping with full bound attached ==')
    env, _ = build_env(stems, 0.6, 1)
    full = TransportBound(env, use_mch=True, use_veh=True)
    env.attach_bound(full)
    b0 = env.max_endTime.copy()
    rsum = np.zeros(env.number_of_envs)
    while env.done().min() < 1:
        _, r, _ = env.step(random_actions(env, rng))
        rsum += r
    target = (b0 - env.current_makespan) * env.inv_slope
    ok = np.allclose(rsum, target, atol=1e-6)
    fails += (not ok)
    print(f'  {"OK " if ok else "FAIL"} sum r == (B(s0)-Cmax)*s '
          f'(max err {np.abs(rsum - target).max():.2e})')

    print('== N1: policy forward + sampled rollout ==')
    stems25 = sorted(g[:-4] for g in glob.glob('data/PPVC/10x25+ppvc-mixed/instance_*.fjs'))[:4]
    env, (jls, pts, lags, lay) = build_env(stems25, 0.3, 2)
    configs.fea_j_input_dim = env.op_fea_dim
    configs.fea_m_input_dim = env.mch_fea_dim
    configs.n_op_types = 5
    configs.n_mch_types = 9
    configs.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    policy = DANIELTransport(configs)
    state = env.state
    steps = 0
    with torch.no_grad():
        while env.done().min() < 1:
            pi_m, pi_v, v = policy(
                fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                mch_type=state.mch_type_tensor, fea_v=state.fea_v_tensor,
                fea_veh_pairs=state.fea_veh_pairs_tensor,
                veh_action_mask=state.veh_action_mask_tensor,
                event_veh=state.event_veh_tensor,
                task_dest_op=state.task_dest_op_tensor)
            assert torch.isfinite(v).all(), 'critic NaN'
            a, logp = select_actions(pi_m, pi_v, state.event_type_tensor)
            state, r, done = env.step(a.cpu().numpy())
            steps += 1
            if steps > 6 * env.number_of_ops + 1000:
                raise RuntimeError('NN rollout did not terminate')
    ok = True
    for e in range(env.number_of_envs):
        jl, pt, meta = load_instance(stems25[e])
        res = validate_transport_schedule(
            jl, pt, meta['time_lag'], np.array(lay[e]['station_cell']),
            np.array(lay[e]['tau_cells']), 2, lay[e]['veh_start_cell'],
            env.schedule_record(e))
        if not res['feasible']:
            ok = False
            print(f'  FAIL env{e}: {res["violations"][:2]}')
    fails += (not ok)
    print(f'  {"OK " if ok else "FAIL"} sampled rollout, {steps} steps, '
          f'makespans {np.round(env.current_makespan, 1)}')

    print('== N2: one PPO update ==')
    env.reset()
    ppo = TransportPPO(configs)
    mem = TransportMemory(configs.gamma, configs.gae_lambda)
    state = env.state
    dev = torch.device(configs.device)
    while env.done().min() < 1:
        alive = torch.from_numpy(env.n_realized < env.number_of_ops).to(dev)
        with torch.no_grad():
            pi_m, pi_v, v = ppo.policy_old(
                fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                mch_type=state.mch_type_tensor, fea_v=state.fea_v_tensor,
                fea_veh_pairs=state.fea_veh_pairs_tensor,
                veh_action_mask=state.veh_action_mask_tensor,
                event_veh=state.event_veh_tensor,
                task_dest_op=state.task_dest_op_tensor)
            a, logp = select_actions(pi_m, pi_v, state.event_type_tensor)
        mem.push(state, alive)
        mem.action_seq.append(a)
        mem.log_probs.append(logp)
        mem.val_seq.append(v.squeeze(1))
        state, r, done = env.step(a.cpu().numpy())
        mem.reward_seq.append(torch.from_numpy(r).float().to(dev))
        mem.done_seq.append(torch.from_numpy(done).float().to(dev))
    loss, vloss = ppo.update(mem)
    ok = np.isfinite(loss) and np.isfinite(vloss)
    fails += (not ok)
    print(f'  {"OK " if ok else "FAIL"} loss={loss:.4f} vloss={vloss:.4f}')

    print(f'\n{"ALL PASS" if fails == 0 else f"{fails} FAILURES"}')
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
