"""Unit tests for the bound-guided action prior (transport_marl/guide.py).

G1  shape/mask: with use_guide, fea_pairs is [E,J,M,9] and fea_veh_pairs
    [E,J,7]; the guide channel is 0 on masked pairs / inactive tasks.
G2  admissible pricing: along random rollouts, the priced Delta of the
    EXECUTED action never exceeds the realized one-step floor rise
    (-reward): the only step effect guide.py does not reproduce is the
    successor tau_in_min refresh, which can only raise B further.
G3  tightness: mean priced/realized ratio reported (should be near 1;
    hard-asserted > 0.5 as a wiring smoke, not a theorem).
G4  cost: per-step overhead of the guide computation reported.

Run: python scripts/p4_guide_tests.py
"""

import sys, time

import numpy as np

sys.path.insert(0, '.')
from params import configs
configs.device = 'cpu'

from ppvc_instance_generator import ppvc_instance_generator
from transport_marl.layout import build_transport_layout
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound

rng = np.random.default_rng(7)


def make_env(n_env, n_veh, ratio, seed0, use_guide):
    jls, pts, lags, opt, mct, lay = [], [], [], [], [], []
    for e in range(n_env):
        jl, pt, meta = ppvc_instance_generator(
            n_modules=10, class_mix='mixed', seed=seed0 + e)
        layout = build_transport_layout(jl, pt, np.asarray(meta['mch_type']),
                                        ratio, n_veh)
        jls.append(np.asarray(jl)); pts.append(np.asarray(pt, dtype=float))
        lags.append(np.asarray(meta['time_lag'], dtype=float))
        opt.append(np.asarray(meta['op_type']))
        mct.append(np.asarray(meta['mch_type']))
        lay.append(layout)
    env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1],
                           use_lag_features=True, use_guide=use_guide)
    env.set_initial_data(jls, pts, lags, opt, mct, lay)
    env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
    return env


def random_actions(env):
    a = np.zeros(env.number_of_envs, dtype=int)
    M = env.number_of_machines
    for e in range(env.number_of_envs):
        if env.n_realized[e] >= env.number_of_ops:
            continue
        if env.event_type[e] == 0:
            legal = np.nonzero(~env.dynamic_pair_mask[e].reshape(-1))[0]
        else:
            legal = np.nonzero(env.task_active[e])[0]
        a[e] = rng.choice(legal)
    return a


def main():
    env = make_env(n_env=8, n_veh=2, ratio=0.6, seed0=990000, use_guide=True)

    # --- G1 shapes + masked zeros (checked along the rollout too) ---
    E, J, M = env.number_of_envs, env.number_of_jobs, env.number_of_machines
    assert env.fea_pairs.shape == (E, J, M, 9), env.fea_pairs.shape
    assert env.fea_veh_pairs.shape == (E, J, 7), env.fea_veh_pairs.shape
    print(f'G1 PASS: fea_pairs {env.fea_pairs.shape}, '
          f'fea_veh_pairs {env.fea_veh_pairs.shape}')

    # --- G2/G3 admissible pricing along a random rollout ---
    viol, ratios, n_checked = 0, [], 0
    guide_time, steps = 0.0, 0
    while env.done().min() < 1:
        alive = env.n_realized < env.number_of_ops
        gp = env.fea_pairs[:, :, :, -1]
        gt = env.fea_veh_pairs[:, :, -1]
        # done envs keep frozen (stale) features by design -- check alive only
        masked_p = env.dynamic_pair_mask & alive[:, None, None]
        assert np.all(gp[masked_p] == 0.0), 'guide channel nonzero on masked pair'
        dead_t = ~env.task_active & alive[:, None]
        assert np.all(gt[dead_t] == 0.0), 'guide nonzero on dead task'
        a = random_actions(env)
        pred = np.zeros(E)
        for e in range(E):
            if not alive[e]:
                continue
            if env.event_type[e] == 0:
                pred[e] = gp[e, a[e] // M, a[e] % M]
            else:
                pred[e] = gt[e, a[e]]
        t0 = time.perf_counter()
        _, r, _ = env.step(a)          # includes the guide rebuild
        guide_time += time.perf_counter() - t0
        steps += 1
        realized = -np.asarray(r)      # (B(s') - B(s)) * inv_slope
        for e in range(E):
            if not alive[e]:
                continue
            n_checked += 1
            if pred[e] > realized[e] + 1e-6:
                viol += 1
            if realized[e] > 1e-9:
                ratios.append(pred[e] / realized[e])
    assert viol == 0, f'G2 FAIL: {viol}/{n_checked} priced above realized rise'
    mean_ratio = float(np.mean(ratios)) if ratios else float('nan')
    print(f'G2 PASS: 0/{n_checked} admissibility violations')
    print(f'G3 tightness: priced/realized mean {mean_ratio:.3f} '
          f'(n={len(ratios)} rises)')
    assert mean_ratio > 0.5, 'G3 FAIL: pricing far looser than expected'

    # --- G4 cost: same rollout without guide ---
    env0 = make_env(n_env=8, n_veh=2, ratio=0.6, seed0=990000, use_guide=False)
    t0, steps0 = time.perf_counter(), 0
    while env0.done().min() < 1:
        env0.step(random_actions(env0))
        steps0 += 1
    base = (time.perf_counter() - t0) / steps0
    with_g = guide_time / steps
    print(f'G4 cost: step {base*1e3:.1f} ms -> {with_g*1e3:.1f} ms with guide '
          f'({(with_g-base)*1e3:+.1f} ms/step, batch of {E})')
    print('ALL GUIDE TESTS PASS')


if __name__ == '__main__':
    main()
