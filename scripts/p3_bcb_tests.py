"""P3 acceptance tests for the Bound-Counterfactual Baseline (bcb.py).

C1  Pass floor never below the current bound: b_t >= B(s_t) - eps along
    random rollouts (passing can only delay), machine and vehicle events.
C2  Determinism: two evaluations at the same state agree exactly.
C3  Informativeness: on vehicle events with a scarce fleet the credited
    signal c_t = r_t - r_t^pass is non-degenerate (reported, not asserted:
    mean/std/frac-nonzero per event class).
C4  M1 pipeline: 2 rollout+update cycles with credit='m1' produce finite
    losses (run via p2_train_mappo.py --credit m1 smoke).
"""

import sys, glob
import numpy as np

sys.argv = [sys.argv[0]]
sys.path.insert(0, '.')

import torch
from params import configs
from scripts.p2_bound_and_nn_tests import build_env, random_actions
from transport_marl.bound import TransportBound
from transport_marl.bcb import pass_floor, r_pass


def main():
    rng = np.random.default_rng(11)
    stems = sorted(g[:-4] for g in glob.glob(
        'data/PPVC/10x25+ppvc-mixed/instance_*.fjs'))[:4]
    fails = 0

    for ratio, nv in [(0.3, 2), (0.6, 1)]:
        env, _ = build_env(stems, ratio, nv)
        env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
        viol = 0
        c_mch, c_veh = [], []
        det_bad = 0
        while env.done().min() < 1:
            b1 = pass_floor(env)
            b2 = pass_floor(env)
            det_bad += int(not np.array_equal(b1, b2))
            cur = env.max_endTime
            alive = env.n_realized < env.number_of_ops
            viol += int(((b1 - cur) < -1e-9)[alive].any())
            rp = (cur - b1) * env.inv_slope
            et = env.event_type.copy()
            _, r, _ = env.step(random_actions(env, rng))
            c = r - rp
            for e in np.nonzero(alive)[0]:
                (c_mch if et[e] == 0 else c_veh).append(c[e])
        c_mch, c_veh = np.array(c_mch), np.array(c_veh)
        ok = viol == 0 and det_bad == 0
        fails += (not ok)
        print(f'{"OK " if ok else "FAIL"} C1/C2 r={ratio} V={nv}: '
              f'floor-violations={viol} nondet={det_bad}')
        for name, c in [('mch', c_mch), ('veh', c_veh)]:
            if len(c):
                print(f'   C3 {name}: n={len(c)} mean={c.mean():.4f} '
                      f'std={c.std():.4f} nonzero={np.mean(np.abs(c) > 1e-9):.2%}')

    print(f'\n{"C1-C3 PASS" if fails == 0 else f"{fails} FAILURES"} '
          f'(C4 runs via train-script smoke)')
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
