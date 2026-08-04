"""Bound-guided action prior (Paper X2 E5-G: the certificate's third job).

The same Theorem-1 recursion that pays the team (telescoped reward) and
splits the pay (BCB credit) can also PRICE each candidate action before it
is taken: for every currently-legal action a at the decision event, evaluate
the bound at the committed successor state,

    Delta_a = max(B(s . a) - B(s), 0),

and feed Delta_a (inv_slope-scaled) to the acting head as one extra
per-candidate feature channel. Delta_a is exactly the certified-floor rise
the commitment itself causes -- i.e. an a-priori estimate of the immediate
telescoped reward of a -- so the policy sees an admissible price tag per
action, the way A* sees an admissible heuristic per successor.

Fidelity: the machine branch mirrors _step_machine (commit at
max(job_ready, mch_free); co-located ops realize, cross-cell ops get the
optimistic in-transit envelope release + loaded-tau + pt); the vehicle
branch mirrors _step_vehicle (pickup + loaded move, then realize at
arrival). The ONLY post-step effect not reproduced is the tau_in_min
refresh of the realized op's successor (realized-origin min >= static min),
which can raise B_veh further; Delta_a is therefore a LOWER estimate of the
realized one-step floor rise (tested in scripts/p4_guide_tests.py) and an
exact one when travel is trivial. Being an input feature only, it changes
neither the reward nor Lemma 1: faithfulness and no-added-bias hold as
stated.

Cost: O(1) per candidate after O(N + M + V) per-env aggregates (per-job
chain tails and per-group busy/work sums, kept as top-2 so replacing one
entry is O(1)).
"""

import numpy as np


def _top2(vals):
    """(max, runner-up, argmax) of a 1-D array; runner-up = -inf if len 1."""
    i = int(np.argmax(vals))
    m1 = float(vals[i])
    if len(vals) == 1:
        return m1, -np.inf, i
    m2 = float(np.max(np.delete(vals, i)))
    return m1, m2, i


def _max_excluding(m1, m2, i, j):
    return m2 if j == i else m1


def guide_features(env):
    """Per-candidate certified price tags for the current decision events.

    Returns (pair_delta [E, J, M], task_delta [E, J]), inv_slope-scaled,
    zero on masked/irrelevant entries. Rows with event_type 0 fill
    pair_delta; rows with event_type 1 fill task_delta for the acting
    vehicle. Uses env._bound's component set when attached (headline:
    use_mch and use_veh, flags off); falls back to B_chain alone otherwise,
    matching _bound_value().
    """
    E, J, M = env.number_of_envs, env.number_of_jobs, env.number_of_machines
    V = env.n_veh
    s = env.inv_slope
    pair_delta = np.zeros((E, J, M))
    task_delta = np.zeros((E, J))
    bound = getattr(env, '_bound', None)
    use_mch = bound is not None and bound.use_mch
    use_veh = bound is not None and bound.use_veh and V > 0
    agg = bound.flag_fleet_aggregate if bound is not None else False
    # B(s_t) recomputed here: env.max_endTime is one step STALE at feature
    # time (step() rebuilds features before it re-bases the telescope)
    B_all = env._bound_value()

    for e in range(E):
        if env.n_realized[e] >= env.number_of_ops:
            continue
        B_cur = float(B_all[e])
        lastN = env.job_last_op_id[e]                       # [J]
        tails = env.op_ct_lb[e][lastN]                      # per-job chain tails
        t1, t2, ti = _top2(tails)

        if use_mch:
            G = bound.n_groups[e]
            busy = np.zeros(G); size = np.zeros(G); W_g = np.zeros(G)
            np.add.at(busy, bound.group_id[e], env.true_mch_free_time[e])
            np.add.at(size, bound.group_id[e], 1.0)
            unst = env.op_scheduled_flag[e] < 0.5
            if unst.any():
                np.add.at(W_g, bound.op_group[e][unst], env.true_op_min_pt[e][unst])
            gval = (busy + W_g) / size
            g1, g2, gi = _top2(gval)

        if use_veh:
            W_tr = 0.0
            for j in np.nonzero(env.task_active[e])[0]:
                W_tr += env.tau_cells[e, env.task_from[e, j], env.task_to[e, j]]
            unc = env.op_scheduled_flag[e] < 0.5
            mand = unc & (env.tau_in_min[e] > 0)
            W_tr += float(env.tau_in_min[e][mand].sum())
            veh_min = float(env.veh_free[e].min())
            veh_sum = float(env.veh_free[e].sum())

        if env.event_type[e] == 0:
            avail = ~env.dynamic_pair_mask[e]               # [J, M]
            for j in np.nonzero(avail.any(axis=1))[0]:
                o = env.candidate[e, j]
                cur = env.job_cell[e, j]
                for m in np.nonzero(avail[j])[0]:
                    commit_t = max(env.true_candidate_free_time[e, j],
                                   env.true_mch_free_time[e, m])
                    dest = env.station_cell[e, m]
                    tau_l = 0.0
                    if cur >= 0 and cur != dest:
                        tau_l = env.tau_cells[e, cur, dest]
                    ct = commit_t + tau_l + env.true_op_pt[e, o, m]
                    # chain: _raise_ct_lb shifts the job suffix by diff
                    b = _max_excluding(t1, t2, ti, j) if J > 1 else -np.inf
                    b = max(b, tails[j] + (ct - env.op_ct_lb[e, o]))
                    if use_mch:
                        g = bound.group_id[e, m]
                        ng = (busy[g] - env.true_mch_free_time[e, m] + ct
                              + W_g[g] - env.true_op_min_pt[e, o]) / size[g]
                        b = max(b, _max_excluding(g1, g2, gi, g), ng)
                    if use_veh:
                        Wp = W_tr + tau_l
                        if env.tau_in_min[e, o] > 0:
                            Wp -= env.tau_in_min[e, o]
                        if Wp > 0:
                            if agg:
                                b = max(b, (veh_sum + Wp) / V)
                            else:
                                b = max(b, veh_min + Wp / V)
                    pair_delta[e, j, m] = max(b - B_cur, 0.0) * s
        else:
            v = env.event_veh[e]
            vloc = env.veh_cell[e, v]
            for j in np.nonzero(env.task_active[e])[0]:
                frm, to = env.task_from[e, j], env.task_to[e, j]
                tau_l = env.tau_cells[e, frm, to]
                depart = max(env.veh_free[e, v],
                             env.task_release[e, j]) + env.tau_cells[e, vloc, frm]
                arrival = depart + tau_l
                o = env.task_dest_op[e, j]
                m = env.task_dest_mch[e, j]
                ct = arrival + env.true_op_pt[e, o, m]
                b = _max_excluding(t1, t2, ti, j) if J > 1 else -np.inf
                b = max(b, tails[j] + (ct - env.op_ct_lb[e, o]))
                if use_mch:
                    g = bound.group_id[e, m]
                    ng = (busy[g] - env.true_mch_free_time[e, m] + ct
                          + W_g[g]) / size[g]
                    b = max(b, _max_excluding(g1, g2, gi, g), ng)
                if use_veh:
                    Wp = W_tr - tau_l
                    if Wp > 0:
                        if agg:
                            b = max(b, (veh_sum - env.veh_free[e, v]
                                        + arrival + Wp) / V)
                        else:
                            nv_min = min(arrival, float(np.delete(
                                env.veh_free[e], v).min())) if V > 1 else arrival
                            b = max(b, nv_min + Wp / V)
                task_delta[e, j] = max(b - B_cur, 0.0) * s
    return pair_delta, task_delta


def naive_price_features(env):
    """NON-ADMISSIBLE control channel (Paper X2 arm c).

    Same shapes, same clamp, same inv_slope scaling and the same call site as
    guide_features(), but the number is a myopic DURATION rather than a
    certified floor rise: the wall time the committed action itself occupies,
    measured from the current decision time.

        machine event : (commit time - now) + loaded travel + processing time
        vehicle event : (arrival  - now) + processing time

    It is deliberately NOT a lower bound on B(s.a) - B(s): it charges work the
    certificate shows the rest of the shop absorbs in parallel, so it exceeds
    the certified price on most (s, a) and its ranking of candidates need not
    agree with the certified ranking. Corollary 2's admissibility therefore
    fails by construction, which is the point of the arm.

    env.guide_price_scale multiplies the result so that the channel's mean
    magnitude matches the certified channel's (calibrated before launch and
    recorded in the config snapshot). Without that match the arm would test
    feature scale rather than admissibility.
    """
    E, J, M = env.number_of_envs, env.number_of_jobs, env.number_of_machines
    s = env.inv_slope * float(getattr(env, 'guide_price_scale', 1.0))
    pair_delta = np.zeros((E, J, M))
    task_delta = np.zeros((E, J))
    now = env.next_event_time
    for e in range(E):
        if env.n_realized[e] >= env.number_of_ops:
            continue
        if env.event_type[e] == 0:
            avail = ~env.dynamic_pair_mask[e]                # [J, M]
            for j in np.nonzero(avail.any(axis=1))[0]:
                o = env.candidate[e, j]
                cur = env.job_cell[e, j]
                for m in np.nonzero(avail[j])[0]:
                    commit_t = max(env.true_candidate_free_time[e, j],
                                   env.true_mch_free_time[e, m])
                    dest = env.station_cell[e, m]
                    tau_l = 0.0
                    if cur >= 0 and cur != dest:
                        tau_l = env.tau_cells[e, cur, dest]
                    dur = (commit_t - now[e]) + tau_l + env.true_op_pt[e, o, m]
                    pair_delta[e, j, m] = max(dur, 0.0) * s
        else:
            v = env.event_veh[e]
            vloc = env.veh_cell[e, v]
            for j in np.nonzero(env.task_active[e])[0]:
                frm, to = env.task_from[e, j], env.task_to[e, j]
                depart = max(env.veh_free[e, v],
                             env.task_release[e, j]) + env.tau_cells[e, vloc, frm]
                arrival = depart + env.tau_cells[e, frm, to]
                o = env.task_dest_op[e, j]
                m = env.task_dest_mch[e, j]
                dur = (arrival - now[e]) + env.true_op_pt[e, o, m]
                task_delta[e, j] = max(dur, 0.0) * s
    return pair_delta, task_delta
