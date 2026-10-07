"""Bound-Counterfactual Baseline (BCB) — Lemma 1 machinery (Paper X2 §5).

At decision step t the acting agent is either the machine class (event 0) or
vehicle event_veh (event 1). The VIRTUAL DEFER state s_t^pass consumes the
event with no commitment:

- machine event at time t: no pair commits at t; every currently-available
  candidate op can then start no earlier than t' (the next event time after
  the pass). Its completion floor becomes
      min over eligible m of  max(t', job_ready, mch_free[m])
                              + tau(job_cell -> cell(m)) + pt[o, m]
  (travel term 0 when co-located / job not materialized).
- vehicle event (v, t): v skips its turn; it is next available at t'. Every
  pool task's pickup floor becomes
      min over vehicles w of  max(release, avail_w + tau_e(cell_w -> from))
  with avail_v raised to t' (Manhattan tau is a metric, so any detour of w
  is >= the direct empty move — the floor is admissible for the pass state).
  Destination-op floors rise by the loaded move + pt accordingly; B_veh's
  earliest-free uses the raised avail_v.

b_t := B(s_t^pass) is a function of (state, event) only — independent of any
agent's sampled action — so subtracting the induced counterfactual reward
    r_t^pass := (B(s_t) - b_t) * inv_slope        (<= 0: passing delays)
from the policy-gradient advantage is unbiased (Lemma 1, state baseline).
The credited signal r_t - r_t^pass reads "how much did YOUR commitment move
the certified floor relative to passing the event".

Note (paper discipline): the pass floor uses vehicle-availability tightening
that the HEADLINE Theorem-1 reward deliberately omits (capacity relaxation).
Both are admissible bounds of their respective states; the baseline needs
only action-independence, and the tightening is what makes vehicle credit
informative.

Cost: O(candidates x M + pool x V + N) per env per step.
"""

import numpy as np


def _next_time_after(env, e, t, skip_vehicle=None):
    """Earliest potential event time STRICTLY after t in env e (pass-state).
    When another vehicle can act at the same instant, the pass changes
    nothing for the pool and t' collapses to t."""
    cands = []
    pft = env.true_pair_free_time[e]
    blocked = env._blocked_pairs[e]
    open_pairs = pft[~blocked]
    if open_pairs.size:
        later = open_pairs[open_pairs > t + 1e-12]
        if later.size:
            cands.append(float(later.min()))
    if env.task_active[e].any():
        rel = float(env.task_release[e][env.task_active[e]].min())
        for w in range(env.n_veh):
            if w == skip_vehicle:
                continue
            tw = max(float(env.veh_free[e, w]), rel)
            cands.append(tw if tw > t + 1e-12 else t)
    mfree = env.true_mch_free_time[e]
    later_m = mfree[mfree > t + 1e-12]
    if later_m.size:
        cands.append(float(later_m.min()))
    live_jobs = ~env.mask[e].astype(bool)
    if live_jobs.any():
        jready = env.true_candidate_free_time[e][live_jobs]
        later_j = jready[jready > t + 1e-12]
        if later_j.size:
            cands.append(float(later_j.min()))
    return min(cands) if cands else t


def pass_floor(env):
    """b_t = B(s_t^pass) for every env in the batch (true units, [E])."""
    E = env.number_of_envs
    base = env._bound_value()
    out = np.zeros(E)
    for e in range(E):
        if env.n_realized[e] >= env.number_of_ops:
            out[e] = base[e]
            continue
        t = env.next_event_time[e]
        ct_lb = env.op_ct_lb[e].copy()
        veh_free = env.veh_free[e].copy()

        if env.event_type[e] == 0:
            tp = _next_time_after(env, e, t)
            avail = ~env.dynamic_pair_mask[e]                  # [J, M]
            for j in np.nonzero(avail.any(axis=1))[0]:
                o = env.candidate[e, j]
                floor = np.inf
                for m in np.nonzero(avail[j])[0]:
                    tr = 0.0
                    if env.job_cell[e, j] >= 0:
                        tr = env.tau_cells[e, env.job_cell[e, j],
                                           env.station_cell[e, m]]
                    st = max(tp, env.true_candidate_free_time[e, j],
                             env.true_mch_free_time[e, m])
                    floor = min(floor, st + tr + env.true_op_pt[e, o, m])
                if floor > ct_lb[o]:
                    last = env.job_last_op_id[e, j]
                    ct_lb[o:last + 1] += floor - ct_lb[o]
        else:
            v = env.event_veh[e]
            tp = _next_time_after(env, e, t, skip_vehicle=v)
            veh_free[v] = max(veh_free[v], tp)
            for j in np.nonzero(env.task_active[e])[0]:
                frm = env.task_from[e, j]
                to = env.task_to[e, j]
                rel = env.task_release[e, j]
                pickup = np.inf
                for w in range(env.n_veh):
                    pickup = min(pickup, max(
                        rel, veh_free[w] + env.tau_cells[e, env.veh_cell[e, w], frm]))
                o = env.task_dest_op[e, j]
                floor = pickup + env.tau_cells[e, frm, to] + \
                    env.true_op_pt[e, o, env.task_dest_mch[e, j]]
                if floor > ct_lb[o]:
                    last = env.job_last_op_id[e, j]      # pool slot j IS the job
                    ct_lb[o:last + 1] += floor - ct_lb[o]

        b = float(ct_lb.max())                                 # B_chain^pass
        bound = getattr(env, '_bound', None)
        if bound is not None:
            if bound.use_mch:
                b = max(b, float(bound._b_mch()[e]))
            if bound.use_veh and env.n_veh > 0:
                b = max(b, _b_veh_with(env, bound, e, veh_free))
        out[e] = b
    return out


def _b_veh_with(env, bound, e, veh_free):
    """bound._b_veh for env e with a modified vehicle availability vector."""
    W = 0.0
    for j in np.nonzero(env.task_active[e])[0]:
        W += env.tau_cells[e, env.task_from[e, j], env.task_to[e, j]]
    uncommitted = env.op_scheduled_flag[e] < 0.5
    mand = uncommitted & (env.tau_in_min[e] > 0)
    W += float(env.tau_in_min[e][mand].sum())
    if W <= 0:
        return 0.0
    if bound.flag_fleet_aggregate:
        return (float(veh_free.sum()) + W) / env.n_veh
    return float(veh_free.min()) + W / env.n_veh


def r_pass(env):
    """Counterfactual pass-reward r_t^pass = (B(s_t) - b_t) * inv_slope, [E].
    env.max_endTime holds B(s_t) (the env updates it AFTER each step, so call
    this BEFORE env.step at the decision point)."""
    b = pass_floor(env)
    return (env.max_endTime - b) * env.inv_slope
