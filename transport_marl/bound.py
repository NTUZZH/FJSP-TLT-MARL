"""Theorem-1 bound B(s) = max(B_chain, B_mch, B_veh) for FJSP-TL-T (P2).

All three components are lower bounds on the makespan of ANY feasible
completion of the partial schedule s (admissibility); B_chain is tight at
termination (it then equals the realized makespan), hence so is the max.

- B_chain: the env's travel-aware op_ct_lb (chain recursion, realized values
  for committed decisions, lower envelopes for uncommitted ones; in-transit
  ops carry release + loaded-tau + pt). Maintained incrementally by the env.
- B_mch: eligibility-group energetic bound. PPVC eligibility is type-pure,
  so machines partition into groups; remaining work W_g (min-eligible pt of
  unstarted ops) must fit into the group's availability windows:
  C >= (sum_m busy_until(m) + W_g) / |g|.  busy_until uses the same
  optimistic envelope for reserved machines (a lower bound on the true
  busy-until, so the bound stays admissible).
- B_veh (headline, flags OFF): earliest-vehicle-free + W_tr / |V| where W_tr
  sums minimal loaded times of remaining mandatory moves (pool tasks use
  their committed loaded tau; uncommitted transitions use delta_tr_min,
  counted only if mandatory i.e. delta_tr_min > 0).
  Tightening flags (E5 ablation, OFF for the headline theorem):
    flag_fleet_aggregate: (sum_v busy_until(v) + W_tr)/|V|  (dominates).
    flag_min_residual: + min residual processing over remaining moved ops.

Vectorized over the env batch; call value(env) after env state updates.
"""

import numpy as np


class TransportBound:
    def __init__(self, env, use_mch=True, use_veh=True,
                 flag_fleet_aggregate=False, flag_min_residual=False):
        self.env = env
        self.use_mch = use_mch
        self.use_veh = use_veh
        self.flag_fleet_aggregate = flag_fleet_aggregate
        self.flag_min_residual = flag_min_residual
        e = env
        E, M = e.number_of_envs, e.number_of_machines
        # eligibility groups: machines with identical eligible-op sets share a
        # group in PPVC (type-pure); build per-env machine->group ids
        self.group_id = np.zeros((E, M), dtype=int)
        self.n_groups = np.zeros(E, dtype=int)
        for k in range(E):
            sig = {}
            for m in range(M):
                key = e.process_relation[k, :, m].tobytes()
                if key not in sig:
                    sig[key] = len(sig)
                self.group_id[k, m] = sig[key]
            self.n_groups[k] = len(sig)
        # per-op group (via any eligible machine)
        self.op_group = np.zeros((E, e.number_of_ops), dtype=int)
        for k in range(E):
            for o in range(e.number_of_ops):
                m0 = np.nonzero(e.process_relation[k, o])[0][0]
                self.op_group[k, o] = self.group_id[k, m0]

    def value(self):
        e = self.env
        B = np.max(e.op_ct_lb, axis=1)                      # B_chain
        if self.use_mch:
            B = np.maximum(B, self._b_mch())
        if self.use_veh and e.n_veh > 0:
            B = np.maximum(B, self._b_veh())
        return B

    def _b_mch(self):
        e = self.env
        E = e.number_of_envs
        out = np.zeros(E)
        # unstarted = not scheduled (committed-in-transit ops already occupy
        # their machine via busy_until, so exclude them from W_g)
        unstarted = e.op_scheduled_flag < 0.5                # [E, N]
        for k in range(E):
            G = self.n_groups[k]
            busy = np.zeros(G); size = np.zeros(G); W = np.zeros(G)
            np.add.at(busy, self.group_id[k], e.true_mch_free_time[k])
            np.add.at(size, self.group_id[k], 1.0)
            if unstarted[k].any():
                np.add.at(W, self.op_group[k][unstarted[k]],
                          e.true_op_min_pt[k][unstarted[k]])
            out[k] = np.max((busy + W) / size)
        return out

    def _b_veh(self):
        e = self.env
        E = e.number_of_envs
        out = np.zeros(E)
        for k in range(E):
            # remaining mandatory loaded work
            W = 0.0
            res_pt = np.inf
            # pool tasks: committed loaded tau
            for j in np.nonzero(e.task_active[k])[0]:
                W += e.tau_cells[k, e.task_from[k, j], e.task_to[k, j]]
                if self.flag_min_residual:
                    o = e.task_dest_op[k, j]
                    res_pt = min(res_pt, e.true_op_pt[k, o, e.task_dest_mch[k, j]])
            # uncommitted transitions: static min travel, mandatory only.
            # tau_in_min[o] is refreshed to the realized-origin min when the
            # predecessor completes, and delta_tr_min at seed time; ops whose
            # predecessor is not realized use the static value (still a valid
            # lower envelope of that move).
            uncommitted = e.op_scheduled_flag[k] < 0.5
            mand = uncommitted & (e.tau_in_min[k] > 0)
            W += float(e.tau_in_min[k][mand].sum())
            if self.flag_min_residual and mand.any():
                res_pt = min(res_pt, float(e.true_op_min_pt[k][mand].min()))
            if W <= 0:
                out[k] = 0.0
                continue
            if self.flag_fleet_aggregate:
                base = float(e.veh_free[k].sum())
                out[k] = (base + W) / e.n_veh
            else:
                out[k] = float(e.veh_free[k].min()) + W / e.n_veh
            if self.flag_min_residual and np.isfinite(res_pt):
                out[k] += res_pt
        return out
