"""Batched event-driven FJSP-TL-T environment (Paper X2, P1).

Extends the lab's FJSPEnvForSameOpNums with a vehicle fleet and transport
tasks under the v1 decision model (notes/design_v1.md §2):

- event_type[e] == 0 (machine-class event): action = flat job*M + mch pair,
  the parent's encoding, restricted to pairs whose max(job_ready, mch_free)
  equals the event time, machine not reserved, job not in transit.
  Committing to a different cell creates a transport task and RESERVES the
  machine until the job arrives.
- event_type[e] == 1 (vehicle event): the acting vehicle is event_veh[e];
  action = job index of a pending task (task pool is slot-indexed by job).

Design choice (single true clock): ALL dynamics, event times, and action
masks run on raw ("true") time; feature channels are scaled by inv_slope
(= 1 / pt normalization span) at construction so magnitudes match the
Paper-1 feature pipeline. This avoids any float divergence between a
normalized clock and the true clock (ties must break identically in the
reference simulator and here).

Episode lengths differ per env (N commitments + #transports). step() ignores
actions for finished envs; the rollout loop must route actions by
event_type and skip done envs.

Reward: telescoped bound difference r = (B(s_t) - B(s_{t+1})) * inv_slope
with B = B_chain (travel-aware op_ct_lb, true units). The full Theorem-1
B = max(B_chain, B_mch, B_veh) plugs in via _bound_value() (P2, bound.py).

E0a note: this class is NOT a code path of the Paper-1 pipeline; the E0a
anchor runs the parent class itself. A tau==0 & ample-fleet configuration
of THIS class must match no-transport makespans exactly (unit test), which
it does because zero travel makes every task serve instantly at release.
"""

import sys
import numpy as np
import numpy.ma as ma
import copy
import torch
from dataclasses import dataclass

sys.path.insert(0, '.')
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums, EnvState


@dataclass
class TransportEnvState(EnvState):
    """EnvState + vehicle nodes, vehicle-action grid, event routing info."""
    fea_v_tensor: torch.Tensor = None            # [E, V, 4]
    fea_veh_pairs_tensor: torch.Tensor = None    # [E, J, 6]
    veh_action_mask_tensor: torch.Tensor = None  # [E, J] True = masked
    event_type_tensor: torch.Tensor = None       # [E] 0 mch / 1 veh
    event_veh_tensor: torch.Tensor = None        # [E] acting vehicle or -1
    task_dest_op_tensor: torch.Tensor = None     # [E, J] dest op of task or -1

    def update_transport(self, fea_v, fea_veh_pairs, veh_action_mask,
                         event_type, event_veh, task_dest_op):
        device = self.device
        self.fea_v_tensor = torch.from_numpy(np.copy(fea_v)).float().to(device)
        self.fea_veh_pairs_tensor = torch.from_numpy(np.copy(fea_veh_pairs)).float().to(device)
        self.veh_action_mask_tensor = torch.from_numpy(np.copy(veh_action_mask)).to(device)
        self.event_type_tensor = torch.from_numpy(np.copy(event_type)).long().to(device)
        self.event_veh_tensor = torch.from_numpy(np.copy(event_veh)).long().to(device)
        self.task_dest_op_tensor = torch.from_numpy(np.copy(task_dest_op)).long().to(device)


class FJSPEnvTransport(FJSPEnvForSameOpNums):
    """Same-op-nums batched env with transport coupling (always enabled)."""

    def __init__(self, n_j, n_m, use_lag_features=True, use_guide=False,
                 guide_price='certified', guide_price_scale=1.0):
        super().__init__(n_j, n_m, use_lag_features=use_lag_features)
        # +2 transport op channels (tau_in_min, tau_in_max),
        # +2 machine channels (reserved flag, outbound task count)
        self.op_fea_dim += 2
        self.mch_fea_dim += 2
        self.veh_fea_dim = 4
        # bound-guided action prior (guide.py): one extra per-candidate
        # channel on BOTH pair grids when enabled
        self.use_guide = use_guide
        # 'certified' = admissible Theorem-1 price (headline); 'naive' = the
        # non-admissible control. The CHANNEL COUNT is identical either way,
        # so the network and its parameter count are untouched.
        self.guide_price = guide_price
        self.guide_price_scale = float(guide_price_scale)
        self.veh_pair_dim = 6 + (1 if use_guide else 0)

    # ------------------------------------------------------------------
    # data plumbing
    # ------------------------------------------------------------------

    def set_initial_data(self, job_length_list, op_pt_list, time_lag_list=None,
                         op_type_list=None, mch_type_list=None, layout_list=None):
        assert layout_list is not None, 'transport env needs layout_list'
        E = len(job_length_list)
        N = op_pt_list[0].shape[0]
        M0 = op_pt_list[0].shape[1]
        J0 = len(job_length_list[0])
        self.n_veh = int(layout_list[0]['n_vehicles'])
        assert all(int(l['n_vehicles']) == self.n_veh for l in layout_list)
        self.station_cell = np.array([l['station_cell'] for l in layout_list])  # [E, M]
        self.tau_cells = np.array([l['tau_cells'] for l in layout_list])        # [E, C, C] TRUE units
        self.veh_start_cell = np.array([int(l.get('veh_start_cell', 0)) for l in layout_list])
        # external (Link JSSPT) semantics: jobs materialize at a start cell
        # (in-buf) so the FIRST op also needs a transport; -1 = PPVC default
        # (job appears at its first machine, no initial move)
        self.job_start_cell = np.array([int(l.get('job_start_cell', -1))
                                        for l in layout_list])
        self.cell_xy = np.array(layout_list[0].get(
            'cell_xy', [[i // 3, i % 3] for i in range(9)]), dtype=float)

        # placeholders so the parent's feature calls (which hit our overrides)
        # survive before _init_transport_vars runs
        self.inv_slope = 1.0
        self.tau_in_min = np.zeros((E, N))
        self.tau_in_max = np.zeros((E, N))
        self.n_realized = np.zeros(E, dtype=int)
        self.mch_reserved_op = np.full((E, M0), -1, dtype=int)
        self.task_active = np.zeros((E, J0), dtype=bool)
        self.task_from = np.zeros((E, J0), dtype=int)

        state = super().set_initial_data(job_length_list, op_pt_list,
                                         time_lag_list, op_type_list,
                                         mch_type_list)
        assert self.pt_lower_bound == 0, 'transport env requires shift-free pt norm'
        self.inv_slope = 1.0 / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)

        # static min/max travel INTO each op (over eligible pred x succ cells),
        # TRUE units; 0 for job-first ops. [E, N]
        self.delta_tr_min = np.zeros((E, self.number_of_ops))
        self.delta_tr_max = np.zeros((E, self.number_of_ops))
        elig = self.process_relation
        for k in range(E):
            for j in range(self.number_of_jobs):
                first, last = self.job_first_op_id[k][j], self.job_last_op_id[k][j]
                for o in range(first + 1, last + 1):
                    ca = self.station_cell[k][elig[k, o - 1]]
                    cb = self.station_cell[k][elig[k, o]]
                    block = self.tau_cells[k][np.ix_(ca, cb)]
                    self.delta_tr_min[k, o] = block.min()
                    self.delta_tr_max[k, o] = block.max()
                if self.job_start_cell[k] >= 0:
                    # first op's incoming move (start cell -> machine) is
                    # mandatory under external semantics
                    cb = self.station_cell[k][elig[k, first]]
                    block = self.tau_cells[k][self.job_start_cell[k], cb]
                    self.delta_tr_min[k, first] = block.min()
                    self.delta_tr_max[k, first] = block.max()

        # travel-aware chain bound seed, TRUE units:
        # ct_lb[o] = cumsum(true_min_pt[o] + true_lag[o-1] + delta_tr_min[o])
        self.true_op_min_pt = np.min(ma.array(self.true_op_pt,
                                              mask=self.reverse_process_relation),
                                     axis=-1).data
        self.op_ct_lb = np.copy(self.true_op_min_pt)
        for k in range(E):
            for j in range(self.number_of_jobs):
                first, last = self.job_first_op_id[k][j], self.job_last_op_id[k][j]
                seg = self.op_ct_lb[k][first:last + 1]
                seg[0] += self.delta_tr_min[k][first]   # 0 unless job_start_cell
                seg[1:] += self.true_op_lag[k][first:last] + self.delta_tr_min[k][first + 1:last + 1]
                self.op_ct_lb[k][first:last + 1] = np.cumsum(seg)
        self.init_quality = np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.copy(self.init_quality)

        self._init_transport_vars()
        self._rebuild_all_features()

        # snapshots for reset()
        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_init_quality = np.copy(self.init_quality)

        self.old_state = TransportEnvState()
        self.old_state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask,
                              self.dynamic_pair_mask, self.comp_idx, self.candidate,
                              self.fea_pairs, op_type=self.op_type, mch_type=self.mch_type)
        self.old_state.update_transport(self.fea_v, self.fea_veh_pairs,
                                        self.veh_action_mask, self.event_type,
                                        self.event_veh, self.task_dest_op)
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def _init_transport_vars(self):
        E, J, M, V = (self.number_of_envs, self.number_of_jobs,
                      self.number_of_machines, self.n_veh)
        N = self.number_of_ops
        # TRUE-clock dynamic state (candidate/mch free times reuse the parent's
        # true_* arrays as the canonical ones)
        self.veh_free = np.zeros((E, V))
        self.veh_cell = np.tile(self.veh_start_cell[:, None], (1, V))
        # jobs start at job_start_cell (external semantics) or unmaterialized
        self.job_cell = np.tile(self.job_start_cell[:, None], (1, J)).astype(int)
        self.mch_reserved_op = np.full((E, M), -1, dtype=int)
        self.task_active = np.zeros((E, J), dtype=bool)
        self.task_from = np.zeros((E, J), dtype=int)
        self.task_to = np.zeros((E, J), dtype=int)
        self.task_release = np.zeros((E, J))
        self.task_dest_op = np.full((E, J), -1, dtype=int)
        self.task_dest_mch = np.full((E, J), -1, dtype=int)
        self.job_waiting_arrival = np.zeros((E, J), dtype=bool)
        self.op_in_transit = np.zeros((E, N), dtype=bool)
        self.n_realized = np.zeros(E, dtype=int)
        self.tau_in_min = np.copy(self.delta_tr_min)
        self.tau_in_max = np.copy(self.delta_tr_max)
        self.rec_assigned_mch = np.full((E, N), -1, dtype=int)
        self.rec_op_start = np.full((E, N), -1.0)
        self.rec_transports = [[] for _ in range(E)]
        if hasattr(self, '_frozen'):
            del self._frozen
        self._compute_events()

    def reset(self):
        super().reset()
        self.max_endTime = np.copy(self.init_quality)
        self._init_transport_vars()
        self._rebuild_all_features()
        if getattr(self, '_bound', None) is not None:
            self.max_endTime = self._bound_value()
        self.state = copy.deepcopy(self.old_state)
        return self.state

    # ------------------------------------------------------------------
    # event computation (TRUE clock)
    # ------------------------------------------------------------------

    def _compute_events(self):
        E = self.number_of_envs
        candFT = np.expand_dims(self.true_candidate_free_time, 2)
        mchFT = np.expand_dims(self.true_mch_free_time, 1)
        self.true_pair_free_time = np.maximum(candFT, mchFT)   # [E, J, M]
        blocked = np.copy(self.candidate_process_relation)     # True = unusable
        blocked = blocked | (self.mch_reserved_op >= 0)[:, None, :]
        blocked = blocked | self.job_waiting_arrival[:, :, None]
        pft = ma.array(self.true_pair_free_time, mask=blocked)
        t_M = pft.reshape(E, -1).min(axis=1).filled(np.inf)

        rel = ma.array(self.task_release, mask=~self.task_active)
        min_rel = rel.min(axis=1).filled(np.inf)
        t_per_veh = np.maximum(self.veh_free, min_rel[:, None])
        ev_veh = np.argmin(t_per_veh, axis=1)
        t_V = t_per_veh[np.arange(E), ev_veh]
        t_V = np.where(np.isinf(min_rel), np.inf, t_V)

        self.event_type = np.where(t_M <= t_V, 0, 1).astype(int)  # mch wins ties
        done_mask = self.n_realized >= self.number_of_ops
        self.event_type[done_mask] = 0
        self.event_veh = np.where(self.event_type == 1, ev_veh, -1).astype(int)
        self.next_event_time = np.where(self.event_type == 0, t_M, t_V)
        self.next_event_time[done_mask] = self.current_makespan[done_mask]
        self._blocked_pairs = blocked

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def step(self, actions):
        """actions [E]: flat job*M+mch where event_type==0, job idx where 1.
        Done envs ignore their action (episode lengths differ across envs)."""
        actions = np.asarray(actions)
        alive = self.n_realized < self.number_of_ops
        idx_m = np.nonzero((self.event_type == 0) & alive)[0]
        idx_v = np.nonzero((self.event_type == 1) & alive)[0]
        if len(idx_m):
            self._step_machine(idx_m, actions[idx_m])
        if len(idx_v):
            self._step_vehicle(idx_v, actions[idx_v])
        self._after_step()
        new_bound = self._bound_value()
        reward = (self.max_endTime - new_bound) * self.inv_slope
        self.max_endTime = new_bound
        self.state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs, op_type=self.op_type, mch_type=self.mch_type)
        self.state.update_transport(self.fea_v, self.fea_veh_pairs,
                                    self.veh_action_mask, self.event_type,
                                    self.event_veh, self.task_dest_op)
        return self.state, np.array(reward), self.done()

    def attach_bound(self, bound):
        """Plug in the Theorem-1 bound object (transport_marl/bound.py).
        Re-bases max_endTime to B(s_current) so telescoping stays exact.
        With the guide channel on, the initial features were priced against
        the chain-only fallback, so rebuild them (and the reset snapshot)
        against the full bound."""
        self._bound = bound
        self.max_endTime = self._bound_value()
        if self.use_guide:
            self._rebuild_all_features()
            self.old_state.update(self.fea_j, self.op_mask, self.fea_m,
                                  self.mch_mask, self.dynamic_pair_mask,
                                  self.comp_idx, self.candidate, self.fea_pairs,
                                  op_type=self.op_type, mch_type=self.mch_type)
            self.old_state.update_transport(self.fea_v, self.fea_veh_pairs,
                                            self.veh_action_mask, self.event_type,
                                            self.event_veh, self.task_dest_op)
            self.state = copy.deepcopy(self.old_state)

    def _bound_value(self):
        if getattr(self, '_bound', None) is not None:
            return self._bound.value()
        return np.max(self.op_ct_lb, axis=1)   # B_chain only (true units)

    def done(self):
        return (self.n_realized >= self.number_of_ops).astype(float)

    def _step_machine(self, idx, actions):
        M = self.number_of_machines
        job = actions // M
        mch = actions % M
        op = self.candidate[idx, job]
        if (self.reverse_process_relation[idx, op, mch]).any():
            raise RuntimeError('illegal machine action: incompatible pair')
        if (self.mch_reserved_op[idx, mch] >= 0).any() or self.job_waiting_arrival[idx, job].any():
            raise RuntimeError('illegal machine action: reserved mch / waiting job')

        commit_t = np.maximum(self.true_candidate_free_time[idx, job],
                              self.true_mch_free_time[idx, mch])
        self.rec_assigned_mch[idx, op] = mch
        dest = self.station_cell[idx, mch]
        cur = self.job_cell[idx, job]
        needs_tr = (cur >= 0) & (cur != dest)

        # candidate bookkeeping (parent logic on subset)
        add_flag = (op != self.job_last_op_id[idx, job])
        self.candidate[idx, job] += add_flag
        self.mask[idx, job] = (1 - add_flag)
        am = add_flag
        self.candidate_pt[idx[am], job[am]] = self.unmasked_op_pt[idx[am], op[am] + 1]
        self.candidate_process_relation[idx[am], job[am]] = \
            self.reverse_process_relation[idx[am], op[am] + 1]
        self.candidate_process_relation[idx[~am], job[~am]] = 1
        self.op_scheduled_flag[idx, op] = 1
        self.remain_process_relation[idx, op] = 0
        self.mch_current_available_op_nums[idx] -= self.process_relation[idx, op]
        self._op_match_stats_update(idx, job, op)

        d = ~needs_tr
        if d.any():
            self._realize_op(idx[d], job[d], op[d], mch[d], commit_t[d])
        if needs_tr.any():
            it, jt, ot, mt = idx[needs_tr], job[needs_tr], op[needs_tr], mch[needs_tr]
            ct_t = commit_t[needs_tr]
            self.task_active[it, jt] = True
            self.task_from[it, jt] = self.job_cell[it, jt]
            self.task_to[it, jt] = dest[needs_tr]
            self.task_release[it, jt] = ct_t
            self.task_dest_op[it, jt] = ot
            self.task_dest_mch[it, jt] = mt
            self.mch_reserved_op[it, mt] = ot
            self.job_waiting_arrival[it, jt] = True
            self.op_in_transit[it, ot] = True
            # OPTIMISTIC lower envelope for the in-transit op:
            # release + loaded tau + pt  (empty move >= 0, vehicle queue >= 0)
            tau_l = self.tau_cells[it, self.task_from[it, jt], self.task_to[it, jt]]
            opt_ct = ct_t + tau_l + self.true_op_pt[it, ot, mt]
            self.true_op_ct[it, ot] = opt_ct
            self.true_mch_free_time[it, mt] = opt_ct
            self.true_candidate_free_time[it, jt] = opt_ct + self.true_op_lag[it, ot]
            self._raise_ct_lb(it, jt, ot, opt_ct)

    # ------------------------------------------------------------------
    # injected-state entry (transport_marl/disruption.py)
    # ------------------------------------------------------------------

    def force_realize(self, e, job, op, mch, start):
        """Commit and realize one operation at a GIVEN start time.

        Bypasses the event machinery so a frozen prefix of an already-executed
        schedule can be loaded into a fresh env; every derived array is then
        updated by the same code a normal commit uses. The operation must be
        its job's current candidate, and the module must already be at the
        machine's cell (frozen prefixes are replayed in start-time order, so it
        always is), hence no transport task is created here.
        """
        idx = np.array([e])
        jb = np.array([job])
        op_a = np.array([op])
        mch_a = np.array([mch])
        assert self.candidate[e, job] == op, \
            f'force_realize out of order: candidate {self.candidate[e, job]} != op {op}'
        assert not self.reverse_process_relation[e, op, mch], \
            f'force_realize on incompatible pair (op {op}, machine {mch})'
        self.rec_assigned_mch[e, op] = mch
        add_flag = (op_a != self.job_last_op_id[idx, jb])
        self.candidate[idx, jb] += add_flag
        self.mask[idx, jb] = (1 - add_flag)
        am = add_flag
        self.candidate_pt[idx[am], jb[am]] = self.unmasked_op_pt[idx[am], op_a[am] + 1]
        self.candidate_process_relation[idx[am], jb[am]] = \
            self.reverse_process_relation[idx[am], op_a[am] + 1]
        self.candidate_process_relation[idx[~am], jb[~am]] = 1
        self.op_scheduled_flag[e, op] = 1
        self.remain_process_relation[e, op] = 0
        self.mch_current_available_op_nums[idx] -= self.process_relation[idx, op_a]
        self._op_match_stats_update(idx, jb, op_a)
        self._realize_op(idx, jb, op_a, mch_a, np.array([float(start)]))

    def refresh_state(self):
        """Recompute events, features and the exported state after an external
        state injection. Same tail as step(), without the reward telescope;
        max_endTime is re-based so a later step() still telescopes exactly."""
        self._compute_events()
        self._rebuild_all_features()
        self.max_endTime = self._bound_value()
        self.state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs, op_type=self.op_type,
                          mch_type=self.mch_type)
        self.state.update_transport(self.fea_v, self.fea_veh_pairs,
                                    self.veh_action_mask, self.event_type,
                                    self.event_veh, self.task_dest_op)
        return self.state

    def _step_vehicle(self, idx, actions):
        veh = self.event_veh[idx]
        job = actions
        if not self.task_active[idx, job].all():
            raise RuntimeError('illegal vehicle action: no active task for job')
        t = np.maximum(self.veh_free[idx, veh], self.task_release[idx, job])
        frm = self.task_from[idx, job]
        to = self.task_to[idx, job]
        tau_e = self.tau_cells[idx, self.veh_cell[idx, veh], frm]
        tau_l = self.tau_cells[idx, frm, to]
        arrival = t + tau_e + tau_l
        op = self.task_dest_op[idx, job]
        mch = self.task_dest_mch[idx, job]
        for w, e in enumerate(idx):
            self.rec_transports[e].append(dict(
                job=int(job[w]), op=int(op[w]), veh=int(veh[w]),
                frm=int(frm[w]), to=int(to[w]), depart=float(t[w]),
                pickup=float(t[w] + tau_e[w]), arrival=float(arrival[w])))
        self.veh_free[idx, veh] = arrival
        self.veh_cell[idx, veh] = to
        self.task_active[idx, job] = False
        self.mch_reserved_op[idx, mch] = -1
        self.job_waiting_arrival[idx, job] = False
        self.op_in_transit[idx, op] = False
        self._realize_op(idx, job, op, mch, arrival)

    def _realize_op(self, idx, job, op, mch, start):
        ct = start + self.true_op_pt[idx, op, mch]
        self.rec_op_start[idx, op] = start
        self.true_op_ct[idx, op] = ct
        self.true_candidate_free_time[idx, job] = ct + self.true_op_lag[idx, op]
        self.true_mch_free_time[idx, mch] = ct
        self.current_makespan[idx] = np.maximum(self.current_makespan[idx], ct)
        self.job_cell[idx, job] = self.station_cell[idx, mch]
        self.n_realized[idx] += 1
        self._raise_ct_lb(idx, job, op, ct)
        # travel-in bounds for the job's NEXT op tighten: origin now known
        nxt = op + 1
        has_next = nxt <= self.job_last_op_id[idx, job]
        if has_next.any():
            ih, oh = idx[has_next], nxt[has_next]
            cur_cell = self.job_cell[ih, job[has_next]]
            for w in range(len(ih)):
                e, o2 = ih[w], oh[w]
                cb = self.station_cell[e][self.process_relation[e, o2]]
                block = self.tau_cells[e][cur_cell[w], cb]
                self.tau_in_min[e, o2] = block.min()
                self.tau_in_max[e, o2] = block.max()

    def _raise_ct_lb(self, idx, job, op, new_ct):
        """Raise op's ct_lb to new_ct, propagate diff along the job suffix."""
        diff = new_ct - self.op_ct_lb[idx, op]
        last = self.job_last_op_id[idx, job]
        m = (self.op_idx >= op[:, None]) & (self.op_idx < (last + 1)[:, None])
        self.op_ct_lb[idx] += m * diff[:, None]

    def _op_match_stats_update(self, idx, job, op):
        first = self.job_first_op_id[idx, job]
        last = self.job_last_op_id[idx, job]
        m2 = (self.op_idx >= first[:, None]) & (self.op_idx < (last + 1)[:, None])
        self.op_match_job_left_op_nums[idx] -= m2
        self.op_match_job_remain_work[idx] -= m2 * self.op_mean_pt[idx, op][:, None]

    # ------------------------------------------------------------------
    # post-step recompute (features + next event)
    # ------------------------------------------------------------------

    def _after_step(self):
        self.step_count += 1
        self._compute_events()
        self._rebuild_all_features()

    def _rebuild_all_features(self):
        t = self.next_event_time[:, None]
        realized = self.op_scheduled_flag.astype(bool) & ~self.op_in_transit
        self.deleted_op_nodes = (self.true_op_ct <= t) & realized
        self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, None],
                                         (1, 1, self.op_fea_dim))
        self.update_op_mask()

        s = self.inv_slope
        prev_wait = self.op_waiting_time[self.env_job_idx, self.candidate]
        self.op_waiting_time = np.zeros_like(self.op_waiting_time)
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(self.next_event_time[:, None]
                                         - self.true_candidate_free_time, 0) * s + \
            self.mask * prev_wait
        self.op_remain_work = np.maximum(
            self.true_op_ct * self.op_scheduled_flag - t, 0) * s
        if self.use_lag_features:
            self.op_remain_lag = self.op_scheduled_flag * np.clip(
                self.true_op_ct + self.true_op_lag - t, 0, self.true_op_lag) * s
        self.construct_op_features()

        # machine action mask: blocked ∪ not-startable-at-event-time
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)
        self.unavailable_pairs = self.true_pair_free_time > self.next_event_time[:, None, None]
        self.dynamic_pair_mask = self.dynamic_pair_mask | self.unavailable_pairs
        self.dynamic_pair_mask = self.dynamic_pair_mask | self._blocked_pairs
        self.dynamic_pair_mask[self.event_type == 1] = True

        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.update_mch_mask()
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        mch_free_duration = (self.next_event_time[:, None] - self.true_mch_free_time) * s
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration
        self.mch_remain_work = np.maximum(-mch_free_duration, 0)
        self.mch_free_time = self.true_mch_free_time * s
        self.construct_mch_features()
        self.construct_pair_features()
        self._construct_vehicle_features()
        if self.use_guide:
            if getattr(self, 'guide_price', 'certified') == 'naive':
                from transport_marl.guide import naive_price_features
                gp, gt = naive_price_features(self)
            else:
                from transport_marl.guide import guide_features
                gp, gt = guide_features(self)
            self.fea_pairs = np.concatenate(
                [self.fea_pairs, gp[:, :, :, None]], axis=3)
            self.fea_veh_pairs = np.concatenate(
                [self.fea_veh_pairs, gt[:, :, None]], axis=2)

        # freeze features of DONE envs at their last alive values: with every
        # node deleted, the house z-norm divides by zero (NaN) and NaN rows
        # poison pooling / the PPO update even though their losses are masked
        # (NaN * 0 = NaN in torch). Parent never hits this (synchronized
        # termination); here episode lengths differ per env.
        _frz_names = ('fea_j', 'fea_m', 'fea_pairs', 'fea_v', 'fea_veh_pairs',
                      'op_mask', 'mch_mask', 'comp_idx')
        done_envs = self.n_realized >= self.number_of_ops
        if done_envs.any() and hasattr(self, '_frozen'):
            for k in np.nonzero(done_envs)[0]:
                for name in _frz_names:
                    getattr(self, name)[k] = self._frozen[name][k]
        self._frozen = {name: np.copy(getattr(self, name)) for name in _frz_names}

    def construct_op_features(self):
        s = self.inv_slope
        feature_channels = [self.op_scheduled_flag, self.op_ct_lb * s,
                            self.op_min_pt, self.pt_span, self.op_mean_pt,
                            self.op_waiting_time, self.op_remain_work,
                            self.op_match_job_left_op_nums,
                            self.op_match_job_remain_work,
                            self.op_available_mch_nums]
        if self.use_lag_features:
            feature_channels.append(self.op_lag)
            feature_channels.append(self.op_remain_lag)
        feature_channels.append(self.tau_in_min * s)
        feature_channels.append(self.tau_in_max * s)
        self.fea_j = np.stack(feature_channels, axis=2)
        if self.n_realized.min() < self.number_of_ops:
            self.norm_op_features()

    def construct_mch_features(self):
        outbound = np.zeros((self.number_of_envs, self.number_of_machines))
        for e in range(self.number_of_envs):
            for j in np.nonzero(self.task_active[e])[0]:
                outbound[e] += (self.station_cell[e] == self.task_from[e, j])
        self.fea_m = np.stack((self.mch_current_available_jc_nums,
                               self.mch_current_available_op_nums,
                               self.mch_min_pt, self.mch_mean_pt,
                               self.mch_waiting_time, self.mch_remain_work,
                               self.mch_free_time, self.mch_working_flag,
                               (self.mch_reserved_op >= 0).astype(float),
                               outbound), axis=2)
        if self.n_realized.min() < self.number_of_ops:
            self.norm_machine_features()

    def construct_pair_features(self):
        # pair_wait_time inside uses op_waiting_time/mch_waiting_time already
        # in scaled units; candidate_pt is normalized pt — parent logic reused.
        super().construct_pair_features()

    def norm_op_features(self):
        """Parent z-norm with a zero-safe divisor: an alive env can have every
        op node deleted-or-in-transit only transiently, but a DONE env in a
        mixed batch has zero live nodes and the parent divides 0/0."""
        self.fea_j[self.delete_mask_fea_j] = 0
        num_delete = np.count_nonzero(self.deleted_op_nodes, axis=1)[:, None]
        num_left = np.maximum(self.number_of_ops - num_delete, 1)
        mean = np.sum(self.fea_j, axis=1) / num_left
        temp = np.where(self.delete_mask_fea_j, mean[:, None, :], self.fea_j)
        var = np.var(temp, axis=1)
        std = np.sqrt(var * self.number_of_ops / num_left)
        self.fea_j = (temp - mean[:, None, :]) / (std[:, None, :] + 1e-8)

    def norm_machine_features(self):
        """Parent z-norm with a zero-safe divisor: when every op of an alive
        env is committed (some in transit), every machine node is deleted and
        the parent divides 0/0 (its termination is synchronized; ours isn't)."""
        self.fea_m[self.delete_mask_fea_m] = 0
        num_delete = np.count_nonzero(self.delete_mask_fea_m[:, :, 0], axis=1)[:, None]
        num_left = np.maximum(self.number_of_machines - num_delete, 1)
        mean = np.sum(self.fea_m, axis=1) / num_left
        temp = np.where(self.delete_mask_fea_m, mean[:, None, :], self.fea_m)
        var = np.var(temp, axis=1)
        std = np.sqrt(var * self.number_of_machines / num_left)
        self.fea_m = (temp - mean[:, None, :]) / (std[:, None, :] + 1e-8)

    def _construct_vehicle_features(self):
        E, J = self.number_of_envs, self.number_of_jobs
        s = self.inv_slope
        xy = self.cell_xy[self.veh_cell] / 2.0                      # [E, V, 2]
        busy = np.maximum(self.veh_free - self.next_event_time[:, None], 0) * s
        moving = (busy > 1e-12).astype(float)
        self.fea_v = np.concatenate([xy, busy[:, :, None], moving[:, :, None]], axis=2)

        self.veh_action_mask = ~self.task_active
        self.veh_action_mask[self.event_type == 0] = True
        av = np.maximum(self.event_veh, 0)
        vloc = self.veh_cell[np.arange(E), av]
        tau_e = self.tau_cells[np.arange(E)[:, None], vloc[:, None], self.task_from] * s
        tau_l = self.tau_cells[np.arange(E)[:, None], self.task_from, self.task_to] * s
        wait = np.maximum(self.next_event_time[:, None] - self.task_release, 0) * s
        cand_clip = np.minimum(self.candidate, self.number_of_ops - 1)
        jrw = self.op_match_job_remain_work[self.env_job_idx, cand_clip]
        jlo = self.op_match_job_left_op_nums[self.env_job_idx, cand_clip]
        dest_pt = np.zeros((E, J))
        act = self.task_active
        if act.any():
            ee, jj = np.nonzero(act)
            dest_pt[ee, jj] = self.true_op_pt[ee, self.task_dest_op[ee, jj],
                                              self.task_dest_mch[ee, jj]] * s
        feats = np.stack([tau_e, tau_l, wait, jrw, jlo, dest_pt], axis=2)
        feats[~act] = 0.0
        self.fea_veh_pairs = feats

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------

    def schedule_record(self, e):
        return dict(assigned_mch=self.rec_assigned_mch[e].copy(),
                    op_start=self.rec_op_start[e].copy(),
                    op_ct=self.true_op_ct[e].copy(),
                    transports=[dict(tr) for tr in self.rec_transports[e]],
                    makespan=float(self.current_makespan[e]))
