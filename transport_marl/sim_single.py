"""Single-instance event-driven simulator for FJSP-TL-T (reference oracle).

Implements the v1 decision model pinned in notes/design_v1.md §2:
- Machine-class event (global DANIEL-style pair pick): fires at
  t_M = min over eligible (ready job, machine) pairs of max(job_ready, mch_free);
  action space = all pairs achieving <= t_M. Committing to a different cell
  creates a transport task (released now) and RESERVES the machine until the
  job arrives (v1 reservation semantics).
- Vehicle event (per-vehicle): fires at t_v = max(veh_free[v], earliest task
  release); action space = the released task pool (at most one task per job).
- Event order: (time, class[mch=0 < veh=1], index). No no-op/defer.

This module is plain numpy, torch-free. The batched training env must agree
with it on random instances (unit-tested), and every schedule it emits must
pass transport_marl/validator_t.py (which shares no code with this file).

A simulator can also be pre-loaded with a mid-execution state through
load_residual_state(), so a rollout continues from a disrupted schedule
instead of from an empty factory (transport_marl/disruption.py builds the
state). reset() then restores that state, which is what makes the PDR rules
and the GA decode re-plan a residual problem without knowing about it.
"""

import numpy as np

EPS = 1e-9


class TransportSim:
    def __init__(self, job_length, op_pt, time_lag, station_cell, tau_cells,
                 n_vehicles, veh_start_cell=0):
        self.job_length = np.asarray(job_length, dtype=int)
        self.op_pt = np.asarray(op_pt, dtype=float)
        self.time_lag = np.asarray(time_lag, dtype=float)
        self.station_cell = np.asarray(station_cell, dtype=int)
        self.tau = np.asarray(tau_cells, dtype=float)
        self.n_j = len(self.job_length)
        self.n_m = self.op_pt.shape[1]
        self.n_v = int(n_vehicles)
        self.veh_start_cell = int(veh_start_cell)
        self.n_ops = int(self.job_length.sum())
        self.first_op = np.zeros(self.n_j, dtype=int)
        self.first_op[1:] = np.cumsum(self.job_length)[:-1]
        self.last_op = self.first_op + self.job_length - 1
        self.job_of_op = np.repeat(np.arange(self.n_j), self.job_length)
        self.reset()

    def reset(self):
        J, M, V, N = self.n_j, self.n_m, self.n_v, self.n_ops
        self.next_op = self.first_op.copy()          # global idx of next uncommitted op; -1 = job done committing
        self.job_ready = np.zeros(J)                 # time next op becomes commit-ready
        self.job_loc = np.full(J, -1, dtype=int)     # cell; -1 = not materialized
        self.job_waiting_arrival = np.zeros(J, dtype=bool)
        self.mch_free = np.zeros(M)
        self.mch_reserved = np.full(M, -1, dtype=int)  # op idx reserved for, -1 = none
        self.veh_free = np.zeros(V)
        self.veh_loc = np.full(V, self.veh_start_cell, dtype=int)
        self.pool = {}                                # job -> dict(from,to,release,op,mch)
        self.assigned_mch = np.full(N, -1, dtype=int)
        self.assigned_veh = np.full(N, -1, dtype=int)  # vehicle that moved the job TO this op
        self.op_start = np.full(N, -1.0)
        self.op_ct = np.full(N, -1.0)
        self.transports = []
        self.n_committed = 0
        self.n_completed = 0
        self.now = 0.0
        res = getattr(self, '_residual', None)
        if res is not None:
            self._apply_residual(res)

    # ---------- residual (mid-execution) state ----------

    def load_residual_state(self, residual):
        """Pre-load the mid-execution state built by disruption.build_residual.

        After this call the simulator holds the frozen prefix of a disrupted
        schedule and every subsequent reset() restores it, so run() and the GA
        decode roll out the residual problem. schedule_record() then returns
        the FULL recovered schedule, frozen operations included.
        """
        self._residual = residual
        self.reset()

    def _apply_residual(self, res):
        self.next_op = np.asarray(res['next_op'], dtype=int).copy()
        self.job_ready = np.asarray(res['job_ready'], dtype=float).copy()
        self.job_loc = np.asarray(res['job_loc'], dtype=int).copy()
        self.mch_free = np.asarray(res['mch_free'], dtype=float).copy()
        self.veh_free = np.asarray(res['veh_free'], dtype=float).copy()
        self.veh_loc = np.asarray(res['veh_cell'], dtype=int).copy()
        self.assigned_mch = np.asarray(res['assigned_mch'], dtype=int).copy()
        self.assigned_veh = np.asarray(res['assigned_veh'], dtype=int).copy()
        self.op_start = np.asarray(res['op_start'], dtype=float).copy()
        self.op_ct = np.asarray(res['op_ct'], dtype=float).copy()
        self.transports = [dict(tr) for tr in res['transports']]
        self.n_committed = int(res['n_frozen'])
        self.n_completed = int(res['n_frozen'])
        self.now = float(res['t'])

    # ---------- event machinery ----------

    def _machine_pairs(self):
        """(t_M, [(job, mch, op)]) over commit-ready jobs x usable machines."""
        best_t, pairs = np.inf, []
        for j in range(self.n_j):
            o = self.next_op[j]
            if o < 0 or self.job_waiting_arrival[j]:
                continue
            for m in np.nonzero(self.op_pt[o] > 0)[0]:
                if self.mch_reserved[m] >= 0:
                    continue
                t = max(self.job_ready[j], self.mch_free[m])
                best_t = min(best_t, t)
                pairs.append((t, j, int(m), int(o)))
        avail = [(j, m, o) for (t, j, m, o) in pairs if t <= best_t + EPS]
        return best_t, avail

    def _vehicle_event(self):
        """(t_v, veh) for the earliest vehicle able to serve a released task."""
        if not self.pool:
            return np.inf, -1
        min_rel = min(task['release'] for task in self.pool.values())
        best_t, best_v = np.inf, -1
        for v in range(self.n_v):
            t = max(self.veh_free[v], min_rel)
            if t < best_t - EPS:
                best_t, best_v = t, v
        return best_t, best_v

    def peek_event(self):
        """Next decision point: ('mch', t, pairs) | ('veh', t, v, tasks) | ('done',) ."""
        if self.n_completed == self.n_ops:
            return ('done',)
        t_m, pairs = self._machine_pairs()
        t_v, v = self._vehicle_event()
        if np.isinf(t_m) and np.isinf(t_v):
            raise RuntimeError('deadlock: no machine or vehicle event but not done')
        if pairs and t_m <= t_v + EPS:           # machine class wins ties
            return ('mch', t_m, pairs)
        tasks = sorted(self.pool.keys())
        return ('veh', t_v, v, [self.pool[j] for j in tasks])

    # ---------- actions ----------

    def commit(self, job, mch):
        """Machine-class action: commit job's next op to mch at the event time."""
        o = self.next_op[job]
        t = max(self.job_ready[job], self.mch_free[mch])
        self.now = t
        self.assigned_mch[o] = mch
        dest = self.station_cell[mch]
        same_cell = (self.job_loc[job] == -1) or (self.job_loc[job] == dest)
        if same_cell:
            self._start_op(job, o, mch, t)
        else:
            self.pool[job] = dict(job=job, op=int(o), mch=int(mch),
                                  frm=int(self.job_loc[job]), to=int(dest),
                                  release=t)
            self.mch_reserved[mch] = o
            self.mch_free[mch] = np.inf          # blocked until arrival
            self.job_waiting_arrival[job] = True
        self.n_committed += 1

    def serve(self, veh, job):
        """Vehicle action: vehicle veh serves job's pending task."""
        task = self.pool.pop(job)
        t = max(self.veh_free[veh], task['release'])
        self.now = t
        tau_e = self.tau[self.veh_loc[veh], task['frm']]
        tau_l = self.tau[task['frm'], task['to']]
        pickup = t + tau_e
        arrival = pickup + tau_l
        self.transports.append(dict(job=int(job), op=task['op'], veh=int(veh),
                                    frm=task['frm'], to=task['to'],
                                    depart=t, pickup=pickup, arrival=arrival))
        self.veh_free[veh] = arrival
        self.veh_loc[veh] = task['to']
        self.assigned_veh[task['op']] = veh
        m = task['mch']
        self.mch_reserved[m] = -1
        self.job_waiting_arrival[job] = False
        self._start_op(job, task['op'], m, arrival)

    def _start_op(self, job, o, mch, start):
        pt = self.op_pt[o, mch]
        ct = start + pt
        self.op_start[o] = start
        self.op_ct[o] = ct
        self.mch_free[mch] = ct
        self.job_loc[job] = self.station_cell[mch]
        self.n_completed += 1
        if o == self.last_op[job]:
            self.next_op[job] = -1
        else:
            self.next_op[job] = o + 1
            self.job_ready[job] = ct + self.time_lag[o]

    # ---------- drivers ----------

    def run(self, mch_rule, veh_rule):
        """Roll out with priority rules. mch_rule(sim, pairs) -> (job, mch);
        veh_rule(sim, veh, tasks) -> job. Returns makespan."""
        while True:
            ev = self.peek_event()
            if ev[0] == 'done':
                break
            if ev[0] == 'mch':
                j, m = mch_rule(self, ev[2])
                self.commit(j, m)
            else:
                job = veh_rule(self, ev[2], ev[3])
                self.serve(ev[2], job)
        return float(self.op_ct.max())

    def schedule_record(self):
        return dict(assigned_mch=self.assigned_mch.copy(),
                    assigned_veh=self.assigned_veh.copy(),
                    op_start=self.op_start.copy(),
                    op_ct=self.op_ct.copy(),
                    transports=[dict(tr) for tr in self.transports],
                    makespan=float(self.op_ct.max()))
