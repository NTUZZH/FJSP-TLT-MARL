"""CP-SAT reference model for FJSP-TL-T (Paper X2, proposal §8).

PPVC-specialized v1: operation eligibility is station-type-pure, so each
consecutive-op transition either never needs transport (same type cell) or
always does, with CONSTANT (from, to) cells and duration. What remains free
is machine choice within the type, op timing, vehicle assignment, and
vehicle sequencing (empty moves are sequence-dependent).

Model:
- per op: master interval + optional per-machine alternatives (house style);
  NoOverlap per machine.
- per needed move k (job j, ops o-1 -> o): interval [s_k, s_k + tau_k];
  s_k >= end_{o-1} + lag_{o-1}; start_o >= s_k + tau_k.
  (No machine reservation: CP-SAT searches the full schedule space, a
  superset of the env's constructive class - gaps reported vs this.)
- fleet: one AddCircuit per vehicle over {depot} + moves; arc (a -> b)
  implies s_b >= s_a + tau_a + tau_empty(to_a, from_b); a move is served by
  exactly one vehicle. Times are integerized by round(x * SCALE).

Entry: solve_transport_instance(jl, pt, meta, time_limit, ...) -> dict.

Residual mode (Paper X2 disruption experiment): six optional arguments turn
the same model into the residual of a partially executed schedule, without
touching a single default. See their docstring below; when all six are absent
the model built here is byte-identical to the one that produced
or_solution/PPVCT/*.jsonl.
"""

import numpy as np
from ortools.sat.python import cp_model

# speed constants are dyadic (k/128, see layout.calibrate_speed), so at
# SCALE=128 every integerization below is EXACT (round() is a formality)
SCALE = 128


def solve_transport_instance(job_length, op_pt, meta, time_limit=300.0,
                             n_workers=8, log=False, warmstart=None,
                             strengthen=False, energy_cut=True, fleet=True,
                             op_release=None, mch_ready=None, fixed_ops=None,
                             veh_ready=None, veh_cells=None,
                             delivered_ops=None):
    """RESIDUAL ARGUMENTS (all default None, and every one of them only ever
    ADDS constraints, so with all six absent the model, the parameters and the
    search are exactly what they were before they existed):
      op_release   [N]  earliest start time of each operation.
      mch_ready    [M]  earliest time each machine is available; a machine
                        down for repair until T carries mch_ready = T.
      fixed_ops    {op: (machine, start)} decisions already executed and not
                        open to the solver.
      veh_ready    [V]  earliest time each vehicle can depart.
      veh_cells    [V]  the cell each vehicle stands in at the start of the
                        residual problem (they no longer share one depot, so
                        vehicle symmetry breaking switches itself off unless
                        they happen to coincide).
      delivered_ops     operations whose incoming move already happened; no
                        move variable is created for them even though their
                        cell differs from their predecessor's.

    warmstart: optional schedule_record dict (sim/env format: assigned_mch,
    op_start, op_ct, transports) hinted via AddHint (proposal §8 warm-started
    variant). Hints must come from the SAME quantized time grid (dyadic).

    strengthen=False reproduces the v1 reference model exactly (do not change
    its semantics: or_solution/PPVCT/*.jsonl was produced with it).
    strengthen=True adds three sound, solution-preserving strengthenings:
      (a) horizon = warm-start makespan.  The warm start is a feasible schedule
          of the same instance, so its makespan is a valid upper bound and
          every optimal solution survives the tightened variable domains.
      (b) redundant fleet capacity: an AddCumulative of unit demand over the
          loaded-move intervals with capacity |V| (a vehicle is occupied for
          the whole loaded leg), plus makespan >= end of every loaded move and
          the energetic corollary |V| * makespan >= sum of loaded durations.
          Empty legs stay encoded in the circuit's arc precedences and are
          deliberately NOT given intervals: a vehicle waiting idle at a cell is
          not occupied, so charging it would be unsound.
      (c) vehicle symmetry breaking (only when |V| >= 2 and all vehicles share
          one start cell, which is the case here): vehicle v may serve move k
          only if vehicle v-1 serves some move with a smaller index.  The
          warm-start vehicle labels are canonicalized to match before hinting.
    """
    jl = np.asarray(job_length, dtype=int)
    pt = np.asarray(op_pt, dtype=float)
    tr = meta['transport']
    station_cell = np.asarray(tr['station_cell'], dtype=int)
    tau = np.asarray(tr['tau_cells'], dtype=float)
    n_veh = int(tr['n_vehicles'])
    depot = int(tr.get('veh_start_cell', 0))
    lag = np.asarray(meta['time_lag'], dtype=float)

    n_j = len(jl)
    n_ops, n_m = pt.shape
    first = np.zeros(n_j, dtype=int)
    first[1:] = np.cumsum(jl)[:-1]
    last = first + jl - 1

    def I(x):  # integerize
        return int(round(x * SCALE))

    ipt = np.vectorize(I)(pt)
    ilag = np.vectorize(I)(lag)
    itau = np.vectorize(I)(tau)

    # eligibility must be cell-pure per op for this specialized model
    elig = [np.nonzero(pt[o] > 0)[0] for o in range(n_ops)]
    op_cell = np.zeros(n_ops, dtype=int)
    for o in range(n_ops):
        cells = set(station_cell[m] for m in elig[o])
        assert len(cells) == 1, 'PPVC-specialized model needs type-pure eligibility'
        op_cell[o] = cells.pop()

    # moves: consecutive pairs with differing cells (job-first ops never move).
    # A residual problem skips the operations whose module has already been
    # delivered: that move is history, not a decision.
    delivered = set(int(o) for o in (delivered_ops or ()))
    moves = []   # (job, dest_op, from_cell, to_cell, dur)
    for j in range(n_j):
        for o in range(first[j] + 1, last[j] + 1):
            if op_cell[o] != op_cell[o - 1] and o not in delivered:
                moves.append((j, o, int(op_cell[o - 1]), int(op_cell[o]),
                              int(itau[op_cell[o - 1], op_cell[o]])))
    K = len(moves)

    ir_op = None if op_release is None else [I(x) for x in op_release]
    ir_mch = None if mch_ready is None else [I(x) for x in mch_ready]
    ir_veh = None if veh_ready is None else [I(x) for x in veh_ready]
    v_cells = ([depot] * n_veh if veh_cells is None
               else [int(c) for c in veh_cells])
    fixed = {} if fixed_ops is None else {int(o): (int(m), I(s))
                                          for o, (m, s) in fixed_ops.items()}
    # vehicles are interchangeable only while they share a start cell and a
    # ready time; a residual problem generally breaks both
    symmetric = len(set(v_cells)) == 1 and (ir_veh is None
                                            or len(set(ir_veh)) == 1)

    horizon = int(ipt.max(initial=0) * n_ops + ilag.sum() +
                  (max(m[4] for m in moves) if moves else 0) * (K + n_ops) +
                  itau.max(initial=0) * (K + 1))
    # the residual clock starts late, so the horizon has to start there too
    horizon += max([0] + [max(x) for x in (ir_op, ir_mch, ir_veh) if x])

    # (a) horizon = warm-start makespan (a valid UB, so no optimum is cut off)
    ws_ub = None
    if strengthen and warmstart is not None:
        ws_ub = I(float(np.max(np.asarray(warmstart['op_ct'], dtype=float))))
        if 0 < ws_ub < horizon:
            horizon = ws_ub

    model = cp_model.CpModel()
    start = [model.NewIntVar(0, horizon, f's{o}') for o in range(n_ops)]
    end = [model.NewIntVar(0, horizon, f'e{o}') for o in range(n_ops)]
    mch_of = [model.NewIntVar(0, n_m - 1, f'm{o}') for o in range(n_ops)]
    per_mch_intervals = [[] for _ in range(n_m)]
    for o in range(n_ops):
        alts = []
        for m in elig[o]:
            p = model.NewBoolVar(f'p{o}_{m}')
            iv = model.NewOptionalIntervalVar(start[o], int(ipt[o, m]), end[o],
                                              p, f'iv{o}_{m}')
            per_mch_intervals[m].append(iv)
            model.Add(mch_of[o] == int(m)).OnlyEnforceIf(p)
            if ir_mch is not None and o not in fixed:
                model.Add(start[o] >= ir_mch[m]).OnlyEnforceIf(p)
            alts.append(p)
        model.AddExactlyOne(alts)
        if ir_op is not None:
            model.Add(start[o] >= ir_op[o])
        if o in fixed:
            model.Add(mch_of[o] == fixed[o][0])
            model.Add(start[o] == fixed[o][1])
    for m in range(n_m):
        if per_mch_intervals[m]:
            model.AddNoOverlap(per_mch_intervals[m])

    # precedence with lags (+ transport where needed)
    mv_start = []
    move_of_op = {}
    for k, (j, o, fc, tc, dur) in enumerate(moves):
        sk = model.NewIntVar(0, horizon, f'mv{k}')
        mv_start.append(sk)
        move_of_op[o] = k
    for j in range(n_j):
        for o in range(first[j] + 1, last[j] + 1):
            if o in move_of_op:
                k = move_of_op[o]
                dur = moves[k][4]
                model.Add(mv_start[k] >= end[o - 1] + int(ilag[o - 1]))
                model.Add(start[o] >= mv_start[k] + dur)
            else:
                model.Add(start[o] >= end[o - 1] + int(ilag[o - 1]))

    # vehicles: circuit per vehicle with empty-move transitions.
    # All-zero tau makes vehicles irrelevant (zero-duration seizure can never
    # delay anything): skip the circuit machinery, keep plain precedences.
    if K and itau.max() > 0:
        veh_of = [[model.NewBoolVar(f'v{k}_{v}') for v in range(n_veh)]
                  for k in range(K)]
        for k in range(K):
            model.AddExactlyOne(veh_of[k])
        # (c) identical-vehicle symmetry breaking. All vehicles share the same
        # start cell `depot` and are otherwise interchangeable, so any solution
        # can be relabelled so that vehicle v is first used later than v-1.
        # In a residual problem the vehicles stand in different cells and are
        # ready at different times, so they are no longer interchangeable and
        # the relabelling argument fails.
        if strengthen and n_veh >= 2 and symmetric:
            for v in range(1, n_veh):
                for k in range(K):
                    if k < v:
                        model.Add(veh_of[k][v] == 0)
                    else:
                        model.AddBoolOr([veh_of[k][v].Not()] +
                                        [veh_of[kp][v - 1] for kp in range(k)])
        for v in range(n_veh):
            arcs = []
            # node 0 = depot, node k+1 = move k
            for k in range(K):
                lit_start = model.NewBoolVar(f'a_dep_{k}_{v}')
                arcs.append((0, k + 1, lit_start))
                first_ok = int(itau[v_cells[v], moves[k][2]])
                if ir_veh is not None:
                    first_ok += ir_veh[v]
                model.Add(mv_start[k] >= first_ok).OnlyEnforceIf(lit_start)
                lit_end = model.NewBoolVar(f'a_{k}_dep_{v}')
                arcs.append((k + 1, 0, lit_end))
                # self-loop when k not on vehicle v
                lit_skip = model.NewBoolVar(f'a_skip_{k}_{v}')
                arcs.append((k + 1, k + 1, lit_skip))
                model.AddImplication(lit_skip, veh_of[k][v].Not())
                model.AddImplication(lit_skip.Not(), veh_of[k][v])
            for a in range(K):
                for b in range(K):
                    if a == b:
                        continue
                    lit = model.NewBoolVar(f'a{a}_{b}_{v}')
                    arcs.append((a + 1, b + 1, lit))
                    # arrive at a's dest, empty-move to b's origin
                    trans = int(itau[moves[a][3], moves[b][2]])
                    model.Add(mv_start[b] >= mv_start[a] + moves[a][4] + trans
                              ).OnlyEnforceIf(lit)
                    model.AddImplication(lit, veh_of[a][v])
                    model.AddImplication(lit, veh_of[b][v])
            arcs.append((0, 0, model.NewBoolVar(f'a_empty_{v}')))
            model.AddCircuit(arcs)

    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, [end[last[j]] for j in range(n_j)])

    # (b) redundant fleet-capacity constraint. Each loaded move occupies one
    # vehicle for its whole duration, so at most |V| loaded moves overlap.
    # Implied by the circuits, but the cumulative propagator turns it into an
    # energetic bound on the makespan that the circuits alone do not deliver.
    # fleet=False removes strengthening (b) ENTIRELY (no cumulative, no
    # makespan-to-move links, no energy cut) while keeping (a) and (c). This
    # is the single-variable ablation that isolates the fleet-capacity
    # constraint. energy_cut=False is a different, weaker ablation: it drops
    # only the linear cut and keeps the cumulative.
    if strengthen and fleet and K and itau.max() > 0:
        mv_ivs = []
        for k in range(K):
            dur = moves[k][4]
            mv_end_k = model.NewIntVar(0, horizon, f'mve{k}')
            model.Add(mv_end_k == mv_start[k] + dur)
            mv_ivs.append(model.NewIntervalVar(mv_start[k], dur, mv_end_k,
                                               f'mviv{k}'))
            # a loaded move always finishes before the job's last op ends
            model.Add(makespan >= mv_end_k)
        model.AddCumulative(mv_ivs, [1] * K, n_veh)
        # energetic corollary of the same cumulative, stated linearly so the
        # LP relaxation sees it without waiting for the propagator.
        # energy_cut=False leaves it to the cumulative propagator alone (used
        # to check that the LB lift is not an artifact of this one line).
        if energy_cut:
            model.Add(n_veh * makespan >= int(sum(m[4] for m in moves)))

    model.Minimize(makespan)

    if warmstart is not None:
        ws_mch = np.asarray(warmstart['assigned_mch'], dtype=int)
        ws_start = np.asarray(warmstart['op_start'], dtype=float)
        for o in range(n_ops):
            model.AddHint(start[o], I(ws_start[o]))
            model.AddHint(mch_of[o], int(ws_mch[o]))
        ws_moves = {t['op']: t for t in warmstart.get('transports', [])}
        # under (c) the hinted vehicle labels must be relabelled into the
        # canonical order (first use by increasing move index), or the hint
        # contradicts the symmetry-breaking clauses and is thrown away.
        relabel = {}
        if strengthen and n_veh >= 2 and symmetric:
            for k, (j, o, fc, tc, dur) in enumerate(moves):
                t = ws_moves.get(o)
                if t is None:
                    continue
                v0 = t.get('veh', -1)
                if 0 <= v0 < n_veh and v0 not in relabel:
                    relabel[v0] = len(relabel)
        for k, (j, o, fc, tc, dur) in enumerate(moves):
            t = ws_moves.get(o)
            if t is not None:
                model.AddHint(mv_start[k], I(t['pickup']))
                if itau.max() > 0 and 0 <= t.get('veh', -1) < n_veh:
                    vh = relabel.get(t['veh'], t['veh'])
                    for v in range(n_veh):
                        model.AddHint(veh_of[k][v], int(v == vh))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(n_workers)
    solver.parameters.log_search_progress = bool(log)
    status = solver.Solve(model)
    name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(status=name, makespan=None)
    mv_veh = [-1] * K   # -1 = vehicle irrelevant (tau all-zero shortcut)
    if K and itau.max() > 0:
        for k in range(K):
            mv_veh[k] = next(v for v in range(n_veh)
                             if solver.Value(veh_of[k][v]))
    out = dict(
        status=name,
        strengthened=bool(strengthen),
        horizon=horizon / SCALE,
        makespan=solver.Value(makespan) / SCALE,
        objective_bound=solver.BestObjectiveBound() / SCALE,
        walltime=solver.WallTime(),
        assigned_mch=[int(solver.Value(mch_of[o])) for o in range(n_ops)],
        op_start=[solver.Value(start[o]) / SCALE for o in range(n_ops)],
        op_ct=[solver.Value(end[o]) / SCALE for o in range(n_ops)],
        moves=[dict(job=m[0], op=m[1], frm=m[2], to=m[3], veh=mv_veh[k],
                    dur=m[4] / SCALE, start=solver.Value(mv_start[k]) / SCALE)
               for k, m in enumerate(moves)],
    )
    return out
