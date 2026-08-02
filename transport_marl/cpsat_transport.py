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
"""

import numpy as np
from ortools.sat.python import cp_model

# speed constants are dyadic (k/128, see layout.calibrate_speed), so at
# SCALE=128 every integerization below is EXACT (round() is a formality)
SCALE = 128


def solve_transport_instance(job_length, op_pt, meta, time_limit=300.0,
                             n_workers=8, log=False, warmstart=None):
    """warmstart: optional schedule_record dict (sim/env format: assigned_mch,
    op_start, op_ct, transports) hinted via AddHint (proposal §8 warm-started
    variant). Hints must come from the SAME quantized time grid (dyadic)."""
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

    # moves: consecutive pairs with differing cells (job-first ops never move)
    moves = []   # (job, dest_op, from_cell, to_cell, dur)
    for j in range(n_j):
        for o in range(first[j] + 1, last[j] + 1):
            if op_cell[o] != op_cell[o - 1]:
                moves.append((j, o, int(op_cell[o - 1]), int(op_cell[o]),
                              int(itau[op_cell[o - 1], op_cell[o]])))
    K = len(moves)

    horizon = int(ipt.max(initial=0) * n_ops + ilag.sum() +
                  (max(m[4] for m in moves) if moves else 0) * (K + n_ops) +
                  itau.max(initial=0) * (K + 1))

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
            alts.append(p)
        model.AddExactlyOne(alts)
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
        for v in range(n_veh):
            arcs = []
            # node 0 = depot, node k+1 = move k
            for k in range(K):
                lit_start = model.NewBoolVar(f'a_dep_{k}_{v}')
                arcs.append((0, k + 1, lit_start))
                model.Add(mv_start[k] >= int(itau[depot, moves[k][2]])
                          ).OnlyEnforceIf(lit_start)
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
    model.Minimize(makespan)

    if warmstart is not None:
        ws_mch = np.asarray(warmstart['assigned_mch'], dtype=int)
        ws_start = np.asarray(warmstart['op_start'], dtype=float)
        for o in range(n_ops):
            model.AddHint(start[o], I(ws_start[o]))
            model.AddHint(mch_of[o], int(ws_mch[o]))
        ws_moves = {t['op']: t for t in warmstart.get('transports', [])}
        for k, (j, o, fc, tc, dur) in enumerate(moves):
            t = ws_moves.get(o)
            if t is not None:
                model.AddHint(mv_start[k], I(t['pickup']))
                if itau.max() > 0 and 0 <= t.get('veh', -1) < n_veh:
                    for v in range(n_veh):
                        model.AddHint(veh_of[k][v], int(v == t['veh']))

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
