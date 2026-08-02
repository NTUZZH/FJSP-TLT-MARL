"""Independent feasibility validator for FJSP-TL-T schedules (Paper X2, P1).

Deliberately shares NO code with sim_single.py / the training env / layout.py:
plain numpy, everything re-derived from raw inputs. Mirrors the house
schedule_validator.py contract (violation strings with [tag] prefixes).

Checks: [assign] completeness, [compat] eligibility, [ct] start/ct/pt
consistency, [precedence] job order with lags (same-cell case),
[transport] existence/uniqueness/timing of moves for cell-changing pairs,
[vehicle] per-vehicle chain consistency (empty-move duration, no overlap,
valid ids), [spurious] no transport for same-cell pairs, [bound] makespan
>= chain lower bound with lags and mandatory minimal travel.
"""

import numpy as np


def validate_transport_schedule(job_length, op_pt, time_lag, station_cell,
                                tau_cells, n_vehicles, veh_start_cell,
                                record, tol=1e-6, job_start_cell=-1):
    """job_start_cell >= 0 switches on external (Link JSSPT) semantics: every
    job materializes at that cell, so a first op on a different cell REQUIRES
    a transport (release 0). Default -1 keeps PPVC semantics (first op needs
    and tolerates no transport)."""
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt, dtype=float)
    time_lag = np.asarray(time_lag, dtype=float)
    station_cell = np.asarray(station_cell, dtype=int)
    tau = np.asarray(tau_cells, dtype=float)
    amch = np.asarray(record['assigned_mch'], dtype=int)
    start = np.asarray(record['op_start'], dtype=float)
    ct = np.asarray(record['op_ct'], dtype=float)
    transports = record['transports']
    n_ops = int(job_length.sum())
    v = []

    # [assign] completeness
    if (amch < 0).any():
        v.append(f"[assign] unassigned ops: {np.nonzero(amch < 0)[0].tolist()}")
    if (start < -tol).any() or (ct < -tol).any():
        v.append("[assign] negative start/completion times")
    if v:
        return dict(feasible=False, violations=v, makespan=float(ct.max()))

    # [compat] + [ct]
    for o in range(n_ops):
        if op_pt[o, amch[o]] <= 0:
            v.append(f"[compat] op {o} on incompatible machine {amch[o]}")
        elif abs(ct[o] - start[o] - op_pt[o, amch[o]]) > tol:
            v.append(f"[ct] op {o}: ct != start + pt "
                     f"({ct[o]:.6f} != {start[o]:.6f} + {op_pt[o, amch[o]]:.6f})")

    # transports indexed by destination op
    tr_by_op = {}
    for tr in transports:
        if tr['op'] in tr_by_op:
            v.append(f"[transport] duplicate transport for op {tr['op']}")
        tr_by_op[tr['op']] = tr

    # [precedence] / [transport] / [spurious] along each job chain
    first = 0
    for j, L in enumerate(job_length):
        for k in range(L):
            o = first + k
            if k == 0:
                if job_start_cell < 0:
                    if o in tr_by_op:
                        v.append(f"[spurious] transport for first op {o} of job {j}")
                elif station_cell[amch[o]] == job_start_cell:
                    if o in tr_by_op:
                        v.append(f"[spurious] transport for co-located first op {o}")
                else:
                    tr = tr_by_op.get(o)
                    if tr is None:
                        v.append(f"[transport] missing initial transport for op {o} "
                                 f"(cells {job_start_cell}->{station_cell[amch[o]]})")
                    else:
                        if tr['frm'] != job_start_cell or tr['to'] != station_cell[amch[o]]:
                            v.append(f"[transport] op {o} initial move {tr['frm']}->"
                                     f"{tr['to']} does not match "
                                     f"{job_start_cell}->{station_cell[amch[o]]}")
                        if tr['pickup'] < -tol:
                            v.append(f"[transport] op {o} picked up before release 0")
                        if abs(tr['arrival'] - tr['pickup'] - tau[tr['frm'], tr['to']]) > tol:
                            v.append(f"[transport] op {o} loaded move duration wrong")
                        if start[o] < tr['arrival'] - tol:
                            v.append(f"[transport] op {o} starts {start[o]:.6f} < "
                                     f"arrival {tr['arrival']:.6f}")
                continue
            p = o - 1
            release = ct[p] + time_lag[p]
            if station_cell[amch[p]] == station_cell[amch[o]]:
                if o in tr_by_op:
                    v.append(f"[spurious] transport for same-cell pair {p}->{o}")
                if start[o] < release - tol:
                    v.append(f"[precedence] op {o} starts {start[o]:.6f} < "
                             f"pred ct+lag {release:.6f}")
            else:
                tr = tr_by_op.get(o)
                if tr is None:
                    v.append(f"[transport] missing transport for pair {p}->{o} "
                             f"(cells {station_cell[amch[p]]}->{station_cell[amch[o]]})")
                    continue
                if tr['frm'] != station_cell[amch[p]] or tr['to'] != station_cell[amch[o]]:
                    v.append(f"[transport] op {o} move {tr['frm']}->{tr['to']} does not "
                             f"match machine cells {station_cell[amch[p]]}->{station_cell[amch[o]]}")
                if tr['pickup'] < release - tol:
                    v.append(f"[transport] op {o} picked up {tr['pickup']:.6f} < "
                             f"pred ct+lag {release:.6f}")
                if abs(tr['arrival'] - tr['pickup'] - tau[tr['frm'], tr['to']]) > tol:
                    v.append(f"[transport] op {o} loaded move duration wrong")
                if start[o] < tr['arrival'] - tol:
                    v.append(f"[transport] op {o} starts {start[o]:.6f} < arrival "
                             f"{tr['arrival']:.6f}")
        first += L

    # [overlap] machines
    for m in range(op_pt.shape[1]):
        ops_m = np.nonzero(amch == m)[0]
        order = ops_m[np.argsort(start[ops_m])]
        for a, b in zip(order[:-1], order[1:]):
            if start[b] < ct[a] - tol:
                v.append(f"[overlap] machine {m}: ops {a},{b} overlap")

    # [vehicle] chains
    for tr in transports:
        if not (0 <= tr['veh'] < n_vehicles):
            v.append(f"[vehicle] transport for op {tr['op']} uses invalid vehicle {tr['veh']}")
    for veh in range(n_vehicles):
        chain = sorted([t for t in transports if t['veh'] == veh],
                       key=lambda t: t['depart'])
        loc = veh_start_cell
        prev_arrival = 0.0
        for t in chain:
            if t['depart'] < prev_arrival - tol:
                v.append(f"[vehicle] veh {veh} departs {t['depart']:.6f} < previous "
                         f"arrival {prev_arrival:.6f} (op {t['op']})")
            if abs(t['pickup'] - t['depart'] - tau[loc, t['frm']]) > tol:
                v.append(f"[vehicle] veh {veh} empty-move duration wrong before op "
                         f"{t['op']} (from cell {loc} to {t['frm']})")
            loc = t['to']
            prev_arrival = t['arrival']

    # [bound] chain LB with lags + mandatory minimal travel
    makespan = float(ct.max())
    first = 0
    for j, L in enumerate(job_length):
        lb = 0.0
        for k in range(L):
            o = first + k
            if k == 0 and job_start_cell >= 0:
                cb = station_cell[np.nonzero(op_pt[o] > 0)[0]]
                lb += tau[job_start_cell, cb].min()
            lb += op_pt[o][op_pt[o] > 0].min()
            if k < L - 1:
                lb += time_lag[o]
                ca = station_cell[np.nonzero(op_pt[o] > 0)[0]]
                cb = station_cell[np.nonzero(op_pt[o + 1] > 0)[0]]
                lb += tau[np.ix_(ca, cb)].min()
        if makespan < lb - tol:
            v.append(f"[bound] makespan {makespan:.6f} < job {j} chain LB {lb:.6f}")
        first += L

    if abs(makespan - float(np.max(ct))) > tol:
        v.append("[bound] reported makespan != max completion")

    return dict(feasible=len(v) == 0, violations=v, makespan=makespan)
