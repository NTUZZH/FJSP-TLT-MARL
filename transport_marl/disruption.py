"""Mid-execution machine breakdown and residual re-planning for FJSP-TL-T.

WHY THIS FILE EXISTS
  On final makespan with a per-instance search budget of 60 to 3600 CPU
  seconds, the learned policy is at best level with the search baselines. The
  regime where a constructive policy should be structurally ahead is recovery:
  a station breaks down halfway through the week, the plan is invalid, and a
  new plan is needed within a wall-clock budget measured in seconds. This
  module builds that disrupted state and hands every scheduling method the
  same residual problem.

THE PROTOCOL (pre-specified; scripts/x2_disruption.py runs it)
  1. A baseline schedule of the instance gives its makespan C0.
  2. At t = 0.30 * C0 the machine with the largest remaining processing
     workload at t breaks down and is unavailable for d = 0.20 * C0.
     Remaining workload of a machine at t is the sum over its operations of
     the processing still to be done at t, so an operation half finished at t
     contributes its second half. Ties go to the lowest machine index.
  3. Every operation that started strictly before t keeps its machine, its
     vehicle and its start time. Everything else is re-planned from t.
  4. Each method re-plans under a wall-clock budget and the recovered
     schedule is re-checked by validator_t.validate_recovered_schedule.

MODELING CHOICES, all of which apply identically to every method
  * An operation running on the broken machine at t is ABORTED: its processing
    restarts from scratch, and its machine and start time are both re-planned.
    No partial credit is given for the work already done on it.
  * The module of an aborted operation is physically at the broken machine's
    station, so the transport that delivered it stays in the recovered
    schedule and no new move is needed. Operation eligibility in this
    benchmark is station-type-pure (every eligible machine of an operation
    sits in one cell, asserted below), so re-assigning an aborted operation to
    another eligible machine never needs another move either.
  * A loaded move that has not delivered its module by t is CANCELLED: the
    module is treated as still at the station it came from and the vehicle
    returns to service at t at the cell of its last completed delivery. The
    alternative, letting the move land and then pinning the destination
    machine, would fix a decision the re-plan is supposed to make.
  * t and d are rounded onto the same dyadic time grid the CP-SAT model
    integerizes on (SCALE = 128 in cpsat_transport.py), so the solver arm sees
    exactly the same disruption as the simulator arms rather than a rounded
    one.

Entry points: build_residual() -> residual dict; residual_sim() -> a
TransportSim rolling out the residual; right_shift_repair() -> the no-search
industrial reference recovery; load_env_residual() -> the same state inside a
batched FJSPEnvTransport so the policy continues its rollout.
"""

import numpy as np

from transport_marl.sim_single import TransportSim

EPS = 1e-9

# cpsat_transport.SCALE. Imported by value rather than by import so this file
# stays free of the ortools dependency.
TIME_GRID = 128


def quantize(x):
    """Round a duration onto the CP-SAT time grid so every arm sees one t."""
    return float(round(float(x) * TIME_GRID)) / TIME_GRID


def _instance_geometry(job_length, op_pt, meta):
    jl = np.asarray(job_length, dtype=int)
    pt = np.asarray(op_pt, dtype=float)
    tr = meta['transport']
    geo = dict(
        jl=jl, pt=pt,
        lag=np.asarray(meta['time_lag'], dtype=float),
        station_cell=np.asarray(tr['station_cell'], dtype=int),
        tau=np.asarray(tr['tau_cells'], dtype=float),
        n_v=int(tr['n_vehicles']),
        veh_start_cell=int(tr.get('veh_start_cell', 0)),
        job_start_cell=int(tr.get('job_start_cell', -1)),
        n_j=len(jl), n_m=pt.shape[1], n_ops=pt.shape[0])
    geo['first'] = np.zeros(geo['n_j'], dtype=int)
    geo['first'][1:] = np.cumsum(jl)[:-1]
    geo['last'] = geo['first'] + jl - 1
    geo['job_of_op'] = np.repeat(np.arange(geo['n_j']), jl)
    geo['op_cell'] = np.full(geo['n_ops'], -1, dtype=int)
    for o in range(geo['n_ops']):
        cells = set(int(c) for c in geo['station_cell'][np.nonzero(pt[o] > 0)[0]])
        assert len(cells) == 1, ('disruption needs type-pure eligibility '
                                 f'(op {o} spans cells {sorted(cells)})')
        geo['op_cell'][o] = cells.pop()
    return geo


def machine_remaining_workload(assigned_mch, op_start, op_ct, n_m, t):
    """Processing still to be done on each machine at time t."""
    rem = np.zeros(n_m)
    for o in range(len(assigned_mch)):
        left = op_ct[o] - max(op_start[o], t)
        if left > 0:
            rem[assigned_mch[o]] += left
    return rem


def commit_times(job_length, op_pt, meta, record):
    """When the decision model commits each operation of a schedule.

    An operation is committed to its machine once its job is ready (the
    previous operation done and its lag elapsed; time 0 for a first
    operation) and the previous operation on that machine has completed,
    whichever is later. For a cell-changing operation this is the moment its
    machine is reserved and its transport task released.
    """
    g = _instance_geometry(job_length, op_pt, meta)
    amch = np.asarray(record['assigned_mch'], dtype=int)
    st = np.asarray(record['op_start'], dtype=float)
    ct = np.asarray(record['op_ct'], dtype=float)
    prev_ct = np.zeros(g['n_ops'])
    for m in range(g['n_m']):
        seq = np.nonzero(amch == m)[0]
        seq = seq[np.argsort(st[seq], kind='stable')]
        prev_ct[seq[1:]] = ct[seq[:-1]]
    ready = np.zeros(g['n_ops'])
    nf = np.ones(g['n_ops'], dtype=bool)
    nf[g['first']] = False
    idx = np.nonzero(nf)[0]
    ready[idx] = ct[idx - 1] + g['lag'][idx - 1]
    return np.maximum(ready, prev_ct)


def build_residual(job_length, op_pt, meta, record, t_frac=0.30, d_frac=0.20,
                   at=None, breakdown=True, keep_in_transit=False,
                   keep_commitments=False):
    """Disrupt a baseline schedule and describe the residual problem.

    `record` is a schedule_record() dict from TransportSim or from the batched
    env. The returned dict carries both the state the simulator arms load and
    the per-operation release times, per-machine ready times and fixed
    decisions the solver arms need.

    `at` replaces t_frac * C0 by an absolute time, and breakdown=False cuts the
    schedule at t without breaking any machine (broken = -1, nothing aborted,
    d = 0). Together they give the re-planning state of an execution
    checkpoint (scripts/x2_execution_noise.py).

    keep_in_transit=True replaces the cancel rule for a move whose vehicle
    left before t (its 'leave' time when recorded, else its departure) to
    fetch a module whose previous operation started before t: that move is
    completed as executed. Its vehicle is busy until the arrival and ends at
    the destination cell, and the module is ready there at the arrival. The
    destination operation itself stays open, and with type-pure eligibility
    every machine it can be re-assigned to is in that cell, so it needs no
    further move (the same argument as for an aborted operation). Such
    operations are listed in 'in_transit'. A vehicle sent for a module whose
    previous operation has not started is released at t.

    keep_commitments=True (no breakdown) hands over the state the decision
    model itself holds at t. An operation committed before t (commit_times)
    keeps its machine: if its move is already under way it is placed, i.e.
    frozen at its arrival-time start like an operation that has begun;
    otherwise it is listed in 'committed' with its machine reserved and its
    transport task released at its commitment time, to be served by a
    vehicle the planner chooses. Free times stay raw (a machine idle since
    40 reads 40, not t). Only where a raw value would let an event fall
    before t is it raised to t: a job and a machine that could be paired
    before t, and every vehicle while a released task waits; these are
    counted in 'floored'. A vehicle that set off before t for a module whose
    machine is not yet committed (it may leave early under right_shift_repair
    with anticipate) completes its trip, and the operation stays open in the
    delivered cell ('in_transit'), as under keep_in_transit; a plan built by
    the decision model never contains this case. The freeze rule for started
    operations is unchanged.
    """
    g = _instance_geometry(job_length, op_pt, meta)
    n_ops, n_m, n_j = g['n_ops'], g['n_m'], g['n_j']

    amch = np.asarray(record['assigned_mch'], dtype=int)
    st = np.asarray(record['op_start'], dtype=float)
    ct = np.asarray(record['op_ct'], dtype=float)
    assert (amch >= 0).all(), 'baseline schedule is incomplete'
    c0 = float(ct.max())
    t = quantize(t_frac * c0 if at is None else at)
    d = quantize(d_frac * c0) if breakdown else 0.0

    rem = machine_remaining_workload(amch, st, ct, n_m, t)
    broken = int(np.argmax(rem)) if breakdown else -1

    started = st < t - EPS
    committed = []
    in_transit = []
    if keep_commitments:
        assert not breakdown and not keep_in_transit
        c_time = commit_times(job_length, op_pt, meta, record)
        move_into = {int(x['op']): x for x in record['transports']}
        for o in np.nonzero(~started)[0]:
            if o not in move_into or c_time[o] >= t - EPS:
                continue
            x = move_into[o]
            if float(x.get('leave', x['depart'])) < t - EPS:
                started[o] = True               # placed, its move under way
            else:
                committed.append(dict(
                    job=int(g['job_of_op'][o]), op=int(o), mch=int(amch[o]),
                    frm=int(x['frm']), to=int(x['to']),
                    release=float(c_time[o])))
    # the clock floor of the hand-off; raw (zero) when commitments are kept
    floor = 0.0 if keep_commitments else t
    aborted_mask = started & (ct > t + EPS) & (amch == broken)
    frozen_mask = started & ~aborted_mask
    frozen = np.nonzero(frozen_mask)[0]
    aborted = np.nonzero(aborted_mask)[0]

    # a move is kept exactly when it delivered its module before t; that is the
    # same predicate as "the destination operation started before t", so the
    # aborted operations keep the move that put them on the broken station
    if keep_commitments:
        # a vehicle can set off, and even deliver, before the machine is
        # committed; its trip is completed and the operation stays open in
        # the delivered cell, as under keep_in_transit
        for o, x in move_into.items():
            if started[o] or o in [c['op'] for c in committed]:
                continue
            if float(x.get('leave', x['depart'])) >= t - EPS:
                continue
            if o != g['first'][g['job_of_op'][o]] and not started[o - 1]:
                continue
            in_transit.append(o)
    kept_tr = [dict(x) for x in record['transports'] if started[int(x['op'])]]
    kept_tr += [dict(move_into[o]) for o in in_transit]
    if keep_in_transit:
        for x in record['transports']:
            o = int(x['op'])
            if started[o] or float(x.get('leave', x['depart'])) >= t - EPS:
                continue
            if o != g['first'][g['job_of_op'][o]] and not started[o - 1]:
                continue
            kept_tr.append(dict(x))
            in_transit.append(o)

    next_op = np.full(n_j, -1, dtype=int)
    job_ready = np.zeros(n_j)
    job_loc = np.full(n_j, g['job_start_cell'], dtype=int)
    for j in range(n_j):
        span = np.arange(g['first'][j], g['last'][j] + 1)
        run = span[started[span]]
        if len(run) == 0:
            next_op[j] = int(g['first'][j])
            job_ready[j] = floor
            continue
        o = int(run[-1])
        job_loc[j] = int(g['station_cell'][amch[o]])
        if aborted_mask[o]:
            next_op[j] = o                      # redone from scratch, same cell
            job_ready[j] = t
        elif o == g['last'][j]:
            next_op[j] = -1                     # job finished before t
        else:
            next_op[j] = o + 1
            job_ready[j] = max(floor, ct[o] + g['lag'][o])
    for x in kept_tr:
        o = int(x['op'])
        if o in in_transit:
            j = int(g['job_of_op'][o])
            assert next_op[j] == o, f'move into op {o} left before its job did'
            job_loc[j] = int(x['to'])
            job_ready[j] = max(floor, float(x['arrival']))

    mch_free = np.full(n_m, floor)
    for o in frozen:
        mch_free[amch[o]] = max(mch_free[amch[o]], ct[o])
    if breakdown:
        mch_free[broken] = max(mch_free[broken], t + d)

    veh_free = np.full(g['n_v'], floor)
    veh_cell = np.full(g['n_v'], g['veh_start_cell'], dtype=int)
    for v in range(g['n_v']):
        chain = sorted([x for x in kept_tr if int(x['veh']) == v],
                       key=lambda x: x['arrival'])
        if chain:
            veh_cell[v] = int(chain[-1]['to'])
            veh_free[v] = max(floor, float(chain[-1]['arrival']))

    floored = dict(jobs=[], machines=[], vehicles=[])
    if keep_commitments:
        reserved = set(c['mch'] for c in committed)
        waiting = set(c['job'] for c in committed)
        fj, fm = set(), set()
        for j in range(n_j):
            o = next_op[j]
            if o < 0 or j in waiting:
                continue
            for m in np.nonzero(g['pt'][o] > 0)[0]:
                if int(m) not in reserved and \
                        max(job_ready[j], mch_free[m]) < t - EPS:
                    fj.add(j)
                    fm.add(int(m))
        for j in fj:
            job_ready[j] = t
        for m in fm:
            mch_free[m] = t
        fv = ([v for v in range(g['n_v']) if veh_free[v] < t - EPS]
              if committed else [])
        for v in fv:
            veh_free[v] = t
        floored = dict(jobs=sorted(fj), machines=sorted(fm), vehicles=fv)

    r_mch = np.full(n_ops, -1, dtype=int)
    r_veh = np.full(n_ops, -1, dtype=int)
    r_start = np.full(n_ops, -1.0)
    r_ct = np.full(n_ops, -1.0)
    r_mch[frozen] = amch[frozen]
    r_start[frozen] = st[frozen]
    r_ct[frozen] = ct[frozen]
    for x in kept_tr:
        r_veh[int(x['op'])] = int(x['veh'])

    frozen_ct_max = float(ct[frozen].max()) if len(frozen) else 0.0
    op_release = np.where(frozen_mask, st, t)
    for o in in_transit:
        op_release[o] = max(t, job_ready[int(g['job_of_op'][o])])
    return dict(
        t=t, downtime=d, t_up=t + d, broken=broken, baseline_makespan=c0,
        t_frac=float(t_frac), d_frac=float(d_frac),
        broken_workload=float(rem[broken]) if breakdown else 0.0,
        frozen=frozen.tolist(), aborted=aborted.tolist(),
        in_transit=sorted(in_transit), committed=committed, floored=floored,
        handoff='commit' if keep_commitments else 'release',
        n_frozen=int(len(frozen)),
        next_op=next_op, job_ready=job_ready, job_loc=job_loc,
        mch_free=mch_free, veh_free=veh_free, veh_cell=veh_cell,
        assigned_mch=r_mch, assigned_veh=r_veh, op_start=r_start, op_ct=r_ct,
        transports=kept_tr,
        op_release=op_release,
        mch_ready=np.where(np.arange(n_m) == broken, t + d, t),
        delivered_ops=sorted([int(o) for o in np.nonzero(started)[0]]
                             + in_transit),
        fixed_ops={int(o): (int(amch[o]), float(st[o])) for o in frozen},
        lower_bound=max(t + d, frozen_ct_max),
        frozen_ct_max=frozen_ct_max,
        base_assigned_mch=amch.copy(), base_op_start=st.copy(),
        base_op_ct=ct.copy(),
        base_transports=[dict(x) for x in record['transports']])


def residual_sim(job_length, op_pt, meta, residual):
    """A TransportSim holding the disrupted state, ready to be rolled out."""
    tr = meta['transport']
    sim = TransportSim(job_length, op_pt, meta['time_lag'], tr['station_cell'],
                       tr['tau_cells'], int(tr['n_vehicles']),
                       int(tr.get('veh_start_cell', 0)))
    sim.load_residual_state(residual)
    return sim


# --------------------------------------------------------------------------- #
# Right-shift repair: the no-search industrial reference
# --------------------------------------------------------------------------- #
def right_shift_repair(job_length, op_pt, meta, residual, move_time=None,
                       anticipate=False):
    """Push the invalidated part of the baseline later, changing no decision.

    Machine assignments, the operation order on every machine, vehicle
    assignments and the move order on every vehicle all stay exactly as the
    baseline had them; only times move, and only forward. The recovered start
    times are the longest path through the fixed arc set, which is what a
    planner does when it refuses to reschedule and simply delays.

    An aborted operation keeps the broken machine and waits for the repair,
    which is the classic right-shift response to a breakdown.

    move_time(op, frm, to), when given, is the loaded duration of the move
    into op; the default is tau[frm, to]. Executing a plan under realized
    durations passes realized op_pt and lags and this hook, since a realized
    loaded leg belongs to one move and not to the cell pair.

    anticipate=False dispatches a vehicle once its module is ready, as the
    decision model does. anticipate=True lets it drive to the pickup as soon
    as it is free and wait there, so the pickup is the later of its arrival
    and the module's ready time; the record shows it leaving one empty leg
    before the pickup, which is the same timing, and keeps the moment it
    actually set off as 'leave'. This is the dispatch a CP-SAT schedule uses,
    and it needs no foresight of the ready time.
    """
    g = _instance_geometry(job_length, op_pt, meta)
    n_ops = g['n_ops']
    t, broken = residual['t'], residual['broken']
    amch = residual['base_assigned_mch']
    b_start, b_ct = residual['base_op_start'], residual['base_op_ct']
    frozen = set(residual['frozen'])
    delivered = set(residual['delivered_ops'])
    tau, lag = g['tau'], g['lag']

    ct_new = np.full(n_ops, -1.0)
    start_new = np.full(n_ops, -1.0)
    for o in frozen:
        start_new[o] = b_start[o]
        ct_new[o] = b_ct[o]

    # moves that still have to happen: the baseline transports whose module was
    # not yet delivered at t
    moves = [dict(x) for x in residual['base_transports']
             if int(x['op']) not in delivered]
    move_of_op = {int(x['op']): k for k, x in enumerate(moves)}
    mv_depart = np.full(len(moves), -1.0)
    mv_pickup = np.full(len(moves), -1.0)
    mv_arrival = np.full(len(moves), -1.0)
    mv_leave = np.full(len(moves), -1.0)

    prev_on_mch = {}
    for m in range(g['n_m']):
        seq = sorted((o for o in range(n_ops)
                      if amch[o] == m and o not in frozen),
                     key=lambda o: b_start[o])
        for a, b in zip(seq[:-1], seq[1:]):
            prev_on_mch[b] = a
    prev_on_veh, veh_of_first = {}, {}
    for v in range(g['n_v']):
        seq = sorted((k for k, x in enumerate(moves) if int(x['veh']) == v),
                     key=lambda k: moves[k]['depart'])
        for a, b in zip(seq[:-1], seq[1:]):
            prev_on_veh[b] = a
        if seq:
            veh_of_first[seq[0]] = v

    # every arc of the fixed arc set runs strictly forward in baseline time
    # (processing times and inter-cell travel times are strictly positive), so
    # the baseline order is a topological order of it. A move is placed at
    # its pickup, which follows its job's predecessor in every feasible
    # schedule; its departure need not, once a vehicle may leave early.
    nodes = [('op', o, b_start[o]) for o in range(n_ops) if o not in frozen]
    nodes += [('mv', k, moves[k]['pickup']) for k in range(len(moves))]
    nodes.sort(key=lambda x: (x[2], x[0], x[1]))

    for kind, i, _ in nodes:
        if kind == 'mv':
            o = int(moves[i]['op'])
            j = int(g['job_of_op'][o])
            if o == g['first'][j]:
                release = residual['job_ready'][j]
            else:
                release = ct_new[o - 1] + lag[o - 1]
            if i in prev_on_veh:
                q = prev_on_veh[i]
                ready, at_cell = mv_arrival[q], int(moves[q]['to'])
            else:
                v = veh_of_first[i]
                ready = residual['veh_free'][v]
                at_cell = int(residual['veh_cell'][v])
            empty = tau[at_cell, int(moves[i]['frm'])]
            if anticipate:
                mv_leave[i] = max(t, ready)
                depart = max(mv_leave[i], release - empty)
            else:
                depart = max(t, release, ready)
            pickup = depart + empty
            mv_depart[i], mv_pickup[i] = depart, pickup
            frm, to = int(moves[i]['frm']), int(moves[i]['to'])
            mv_arrival[i] = pickup + (tau[frm, to] if move_time is None
                                      else move_time(o, frm, to))
            continue
        o, m = i, int(amch[i])
        j = int(g['job_of_op'][o])
        if o in move_of_op:
            ready = mv_arrival[move_of_op[o]]
        elif o == g['first'][j]:
            ready = residual['job_ready'][j]
        else:
            ready = ct_new[o - 1] + lag[o - 1]
        if o == residual['next_op'][j]:
            # the module of an in-transit operation is ready on arrival; for
            # every other next operation this bound is already implied
            ready = max(ready, residual['job_ready'][j])
        floor = residual['mch_free'][m] if o not in prev_on_mch \
            else ct_new[prev_on_mch[o]]
        s = max(t, ready, floor)
        if m == broken:
            s = max(s, residual['t_up'])
        start_new[o] = s
        ct_new[o] = s + g['pt'][o, m]

    assert (ct_new >= 0).all(), 'right-shift left an operation unscheduled'
    transports = [dict(x) for x in residual['transports']]
    for k, x in enumerate(moves):
        transports.append(dict(job=int(x['job']), op=int(x['op']),
                               veh=int(x['veh']), frm=int(x['frm']),
                               to=int(x['to']), depart=float(mv_depart[k]),
                               pickup=float(mv_pickup[k]),
                               arrival=float(mv_arrival[k])))
        if anticipate:
            transports[-1]['leave'] = float(mv_leave[k])
    return dict(assigned_mch=amch.copy(), op_start=start_new, op_ct=ct_new,
                transports=transports, makespan=float(ct_new.max()))


# --------------------------------------------------------------------------- #
# Batched env: load the same state so the policy continues its rollout
# --------------------------------------------------------------------------- #
def load_env_residual(env, e, job_length, op_pt, meta, residual):
    """Load the disrupted state into env slot e of a fresh FJSPEnvTransport.

    The frozen operations are replayed through the env's own commit path
    (force_realize), so every derived array the network reads is built by the
    code that builds it during training; only the residual clock, the vehicle
    positions and the broken machine's ready time are written directly.

    A residual built with keep_commitments also carries 'committed'
    operations. They are committed through the env's own machine step after
    the replay, so the reservation, the released task and its release time
    are the ones the env computes itself (checked against the residual), and
    the chain bound is then only raised where a free time was floored.
    """
    g = _instance_geometry(job_length, op_pt, meta)
    frozen = sorted(residual['frozen'], key=lambda o: residual['base_op_start'][o])
    amch = residual['base_assigned_mch']
    for o in frozen:
        env.force_realize(e, int(g['job_of_op'][o]), int(o), int(amch[o]),
                          float(residual['base_op_start'][o]))
    for o in residual['aborted']:
        j = int(g['job_of_op'][o])
        # the module sits on the broken station, so the operation needs no
        # travel wherever inside that cell it is re-assigned
        env.job_cell[e, j] = int(g['station_cell'][amch[o]])
        env.tau_in_min[e, o] = 0.0
        env.tau_in_max[e, o] = 0.0
    for o in residual.get('in_transit', []):
        j = int(g['job_of_op'][o])
        # the module is on its way into this operation's cell, so the
        # operation needs no travel wherever in that cell it is assigned
        env.job_cell[e, j] = int(g['op_cell'][o])
        env.tau_in_min[e, o] = 0.0
        env.tau_in_max[e, o] = 0.0
    for c in residual.get('committed', []):
        env._step_machine(np.array([e]),
                          np.array([c['job'] * env.number_of_machines
                                    + c['mch']]))
        assert abs(env.task_release[e, c['job']] - c['release']) < 1e-6, \
            f"op {c['op']}: env commits at {env.task_release[e, c['job']]}, " \
            f"residual at {c['release']}"
    env.true_mch_free_time[e] = np.maximum(env.true_mch_free_time[e],
                                           residual['mch_free'])
    env.true_candidate_free_time[e] = np.maximum(
        env.true_candidate_free_time[e], residual['job_ready'])
    env.veh_free[e] = residual['veh_free']
    env.veh_cell[e] = residual['veh_cell']
    env.rec_transports[e] = [dict(x) for x in residual['transports']]
    env.current_makespan[e] = max(float(env.current_makespan[e]), 0.0)
    if _kept(residual):
        for o in residual['in_transit']:
            # the module needs no further travel: restate this operation's
            # chain bound without it (the env built it with the old cell)
            j = int(g['job_of_op'][o])
            lb = (env.true_candidate_free_time[e, j]
                  + env.true_op_min_pt[e, o])
            env._raise_ct_lb(np.array([e]), np.array([j]), np.array([o]),
                             np.array([lb]))
        for j in residual['floored']['jobs']:
            o = int(env.candidate[e, j])
            lb = (env.true_candidate_free_time[e, j] + env.tau_in_min[e, o]
                  + env.true_op_min_pt[e, o])
            if lb > env.op_ct_lb[e, o]:
                env._raise_ct_lb(np.array([e]), np.array([j]), np.array([o]),
                                 np.array([lb]))
    else:
        _reseed_ct_lb(env, e)
    return env


def _kept(residual):
    """True for a residual built with keep_commitments (raw clock)."""
    return residual.get('handoff') == 'commit'


def _reseed_ct_lb(env, e):
    """Rebuild the travel-aware chain bound from the residual clock.

    Same recursion the env maintains incrementally, restated from the loaded
    state: realized operations contribute their true completion, the first
    unrealized operation of a job starts no earlier than the job's residual
    ready time, and the rest chain on through lags and minimum travel. Without
    this the bound would still carry the travel an aborted operation no longer
    has to make, which would leave it above the true optimum instead of below.
    """
    for j in range(env.number_of_jobs):
        f = int(env.job_first_op_id[e, j])
        l = int(env.job_last_op_id[e, j])
        for o in range(f, l + 1):
            if env.op_scheduled_flag[e, o] > 0.5:
                env.op_ct_lb[e, o] = env.true_op_ct[e, o]
            elif o == f or env.op_scheduled_flag[e, o - 1] > 0.5:
                env.op_ct_lb[e, o] = (env.true_candidate_free_time[e, j]
                                      + env.tau_in_min[e, o]
                                      + env.true_op_min_pt[e, o])
            else:
                env.op_ct_lb[e, o] = (env.op_ct_lb[e, o - 1]
                                      + env.true_op_lag[e, o - 1]
                                      + env.tau_in_min[e, o]
                                      + env.true_op_min_pt[e, o])
