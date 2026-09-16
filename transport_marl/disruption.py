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


def build_residual(job_length, op_pt, meta, record, t_frac=0.30, d_frac=0.20):
    """Disrupt a baseline schedule and describe the residual problem.

    `record` is a schedule_record() dict from TransportSim or from the batched
    env. The returned dict carries both the state the simulator arms load and
    the per-operation release times, per-machine ready times and fixed
    decisions the solver arms need.
    """
    g = _instance_geometry(job_length, op_pt, meta)
    n_ops, n_m, n_j = g['n_ops'], g['n_m'], g['n_j']

    amch = np.asarray(record['assigned_mch'], dtype=int)
    st = np.asarray(record['op_start'], dtype=float)
    ct = np.asarray(record['op_ct'], dtype=float)
    assert (amch >= 0).all(), 'baseline schedule is incomplete'
    c0 = float(ct.max())
    t = quantize(t_frac * c0)
    d = quantize(d_frac * c0)

    rem = machine_remaining_workload(amch, st, ct, n_m, t)
    broken = int(np.argmax(rem))

    started = st < t - EPS
    aborted_mask = started & (ct > t + EPS) & (amch == broken)
    frozen_mask = started & ~aborted_mask
    frozen = np.nonzero(frozen_mask)[0]
    aborted = np.nonzero(aborted_mask)[0]

    # a move is kept exactly when it delivered its module before t; that is the
    # same predicate as "the destination operation started before t", so the
    # aborted operations keep the move that put them on the broken station
    kept_tr = [dict(x) for x in record['transports'] if started[int(x['op'])]]

    next_op = np.full(n_j, -1, dtype=int)
    job_ready = np.zeros(n_j)
    job_loc = np.full(n_j, g['job_start_cell'], dtype=int)
    for j in range(n_j):
        span = np.arange(g['first'][j], g['last'][j] + 1)
        run = span[started[span]]
        if len(run) == 0:
            next_op[j] = int(g['first'][j])
            job_ready[j] = t
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
            job_ready[j] = max(t, ct[o] + g['lag'][o])

    mch_free = np.full(n_m, t)
    for o in frozen:
        mch_free[amch[o]] = max(mch_free[amch[o]], ct[o])
    mch_free[broken] = max(mch_free[broken], t + d)

    veh_free = np.full(g['n_v'], t)
    veh_cell = np.full(g['n_v'], g['veh_start_cell'], dtype=int)
    for v in range(g['n_v']):
        chain = sorted([x for x in kept_tr if int(x['veh']) == v],
                       key=lambda x: x['arrival'])
        if chain:
            veh_cell[v] = int(chain[-1]['to'])

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
    return dict(
        t=t, downtime=d, t_up=t + d, broken=broken, baseline_makespan=c0,
        t_frac=float(t_frac), d_frac=float(d_frac),
        broken_workload=float(rem[broken]),
        frozen=frozen.tolist(), aborted=aborted.tolist(),
        n_frozen=int(len(frozen)),
        next_op=next_op, job_ready=job_ready, job_loc=job_loc,
        mch_free=mch_free, veh_free=veh_free, veh_cell=veh_cell,
        assigned_mch=r_mch, assigned_veh=r_veh, op_start=r_start, op_ct=r_ct,
        transports=kept_tr,
        op_release=np.where(frozen_mask, st, t),
        mch_ready=np.where(np.arange(n_m) == broken, t + d, t),
        delivered_ops=sorted(int(o) for o in np.nonzero(started)[0]),
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
def right_shift_repair(job_length, op_pt, meta, residual):
    """Push the invalidated part of the baseline later, changing no decision.

    Machine assignments, the operation order on every machine, vehicle
    assignments and the move order on every vehicle all stay exactly as the
    baseline had them; only times move, and only forward. The recovered start
    times are the longest path through the fixed arc set, which is what a
    planner does when it refuses to reschedule and simply delays.

    An aborted operation keeps the broken machine and waits for the repair,
    which is the classic right-shift response to a breakdown.
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
    # the baseline order is a topological order of it
    nodes = [('op', o, b_start[o]) for o in range(n_ops) if o not in frozen]
    nodes += [('mv', k, moves[k]['depart']) for k in range(len(moves))]
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
            depart = max(t, release, ready)
            pickup = depart + tau[at_cell, int(moves[i]['frm'])]
            mv_depart[i], mv_pickup[i] = depart, pickup
            mv_arrival[i] = pickup + tau[int(moves[i]['frm']),
                                         int(moves[i]['to'])]
            continue
        o, m = i, int(amch[i])
        j = int(g['job_of_op'][o])
        if o in move_of_op:
            ready = mv_arrival[move_of_op[o]]
        elif o == g['first'][j]:
            ready = residual['job_ready'][j]
        else:
            ready = ct_new[o - 1] + lag[o - 1]
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
    env.true_mch_free_time[e] = np.maximum(env.true_mch_free_time[e],
                                           residual['mch_free'])
    env.true_candidate_free_time[e] = np.maximum(
        env.true_candidate_free_time[e], residual['job_ready'])
    env.veh_free[e] = residual['veh_free']
    env.veh_cell[e] = residual['veh_cell']
    env.rec_transports[e] = [dict(x) for x in residual['transports']]
    env.current_makespan[e] = max(float(env.current_makespan[e]), 0.0)
    _reseed_ct_lb(env, e)
    return env


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
