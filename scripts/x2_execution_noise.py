"""Realized makespan under uncertain durations, with and without re-planning.

WHAT THIS MEASURES AND WHY
  Every schedule in the paper is computed and scored on nominal durations. On
  a real line, curing and drying lags, processing and vehicle trips all run
  longer or shorter than planned. This script executes each method's plan
  under perturbed durations and reports the makespan that is actually
  realized, either keeping the plan's decisions (right shift) or re-planning
  the residual from the realized state at fixed checkpoints.

NOISE MODEL
  Three duration classes of the model are perturbed, each multiplicatively:
    processing  one multiplier per operation, applied to the processing time
                on whichever eligible machine runs it;
    lag         one multiplier per operation on its machine-free lag, so a
                zero lag stays zero;
    loaded leg  one multiplier per destination operation on the loaded leg
                tau(a, b) of the move into it, whatever cells a and b are.
  Empty repositioning legs stay nominal. realized = q(nominal * (1 + sigma*u)),
  u ~ U(-1, 1), drawn once per (instance, replicate) from the integer seed
  NOISE_BASE + 1000 * instance + replicate; every sigma level and every arm
  scales the same u, so all arms meet the same realization. q rounds onto the
  1/128 grid CP-SAT integerizes on (disruption.quantize), so the solver arm
  sees exactly the durations the simulator arms see. Because a multiplier
  belongs to an operation, an arm that picks another machine or route meets
  the same relative deviation.

EXECUTION (right shift)
  A plan's machine assignments, machine orders, vehicle assignments and
  vehicle orders are kept; every start is the longest path through those arcs
  under realized durations. That is disruption.right_shift_repair called with
  realized op_pt and lags, a per-move loaded-time hook and anticipate=True: a
  vehicle drives to its next pickup as soon as it is free and waits there.
  That is the dispatch a CP-SAT schedule already uses; with the decision
  model's dispatch (leave once the module is ready) every CP-SAT plan would
  lose one empty leg per move in execution. It needs no foresight, and every
  arm is executed by the same rule.

RE-PLANNING
  Checkpoint k (k = 1..K-1, K = 10) is the realized completion time of
  operation number ceil(k * N / K) in the schedule being executed. At a
  checkpoint t the planner receives the state the decision model itself
  holds at t (--handoff commit, the default; disruption.build_residual with
  keep_commitments): every operation that started before t is frozen, every
  operation committed to its machine before t keeps that machine (placed if
  its move is under way, otherwise with its machine reserved and its task
  released), and free times stay raw except where an event could otherwise
  fall before t. A long lag does not cancel a reservation in a plant, and
  handing the policy released reservations and a floored clock, a state it
  never meets in its own rollout, cost greedy re-planning up to 4.6 percent
  of makespan with no noise at all. --handoff release keeps the earlier
  protocol: commitments are released, the clock is floored at t, and the
  destination of a move under way stays open inside its cell. The
  planner sees the realized durations of finished operations, elapsed lags
  and moves that have arrived, and nominal values for everything else
  (running operations, lags still running, moves still under way, all future
  activities). Its plan is validated against that information, then executed
  under realized durations until the next checkpoint; every executed segment
  is validated against the realized durations and the frozen prefix.

ARMS (initial plan -> online response)
  policy_rs      policy greedy rollout            -> right shift
  policy_replan  policy greedy rollout            -> greedy policy re-plan
  policy_threshold policy greedy rollout          -> right shift, and a greedy
                 policy re-plan only at a checkpoint where the executed
                 prefix runs late against the plan in force by more than
                 --replan_threshold (0.05, fixed before any run) of the
                 initial plan's makespan. Lateness is the largest realized
                 minus planned completion time over the operations finished
                 by the checkpoint.
  pdr_replan     best of the nine rule pairs      -> best-of-nine re-dispatch
  ga60_rs        PDR-seeded GA v2, 60 CPU-s       -> right shift
  ga_replan      PDR-seeded GA v2, 60 CPU-s       -> GA, --ga_replan_budget
                 CPU-s, the current plan kept when it is better
  cpsat_replan   PDR-seeded GA v2, 60 CPU-s       -> warm-started residual
                 CP-SAT, --cpsat_budget wall-s, current plan as warm start
  The initial plans are computed once per instance and cached under
  {outdir}/plans/{cell}/, so every replicate and sigma level executes the same
  plan. The GA budgets are CPU seconds (time.process_time), as in
  scripts/x2_scale_ga.py, so co-tenancy cannot weaken them.

THREADS. The policy arms run one batch-1 forward pass per event. That is
fastest at one torch thread (1.7 ms on a 10-module instance, 142 ms at eight),
so --torch_threads (default 1) is kept apart from --threads, which caps the
numerical libraries and should equal the pinned core count.

Output: {outdir}/{cell}.jsonl, one record per (instance, replicate, sigma,
arm), appended under an exclusive lock and fsynced, keyed so a rerun skips
finished work. Several shards (--shard i/W) may append to one file.

Usage:
  python -u scripts/x2_execution_noise.py --cell 50x25+ppvct-mixed+v2+t1.0 \\
      --model_name mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-s301 \\
      --arms policy_rs,policy_replan,pdr_replan,ga60_rs,ga_replan \\
      --shard 0/4 --cores 18-18 --threads 1
"""

import argparse
import fcntl
import glob
import json
import math
import os
import sys
import time


ALL_ARMS = ('policy_rs', 'policy_replan', 'policy_threshold', 'pdr_replan',
            'ga60_rs', 'ga_replan', 'cpsat_replan')
INITIAL = {'policy_rs': 'policy', 'policy_replan': 'policy',
           'policy_threshold': 'policy', 'pdr_replan': 'pdr',
           'ga60_rs': 'ga60', 'ga_replan': 'ga60', 'cpsat_replan': 'ga60'}
NOISE_BASE = 9100000


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', type=str, required=True)
    ap.add_argument('--model_name', type=str,
                    default='mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-s301')
    ap.add_argument('--arms', type=str, default=','.join(ALL_ARMS))
    ap.add_argument('--sigmas', type=str, default='0.1,0.2,0.3')
    ap.add_argument('--replicates', type=int, default=3)
    ap.add_argument('--n', type=int, default=0, help='first N instances; 0=all')
    ap.add_argument('--instances', type=str, default='',
                    help='comma list of instance names; overrides --n')
    ap.add_argument('--checkpoints', type=int, default=10,
                    help='K: re-plan at every completed 1/K of the operations')
    ap.add_argument('--handoff', type=str, default='commit',
                    choices=['commit', 'release'],
                    help='state handed to a re-plan: commit keeps every '
                         'commitment made before the checkpoint and the raw '
                         'free times; release frees them and floors the clock')
    ap.add_argument('--replan_threshold', type=float, default=0.05,
                    help='policy_threshold re-plans when the lateness of the '
                         'executed prefix exceeds this fraction of the '
                         'initial plan makespan')
    ap.add_argument('--ga_nominal_budget', type=float, default=60.0)
    ap.add_argument('--ga_replan_budget', type=float, default=10.0)
    ap.add_argument('--cpsat_budget', type=float, default=10.0)
    ap.add_argument('--cpsat_workers', type=int, default=4)
    ap.add_argument('--ga_pop', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shard', type=str, default='0/1',
                    help='i/W: this process takes the instances whose index '
                         'is i modulo W')
    ap.add_argument('--device', type=str, default='cpu',
                    choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--cores', type=str, default='',
                    help='cpu range to pin to, e.g. 18-21')
    ap.add_argument('--threads', type=int, default=1,
                    help='OMP/MKL cap; must match the pinned core count')
    ap.add_argument('--torch_threads', type=int, default=1)
    ap.add_argument('--outdir', type=str, default='results/execution_noise')
    args = ap.parse_args()
    args.arms = [a for a in args.arms.split(',') if a]
    bad = [a for a in args.arms if a not in ALL_ARMS]
    assert not bad, f'unknown arms {bad}; pick from {ALL_ARMS}'
    args.sigmas = [float(s) for s in args.sigmas.split(',')]
    i, w = (int(x) for x in args.shard.split('/'))
    assert 0 <= i < w, f'bad --shard {args.shard}'
    args.shard_i, args.shard_w = i, w
    sys.argv = [sys.argv[0]]
    return args


A = parse_cli()

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = str(A.threads)
if A.cores:
    lo, hi = (int(x) for x in A.cores.split('-'))
    os.sched_setaffinity(0, list(range(lo, hi + 1)))

sys.path.insert(0, '.')

import numpy as np
import torch

from params import configs

ARCH_KEYS = ['fea_j_input_dim', 'fea_m_input_dim', 'n_op_types', 'n_mch_types',
             'type_emb_dim', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor', 'hidden_dim_actor',
             'num_mlp_layers_critic', 'hidden_dim_critic', 'dropout_prob',
             'guide']
NEEDS_POLICY = any(INITIAL[a] == 'policy' for a in A.arms)


def load_snapshot(model_name):
    with open(f'train_log/PPVCT/config_{model_name}.json') as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            val = snap[k]
            if isinstance(val, str) and val.startswith('['):
                val = json.loads(val)
            setattr(configs, k, val)
    return snap


SNAP = load_snapshot(A.model_name) if NEEDS_POLICY else {}
# configs.device must be set before fjsp_env_same_op_nums is imported: its
# EnvState reads it once, at class-definition time.
if A.device == 'auto':
    configs.device = 'cuda' if torch.cuda.device_count() else 'cpu'
else:
    configs.device = A.device
assert configs.device != 'cuda' or torch.cuda.device_count(), \
    '--device cuda requested but torch sees no GPU'
torch.set_num_threads(A.torch_threads)
try:
    torch.set_num_interop_threads(A.torch_threads)
except RuntimeError:
    pass

from ppvc_instance_generator import load_instance
from transport_marl import disruption as dis
from transport_marl import ga_transport_v2 as ga
from transport_marl.cpsat_transport import solve_transport_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.mappo import select_actions
from transport_marl.model_transport import DANIELTransport
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES
from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_recovered_schedule

EPS = 1e-9
GRID = 1.0 / dis.TIME_GRID
HANDOFF = (dict(breakdown=False, keep_commitments=True)
           if A.handoff == 'commit'
           else dict(breakdown=False, keep_in_transit=True))


# --------------------------------------------------------------------------- #
# instance, noise and belief
# --------------------------------------------------------------------------- #
class Instance:
    def __init__(self, stem):
        jl, pt, meta = load_instance(stem)
        self.jl = np.asarray(jl, dtype=int)
        self.pt = np.asarray(pt, dtype=float)
        self.meta = meta
        self.lag = np.asarray(meta['time_lag'], dtype=float)
        tr = meta['transport']
        self.cell = np.asarray(tr['station_cell'], dtype=int)
        self.tau = np.asarray(tr['tau_cells'], dtype=float)
        self.n_v = int(tr['n_vehicles'])
        self.vsc = int(tr.get('veh_start_cell', 0))
        self.jsc = int(tr.get('job_start_cell', -1))
        self.n_ops = self.pt.shape[0]
        self.first_ops = set(np.concatenate([[0], np.cumsum(self.jl)[:-1]])
                             .astype(int).tolist())
        for x in (self.pt, self.lag, self.tau):
            assert np.allclose(x * dis.TIME_GRID, np.round(x * dis.TIME_GRID)), \
                'nominal durations must sit on the CP-SAT time grid'

    def with_durations(self, pt, lag):
        return pt, dict(self.meta, time_lag=lag)


def _q(x, floor):
    return np.maximum(np.round(x * dis.TIME_GRID) / dis.TIME_GRID, floor)


class Realization:
    """Realized durations of one (instance, replicate, sigma)."""

    def __init__(self, inst, u, sigma):
        u_pt, u_lag, u_mv = u
        self.sigma = sigma
        scale = 1.0 + sigma * u_pt
        self.pt = np.where(inst.pt > 0, _q(inst.pt * scale[:, None], GRID), 0.0)
        self.lag = _q(inst.lag * (1.0 + sigma * u_lag), 0.0)
        self.mv_scale = 1.0 + sigma * u_mv
        self.tau = inst.tau

    def loaded(self, o, a, b):
        return float(_q(self.tau[a, b] * self.mv_scale[o], GRID))


def noise_draw(inst_idx, rep, n_ops):
    seed = NOISE_BASE + 1000 * inst_idx + rep
    u = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(3, n_ops))
    return seed, u


def _believe_move(inst, x, t, rec_b, lag_b):
    """A move under way at t as the planner believes it, in place.

    If its module is not yet picked up, the pickup waits for the module's
    believed ready time; the loaded leg is nominal and the arrival is not
    before t."""
    o = int(x['op'])
    if x['pickup'] > t + EPS and o not in inst.first_ops:
        empty = x['pickup'] - x['depart']
        x['pickup'] = max(x.get('leave', x['depart']) + empty,
                          rec_b['op_ct'][o - 1] + lag_b[o - 1])
        x['depart'] = x['pickup'] - empty
    x['arrival'] = max(t, x['pickup'] + float(inst.tau[x['frm'], x['to']]))


def belief(inst, real, hist, t):
    """What the planner knows at checkpoint t.

    Finished operations, elapsed lags and moves that have arrived carry their
    realized durations; everything else is nominal, except that what is
    observed unfinished at t is not believed to end before t (a running
    operation, a lag still running, a move still under way). Returns the
    belief durations, the belief copy of the executed record (a move under
    way whose module is not yet picked up waits for the module's believed
    ready time) and the loaded-time function for the validator.
    """
    st = np.asarray(hist['op_start'], dtype=float)
    ct = np.asarray(hist['op_ct'], dtype=float)
    amch = np.asarray(hist['assigned_mch'], dtype=int)
    started = st < t - EPS
    done = started & (ct <= t + EPS)
    running = started & ~done
    pt_b = inst.pt.copy()
    pt_b[done] = real.pt[done]
    for o in np.nonzero(running)[0]:
        pt_b[o, amch[o]] = max(inst.pt[o, amch[o]], t - st[o])
    lag_b = inst.lag.copy()
    elapsed = done & (ct + real.lag <= t + EPS)
    lag_b[elapsed] = real.lag[elapsed]
    pend = done & ~elapsed
    lag_b[pend] = np.maximum(inst.lag[pend], t - ct[pend])
    rec_b = dict(hist, op_ct=ct.copy(), op_start=st.copy(),
                 transports=[dict(x) for x in hist['transports']])
    c_time = dis.commit_times(inst.jl, real.pt,
                              dict(inst.meta, time_lag=real.lag), hist)
    rec_b['op_ct'][running] = st[running] + pt_b[running, amch[running]]
    arrived = set(int(x['op']) for x in hist['transports']
                  if x['arrival'] <= t + EPS)
    leg_b = {}
    # placed operations (handoff commit: committed, move under way, not
    # started by t) are frozen by the hand-off. Their moves are handled
    # first, because a later move of the same job picks its module up after
    # the placed operation completes, and must read the believed completion.
    under_way = [x for x in rec_b['transports']
                 if x.get('leave', x['depart']) < t - EPS
                 and not started[int(x['op'])]]
    placed = set(int(x['op']) for x in under_way
                 if A.handoff == 'commit' and c_time[int(x['op'])] < t - EPS)
    under_way.sort(key=lambda x: int(x['op']) not in placed)
    for x in under_way:
        o = int(x['op'])
        if o not in arrived:
            _believe_move(inst, x, t, rec_b, lag_b)
            # only a move the hand-off keeps carries its believed leg; a
            # move whose vehicle is released is re-planned at its nominal leg
            if o in inst.first_ops or started[o - 1] or (o - 1) in placed:
                leg_b[o] = x['arrival'] - x['pickup']
        if o in placed:
            # the reserved machine waits for the module; the operation is
            # believed nominal, also when the module arrives exactly at t
            rec_b['op_start'][o] = max(x['arrival'], c_time[o])
            rec_b['op_ct'][o] = rec_b['op_start'][o] + inst.pt[o, amch[o]]

    def loaded_b(o, a, b):
        if o in arrived:
            return real.loaded(o, a, b)
        return leg_b.get(o, float(inst.tau[a, b]))
    return pt_b, lag_b, rec_b, loaded_b


def with_plan(res, plan):
    """The residual `res` with `plan` as the schedule to be right-shifted.

    Frozen operations keep the history `res` was cut from; the rest take the
    plan's machines and order, and the plan's moves that are not delivered."""
    frozen = np.zeros(len(res['base_op_start']), dtype=bool)
    frozen[res['frozen']] = True
    amch = np.asarray(plan['assigned_mch'], dtype=int)
    assert (amch[frozen] == res['base_assigned_mch'][frozen]).all(), \
        'plan moved a frozen operation'
    return dict(res, base_assigned_mch=amch,
                base_op_start=np.where(frozen, res['base_op_start'],
                                       np.asarray(plan['op_start'], dtype=float)),
                base_op_ct=np.where(frozen, res['base_op_ct'],
                                    np.asarray(plan['op_ct'], dtype=float)),
                base_transports=[dict(x) for x in plan['transports']])


def check(inst, pt, lag, rec, res, loaded, tag):
    out = validate_recovered_schedule(
        inst.jl, pt, lag, inst.cell, inst.tau, inst.n_v, inst.vsc, rec, res,
        job_start_cell=inst.jsc, loaded_time=loaded)
    if not out['feasible']:
        raise AssertionError(f'{tag}: schedule INVALID: {out["violations"][:5]}')
    return out['makespan']


def execute(inst, real, plan, res_real, tag):
    """Right-shift `plan` from the realized state `res_real`; validated."""
    pt_r, meta_r = inst.with_durations(real.pt, real.lag)
    rec = dis.right_shift_repair(inst.jl, pt_r, meta_r,
                                 with_plan(res_real, plan),
                                 move_time=real.loaded, anticipate=True)
    check(inst, real.pt, real.lag, rec, res_real, real.loaded, tag)
    return rec


# --------------------------------------------------------------------------- #
# planners
# --------------------------------------------------------------------------- #
def build_policy():
    algo = SNAP.get('algo', 'mappo')
    if algo == 'single':
        from transport_marl.single_agent import DANIELSingle
        policy = DANIELSingle(configs)
    elif algo in ('mappo', 'coma'):
        policy = DANIELTransport(configs)
    else:
        raise ValueError(f"unknown algo '{algo}'")
    sd = torch.load(f'trained_network/PPVCT/{A.model_name}.pth',
                    map_location=torch.device(configs.device))
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'checkpoint/architecture mismatch for '
                           f'{A.model_name}: {len(missing)} missing, '
                           f'{len(unexpected)} unexpected')
    policy.eval()
    return policy


def make_env(inst, pt, meta):
    guide = bool(getattr(configs, 'guide', False))
    env = FJSPEnvTransport(len(inst.jl), pt.shape[1], use_lag_features=True,
                           use_guide=guide,
                           guide_price=SNAP.get('guide_price', 'certified'),
                           guide_price_scale=float(
                               SNAP.get('guide_price_scale', 1.0)))
    env.set_initial_data([inst.jl], [np.asarray(pt, dtype=float)],
                         [np.asarray(meta['time_lag'], dtype=float)],
                         [np.asarray(meta['op_type'])],
                         [np.asarray(meta['mch_type'])],
                         [meta['transport']])
    if guide:
        from transport_marl.bound import TransportBound
        env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
    return env


def rollout(env, policy):
    """Greedy two-head rollout to completion; returns the decision count."""
    state = env.state
    steps = 0
    while env.done().min() < 1:
        steps += 1
        with torch.no_grad():
            pi_m, pi_v, _ = policy(
                fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                mch_type=state.mch_type_tensor, fea_v=state.fea_v_tensor,
                fea_veh_pairs=state.fea_veh_pairs_tensor,
                veh_action_mask=state.veh_action_mask_tensor,
                event_veh=state.event_veh_tensor,
                task_dest_op=state.task_dest_op_tensor)
            a, _ = select_actions(pi_m, pi_v, state.event_type_tensor,
                                  greedy=True)
        state, _, _ = env.step(a.cpu().numpy())
    if configs.device == 'cuda':
        torch.cuda.synchronize()
    return steps


def best_pdr(sim):
    best_ms, best_rec, best_pair = np.inf, None, None
    for mn in MCH_RULES:
        for vn in VEH_RULES:
            sim.reset()
            ms = sim.run(MCH_RULES[mn], VEH_RULES[vn])
            if ms < best_ms:
                best_ms, best_rec = ms, sim.schedule_record()
                best_pair = f'{mn}+{vn}'
    return best_rec, best_pair


def run_ga_on(sim, pt, budget, rng):
    """PDR-seeded GA v2 on whatever state `sim` restores to (x2_scale_ga)."""
    elig = ga.eligible_machines(np.asarray(pt, dtype=float))
    seeds, pair_ms = ga.pdr_seed_chromosomes(sim, elig, rng)
    for nm, s in zip(pair_ms.keys(), seeds):
        d = ga.decode(sim, elig, *s)
        assert abs(d - pair_ms[nm]) < 1e-9, \
            f'GA seed {nm} decodes {d} != PDR {pair_ms[nm]}'
    ga_ms, best, gens = ga.run_ga(sim, elig, rng, budget, pop_size=A.ga_pop,
                                  seeds=seeds)
    ga.decode(sim, elig, *best)
    rec = sim.schedule_record()
    assert abs(rec['makespan'] - ga_ms) < 1e-6, 'GA re-decode is not stable'
    return rec, dict(gens=int(gens), ga_own=float(ga_ms),
                     seed_best=float(min(pair_ms.values())))


def plan_policy(inst, pt, meta, res, policy):
    env = make_env(inst, pt, meta)
    if res is not None:
        dis.load_env_residual(env, 0, inst.jl, pt, meta, res)
        env.refresh_state()
    steps = rollout(env, policy)
    rec = env.schedule_record(0)
    del env
    return rec, dict(decisions=int(steps))


def plan_pdr(inst, pt, meta, res):
    tr = meta['transport']
    if res is None:
        sim = TransportSim(inst.jl, pt, meta['time_lag'], tr['station_cell'],
                           tr['tau_cells'], inst.n_v, inst.vsc)
    else:
        sim = dis.residual_sim(inst.jl, pt, meta, res)
    rec, pair = best_pdr(sim)
    return rec, dict(pdr_pair=pair)


def plan_ga(inst, pt, meta, res, budget, rng, incumbent=None):
    tr = meta['transport']
    if res is None:
        sim = TransportSim(inst.jl, pt, meta['time_lag'], tr['station_cell'],
                           tr['tau_cells'], inst.n_v, inst.vsc)
    else:
        sim = dis.residual_sim(inst.jl, pt, meta, res)
    rec, info = run_ga_on(sim, pt, budget, rng)
    info['kept_current'] = False
    if incumbent is not None and incumbent['makespan'] < rec['makespan'] - 1e-9:
        # the right-shifted current plan is outside the GA's non-delay decode
        # class, so it is carried as an incumbent rather than seeded
        rec, info['kept_current'] = incumbent, True
    return rec, info


def plan_cpsat(inst, pt, meta, res, warm):
    sol = solve_transport_instance(
        inst.jl, pt, meta, time_limit=A.cpsat_budget,
        n_workers=A.cpsat_workers, warmstart=warm, strengthen=True,
        op_release=res['op_release'], mch_ready=res['mch_ready'],
        fixed_ops=res['fixed_ops'],
        veh_ready=np.maximum(res['veh_free'], res['t']),
        veh_cells=res['veh_cell'], delivered_ops=res['delivered_ops'],
        fixed_mch={c['op']: c['mch'] for c in res['committed']})
    info = dict(status=sol['status'], cpsat_own=sol.get('makespan'),
                kept_current=False)
    if sol['makespan'] is None or sol['makespan'] > warm['makespan'] - 1e-9:
        return warm, dict(info, kept_current=True)
    transports = [dict(x) for x in res['transports']]
    by_veh = {}
    for m in sol['moves']:
        by_veh.setdefault(int(m['veh']), []).append(m)
    for v, chain in by_veh.items():
        chain.sort(key=lambda x: x['start'])
        loc = int(res['veh_cell'][v])
        for m in chain:
            pickup = float(m['start'])
            transports.append(dict(
                job=int(m['job']), op=int(m['op']), veh=v, frm=int(m['frm']),
                to=int(m['to']),
                depart=pickup - float(inst.tau[loc, int(m['frm'])]),
                pickup=pickup,
                arrival=pickup + float(inst.tau[int(m['frm']), int(m['to'])])))
            loc = int(m['to'])
    ct = np.asarray(sol['op_ct'], dtype=float)
    rec = dict(assigned_mch=np.asarray(sol['assigned_mch'], dtype=int),
               op_start=np.asarray(sol['op_start'], dtype=float), op_ct=ct,
               transports=transports, makespan=float(ct.max()))
    assert abs(rec['makespan'] - sol['makespan']) < 1e-6, \
        f"CP-SAT objective {sol['makespan']} != rebuilt {rec['makespan']}"
    return rec, info


def timed(fn, *args, **kw):
    w0, c0 = time.perf_counter(), time.process_time()
    rec, info = fn(*args, **kw)
    return rec, info, time.perf_counter() - w0, time.process_time() - c0


# --------------------------------------------------------------------------- #
# initial plans, cached per instance
# --------------------------------------------------------------------------- #
def _to_json(rec):
    return dict(assigned_mch=np.asarray(rec['assigned_mch']).tolist(),
                op_start=np.asarray(rec['op_start'], dtype=float).tolist(),
                op_ct=np.asarray(rec['op_ct'], dtype=float).tolist(),
                transports=[{k: (float(v) if isinstance(v, (float, np.floating))
                                 else int(v)) for k, v in x.items()}
                            for x in rec['transports']],
                makespan=float(rec['makespan']))


def _from_json(d):
    return dict(assigned_mch=np.asarray(d['assigned_mch'], dtype=int),
                op_start=np.asarray(d['op_start'], dtype=float),
                op_ct=np.asarray(d['op_ct'], dtype=float),
                transports=[dict(x) for x in d['transports']],
                makespan=float(d['makespan']))


def plan_key(kind):
    if kind == 'policy':
        return f'policy|{A.model_name}'
    if kind == 'ga60':
        return f'ga|B{A.ga_nominal_budget:g}|pop{A.ga_pop}|seed{A.seed}'
    return 'pdr'


def initial_plan(inst, name, idx, kind, policy):
    path = f'{A.outdir}/plans/{A.cell}/{name}.json'
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            cache = json.load(f)
    key = plan_key(kind)
    if key not in cache:
        if kind == 'policy':
            rec, info, w, c = timed(plan_policy, inst, inst.pt, inst.meta,
                                    None, policy)
        elif kind == 'pdr':
            rec, info, w, c = timed(plan_pdr, inst, inst.pt, inst.meta, None)
        else:
            # same RNG stream as scripts/x2_scale_ga.py for this instance
            rng = np.random.default_rng(A.seed * 100003 + idx)
            rec, info, w, c = timed(plan_ga, inst, inst.pt, inst.meta, None,
                                    A.ga_nominal_budget, rng)
        res0 = dis.build_residual(inst.jl, inst.pt, inst.meta, rec, at=0.0,
                                  breakdown=False)
        check(inst, inst.pt, inst.lag, rec, res0, None, f'{name}/{kind} plan')
        cache[key] = dict(record=_to_json(rec), wall_s=round(w, 4),
                          cpu_s=round(c, 4), **info)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    entry = dict(cache[key])
    return _from_json(entry.pop('record')), entry


# --------------------------------------------------------------------------- #
# one closed loop
# --------------------------------------------------------------------------- #
def closed_loop(inst, name, idx, rep, real, arm, plan0, policy):
    pt_r, meta_r = inst.with_durations(real.pt, real.lag)
    res0 = dis.build_residual(inst.jl, pt_r, meta_r, plan0, at=0.0,
                              breakdown=False)
    hist = execute(inst, real, plan0, res0, f'{name}/{arm}/exec0')
    plan = plan0
    log = dict(checkpoints=[], plan_wall_s=[], plan_cpu_s=[],
               in_transit_moves=[], committed_ops=[], floored=[],
               replan_info=[])
    if arm in ('policy_rs', 'ga60_rs'):
        return hist, log
    if arm == 'policy_threshold':
        log['lateness'] = []
        limit = A.replan_threshold * float(plan0['makespan'])

    n = inst.n_ops
    t_prev = 0.0
    for k in range(1, A.checkpoints):
        ct_sorted = np.sort(np.asarray(hist['op_ct'], dtype=float))
        t = float(ct_sorted[math.ceil(k * n / A.checkpoints) - 1])
        if t <= t_prev + EPS:
            continue
        st = np.asarray(hist['op_start'], dtype=float)
        if (st < t - EPS).all():
            break
        if arm == 'policy_threshold':
            ct_h = np.asarray(hist['op_ct'], dtype=float)
            fin = ct_h <= t + EPS
            late = float((ct_h[fin] - np.asarray(plan['op_ct'])[fin]).max())
            log['lateness'].append([round(t, 6), round(late, 6)])
            if late <= limit:
                continue
        res_r = dis.build_residual(inst.jl, pt_r, meta_r, hist, at=t,
                                   **HANDOFF)
        pt_b, lag_b, rec_b, loaded_b = belief(inst, real, hist, t)
        pt_b_, meta_b = inst.with_durations(pt_b, lag_b)
        res_b = dis.build_residual(inst.jl, pt_b, meta_b, rec_b, at=t,
                                   **HANDOFF)
        assert res_b['frozen'] == res_r['frozen']
        assert res_b['in_transit'] == res_r['in_transit']
        assert res_b['committed'] == res_r['committed']
        kept_mch = {c['op']: c['mch'] for c in res_r['committed']}

        if arm in ('policy_replan', 'policy_threshold'):
            new, info, w, c = timed(plan_policy, inst, pt_b, meta_b, res_b,
                                    policy)
        elif arm == 'pdr_replan':
            new, info, w, c = timed(plan_pdr, inst, pt_b, meta_b, res_b)
        else:
            w0, c0 = time.perf_counter(), time.process_time()
            # the plan in force, right-shifted on what the planner knows: the
            # warm start of CP-SAT and the incumbent of the GA
            warm = dis.right_shift_repair(inst.jl, pt_b, meta_b,
                                          with_plan(res_b, plan),
                                          anticipate=True)
            check(inst, pt_b, lag_b, warm, res_b, loaded_b,
                  f'{name}/{arm}/warm{k}')
            if arm == 'ga_replan':
                rng = np.random.default_rng(
                    (A.seed, idx, rep, int(round(real.sigma * 1000)), k))
                new, info = plan_ga(inst, pt_b, meta_b, res_b,
                                    A.ga_replan_budget, rng, incumbent=warm)
            else:
                new, info = plan_cpsat(inst, pt_b, meta_b, res_b, warm)
            w, c = time.perf_counter() - w0, time.process_time() - c0
        check(inst, pt_b, lag_b, new, res_b, loaded_b, f'{name}/{arm}/plan{k}')
        assert all(int(new['assigned_mch'][o]) == m
                   for o, m in kept_mch.items()), \
            f'{name}/{arm}/plan{k}: a committed machine was changed'
        hist = execute(inst, real, new, res_r, f'{name}/{arm}/exec{k}')
        plan = new
        t_prev = t
        log['checkpoints'].append(round(t, 6))
        log['plan_wall_s'].append(round(w, 4))
        log['plan_cpu_s'].append(round(c, 4))
        log['in_transit_moves'].append(len(res_r['in_transit']))
        log['committed_ops'].append(len(res_r['committed']))
        log['floored'].append([len(res_r['floored'][k2])
                               for k2 in ('jobs', 'machines', 'vehicles')])
        log['replan_info'].append(info)
    return hist, log


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def append_row(jsonl, row):
    with open(jsonl, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row) + '\n')
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def rec_key(name, rep, sigma, arm):
    tag = f'|{A.model_name}' if INITIAL[arm] == 'policy' else ''
    return f'{A.cell}|{name}|r{rep}|s{sigma:g}|{arm}{tag}'


def main():
    os.makedirs(A.outdir, exist_ok=True)
    jsonl = f'{A.outdir}/{A.cell}.jsonl'
    done = set()
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            done = {json.loads(l)['key'] for l in f if l.strip()}
    stems = sorted(g[:-4] for g in
                   glob.glob(f'data/PPVCT/{A.cell}/test/instance_*.fjs'))
    assert stems, f'no instances in data/PPVCT/{A.cell}/test'
    if A.instances:
        want = set(A.instances.split(','))
        stems = [s for s in stems if os.path.basename(s) in want]
    elif A.n:
        stems = stems[:A.n]
    policy = build_policy() if NEEDS_POLICY else None
    print(f'[x2_execution_noise] cell={A.cell} arms={",".join(A.arms)} '
          f'sigmas={A.sigmas} reps={A.replicates} n={len(stems)} '
          f'shard={A.shard} device={configs.device} done={len(done)}',
          flush=True)

    for stem in stems:
        name = os.path.basename(stem)
        idx = int(name.split('_')[-1])
        if idx % A.shard_w != A.shard_i:
            continue
        wanted = [(r, s, a) for r in range(A.replicates) for s in A.sigmas
                  for a in A.arms]
        if all(rec_key(name, r, s, a) in done for r, s, a in wanted):
            continue
        inst = Instance(stem)
        plans = {}
        for kind in sorted({INITIAL[a] for a in A.arms}):
            rec, info = initial_plan(inst, name, idx, kind, policy)
            # zero-noise execution reproduces the plan or improves it (a
            # vehicle may head for its next pickup before the module is ready
            # or its machine is committed); anything else is an executor bug
            zero = Realization(inst, np.zeros((3, inst.n_ops)), 0.0)
            res0 = dis.build_residual(inst.jl, inst.pt, inst.meta, rec,
                                      at=0.0, breakdown=False)
            ms0 = float(execute(inst, zero, rec, res0,
                                f'{name}/{kind}/zero')['op_ct'].max())
            assert ms0 <= rec['makespan'] + 1e-6, \
                f'{name}/{kind}: zero-noise execution {ms0} > plan ' \
                f'{rec["makespan"]}'
            plans[kind] = (rec, info, ms0)

        for rep in range(A.replicates):
            noise_seed, u = noise_draw(idx, rep, inst.n_ops)
            for sigma in A.sigmas:
                real = Realization(inst, u, sigma)
                for arm in A.arms:
                    key = rec_key(name, rep, sigma, arm)
                    if key in done:
                        continue
                    plan0, pinfo, ms0 = plans[INITIAL[arm]]
                    w0 = time.perf_counter()
                    hist, log = closed_loop(inst, name, idx, rep, real, arm,
                                            plan0, policy)
                    ms = float(np.asarray(hist['op_ct']).max())
                    row = dict(
                        key=key, cell=A.cell, instance=name, replicate=rep,
                        sigma=sigma, arm=arm, noise_seed=noise_seed,
                        realized_makespan=round(ms, 6),
                        nominal_makespan=round(float(plan0['makespan']), 6),
                        zero_noise_makespan=round(ms0, 6),
                        n_replans=len(log['checkpoints']),
                        replan_wall_mean=(round(float(np.mean(
                            log['plan_wall_s'])), 4)
                            if log['plan_wall_s'] else None),
                        replan_cpu_mean=(round(float(np.mean(
                            log['plan_cpu_s'])), 4)
                            if log['plan_cpu_s'] else None),
                        **log,
                        initial_plan=INITIAL[arm],
                        initial_plan_wall_s=pinfo['wall_s'],
                        initial_plan_cpu_s=pinfo['cpu_s'],
                        loop_wall_s=round(time.perf_counter() - w0, 3),
                        validated=True, checkpoints_k=A.checkpoints,
                        handoff=A.handoff,
                        replan_threshold=(A.replan_threshold
                                          if arm == 'policy_threshold'
                                          else None),
                        ga_nominal_budget=A.ga_nominal_budget,
                        ga_replan_budget=A.ga_replan_budget,
                        cpsat_budget=A.cpsat_budget,
                        cpsat_workers=A.cpsat_workers, ga_pop=A.ga_pop,
                        model=(A.model_name if INITIAL[arm] == 'policy'
                               else None),
                        device=configs.device, threads=A.threads,
                        torch_threads=A.torch_threads, cores=A.cores)
                    append_row(jsonl, row)
                    done.add(key)
                    print(f'{name} r{rep} s={sigma:g} {arm:13s} '
                          f'ms={ms:9.3f} nominal={plan0["makespan"]:9.3f} '
                          f'replans={row["n_replans"]} '
                          f'plan_cpu={row["replan_cpu_mean"]} '
                          f'loop={row["loop_wall_s"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()
