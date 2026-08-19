"""Recovery from a mid-execution machine breakdown under a wall-clock budget.

WHAT THIS MEASURES AND WHY
  Given 60 to 3600 CPU seconds per instance, search finds schedules as good as
  the learned policy's. A factory does not have 3600 seconds when a station
  fails halfway through the week: it has the time between the alarm and the
  next handling decision. This script re-plans a disrupted schedule under
  wall-clock budgets of 1, 10 and 60 seconds and reports the makespan each
  method actually realizes.

PROTOCOL (transport_marl/disruption.py implements it; do not vary it here)
  A baseline schedule of the instance gives C0. At t = 0.30 * C0 the machine
  with the largest remaining processing workload breaks down for d = 0.20 * C0.
  Operations that started before t keep their machine, vehicle and start time;
  an operation running on the broken machine is aborted and re-planned with its
  processing restarted from scratch. Five methods then re-plan the residual:

    right_shift  keep every decision, delay what the breakdown invalidated.
                 The no-search industrial reference, and the warm start of the
                 two searches below.
    pdr          re-dispatch the residual with the nine rule pairs, best taken.
    ga           PDR-seeded GA v2 on the residual, wall-clock budget B, with
                 the right-shift schedule carried as an incumbent (the GA's
                 key-decode class is non-delay and cannot represent a
                 right-shifted schedule, so it is carried rather than seeded).
    cpsat        the strengthened residual CP-SAT model warm-started from the
                 right-shift schedule, wall-clock budget B. Its horizon is the
                 warm-start makespan, so it can only return that or better.
    policy       the trained policy continues its event-driven rollout from
                 the disrupted state. It gets no budget; the wall time it
                 needs is reported.
    policy_sample the same continuation decoded best-of-N: candidate 0 is the
                 greedy rollout and the rest are multinomial draws from the
                 same policy, under wall-clock budget B. This is the
                 matched-compute lever of scripts/x2_eval_sample.py, and it
                 is the only policy arm that spends a budget, so the number
                 of candidates B actually bought is recorded next to the
                 makespan.

  Every recovered schedule is re-checked by validator_t.validate_recovered_
  schedule and the run aborts on the first failure.

THREADS. The policy arm's wall time is one forward pass per event, and a
batch-1 forward is far too small to parallelize: on a 10-module instance it
costs 1.7 ms at one torch thread and 142 ms at eight, because the per-operator
thread barriers dominate the arithmetic. --torch_threads (default 1) therefore
sizes torch independently of --threads, which stays matched to the pinned core
set for the numerical libraries the GA and CP-SAT arms use. Neither the
residual state injection nor the environment is the cost: injection takes
under 10 ms and one env.step is 0.8 ms.

Output: results/disruption/{cell}.jsonl, one JSON record per (instance,
method, budget), appended and fsynced as it is produced, keyed so a rerun
skips finished work.

Usage:
  python -u scripts/x2_disruption.py --cell 10x25+ppvct-mixed+v2+t1.0 \
      --model_name 10x25+ppvct-mixed+m1-bcb-guide-s301 \
      --budgets 1,10,60 --n 3 --device cuda --cores 0-7
  python -u scripts/x2_disruption.py --cell CELL --model_name M \
      --baseline policy --methods policy,policy_sample --budgets 60
"""

import argparse
import glob
import json
import os
import sys
import time


ALL_METHODS = ('right_shift', 'pdr', 'policy', 'policy_sample', 'ga', 'cpsat')
BUDGETED = ('policy_sample', 'ga', 'cpsat')


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', type=str, required=True)
    ap.add_argument('--model_name', type=str, required=True)
    ap.add_argument('--budgets', type=str, default='1,10,60')
    ap.add_argument('--n', type=int, default=0, help='first N instances; 0=all')
    ap.add_argument('--device', type=str, default='cpu',
                    choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--cores', type=str, default='',
                    help='cpu range to pin to, e.g. 0-7')
    ap.add_argument('--threads', type=int, default=8,
                    help='OMP/MKL cap for the numerical libraries the solver '
                         'arms use; must match the pinned core count')
    ap.add_argument('--torch_threads', type=int, default=1,
                    help='torch intra-op threads for the policy arms, '
                         'SEPARATE from --threads. One batch-1 forward pass is '
                         'far too small to parallelize: on a 10-module '
                         'instance it costs 1.7 ms at one thread, 3.2 ms at '
                         'four and 142 ms at eight, because the per-operator '
                         'thread barriers dominate the arithmetic. Leave this '
                         'at 1 unless a measurement on the target instance '
                         'size says otherwise.')
    ap.add_argument('--cpsat_workers', type=int, default=4)
    ap.add_argument('--methods', type=str, default=','.join(ALL_METHODS),
                    help='subset of ' + ','.join(ALL_METHODS) + '. The '
                         'right-shift repair is computed either way, because '
                         'it is the warm start of ga and cpsat; --methods '
                         'only decides what gets recorded.')
    ap.add_argument('--sample_cap', type=int, default=0,
                    help='hard cap on policy_sample candidates; 0 = the '
                         'wall-clock budget is the only stop')
    ap.add_argument('--ga_pop', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--baseline', type=str, default='pdr',
                    choices=['pdr', 'policy'],
                    help='which scheduler produces the pre-disruption plan. '
                         'One plan per instance is shared by every recovery '
                         'method, so the disruption and the frozen prefix are '
                         'identical across methods.')
    ap.add_argument('--t_frac', type=float, default=0.30)
    ap.add_argument('--d_frac', type=float, default=0.20)
    ap.add_argument('--outdir', type=str, default='results/disruption')
    args = ap.parse_args()
    args.methods = [m for m in args.methods.split(',') if m]
    bad = [m for m in args.methods if m not in ALL_METHODS]
    assert not bad, f'unknown methods {bad}; pick from {ALL_METHODS}'
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


SNAP = load_snapshot(A.model_name)
# configs.device must be set before fjsp_env_same_op_nums is imported: its
# EnvState reads it once, at class-definition time, to place the state tensors.
# Nothing imported above pulls that module in.
if A.device == 'auto':
    configs.device = 'cuda' if torch.cuda.device_count() else 'cpu'
else:
    configs.device = A.device
assert configs.device != 'cuda' or torch.cuda.device_count(), \
    '--device cuda requested but torch sees no GPU; refusing to fall back to ' \
    'CPU silently, because the policy arm would then report a latency the ' \
    'deployment does not have'
# torch gets its OWN thread count, not the process-wide --threads: the policy
# arm's wall time is the headline quantity of this experiment, and sizing
# torch's pool from the pinned core set inflates it by two orders of magnitude
# (see --torch_threads). set_num_threads overrides OMP_NUM_THREADS for torch
# only, so the solver arms keep the cap they need.
torch.set_num_threads(A.torch_threads)
try:
    torch.set_num_interop_threads(A.torch_threads)
except RuntimeError:
    pass          # already fixed by an earlier parallel region; intra-op is
    # the one that matters here

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
    return policy, algo


def make_env(jl, pt, meta):
    guide = bool(getattr(configs, 'guide', False))
    env = FJSPEnvTransport(len(jl), pt.shape[1], use_lag_features=True,
                           use_guide=guide,
                           guide_price=SNAP.get('guide_price', 'certified'),
                           guide_price_scale=float(
                               SNAP.get('guide_price_scale', 1.0)))
    env.set_initial_data([np.asarray(jl)], [np.asarray(pt, dtype=float)],
                         [np.asarray(meta['time_lag'], dtype=float)],
                         [np.asarray(meta['op_type'])],
                         [np.asarray(meta['mch_type'])],
                         [meta['transport']])
    if guide:
        from transport_marl.bound import TransportBound
        env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
    return env


def rollout(env, policy, greedy=True):
    """Two-head rollout to completion.

    Returns (makespan, events acted on, seconds inside the forward pass). The
    forward-pass total is kept apart from the wall clock because that is the
    quantity scripts/x2_scale_policy.py reports as its latency; the difference
    between the two is the environment's own step cost, and the paper must not
    compare one against the other by accident.

    greedy=False draws each action from the routed head instead of taking its
    argmax, which is the sampled-decoding convention of x2_eval_sample.py."""
    state = env.state
    steps, nn_s = 0, 0.0
    while env.done().min() < 1:
        steps += 1
        t_nn = time.perf_counter()
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
                                  greedy=greedy)
        a = a.cpu().numpy()
        nn_s += time.perf_counter() - t_nn
        state, _, _ = env.step(a)
    if configs.device == 'cuda':
        torch.cuda.synchronize()
    return float(env.current_makespan[0]), steps, nn_s


# --------------------------------------------------------------------------- #
# baseline plan (one per instance, shared by every recovery method)
# --------------------------------------------------------------------------- #
def fresh_sim(jl, pt, meta):
    tr = meta['transport']
    return TransportSim(jl, pt, meta['time_lag'], tr['station_cell'],
                        tr['tau_cells'], int(tr['n_vehicles']),
                        int(tr.get('veh_start_cell', 0)))


def best_pdr(sim):
    """Best of the nine rule pairs on whatever state `sim` restores to."""
    best_ms, best_rec, best_pair = np.inf, None, None
    for mn in MCH_RULES:
        for vn in VEH_RULES:
            sim.reset()
            ms = sim.run(MCH_RULES[mn], VEH_RULES[vn])
            if ms < best_ms:
                best_ms, best_rec = ms, sim.schedule_record()
                best_pair = f'{mn}+{vn}'
    return float(best_ms), best_rec, best_pair


def baseline_plan(jl, pt, meta, policy):
    if A.baseline == 'pdr':
        ms, rec, pair = best_pdr(fresh_sim(jl, pt, meta))
        return rec, ms, dict(baseline_method='pdr', baseline_pair=pair)
    env = make_env(jl, pt, meta)
    ms, _, _ = rollout(env, policy)
    rec = env.schedule_record(0)
    del env
    return rec, float(ms), dict(baseline_method='policy')


# --------------------------------------------------------------------------- #
# recovery methods
# --------------------------------------------------------------------------- #
def recover_right_shift(jl, pt, meta, res):
    t0 = time.time()
    rec = dis.right_shift_repair(jl, pt, meta, res)
    return rec, time.time() - t0, {}


def recover_pdr(jl, pt, meta, res):
    t0 = time.time()
    sim = dis.residual_sim(jl, pt, meta, res)
    ms, rec, pair = best_pdr(sim)
    return rec, time.time() - t0, dict(pdr_pair=pair)


def recover_ga(jl, pt, meta, res, budget, rs_rec, idx):
    t0 = time.time()
    sim = dis.residual_sim(jl, pt, meta, res)
    elig = ga.eligible_machines(np.asarray(pt, dtype=float))
    rng = np.random.default_rng(A.seed * 100003 + idx)
    seeds, pair_ms = ga.pdr_seed_chromosomes(sim, elig, rng)
    for nm, s in zip(pair_ms.keys(), seeds):
        d = ga.decode(sim, elig, *s)
        assert abs(d - pair_ms[nm]) < 1e-9, \
            f'residual GA seed {nm} decodes {d} != PDR {pair_ms[nm]}'
    t_search = time.time()
    ga_ms, best, gens = ga.run_ga(sim, elig, rng, budget, pop_size=A.ga_pop,
                                  seeds=seeds, clock=time.time)
    search_s = time.time() - t_search
    ga.decode(sim, elig, *best)
    rec = sim.schedule_record()
    assert abs(rec['makespan'] - ga_ms) < 1e-6, 'GA re-decode is not stable'
    extra = dict(gens=int(gens), search_s=round(search_s, 3),
                 seed_best=float(min(pair_ms.values())),
                 ga_own=float(ga_ms), used_warmstart=False)
    if rs_rec['makespan'] < ga_ms - 1e-9:
        # the right-shift schedule is not in the GA's non-delay decode class,
        # so it is carried as an incumbent rather than encoded as a seed
        rec, extra['used_warmstart'] = rs_rec, True
    return rec, time.time() - t0, extra


def recover_cpsat(jl, pt, meta, res, budget, rs_rec):
    t0 = time.time()
    sol = solve_transport_instance(
        jl, pt, meta, time_limit=budget, n_workers=A.cpsat_workers,
        warmstart=rs_rec, strengthen=True,
        op_release=res['op_release'], mch_ready=res['mch_ready'],
        fixed_ops=res['fixed_ops'], veh_ready=res['veh_free'],
        veh_cells=res['veh_cell'], delivered_ops=res['delivered_ops'])
    sw = sol.get('walltime')
    extra = dict(status=sol['status'], lb=sol.get('objective_bound'),
                 solver_walltime=None if sw is None else round(sw, 3),
                 cpsat_own=sol.get('makespan'), used_warmstart=False)
    if sol['makespan'] is None or sol['makespan'] > rs_rec['makespan'] - 1e-9:
        # a warm-started anytime solver never gives up its incumbent
        return rs_rec, time.time() - t0, dict(extra, used_warmstart=True)
    tau = np.asarray(meta['transport']['tau_cells'], dtype=float)
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
                to=int(m['to']), depart=pickup - float(tau[loc, int(m['frm'])]),
                pickup=pickup,
                arrival=pickup + float(tau[int(m['frm']), int(m['to'])])))
            loc = int(m['to'])
    ct = np.asarray(sol['op_ct'], dtype=float)
    rec = dict(assigned_mch=np.asarray(sol['assigned_mch'], dtype=int),
               op_start=np.asarray(sol['op_start'], dtype=float), op_ct=ct,
               transports=transports, makespan=float(ct.max()))
    assert abs(rec['makespan'] - sol['makespan']) < 1e-6, \
        f"CP-SAT objective {sol['makespan']} != rebuilt {rec['makespan']}"
    return rec, time.time() - t0, extra


def residual_env(jl, pt, meta, res):
    env = make_env(jl, pt, meta)
    dis.load_env_residual(env, 0, jl, pt, meta, res)
    env.refresh_state()
    return env


def recover_policy(jl, pt, meta, res, policy):
    t0 = time.time()
    env = residual_env(jl, pt, meta, res)
    t_roll = time.time()
    ms, steps, nn_s = rollout(env, policy)
    rollout_s = time.time() - t_roll
    rec = env.schedule_record(0)
    assert abs(rec['makespan'] - ms) < 1e-6, 'env makespan disagrees with record'
    del env
    return rec, time.time() - t0, dict(
        rollout_s=round(rollout_s, 4), decisions=int(steps),
        # s_per_decision is forward pass PLUS environment step, which is what a
        # deployment pays; s_per_decision_nn is the forward pass alone, the
        # quantity scripts/x2_scale_policy.py reports
        s_per_decision=round(rollout_s / max(steps, 1), 6),
        s_per_decision_nn=round(nn_s / max(steps, 1), 6),
        nn_s=round(nn_s, 4))


def recover_policy_sample(jl, pt, meta, res, budget, policy, checker):
    """Best-of-N sampled continuation under a wall-clock budget.

    Candidate 0 is the greedy continuation, so this arm can never be worse
    than the `policy` arm. Candidates 1.. are multinomial draws seeded
    torch.manual_seed(1000 + p), the convention scripts/x2_eval_sample.py
    fixed, so the draw sequence is reproducible. Only an improving candidate
    is validated and kept.

    The clock is tested between candidates, never inside one: a half-finished
    rollout is not a schedule. The arm therefore overruns B by at most one
    candidate, exactly as the GA overruns it by at most one generation, and
    `candidates` plus the realized `wall_s` say what B actually bought.
    """
    t0 = time.time()
    best_rec, best_ms, drawn, cand_ms = None, np.inf, 0, []
    greedy_ms, greedy_steps = None, 0
    while True:
        torch.manual_seed(1000 + drawn)
        env = residual_env(jl, pt, meta, res)
        ms, steps, _ = rollout(env, policy, greedy=(drawn == 0))
        rec = env.schedule_record(0)
        del env
        if drawn == 0:
            # keep the greedy makespan UNROUNDED: comparing the running best
            # against a rounded copy of itself reports a spurious improvement
            greedy_ms, greedy_steps = float(ms), steps
        drawn += 1
        cand_ms.append(round(float(ms), 6))
        if ms < best_ms - 1e-9:
            checker(rec, f'policy_sample cand {drawn - 1}')
            best_rec, best_ms = rec, float(ms)
        if time.time() - t0 >= budget:
            break
        if A.sample_cap and drawn >= A.sample_cap:
            break
    return best_rec, time.time() - t0, dict(
        candidates=drawn, greedy_ms=round(greedy_ms, 6),
        improved_on_greedy=bool(best_ms < greedy_ms - 1e-9),
        candidate_ms=cand_ms, decisions=int(greedy_steps))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def check(jl, pt, meta, res, rec, tag):
    tr = meta['transport']
    out = validate_recovered_schedule(
        jl, pt, meta['time_lag'], np.asarray(tr['station_cell']),
        np.asarray(tr['tau_cells']), int(tr['n_vehicles']),
        int(tr.get('veh_start_cell', 0)), rec, res,
        job_start_cell=int(tr.get('job_start_cell', -1)))
    if not out['feasible']:
        raise AssertionError(f'{tag}: recovered schedule INVALID: '
                             f'{out["violations"][:5]}')
    return out['makespan']


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
    if A.n:
        stems = stems[:A.n]
    budgets = [float(b) for b in A.budgets.split(',')]
    policy, algo = build_policy()
    print(f'[x2_disruption] cell={A.cell} model={A.model_name} algo={algo} '
          f'device={configs.device} baseline={A.baseline} '
          f'methods={",".join(A.methods)} budgets={budgets} '
          f'n={len(stems)} done={len(done)}', flush=True)

    def rec_key(name, method, budget):
        # The key must name the checkpoint whenever the record depends on it,
        # or a second training seed writing to the same cell file would find
        # every key already present and skip the whole run: the three-seed
        # design would silently collapse to one seed. Two ways a record
        # depends on the checkpoint: the arm is a policy arm, or the plan the
        # breakdown hits is the policy's own schedule, which makes even the
        # dispatching and search arms seed-dependent through their residual.
        # Under --baseline pdr the non-policy arms are seed-independent and
        # stay unkeyed, so they are computed once and reused across seeds.
        seeded = A.baseline == 'policy' or method in ('policy', 'policy_sample')
        tag = f'|{A.model_name}' if seeded else ''
        return (f'{A.cell}|{name}|{A.baseline}|{method}|'
                f'{"na" if budget is None else f"B{budget:g}"}{tag}')

    wanted = [(m, None) for m in A.methods if m not in BUDGETED]
    wanted += [(m, b) for b in budgets for m in A.methods if m in BUDGETED]

    for idx, stem in enumerate(stems):
        name = os.path.basename(stem)
        # the baseline plan and the right-shift are shared by every method, so
        # skip the instance before paying for them when nothing is left to do
        if all(rec_key(name, m, b) in done for m, b in wanted):
            continue
        jl, pt, meta = load_instance(stem)
        pt = np.asarray(pt, dtype=float)
        base_rec, c0, base_info = baseline_plan(jl, pt, meta, policy)
        res = dis.build_residual(jl, pt, meta, base_rec,
                                 t_frac=A.t_frac, d_frac=A.d_frac)
        rs_rec, rs_wall, _ = recover_right_shift(jl, pt, meta, res)
        rs_ms = check(jl, pt, meta, res, rs_rec, f'{name}/right_shift')

        def checker(rec, tag):
            return check(jl, pt, meta, res, rec, f'{name}/{tag}')

        runners = {
            'right_shift': lambda b: (rs_rec, rs_wall, {}),
            'pdr': lambda b: recover_pdr(jl, pt, meta, res),
            'policy': lambda b: recover_policy(jl, pt, meta, res, policy),
            'policy_sample': lambda b: recover_policy_sample(
                jl, pt, meta, res, b, policy, checker),
            'ga': lambda b: recover_ga(jl, pt, meta, res, b, rs_rec, idx),
            'cpsat': lambda b: recover_cpsat(jl, pt, meta, res, b, rs_rec),
        }
        jobs = [(m, b, (lambda m=m, b=b: runners[m](b))) for m, b in wanted]

        for method, budget, run in jobs:
            key = rec_key(name, method, budget)
            if key in done:
                continue
            rec, wall, extra = run()
            ms = check(jl, pt, meta, res, rec, f'{name}/{method}')
            assert ms >= res['lower_bound'] - 1e-6, \
                f'{name}/{method}: makespan {ms} below disrupted lower bound ' \
                f'{res["lower_bound"]}'
            # only ga and cpsat are warm-started from the right shift; the
            # dispatching and policy arms re-plan independently of it
            assert ms <= rs_ms + 1e-6 or method not in ('ga', 'cpsat'), \
                f'{name}/{method}: {ms} worse than the right-shift warm ' \
                f'start {rs_ms}'
            row = dict(
                key=key, cell=A.cell, instance=name, method=method,
                budget_s=budget, makespan=round(float(ms), 6),
                wall_s=round(float(wall), 4),
                baseline_makespan=round(float(c0), 6),
                right_shift_makespan=round(float(rs_ms), 6),
                disruption_t=res['t'], downtime=res['downtime'],
                broken_machine=res['broken'],
                broken_workload=round(res['broken_workload'], 6),
                n_frozen=res['n_frozen'], n_aborted=len(res['aborted']),
                lower_bound=round(float(res['lower_bound']), 6),
                validated=True, t_frac=A.t_frac, d_frac=A.d_frac,
                model=A.model_name, device=configs.device,
                cpsat_workers=A.cpsat_workers, threads=A.threads,
                cores=A.cores, **base_info, **extra)
            with open(jsonl, 'a') as f:
                f.write(json.dumps(row) + '\n')
                f.flush()
                os.fsync(f.fileno())
            done.add(key)
            print(f'{name} {method:11s} '
                  f'B={"-" if budget is None else f"{budget:g}s":>4s} '
                  f'ms={ms:9.3f} wall={wall:7.3f}s '
                  f'(C0={c0:.2f} t={res["t"]:.2f} d={res["downtime"]:.2f} '
                  f'mch={res["broken"]})', flush=True)


if __name__ == '__main__':
    main()
