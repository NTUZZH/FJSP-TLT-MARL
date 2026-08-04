"""GA metaheuristic baseline for FJSP-TL-T, v2 (fixed vehicle encoding).

WHY v2 EXISTS
  The v1 GA (transport_marl/ga_transport.py, results/ga/) encoded the vehicle
  dimension as ONE STATIC KEY PER JOB, veh_prio[J]. At a vehicle event it
  picked the pending task minimising (veh_prio[job], job), so the vehicle
  policy was a fixed total order over jobs, replayed unchanged at every
  vehicle event and blind to where the vehicle actually was.

  The three PDR vehicle rules are location-adaptive: STT and NVF both read the
  vehicle's CURRENT cell. No static job order can express them. Consequently
  the "PDR seeds" did not encode the PDR schedules at all: on 10x25 v1+t0.6
  and v1+t1.0 the seed chromosomes decoded 38-45% WORSE than the PDR
  schedules they were supposed to encode, and were beaten by random
  chromosomes. The GA converged ~25% above best-of-nine PDR, the caller fell
  back to reporting the PDR schedule, and the cell recorded 0/100 improved.
  Diagnosis with evidence: notes/ga_diagnosis.md.

  The loss was entirely on the vehicle side. Decoding the same seed
  chromosomes with the machine-side keys but the true vehicle rule reproduced
  all nine PDR makespans EXACTLY (0.00 error, 36/36 seed-instance pairs), so
  the machine-side encoding (op_prio, mch_pref) is exact and is kept unchanged.

WHAT CHANGED (vehicle side only)
  chromosome = (op_prio [N], mch_pref [N], veh_w [4], veh_prio [N])
    * op_prio, mch_pref : UNCHANGED from v1 (per-op priority key and
      machine-preference key; see ga_transport.py docstring)
    * veh_w [4]  : weights of a composite, state-dependent vehicle key
    * veh_prio [N] : per-OP vehicle priority offset (v1 had J=10 static
      per-job keys for ~114 vehicle events; v2 has one per operation)

  At a vehicle event with vehicle at cell `loc` and candidate task set T, each
  task t is scored on three rule features and one free priority:
      f_stt (t) = tau[loc, frm] + tau[frm, to]     (shortest total travel)
      f_nvf (t) = tau[loc, frm]                    (nearest vehicle first)
      f_fifo(t) = release[t]                       (first released first)
      p     (t) = frac(veh_prio[op(t)])            (free per-op key)
  The three rule features are min-max normalised over T (a monotone map, so
  the argmin of any single feature is unchanged), and
      key(t) = w0*n_stt + w1*n_nvf + w2*n_fifo + w3*p ,  ties by job index.

  This policy class CONTAINS the three PDR vehicle rules exactly:
  w = (1,0,0,0) is STT, (0,1,0,0) is NVF, (0,0,1,0) is FIFO, with the same
  (value, job) tie-break. So the nine PDR seeds now decode to their true PDR
  makespans, elitism makes the incumbent monotone from PDR level, and any
  reported improvement is a real improvement over best-of-nine PDR. w3 > 0
  blends in per-op freedom, which lets the search deviate from a rule exactly
  where the rule is close to indifferent.

UNCHANGED FROM v1 (fairness)
  Population 100, elitism 5, tournament size 2, uniform crossover per vector,
  gaussian per-gene mutation (p = 0.1, sigma = 0.1), 60 s wall-clock budget
  per instance, single-threaded, PDR-seeded with the same nine rule pairs.
  The decode still rolls the reference TransportSim event by event and stays
  inside its constructive class, so every chromosome maps to a feasible
  schedule, no repair is ever needed, and the winner is re-checked by
  transport_marl/validator_t.py before anything is written.
"""

import time

import numpy as np

from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES

MCH_NAMES = ('SPT', 'MWKR', 'FIFO')
VEH_NAMES = ('STT', 'NVF', 'FIFO')

# veh_w index of the rule each vehicle-side PDR rule seeds to
VEH_RULE_SLOT = {'STT': 0, 'NVF': 1, 'FIFO': 2}
N_VEH_W = 4


# --------------------------------------------------------------------------- #
# Eligibility tables (per instance, decode-time lookups)  -- unchanged from v1
# --------------------------------------------------------------------------- #
def eligible_machines(op_pt):
    """Per op: (sorted eligible machine ids, {machine id -> position})."""
    elig, pos = [], []
    for o in range(op_pt.shape[0]):
        E = np.nonzero(op_pt[o] > 0)[0]
        elig.append(E)
        pos.append({int(m): k for k, m in enumerate(E)})
    return elig, pos


# --------------------------------------------------------------------------- #
# Vehicle-side composite key
# --------------------------------------------------------------------------- #
def _norm(vals):
    """Min-max normalise to [0, 1]; constant input -> all zeros.

    Monotone on the candidate set, so a unit weight on one feature selects
    exactly the same task as the corresponding PDR rule.
    """
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return [0.0] * len(vals)
    d = hi - lo
    return [(v - lo) / d for v in vals]


def _pick_vehicle_task(sim, veh, tasks, veh_w, veh_prio):
    """Composite-rule vehicle choice. Returns the chosen job."""
    if len(tasks) == 1:
        return tasks[0]['job']
    loc = sim.veh_loc[veh]
    tau = sim.tau
    f_stt, f_nvf, f_fifo = [], [], []
    for t in tasks:
        e = tau[loc, t['frm']]
        f_nvf.append(e)
        f_stt.append(e + tau[t['frm'], t['to']])
        f_fifo.append(t['release'])
    n_stt, n_nvf, n_fifo = _norm(f_stt), _norm(f_nvf), _norm(f_fifo)
    w0, w1, w2, w3 = veh_w
    best_key, best_job = None, -1
    for i, t in enumerate(tasks):
        p = veh_prio[t['op']] % 1.0
        key = (w0 * n_stt[i] + w1 * n_nvf[i] + w2 * n_fifo[i] + w3 * p,
               t['job'])
        if best_key is None or key < best_key:
            best_key, best_job = key, t['job']
    return best_job


# --------------------------------------------------------------------------- #
# Decode: chromosome -> feasible schedule via the reference sim
# --------------------------------------------------------------------------- #
def decode(sim, elig_tbl, op_prio, mch_pref, veh_w, veh_prio):
    """Deterministic key-decode. Returns the makespan; the sim is left holding
    the full schedule (sim.schedule_record()) for validation/inspection."""
    elig, pos = elig_tbl
    lens = np.array([len(E) for E in elig])
    pref = np.minimum((np.mod(mch_pref, 1.0) * lens).astype(int), lens - 1)

    sim.reset()
    while True:
        ev = sim.peek_event()
        if ev[0] == 'done':
            break
        if ev[0] == 'mch':
            best_key, bj, bm = None, -1, -1
            for (j, m, o) in ev[2]:
                key = (op_prio[o], (pos[o][m] - pref[o]) % lens[o], j, m)
                if best_key is None or key < best_key:
                    best_key, bj, bm = key, j, m
            sim.commit(bj, bm)
        else:
            sim.serve(ev[2], _pick_vehicle_task(sim, ev[2], ev[3],
                                                veh_w, veh_prio))
    return float(sim.op_ct.max())


# --------------------------------------------------------------------------- #
# PDR seeding: roll the 9 rule pairs, encode each schedule into keys
# --------------------------------------------------------------------------- #
def _rollout_recording(sim, mch_name, veh_name):
    """PDR-pair rollout that also records the commit order."""
    mch_rule, veh_rule = MCH_RULES[mch_name], VEH_RULES[veh_name]
    sim.reset()
    commit_seq = []
    while True:
        ev = sim.peek_event()
        if ev[0] == 'done':
            break
        if ev[0] == 'mch':
            j, m = mch_rule(sim, ev[2])
            commit_seq.append(int(sim.next_op[j]))
            sim.commit(j, m)
        else:
            sim.serve(ev[2], veh_rule(sim, ev[2], ev[3]))
    return float(sim.op_ct.max()), commit_seq, sim.assigned_mch.copy()


def _encode_schedule(commit_seq, assigned_mch, elig_tbl, veh_name, rng=None):
    """Exact key-encoding of a PDR schedule.

    Machine side as in v1 (commit-order ranks + the machine-preference key
    that hashes onto the machine the PDR chose). Vehicle side is now the
    one-hot rule weight, which reproduces the rule exactly; veh_prio carries
    zero weight in the seed and is filled with random keys so crossover has
    material to work with.
    """
    elig, pos = elig_tbl
    n_ops = len(assigned_mch)
    op_prio = np.empty(n_ops)
    for r, o in enumerate(commit_seq):
        op_prio[o] = (r + 0.5) / n_ops
    mch_pref = np.array([(pos[o][int(assigned_mch[o])] + 0.5) / len(elig[o])
                         for o in range(n_ops)])
    veh_w = np.zeros(N_VEH_W)
    veh_w[VEH_RULE_SLOT[veh_name]] = 1.0
    veh_prio = (rng.random(n_ops) if rng is not None else np.full(n_ops, 0.5))
    return op_prio, mch_pref, veh_w, veh_prio


def pdr_seed_chromosomes(sim, elig_tbl, rng=None):
    """Returns (seed chromosomes [9], {'SPT+STT': makespan, ...}).

    The makespans are the TRUE PDR-pair results (identical to p1_eval_pdr);
    min over them is the "seed makespan" reported next to the GA result. Each
    seed chromosome decodes EXACTLY to its pair's makespan (asserted by the
    caller), so the GA incumbent starts at PDR level.
    """
    seeds, pair_ms = [], {}
    for mn in MCH_NAMES:
        for vn in VEH_NAMES:
            ms, commits, amch = _rollout_recording(sim, mn, vn)
            seeds.append(_encode_schedule(commits, amch, elig_tbl, vn, rng))
            pair_ms[f'{mn}+{vn}'] = ms
    return seeds, pair_ms


# --------------------------------------------------------------------------- #
# GA operators  -- identical to v1 apart from the extra veh_w vector
# --------------------------------------------------------------------------- #
def random_individual(rng, n_ops, n_jobs):
    return (rng.random(n_ops), rng.random(n_ops),
            rng.random(N_VEH_W), rng.random(n_ops))


def tournament(rng, pop_fit, k=2):
    cand = rng.integers(0, len(pop_fit), size=k)
    return cand[np.argmin(pop_fit[cand])]


def uniform_crossover(rng, p1, p2):
    """Gene-wise 50/50 mix, independently on each key vector."""
    return tuple(np.where(rng.random(len(a)) < 0.5, a, b)
                 for a, b in zip(p1, p2))


def mutate(rng, ind, p=0.1, sigma=0.1):
    """Gaussian per-gene mutation. Any real key vector decodes feasibly, so
    no repair is ever needed (mch_pref and veh_prio are read mod 1)."""
    out = []
    for vec in ind:
        vec = vec.copy()
        mask = rng.random(len(vec)) < p
        vec[mask] += rng.normal(0.0, sigma, int(mask.sum()))
        out.append(vec)
    return tuple(out)


# --------------------------------------------------------------------------- #
# GA driver (one instance, time-budgeted)  -- identical to v1
# --------------------------------------------------------------------------- #
def run_ga(sim, elig_tbl, rng, budget_s, pop_size=100, elite=5, tour_k=2,
           seeds=None, clock=time.process_time):
    """Time-budgeted GA. Returns (best_makespan, best_chromosome, n_gens).

    `clock` defaults to time.process_time (CPU time of this worker process)
    rather than wall-clock. On an idle core the two are the same; under load
    the CPU-time budget still delivers the SAME amount of search, so the
    baseline cannot be silently weakened by whatever else the machine is
    running. That matters here because contention would bias the comparison
    in the policy's favour. Pass clock=time.time for a wall-clock budget.
    """
    n_ops, n_jobs = sim.n_ops, sim.n_j

    pop = [tuple(v.copy() for v in s) for s in (seeds or [])][:pop_size]
    pop += [random_individual(rng, n_ops, n_jobs)
            for _ in range(pop_size - len(pop))]
    pop_fit = np.array([decode(sim, elig_tbl, *ind) for ind in pop])

    best_idx = int(np.argmin(pop_fit))
    best = tuple(v.copy() for v in pop[best_idx])
    best_fit = float(pop_fit[best_idx])

    t0 = clock()
    gen = 0
    while clock() - t0 < budget_s:
        gen += 1
        order = np.argsort(pop_fit)
        new_pop = [tuple(v.copy() for v in pop[i]) for i in order[:elite]]
        new_fit = list(pop_fit[order[:elite]])          # elites keep fitness

        while len(new_pop) < pop_size:
            p1 = pop[tournament(rng, pop_fit, tour_k)]
            p2 = pop[tournament(rng, pop_fit, tour_k)]
            child = mutate(rng, uniform_crossover(rng, p1, p2))
            new_pop.append(child)
            new_fit.append(decode(sim, elig_tbl, *child))

        pop, pop_fit = new_pop, np.array(new_fit)
        cur = int(np.argmin(pop_fit))
        if pop_fit[cur] < best_fit - 1e-9:
            best_fit = float(pop_fit[cur])
            best = tuple(v.copy() for v in pop[cur])

    return best_fit, best, gen
