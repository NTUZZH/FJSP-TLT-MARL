"""GA metaheuristic baseline for FJSP-TL-T (transport-aware), Paper X2 T2/E1.

The transport analogue of the house eval_ga.py (base FJSP-TL GA): same
time-budgeted, PDR-seeded, elitist design, but the chromosome and decoder are
adapted to the joint machine+vehicle decision model of the reference simulator
(proposal Appendix B: "GA pop 100, 60 s, PDR-seeded").

WHY THIS DESIGN
  chromosome = (op_prio [N], mch_pref [N], veh_prio [J])  -- three real-key
  vectors (random-key GA in the Bean tradition):
    * op_prio[o]  : priority key of global op o (smaller = sooner)
    * mch_pref[o] : machine-preference key; frac(mch_pref[o]) hashes to a
                    preferred index into op o's eligible-machine list, and
                    eligible machines are ranked by circular distance from it
    * veh_prio[j] : vehicle-side key of job j (transport tasks are per-job)

  DECODE rolls the reference TransportSim (sim_single.py) event by event:
    - machine-class event: among the AVAILABLE (job, mch) pairs offered by the
      sim, pick the pair minimising (op_prio[o], mch_rank(o, m), j, m)
    - vehicle event: among the pending tasks, pick the job minimising
      (veh_prio[job], job)
  The decode is deterministic and stays inside the sim's constructive class,
  so EVERY chromosome maps to a feasible schedule -- no repair needed. All
  transport/reservation/lag semantics live in the sim (single source of
  truth, same oracle the PDR baselines and the validator cross-check).

INIT (PDR-SEEDED)
  Population 100 = 9 seeds encoding the 9 PDR-pair rollouts (pdr_pairs.py:
  {SPT,MWKR,FIFO} x {STT,NVF,FIFO}) + 91 random chromosomes. Seeds are LOOSE
  encodings: op_prio = commit-order ranks of the PDR run, mch_pref = the key
  that hashes exactly to the machine the PDR chose, veh_prio = first-serve
  ranks. A static per-job vehicle key cannot replay every per-event vehicle
  choice, so a seed decode may differ slightly from its PDR makespan -- the
  seeds put the initial front AT PDR level and the budget is spent beating it.

OPERATORS (proposal Appendix B)
  tournament selection (size 2), uniform crossover per vector, gaussian
  per-gene mutation (p = 0.1, sigma = 0.1), elitism 5, population 100,
  wall-clock budget per instance (default 60 s), single-threaded.

VALIDATION
  The caller re-decodes the winner and must pass the schedule_record through
  transport_marl/validator_t.py (shares no code with the sim) before
  reporting. See scripts/p3_eval_ga.py.
"""

import time

import numpy as np

from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES

MCH_NAMES = ('SPT', 'MWKR', 'FIFO')
VEH_NAMES = ('STT', 'NVF', 'FIFO')


# --------------------------------------------------------------------------- #
# Eligibility tables (per instance, decode-time lookups)
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
# Decode: chromosome -> feasible schedule via the reference sim
# --------------------------------------------------------------------------- #
def decode(sim, elig_tbl, op_prio, mch_pref, veh_prio):
    """Deterministic key-decode. Returns the makespan; the sim is left holding
    the full schedule (sim.schedule_record()) for validation/inspection."""
    elig, pos = elig_tbl
    # preferred eligible-machine index per op: frac(key) hashed onto [0, L)
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
                # machines ranked by circular distance from the preferred one
                key = (op_prio[o], (pos[o][m] - pref[o]) % lens[o], j, m)
                if best_key is None or key < best_key:
                    best_key, bj, bm = key, j, m
            sim.commit(bj, bm)
        else:
            job = min(ev[3], key=lambda t: (veh_prio[t['job']], t['job']))['job']
            sim.serve(ev[2], job)
    return float(sim.op_ct.max())


# --------------------------------------------------------------------------- #
# PDR seeding: roll the 9 rule pairs, encode each schedule loosely into keys
# --------------------------------------------------------------------------- #
def _rollout_recording(sim, mch_name, veh_name):
    """PDR-pair rollout that also records commit order and serve order."""
    mch_rule, veh_rule = MCH_RULES[mch_name], VEH_RULES[veh_name]
    sim.reset()
    commit_seq, serve_seq = [], []
    while True:
        ev = sim.peek_event()
        if ev[0] == 'done':
            break
        if ev[0] == 'mch':
            j, m = mch_rule(sim, ev[2])
            commit_seq.append(int(sim.next_op[j]))
            sim.commit(j, m)
        else:
            job = veh_rule(sim, ev[2], ev[3])
            serve_seq.append(int(job))
            sim.serve(ev[2], job)
    return (float(sim.op_ct.max()), commit_seq, serve_seq,
            sim.assigned_mch.copy())


def _encode_schedule(commit_seq, serve_seq, assigned_mch, elig_tbl, n_jobs):
    """Loose key-encoding of a PDR schedule (see module docstring)."""
    elig, pos = elig_tbl
    n_ops = len(assigned_mch)
    op_prio = np.empty(n_ops)
    for r, o in enumerate(commit_seq):
        op_prio[o] = (r + 0.5) / n_ops
    # mch_pref key whose hash lands exactly on the machine the PDR chose
    mch_pref = np.array([(pos[o][int(assigned_mch[o])] + 0.5) / len(elig[o])
                         for o in range(n_ops)])
    first_serve = {}
    for r, j in enumerate(serve_seq):
        first_serve.setdefault(j, r)
    n_served = max(len(first_serve), 1)
    veh_prio = np.array([(first_serve[j] + 0.5) / n_served
                         if j in first_serve else 1.0 + j / n_jobs
                         for j in range(n_jobs)])
    return op_prio, mch_pref, veh_prio


def pdr_seed_chromosomes(sim, elig_tbl):
    """Returns (seed chromosomes [9], {'SPT+STT': makespan, ...}).

    The makespans are the TRUE PDR-pair results (identical to p1_eval_pdr);
    min over them is the "seed makespan" reported next to the GA result.
    """
    seeds, pair_ms = [], {}
    for mn in MCH_NAMES:
        for vn in VEH_NAMES:
            ms, commits, serves, amch = _rollout_recording(sim, mn, vn)
            seeds.append(_encode_schedule(commits, serves, amch,
                                          elig_tbl, sim.n_j))
            pair_ms[f'{mn}+{vn}'] = ms
    return seeds, pair_ms


# --------------------------------------------------------------------------- #
# GA operators
# --------------------------------------------------------------------------- #
def random_individual(rng, n_ops, n_jobs):
    return (rng.random(n_ops), rng.random(n_ops), rng.random(n_jobs))


def tournament(rng, pop_fit, k=2):
    cand = rng.integers(0, len(pop_fit), size=k)
    return cand[np.argmin(pop_fit[cand])]


def uniform_crossover(rng, p1, p2):
    """Gene-wise 50/50 mix, independently on each of the three key vectors."""
    return tuple(np.where(rng.random(len(a)) < 0.5, a, b)
                 for a, b in zip(p1, p2))


def mutate(rng, ind, p=0.1, sigma=0.1):
    """Gaussian per-gene mutation. Any real key vector decodes feasibly, so
    no repair is ever needed (mch_pref is read mod 1 in the decoder)."""
    out = []
    for vec in ind:
        vec = vec.copy()
        mask = rng.random(len(vec)) < p
        vec[mask] += rng.normal(0.0, sigma, int(mask.sum()))
        out.append(vec)
    return tuple(out)


# --------------------------------------------------------------------------- #
# GA driver (one instance, time-budgeted)
# --------------------------------------------------------------------------- #
def run_ga(sim, elig_tbl, rng, budget_s, pop_size=100, elite=5, tour_k=2,
           seeds=None):
    """Time-budgeted GA. Returns (best_makespan, best_chromosome, n_gens).

    seeds : optional list of chromosomes (the 9 PDR-pair encodings) placed in
            the initial population; elitism keeps the incumbent monotone.
    """
    n_ops, n_jobs = sim.n_ops, sim.n_j

    pop = [tuple(v.copy() for v in s) for s in (seeds or [])][:pop_size]
    pop += [random_individual(rng, n_ops, n_jobs)
            for _ in range(pop_size - len(pop))]
    pop_fit = np.array([decode(sim, elig_tbl, *ind) for ind in pop])

    best_idx = int(np.argmin(pop_fit))
    best = tuple(v.copy() for v in pop[best_idx])
    best_fit = float(pop_fit[best_idx])

    t0 = time.time()
    gen = 0
    while time.time() - t0 < budget_s:
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
