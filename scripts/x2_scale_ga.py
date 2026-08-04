"""PDR-seeded GA v2 on the scale-up cells (Phase A and Phase B).

The per-instance procedure below is copied verbatim from
scripts/p3_eval_ga_v2.py (pop 100, nine PDR-pair seed chromosomes, exact-decode
assertion on every seed, winner independently re-checked by validator_t). Only
the dataset path and the output directory differ, so results/ga_v2/ is never
written to and nothing can be swept into the manuscript's existing GA analysis.

The budget is metered with time.process_time, not wall-clock: a busy box
cannot silently weaken this baseline, which is the direction of bias that
would flatter the policy.

PHASE B ADDITIONS (cost accounting and anytime trace)
Phase A found that at 20-30 modules the fixed price of *starting* the GA
already exceeds the search budget the sweep was varying, so a bare "60 CPU-s"
label understates what the baseline costs a user. Three fields are therefore
recorded separately:

  startup_cpu : CPU seconds from instance load to the first generation, i.e.
                nine PDR rollouts + the nine exact-decode assertions + the
                initial population of 100 decodes. This is paid whatever the
                budget is, and it is NOT part of the metered budget.
  search_cpu  : CPU seconds actually consumed by the generation loop (the
                metered budget).
  cpu         : all-in CPU seconds for the instance, startup + search +
                the final re-decode and validator check. Same field Phase A
                wrote, so the two phases stay comparable.

  anytime     : [[cpu_s_since_search_start, best_makespan], ...], one entry
                per completed generation, plus a generation-0 entry at cpu
                0.0. That entry is the best of the whole initial population
                (nine exact PDR seeds and 91 random chromosomes), which can
                already be below the best PDR seed, and it is reached during
                startup rather than during the metered budget. Elitism makes
                the series monotone from there.

Note on the metered budget: the loop tests the clock only at generation
boundaries, so the last generation overshoots. `search_cpu` is what the search
actually consumed and is the honest number; the budget is a floor, not a cap.

`run_ga_traced` below is `ga_transport_v2.run_ga` with the trace appended and
nothing else changed: the same operators are called in the same order, so the
RNG stream and hence the search are identical. `--selftest` proves it, by
running both functions on a real instance with a deterministic counter clock
(so both do the same number of generations) and asserting that the returned
makespan, generation count and chromosome match exactly.

Output: results/scaleup/ga/{cell}.json      (60 CPU-s headline)
        results/scaleup/ga_long/600s/{cell}.json
        {instance: {ga, seed, gens, improved_on_seed, wall, cpu,
                    startup_cpu, search_cpu, anytime}}

Usage:
  python scripts/x2_scale_ga.py --cells CELL[,CELL] [--budget 60] [--workers 8]
  python scripts/x2_scale_ga.py --selftest
"""

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

CELLS = ['20x25+ppvct-mixed+v1+t0.6', '20x25+ppvct-mixed+v1+t1.0',
         '30x25+ppvct-mixed+v1+t0.6', '30x25+ppvct-mixed+v2+t1.0']


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', type=str, default='')
    ap.add_argument('--budget', type=float, default=60.0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--pop', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default='results/scaleup/ga')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]
    return args


_ARGS = parse_cli()

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

sys.path.insert(0, '.')

import numpy as np
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.pdr_pairs import run_pdr_pair
from transport_marl import ga_transport_v2 as ga


def run_ga_traced(sim, elig_tbl, rng, budget_s, pop_size=100, elite=5,
                  tour_k=2, seeds=None, clock=time.process_time):
    """ga_transport_v2.run_ga with a per-generation anytime trace.

    Returns (best_makespan, best_chromosome, n_gens, pop_cpu, search_cpu,
             trace) where pop_cpu is the CPU time spent building and decoding
    the initial population (before the budget clock starts) and trace is
    [(cpu_s_since_search_start, best_makespan)] with one entry per completed
    generation, prefixed by the generation-0 incumbent at 0.0.

    Operator calls and their order are identical to run_ga, so the RNG stream
    and the search are identical; only clock() calls and list appends are
    added. Verified by --selftest.
    """
    n_ops, n_jobs = sim.n_ops, sim.n_j

    t_pop0 = clock()
    pop = [tuple(v.copy() for v in s) for s in (seeds or [])][:pop_size]
    pop += [ga.random_individual(rng, n_ops, n_jobs)
            for _ in range(pop_size - len(pop))]
    pop_fit = np.array([ga.decode(sim, elig_tbl, *ind) for ind in pop])

    best_idx = int(np.argmin(pop_fit))
    best = tuple(v.copy() for v in pop[best_idx])
    best_fit = float(pop_fit[best_idx])
    pop_cpu = clock() - t_pop0

    trace = [(0.0, best_fit)]
    t0 = clock()
    gen = 0
    tnow = clock()
    while tnow - t0 < budget_s:
        gen += 1
        order = np.argsort(pop_fit)
        new_pop = [tuple(v.copy() for v in pop[i]) for i in order[:elite]]
        new_fit = list(pop_fit[order[:elite]])          # elites keep fitness

        while len(new_pop) < pop_size:
            p1 = pop[ga.tournament(rng, pop_fit, tour_k)]
            p2 = pop[ga.tournament(rng, pop_fit, tour_k)]
            child = ga.mutate(rng, ga.uniform_crossover(rng, p1, p2))
            new_pop.append(child)
            new_fit.append(ga.decode(sim, elig_tbl, *child))

        pop, pop_fit = new_pop, np.array(new_fit)
        cur = int(np.argmin(pop_fit))
        if pop_fit[cur] < best_fit - 1e-9:
            best_fit = float(pop_fit[cur])
            best = tuple(v.copy() for v in pop[cur])
        tnow = clock()
        trace.append((round(tnow - t0, 4), best_fit))

    return best_fit, best, gen, pop_cpu, tnow - t0, trace


def one_instance(task):
    stem, budget_s, pop, seed, idx = task
    t_wall0, t_cpu0 = time.time(), time.process_time()
    jl, pt, meta = load_instance(stem)
    tr = meta['transport']
    sim = TransportSim(jl, pt, meta['time_lag'], tr['station_cell'],
                       tr['tau_cells'], int(tr['n_vehicles']),
                       int(tr['veh_start_cell']))
    elig_tbl = ga.eligible_machines(pt)
    rng = np.random.default_rng(seed * 100003 + idx)

    seeds, pair_ms = ga.pdr_seed_chromosomes(sim, elig_tbl, rng)
    best_pair = min(pair_ms, key=pair_ms.get)
    seed_best = float(pair_ms[best_pair])

    for nm, s in zip(pair_ms.keys(), seeds):
        d = ga.decode(sim, elig_tbl, *s)
        assert abs(d - pair_ms[nm]) < 1e-9, \
            f'{stem}: seed {nm} decodes {d} != PDR {pair_ms[nm]}'
    pre_cpu = time.process_time() - t_cpu0   # PDR seeding + decode assertions

    ga_ms, best_chrom, gens, pop_cpu, search_cpu, trace = run_ga_traced(
        sim, elig_tbl, rng, budget_s, pop_size=pop, seeds=seeds)
    assert ga_ms <= seed_best + 1e-9, \
        f'{stem}: GA {ga_ms} worse than exact seed {seed_best}'

    improved = bool(ga_ms < seed_best - 1e-9)
    if improved:
        best_ms = ga_ms
        ms2 = ga.decode(sim, elig_tbl, *best_chrom)
        assert abs(ms2 - best_ms) < 1e-6, \
            f'{stem}: non-deterministic decode {ms2} != {best_ms}'
    else:
        best_ms = seed_best
        ms2 = run_pdr_pair(sim, *best_pair.split('+'))
        assert abs(ms2 - best_ms) < 1e-6, \
            f'{stem}: PDR re-run {ms2} != recorded {best_ms}'
    res = validate_transport_schedule(
        jl, pt, meta['time_lag'], np.array(tr['station_cell']),
        np.array(tr['tau_cells']), int(tr['n_vehicles']),
        int(tr['veh_start_cell']), sim.schedule_record())
    assert res['feasible'], f'{stem}: GA schedule INFEASIBLE: ' \
                            f'{res["violations"][:3]}'
    assert abs(res['makespan'] - best_ms) < 1e-6, \
        f'{stem}: validator makespan {res["makespan"]} != GA {best_ms}'

    return os.path.basename(stem), dict(
        ga=best_ms, seed=seed_best, gens=gens, improved_on_seed=improved,
        wall=round(time.time() - t_wall0, 2),
        cpu=round(time.process_time() - t_cpu0, 2),
        startup_cpu=round(pre_cpu + pop_cpu, 3),
        pdr_seed_cpu=round(pre_cpu, 3), pop_init_cpu=round(pop_cpu, 3),
        search_cpu=round(search_cpu, 3),
        anytime=[[float(t), float(v)] for t, v in trace])


def selftest():
    """run_ga_traced == run_ga under a deterministic clock.

    Both functions call clock() once before the loop and once per iteration,
    so a counter that returns 1, 2, 3, ... makes both stop after the same
    number of generations with the same RNG stream. Any divergence in the
    operator sequence would show up as a different makespan or chromosome.
    """
    stem = sorted(glob.glob(
        'data/PPVCT/20x25+ppvct-mixed+v1+t0.6/test/instance_*.fjs'))[0][:-4]
    jl, pt, meta = load_instance(stem)
    tr = meta['transport']

    def build():
        sim = TransportSim(jl, pt, meta['time_lag'], tr['station_cell'],
                           tr['tau_cells'], int(tr['n_vehicles']),
                           int(tr['veh_start_cell']))
        elig_tbl = ga.eligible_machines(pt)
        rng = np.random.default_rng(12345)
        seeds, _ = ga.pdr_seed_chromosomes(sim, elig_tbl, rng)
        return sim, elig_tbl, np.random.default_rng(999), seeds

    def counter():
        counter.k += 1.0
        return counter.k

    sim, elig, rng, seeds = build()
    counter.k = 0.0
    a_ms, a_chrom, a_gen = ga.run_ga(sim, elig, rng, 4.0, pop_size=20,
                                     seeds=seeds, clock=counter)
    sim, elig, rng, seeds = build()
    counter.k = 0.0
    b_ms, b_chrom, b_gen, pop_cpu, s_cpu, trace = run_ga_traced(
        sim, elig, rng, 4.0, pop_size=20, seeds=seeds, clock=counter)
    assert a_gen == b_gen, f'gens {a_gen} != {b_gen}'
    assert abs(a_ms - b_ms) < 1e-12, f'makespan {a_ms} != {b_ms}'
    for x, y in zip(a_chrom, b_chrom):
        assert np.array_equal(x, y), 'chromosome differs'
    assert len(trace) == b_gen + 1, f'trace {len(trace)} != {b_gen} + 1'
    assert all(trace[i + 1][1] <= trace[i][1] + 1e-12
               for i in range(len(trace) - 1)), 'trace not monotone'
    print(f'selftest OK: run_ga and run_ga_traced agree '
          f'(gens={a_gen}, makespan={a_ms:.4f}, trace={len(trace)} points)',
          flush=True)


def main():
    args = _ARGS
    if args.selftest:
        selftest()
        return
    cells = args.cells.split(',') if args.cells else CELLS
    os.makedirs(args.outdir, exist_ok=True)
    print(f'x2_scale_ga cells={cells} budget={args.budget:.0f}s(CPU) '
          f'pop={args.pop} workers={args.workers} out={args.outdir}',
          flush=True)
    for cell in cells:
        ds = f'data/PPVCT/{cell}/test'
        out_path = f'{args.outdir}/{cell}.json'
        if os.path.exists(out_path):
            print(f'skip {cell} (exists)', flush=True)
            continue
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        tasks = [(s, args.budget, args.pop, args.seed, i)
                 for i, s in enumerate(stems)]
        t0 = time.time()
        with Pool(args.workers) as pool:
            rows = pool.map(one_instance, tasks)
        data = {name: rec for name, rec in rows}
        # tmp + atomic rename: a reader can never see a half-written cell file
        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
        ga_mean = float(np.mean([r['ga'] for r in data.values()]))
        seed_mean = float(np.mean([r['seed'] for r in data.values()]))
        n_beat = sum(r['improved_on_seed'] for r in data.values())
        cpu_mean = float(np.mean([r['cpu'] for r in data.values()]))
        start_mean = float(np.mean([r['startup_cpu'] for r in data.values()]))
        gen_mean = float(np.mean([r['gens'] for r in data.values()]))
        print(f'{cell}: n={len(data)} GA {ga_mean:.1f} vs seed {seed_mean:.1f} '
              f'({100 * (seed_mean - ga_mean) / seed_mean:+.2f}% better, '
              f'beats seed on {n_beat}/{len(data)}) gens {gen_mean:.1f} '
              f'startup {start_mean:.1f}s all-in cpu/inst {cpu_mean:.0f}s '
              f'[{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
