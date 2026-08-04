"""PDR-seeded GA baseline, v2 (fixed vehicle encoding), over the PPVCT cells.

Same protocol as scripts/p3_eval_ga.py -- pop 100, 60 s per instance, nine
PDR-pair seeds, single-threaded, winner independently re-checked by
validator_t -- but the GA core is transport_marl/ga_transport_v2.py, whose
vehicle-side encoding contains the three PDR vehicle rules exactly. See
notes/ga_diagnosis.md for why v1 stalled at |V|=1 with heavy travel.

BUDGET: 60 s of worker CPU time (time.process_time), not wall-clock. On an
idle core the two coincide; under machine load the CPU-time budget still
buys the same amount of search, so a busy box cannot silently weaken the
baseline. Contention would bias this particular comparison in the policy's
favour, which is exactly what must not happen. Both `wall` and `cpu` are
recorded per instance so the budget actually spent is auditable.

Output: results/ga_v2/10x25+ppvct-mixed+{cell}.json
        {instance: {ga, seed, gens, improved_on_seed, wall, cpu}}
Schema matches results/ga/ plus improved_on_seed (and the wall/cpu audit
fields). Never writes into results/ga/. Resume-safe per cell.

CLI:  python scripts/p3_eval_ga_v2.py all --budget 60 --workers 12
      python scripts/p3_eval_ga_v2.py v1+t0.6 --budget 60 --max_instances 5
      python scripts/p3_eval_ga_v2.py v1+t0.6 --budget 5 --tag '+b5' \
             --outdir results/ga_v2_budget          # budget sweep
"""

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool


def parse_cli():
    ap = argparse.ArgumentParser(
        description='PDR-seeded GA baseline v2 for FJSP-TL-T (T2/E1)')
    ap.add_argument('cells', nargs='+',
                    help="regime cells like v2+t0.3, or 'all' (3x4 grid)")
    ap.add_argument('--budget', type=float, default=60.0,
                    help='per-instance CPU-time budget in seconds')
    ap.add_argument('--workers', type=int, default=12,
                    help='multiprocessing workers across instances')
    ap.add_argument('--pop', type=int, default=100, help='population size')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed')
    ap.add_argument('--outdir', default='results/ga_v2', help='output dir')
    ap.add_argument('--tag', default='',
                    help="suffix appended to the output filename stem, e.g. "
                         "'+b5' -> 10x25+ppvct-mixed+v1+t0.6+b5.json. Lets a "
                         "budget sweep sit beside the 60 s run without ever "
                         "colliding with it.")
    ap.add_argument('--max_instances', type=int, default=None,
                    help='cap instances per cell (smoke; output NOT written)')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]   # scrub argv before params.py's import-time parse
    return args


_ARGS = parse_cli()

# one process per instance; keep numpy single-threaded inside each worker
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

sys.path.insert(0, '.')

import numpy as np
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.pdr_pairs import run_pdr_pair
from transport_marl import ga_transport_v2 as ga

GRID = [f'v{V}+t{r}' for V in (1, 2, 3) for r in ('0.1', '0.3', '0.6', '1.0')]


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

    # 9 PDR-pair rollouts: seed chromosomes + true pair makespans
    seeds, pair_ms = ga.pdr_seed_chromosomes(sim, elig_tbl, rng)
    best_pair = min(pair_ms, key=pair_ms.get)
    seed_best = float(pair_ms[best_pair])

    # v2 invariant: every seed decodes EXACTLY to its pair's makespan, so the
    # GA incumbent starts at PDR level and elitism keeps it monotone. This is
    # what v1 could not do (notes/ga_diagnosis.md).
    for nm, s in zip(pair_ms.keys(), seeds):
        d = ga.decode(sim, elig_tbl, *s)
        assert abs(d - pair_ms[nm]) < 1e-9, \
            f'{stem}: seed {nm} decodes {d} != PDR {pair_ms[nm]}'

    ga_ms, best_chrom, gens = ga.run_ga(
        sim, elig_tbl, rng, budget_s, pop_size=pop, seeds=seeds)
    assert ga_ms <= seed_best + 1e-9, \
        f'{stem}: GA {ga_ms} worse than exact seed {seed_best}'

    improved = bool(ga_ms < seed_best - 1e-9)
    if improved:
        best_ms = ga_ms
        ms2 = ga.decode(sim, elig_tbl, *best_chrom)   # leaves record in sim
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

    # The budget is metered in CPU time, so the search effort is identical
    # under any machine load. wall/cpu are recorded anyway: cpu/wall well
    # below 1 documents how contended the box was while the cell ran.
    return os.path.basename(stem), dict(
        ga=best_ms, seed=seed_best, gens=gens, improved_on_seed=improved,
        wall=round(time.time() - t_wall0, 2),
        cpu=round(time.process_time() - t_cpu0, 2))


def main():
    args = _ARGS
    cells = GRID if args.cells == ['all'] else args.cells
    unknown = [c for c in cells if c not in GRID]
    if unknown:
        sys.exit(f'[p3_eval_ga_v2] unknown cells {unknown}; grid = {GRID}')

    os.makedirs(args.outdir, exist_ok=True)
    print(f'p3_eval_ga_v2  cells={cells}  budget={args.budget:.0f}s  '
          f'pop={args.pop}  workers={args.workers}  seed={args.seed}',
          flush=True)

    for cell in cells:
        ds = f'data/PPVCT/10x25+ppvct-mixed+{cell}/test'
        out_path = f'{args.outdir}/10x25+ppvct-mixed+{cell}{args.tag}.json'
        if os.path.exists(out_path):
            print(f'skip {cell} (exists)', flush=True)
            continue
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        if args.max_instances:
            stems = stems[:args.max_instances]
        tasks = [(s, args.budget, args.pop, args.seed, i)
                 for i, s in enumerate(stems)]

        t0 = time.time()
        with Pool(args.workers) as pool:
            rows = pool.map(one_instance, tasks)
        data = {name: rec for name, rec in rows}

        if args.max_instances:
            for name, rec in data.items():
                print(f'  [smoke] {cell}/{name}  ga={rec["ga"]:.2f}  '
                      f'seed={rec["seed"]:.2f}  gens={rec["gens"]}  '
                      f'improved={rec["improved_on_seed"]} wall={rec["wall"]:.0f}s cpu={rec["cpu"]:.0f}s', flush=True)
            print(f'{cell}: smoke ({len(data)} instances), output NOT written',
                  flush=True)
            continue

        with open(out_path, 'w') as f:
            json.dump(data, f)
        ga_mean = float(np.mean([r['ga'] for r in data.values()]))
        seed_mean = float(np.mean([r['seed'] for r in data.values()]))
        n_beat = sum(r['improved_on_seed'] for r in data.values())
        print(f'{cell}: n={len(data)}  GA {ga_mean:.1f} vs seed {seed_mean:.1f} '
              f'({100 * (seed_mean - ga_mean) / seed_mean:+.2f}% better, '
              f'beats seed on {n_beat}/{len(data)})  '
              f'[{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
