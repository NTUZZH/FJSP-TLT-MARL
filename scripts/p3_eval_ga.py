"""PDR-seeded GA baseline over the PPVCT test cells (T2 metaheuristic row).

Proposal SS8 E1 / Appendix B: GA pop 100, 60 s wall per instance, PDR-seeded.
GA core: transport_marl/ga_transport.py (random-key chromosome decoded by the
reference TransportSim -> always feasible). Per instance: 9 PDR-pair rollouts
seed the population, GA runs single-threaded for the budget, and the winning
schedule is independently re-checked with validator_t before anything is
written. Multiprocessing over instances (mirrors p1_eval_pdr).

CLI:  python scripts/p3_eval_ga.py all --budget 60 --workers 20
      python scripts/p3_eval_ga.py v2+t0.3 v3+t0.6 --budget 10 --workers 2
Output: results/ga/10x25+ppvct-mixed+{cell}.json
        {instance: {ga, seed, gens}}  (seed = best of the 9 PDR pairs)
Resume-safe per cell: existing output files are skipped. With --max_instances
(smoke) nothing is written, so partial runs never shadow a full cell.
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
        description='PDR-seeded GA baseline for FJSP-TL-T (T2/E1)')
    ap.add_argument('cells', nargs='+',
                    help="regime cells like v2+t0.3, or 'all' (3x3 grid)")
    ap.add_argument('--budget', type=float, default=60.0,
                    help='per-instance wall-clock budget in seconds')
    ap.add_argument('--workers', type=int, default=20,
                    help='multiprocessing workers across instances')
    ap.add_argument('--pop', type=int, default=100, help='population size')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed')
    ap.add_argument('--max_instances', type=int, default=None,
                    help='cap instances per cell (smoke; output NOT written)')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]   # scrub argv before params.py's import-time parse
    return args


_ARGS = parse_cli()

# one process per instance; keep numpy single-threaded inside each worker
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

sys.path.insert(0, '.')

import numpy as np
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.pdr_pairs import run_pdr_pair
from transport_marl import ga_transport as ga

GRID = [f'v{V}+t{r}' for V in (1, 2, 3) for r in ('0.1', '0.3', '0.6', '1.0')]


def one_instance(task):
    stem, budget_s, pop, seed, idx = task
    jl, pt, meta = load_instance(stem)
    tr = meta['transport']
    sim = TransportSim(jl, pt, meta['time_lag'], tr['station_cell'],
                       tr['tau_cells'], int(tr['n_vehicles']),
                       int(tr['veh_start_cell']))
    elig_tbl = ga.eligible_machines(pt)

    # 9 PDR-pair rollouts: seed chromosomes + true pair makespans
    seeds, pair_ms = ga.pdr_seed_chromosomes(sim, elig_tbl)
    best_pair = min(pair_ms, key=pair_ms.get)
    seed_best = float(pair_ms[best_pair])

    rng = np.random.default_rng(seed * 100003 + idx)
    ga_ms, best_chrom, gens = ga.run_ga(
        sim, elig_tbl, rng, budget_s, pop_size=pop, seeds=seeds)

    # Seeded-GA guarantee: the 9 PDR schedules are members of the constructive
    # class the GA searches, so the incumbent is never worse than the best
    # seed. The seed encodings are LOOSE (static per-job vehicle keys), so if
    # the GA has not strictly beaten the best pair within the budget, the best
    # PDR schedule itself is the incumbent we report and validate.
    if ga_ms < seed_best - 1e-9:
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

    return os.path.basename(stem), dict(ga=best_ms, seed=seed_best, gens=gens)


def main():
    args = _ARGS
    cells = GRID if args.cells == ['all'] else args.cells
    unknown = [c for c in cells if c not in GRID]
    if unknown:
        sys.exit(f'[p3_eval_ga] unknown cells {unknown}; grid = {GRID}')

    os.makedirs('results/ga', exist_ok=True)
    print(f'p3_eval_ga  cells={cells}  budget={args.budget:.0f}s  '
          f'pop={args.pop}  workers={args.workers}  seed={args.seed}', flush=True)

    for cell in cells:
        ds = f'data/PPVCT/10x25+ppvct-mixed+{cell}/test'
        out_path = f'results/ga/10x25+ppvct-mixed+{cell}.json'
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
                      f'seed={rec["seed"]:.2f}  gens={rec["gens"]}', flush=True)
            print(f'{cell}: smoke ({len(data)} instances), output NOT written',
                  flush=True)
            continue

        with open(out_path, 'w') as f:
            json.dump(data, f)
        ga_mean = float(np.mean([r['ga'] for r in data.values()]))
        seed_mean = float(np.mean([r['seed'] for r in data.values()]))
        n_beat = sum(r['ga'] < r['seed'] - 1e-9 for r in data.values())
        print(f'{cell}: n={len(data)}  GA {ga_mean:.1f} vs seed {seed_mean:.1f} '
              f'({100 * (seed_mean - ga_mean) / seed_mean:+.2f}% better, '
              f'beats seed on {n_beat}/{len(data)})  '
              f'[{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
