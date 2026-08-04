"""Strengthened warm-started CP-SAT references for PPVCT test cells.

Same protocol as scripts/p2_cpsat_refs.py (best-of-9-PDR seed -> AddHint ->
300 s) but calls solve_transport_instance(..., strengthen=True), which adds:
  (a) horizon = warm-start makespan (valid UB),
  (b) redundant AddCumulative over loaded-move intervals, capacity |V|,
  (c) identical-vehicle symmetry breaking.

Writes results/cpsat_v2/{dirname}.json (a dict keyed by instance name) and a
crash-safe results/cpsat_v2/{dirname}.jsonl alongside it. The v1 references in
or_solution/PPVCT/ are never touched.

Usage:
  python -u scripts/p2_cpsat_refs_v2.py v1+t0.6 [--par 3] [--workers 4]
                                        [--time 300] [--cores 12-23]
"""
import sys, os, json, glob, time
from multiprocessing import Pool

ARGS = [a for a in sys.argv[1:]]
sys.argv = [sys.argv[0]]
sys.path.insert(0, '.')

PAR, WORKERS, TLIM, CORES = 3, 4, 300.0, None
ENERGY, TAG, LIMIT_N, CHUNK = True, '', None, None
FLEET = True
cells = []
i = 0
while i < len(ARGS):
    if ARGS[i] == '--chunk':
        CHUNK = int(ARGS[i + 1]); i += 2
    elif ARGS[i] == '--no_energy':
        ENERGY = False; i += 1
    elif ARGS[i] == '--no_fleet':
        FLEET = False; i += 1
    elif ARGS[i] == '--tag':
        TAG = ARGS[i + 1]; i += 2
    elif ARGS[i] == '--n':
        LIMIT_N = int(ARGS[i + 1]); i += 2
    elif ARGS[i] == '--par':
        PAR = int(ARGS[i + 1]); i += 2
    elif ARGS[i] == '--workers':
        WORKERS = int(ARGS[i + 1]); i += 2
    elif ARGS[i] == '--time':
        TLIM = float(ARGS[i + 1]); i += 2
    elif ARGS[i] == '--cores':
        CORES = ARGS[i + 1]; i += 2
    else:
        cells.append(ARGS[i]); i += 1

# thread caps must be set before the numerical runtimes are imported: an
# affinity mask alone does not stop MKL/OMP sizing their pool from nproc.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = str(WORKERS)

import numpy as np
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair
from transport_marl.cpsat_transport import solve_transport_instance

OUT_DIR = 'results/cpsat_v2'


def _pin(core_list):
    try:
        os.sched_setaffinity(0, core_list)
    except OSError:
        pass


def solve_one(task):
    cell, dirname, stem, slot = task
    if CORES:
        lo, hi = (int(x) for x in CORES.split('-'))
        pool_cores = list(range(lo, hi + 1))
        mine = pool_cores[slot * WORKERS:(slot + 1) * WORKERS]
        if mine:
            _pin(mine)
    jl, pt, meta = load_instance(stem)
    tr = meta['transport']
    best_ms, best_rec = np.inf, None
    for mn in MCH_RULES:
        for vn in VEH_RULES:
            sim = TransportSim(jl, pt, meta['time_lag'], tr['station_cell'],
                               tr['tau_cells'], int(tr['n_vehicles']),
                               int(tr['veh_start_cell']))
            ms = run_pdr_pair(sim, mn, vn)
            if ms < best_ms:
                best_ms, best_rec = ms, sim.schedule_record()
    t0 = time.time()
    sol = solve_transport_instance(jl, pt, meta, time_limit=TLIM,
                                   n_workers=WORKERS, warmstart=best_rec,
                                   strengthen=True, energy_cut=ENERGY,
                                   fleet=FLEET)
    return dict(instance=os.path.basename(stem), cell=cell, dataset=dirname,
                pdr_seed=round(float(best_ms), 6), status=sol['status'],
                ub=sol['makespan'], lb=sol.get('objective_bound'),
                horizon=sol.get('horizon'),
                walltime=round(sol.get('walltime', time.time() - t0), 2),
                n_workers=WORKERS, time_limit_s=TLIM, strengthened=True,
                energy_cut=ENERGY, fleet=FLEET)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for cell in cells:
        dirname = cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'
        jsonl = f'{OUT_DIR}/{dirname}{TAG}.jsonl'
        js = f'{OUT_DIR}/{dirname}{TAG}.json'
        done = set()
        if os.path.exists(jsonl):
            with open(jsonl) as f:
                done = {json.loads(l)['instance'] for l in f if l.strip()}
        stems = sorted(g[:-4] for g in glob.glob(
            f'data/PPVCT/{dirname}/test/instance_*.fjs'))
        if LIMIT_N is not None:
            stems = stems[:LIMIT_N]
        todo = [s for s in stems if os.path.basename(s) not in done]
        # --chunk: solve at most this many now and exit, so a supervisor can
        # re-evaluate the core allocation between chunks. Resume is by
        # instance name, so restarting is always safe and never redoes work.
        if CHUNK is not None:
            todo = todo[:CHUNK]
        print(f'{cell}: {len(todo)} to solve ({len(done)} done), par={PAR} '
              f'workers={WORKERS} cores={CORES}', flush=True)
        tasks = [(cell, dirname, s, i % PAR) for i, s in enumerate(todo)]
        if tasks:
            with Pool(PAR) as pool:
                for rec in pool.imap_unordered(solve_one, tasks):
                    with open(jsonl, 'a') as f:
                        f.write(json.dumps(rec) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    print(f'{rec["instance"]}: seed {rec["pdr_seed"]} -> '
                          f'ub {rec["ub"]} lb {rec["lb"]} ({rec["status"]}, '
                          f'{rec["walltime"]}s)', flush=True)
                    recs = {}
                    with open(jsonl) as f:
                        for l in f:
                            if l.strip():
                                r = json.loads(l)
                                recs[r['instance']] = r
                    with open(js, 'w') as f:
                        json.dump({k: recs[k] for k in sorted(recs)}, f,
                                  indent=1)
        print(f'{cell}: done', flush=True)


if __name__ == '__main__':
    main()
