"""Warm-started CP-SAT references for PPVCT test cells (resume-safe).

Per instance: best-PDR seed (9 pairs, sim) -> AddHint -> 300 s solve.
Writes or_solution/PPVCT/{cell}.jsonl (one line per instance, resume-safe)
with ms (UB), lb, status, pdr_seed. Usage:
  python -u scripts/p2_cpsat_refs.py v1+t0.6 v2+t0.6 [--par 2]
"""
import sys, os, json, glob
from multiprocessing import Pool

ARGS = [a for a in sys.argv[1:]]
sys.argv = [sys.argv[0]]
sys.path.insert(0, '.')

import numpy as np
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair
from transport_marl.cpsat_transport import solve_transport_instance

PAR = 2
cells = []
i = 0
while i < len(ARGS):
    if ARGS[i] == '--par':
        PAR = int(ARGS[i + 1]); i += 2
    else:
        cells.append(ARGS[i]); i += 1

def solve_one(task):
    cell, stem = task
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
    sol = solve_transport_instance(jl, pt, meta, time_limit=300, n_workers=10,
                                   warmstart=best_rec)
    return dict(instance=os.path.basename(stem), cell=cell,
                pdr_seed=round(best_ms, 4), status=sol['status'],
                ms=sol['makespan'], lb=sol.get('objective_bound'),
                wall=round(sol.get('walltime', -1), 1))

def main():
    os.makedirs('or_solution/PPVCT', exist_ok=True)
    for cell in cells:
        out = f'or_solution/PPVCT/{cell}.jsonl'
        done = set()
        if os.path.exists(out):
            with open(out) as f:
                done = {json.loads(l)['instance'] for l in f if l.strip()}
        # cell may carry a size prefix ('15x25+ppvct-mixed+v2+t0.6') for the
        # transfer cells; bare cells keep the 10x25 default
        dirname = cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'
        stems = sorted(g[:-4] for g in glob.glob(
            f'data/PPVCT/{dirname}/test/instance_*.fjs'))
        todo = [(cell, s) for s in stems if os.path.basename(s) not in done]
        print(f'{cell}: {len(todo)} to solve ({len(done)} done)', flush=True)
        with Pool(PAR) as pool:
            for rec in pool.imap_unordered(solve_one, todo):
                with open(out, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                print(f'{rec["instance"]}: {rec["pdr_seed"]} -> {rec["ms"]} '
                      f'(lb {rec["lb"]}, {rec["status"]})', flush=True)

if __name__ == '__main__':
    main()
