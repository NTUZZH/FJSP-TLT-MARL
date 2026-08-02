"""Warm-started CP-SAT probe on the same v2+t0.6 instances (P2)."""
import sys, json
sys.argv = [sys.argv[0]]; sys.path.insert(0, '.')
import numpy as np
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair
from transport_marl.cpsat_transport import solve_transport_instance

for cell in ['v2+t0.6', 'v1+t0.6']:
    for i in range(2):
        stem = f'data/PPVCT/10x25+ppvct-mixed+{cell}/test/instance_{i:03d}'
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
        sol = solve_transport_instance(jl, pt, meta, time_limit=300,
                                       n_workers=10, warmstart=best_rec)
        print(json.dumps(dict(cell=cell, inst=i, pdr_seed=round(best_ms, 2),
                              status=sol['status'], ms=sol['makespan'],
                              lb=sol.get('objective_bound'))), flush=True)
