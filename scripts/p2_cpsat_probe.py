"""One-off CP-SAT budget probe: gate-cell instances at 300 s (P2)."""
import sys, json
sys.argv = [sys.argv[0]]; sys.path.insert(0, '.')
from ppvc_instance_generator import load_instance
from transport_marl.cpsat_transport import solve_transport_instance

for cell in ['v1+t0.6', 'v2+t0.6']:
    for i in range(2):
        stem = f'data/PPVCT/10x25+ppvct-mixed+{cell}/test/instance_{i:03d}'
        jl, pt, meta = load_instance(stem)
        sol = solve_transport_instance(jl, pt, meta, time_limit=300, n_workers=10)
        print(json.dumps(dict(cell=cell, inst=i, status=sol['status'],
                              ms=sol['makespan'], lb=sol.get('objective_bound'),
                              wall=round(sol.get('walltime', -1), 1))), flush=True)
