"""CP-SAT transport model sanity (P1, small instances 5x9).

S1  tau=0 & ample fleet == house lag-aware CP-SAT (fjsp_solver) makespan.
S2  transport-tight (V=1, r=0.6): cpsat <= best PDR (sim); >= S1 optimum;
    move timings self-consistent (per-vehicle chain with empty moves).
S3  independent timing re-check of the CP-SAT schedule (numpy).
"""

import sys, glob
import numpy as np

sys.argv = [sys.argv[0]]
sys.path.insert(0, '.')

from ppvc_instance_generator import load_instance
from transport_marl.layout import build_transport_layout
from transport_marl.sim_single import TransportSim
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair
from transport_marl.cpsat_transport import solve_transport_instance
from ortools_solver import fjsp_solver, matrix_to_the_format_for_solving


def check_solution(jl, pt, meta, layout, sol, tol=1e-6):
    """Independent numpy re-check of CP-SAT output timing/logic."""
    v = []
    lag = np.asarray(meta['time_lag'], dtype=float)
    cell = np.asarray(layout['station_cell'])
    tau = np.asarray(layout['tau_cells'])
    jl = np.asarray(jl); first = np.zeros(len(jl), dtype=int)
    first[1:] = np.cumsum(jl)[:-1]
    amch = np.asarray(sol['assigned_mch'])
    st = np.asarray(sol['op_start']); ct = np.asarray(sol['op_ct'])
    pt = np.asarray(pt, dtype=float)
    mv_by_op = {m['op']: m for m in sol['moves']}
    for j, L in enumerate(jl):
        for k in range(L):
            o = first[j] + k
            if pt[o, amch[o]] <= 0:
                v.append(f'compat op{o}')
            if abs(ct[o] - st[o] - pt[o, amch[o]]) > tol:
                v.append(f'ct op{o}')
            if k == 0:
                continue
            rel = ct[o - 1] + lag[o - 1]
            if cell[amch[o - 1]] == cell[amch[o]]:
                if st[o] < rel - tol:
                    v.append(f'prec op{o}')
            else:
                m = mv_by_op.get(o)
                if m is None:
                    v.append(f'missing move op{o}'); continue
                if m['start'] < rel - tol:
                    v.append(f'move-early op{o}')
                if st[o] < m['start'] + m['dur'] - tol:
                    v.append(f'arrive op{o}')
    # machine no-overlap
    for m in range(pt.shape[1]):
        ops = np.nonzero(amch == m)[0]
        order = ops[np.argsort(st[ops])]
        for a, b in zip(order[:-1], order[1:]):
            if st[b] < ct[a] - tol:
                v.append(f'overlap mch{m}')
    # vehicle chains
    for veh in range(int(layout['n_vehicles'])):
        chain = sorted([m for m in sol['moves'] if m['veh'] == veh],
                       key=lambda m: m['start'])
        loc = int(layout.get('veh_start_cell', 0))
        free = 0.0
        for m in chain:
            if m['start'] < free + tau[loc, m['frm']] - tol:
                v.append(f"veh{veh} chain at op{m['op']}")
            free = m['start'] + m['dur']
            loc = m['to']
    return v


def main():
    stems = sorted(g[:-4] for g in glob.glob('data/PPVC/5x9+ppvc-mixed/instance_*.fjs'))[:3]
    fails = 0
    for stem in stems:
        jl, pt, meta = load_instance(stem)
        lag = np.asarray(meta['time_lag'], dtype=float)

        # S1: ample fleet + tau=0 vs house lag-aware solver
        layout0 = build_transport_layout(jl, pt, np.asarray(meta['mch_type']), 0.0, len(jl))
        meta0 = dict(meta); meta0['transport'] = layout0
        sol0 = solve_transport_instance(jl, pt, meta0, time_limit=60, n_workers=8)
        jobs, _nm = matrix_to_the_format_for_solving(jl, pt)
        house = fjsp_solver(jobs, pt.shape[1], 60, time_lag=lag.astype(int).tolist())
        ms_house = house[0]
        ok1 = sol0['makespan'] is not None and abs(sol0['makespan'] - ms_house) < 1e-6
        print(f'{stem.split("/")[-1]} S1 {"OK " if ok1 else "FAIL"} '
              f'transport-model tau0={sol0["makespan"]} vs house={ms_house} '
              f'({sol0["status"]})')
        fails += (not ok1)

        # S2/S3: tight transport
        layout = build_transport_layout(jl, pt, np.asarray(meta['mch_type']), 0.6, 1)
        metaT = dict(meta); metaT['transport'] = layout
        solT = solve_transport_instance(jl, pt, metaT, time_limit=60, n_workers=8)
        best_pdr = np.inf
        for mn in MCH_RULES:
            for vn in VEH_RULES:
                sim = TransportSim(jl, pt, lag, layout['station_cell'],
                                   layout['tau_cells'], 1, layout['veh_start_cell'])
                best_pdr = min(best_pdr, run_pdr_pair(sim, mn, vn))
        ok2 = (solT['makespan'] is not None and
               solT['makespan'] <= best_pdr + 1e-6 and
               solT['makespan'] >= ms_house - 1e-6)
        viol = check_solution(jl, pt, metaT, layout, solT)
        ok3 = len(viol) == 0
        print(f'{stem.split("/")[-1]} S2 {"OK " if ok2 else "FAIL"} '
              f'cpsatT={solT["makespan"]:.2f} ({solT["status"]}) '
              f'<= bestPDR={best_pdr:.2f}, >= tau0-opt={ms_house}')
        print(f'{stem.split("/")[-1]} S3 {"OK " if ok3 else "FAIL"} '
              f'{viol[:3] if viol else "timing consistent"}')
        fails += (not ok2) + (not ok3)

    print(f'\n{"ALL PASS" if fails == 0 else f"{fails} FAILURES"}')
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
