"""P1 acceptance tests for the transport simulator stack (run from repo root).

T1  9 PDR pairs on a real 10x25 PPVC instance + layout; every schedule must pass
    the independent validator; report makespans.
T2  Reduction: tau==0 & ample fleet must reproduce the no-transport makespan
    exactly for every machine rule (vehicle rule irrelevant).
T3  Stress: V=1, tau/p=0.6 on 10 instances x 9 combos; no deadlock; all valid.
T4  Monotonicity sanity: mean makespan non-decreasing as tau/p grows {0->.1->.3->.6}
    at V=2, FIFO+FIFO (weak sanity, reported not asserted).
"""

import sys, json, glob
import numpy as np

sys.path.insert(0, '.')
from transport_marl.layout import build_transport_layout, station_cells_from_mch_type
from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair


def load_instance(stem):
    sys.path.insert(0, '.')
    from ppvc_instance_generator import load_instance as _load
    return _load(stem)


def make_sim(jl, pt, meta, ratio, n_veh, zero_tau=False, single_cell=False):
    mch_type = np.asarray(meta['mch_type'])
    layout = build_transport_layout(jl, pt, mch_type, ratio, n_veh)
    tau = np.array(layout['tau_cells'])
    cell = np.array(layout['station_cell'])
    if zero_tau:
        tau = np.zeros_like(tau)
    if single_cell:
        cell = np.zeros_like(cell)
    return TransportSim(jl, pt, meta['time_lag'], cell, tau,
                        n_veh, layout['veh_start_cell']), layout


def validate(sim, layout, jl, pt, meta):
    rec = sim.schedule_record()
    res = validate_transport_schedule(jl, pt, meta['time_lag'],
                                      sim.station_cell, sim.tau,
                                      sim.n_v, sim.veh_start_cell, rec)
    return res


def main():
    stems = sorted(g[:-4] for g in glob.glob('data/PPVC/10x25+ppvc-mixed/instance_*.fjs'))
    jl, pt, meta = load_instance(stems[0])
    fails = 0

    print('== T1: 9 PDR pairs, instance_000, V=2, tau/p=0.3 ==')
    for mn in MCH_RULES:
        for vn in VEH_RULES:
            sim, layout = make_sim(jl, pt, meta, 0.3, 2)
            ms = run_pdr_pair(sim, mn, vn)
            res = validate(sim, layout, jl, pt, meta)
            n_tr = len(sim.transports)
            flag = 'OK ' if res['feasible'] else 'FAIL'
            if not res['feasible']:
                fails += 1
                print('   ', res['violations'][:3])
            print(f'  {flag} {mn}+{vn}: makespan={ms:.2f} transports={n_tr}')

    print('== T2: tau=0 & ample fleet == no-transport reduction ==')
    for mn in MCH_RULES:
        sim0, _ = make_sim(jl, pt, meta, 0.3, len(jl), zero_tau=True)
        ms0 = run_pdr_pair(sim0, mn, 'FIFO')
        sim1, _ = make_sim(jl, pt, meta, 0.3, len(jl), single_cell=True)
        ms1 = run_pdr_pair(sim1, mn, 'FIFO')
        match = abs(ms0 - ms1) < 1e-9
        fails += (not match)
        print(f'  {"OK " if match else "FAIL"} {mn}: tau0={ms0:.2f} vs no-transport={ms1:.2f}')

    print('== T3: stress V=1, tau/p=0.6, 10 instances x 9 combos ==')
    bad = 0
    for stem in stems[:10]:
        jl2, pt2, meta2 = load_instance(stem)
        for mn in MCH_RULES:
            for vn in VEH_RULES:
                sim, layout = make_sim(jl2, pt2, meta2, 0.6, 1)
                run_pdr_pair(sim, mn, vn)
                res = validate(sim, layout, jl2, pt2, meta2)
                if not res['feasible']:
                    bad += 1
                    print(f'  FAIL {stem} {mn}+{vn}: {res["violations"][:2]}')
    fails += bad
    print(f'  {"OK" if bad == 0 else "FAIL"}: {90 - bad}/90 feasible')

    print('== T4: makespan vs tau/p (V=2, FIFO+FIFO, 10 instances) ==')
    for ratio in [0.0, 0.1, 0.3, 0.6]:
        mss = []
        for stem in stems[:10]:
            jl2, pt2, meta2 = load_instance(stem)
            sim, _ = make_sim(jl2, pt2, meta2, ratio, 2)
            mss.append(run_pdr_pair(sim, 'FIFO', 'FIFO'))
        print(f'  tau/p={ratio}: mean makespan={np.mean(mss):.2f}')

    print(f'\n{"ALL PASS" if fails == 0 else f"{fails} FAILURES"}')
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
