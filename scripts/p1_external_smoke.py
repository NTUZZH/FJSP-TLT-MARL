"""Smoke test: PUBLIC FJSP-T benchmarks -> our sim -> validator (Paper X2 E1).

Loads external instances via transport_marl/external_adapter.py, runs all 9 PDR
pairs (3 machine x 3 vehicle rules) through the reference simulator, validates
EVERY resulting schedule with the independent validator_t, and prints makespans.

Two tracks:
  * Deroussi-Norre  -- parse + pipeline validation only. Travel matrix is NOT
    shipped (caveat C1), so a labeled SYNTHETIC layout is attached; makespans
    here are NOT externally comparable.
  * Berterottiere   -- real shipped travel layout + published best-known Cmax.
    We cross-check that our best PDR makespan is consistent with the published
    optimum/best-known (sanity, given our objective omits the L/U legs, C2).

Run from repo root:  python scripts/p1_external_smoke.py
"""

import sys
import os

sys.argv = [sys.argv[0]]
sys.path.insert(0, '.')

import numpy as np

from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair
from transport_marl import external_adapter as ext


def run_all_pdr(inst):
    """Run 9 PDR pairs; validate each schedule; return dict pair -> makespan.
    Raises AssertionError on the first infeasible schedule."""
    results = {}
    for mn in MCH_RULES:
        for vn in VEH_RULES:
            sim = TransportSim(**inst.to_sim_args())
            ms = run_pdr_pair(sim, mn, vn)
            res = validate_transport_schedule(*inst.to_validator_args(
                sim.schedule_record()))
            assert res['feasible'], (
                f'{inst.name} {mn}+{vn} INFEASIBLE: {res["violations"][:3]}')
            # validator recomputes makespan from the record independently
            assert abs(res['makespan'] - ms) < 1e-6, (
                f'{inst.name} {mn}+{vn}: sim makespan {ms} != '
                f'validator {res["makespan"]}')
            results[f'{mn}+{vn}'] = ms
    return results


def _print_table(name, results, extra=''):
    best_pair = min(results, key=results.get)
    print(f'  {name:<10} best={best_pair:<11} '
          f'Cmax={results[best_pair]:>8.1f}  '
          f'(worst {max(results.values()):>8.1f}){extra}')


def main():
    n_fail = 0
    n_instances = 0

    print('=' * 74)
    print('TRACK 1  Deroussi-Norre  (SYNTHETIC travel layout -- pipeline check '
          'only)')
    print('=' * 74)
    for stem in ['fjsp1', 'fjsp2', 'fjsp5']:
        path = ext.deroussi_norre_path(stem)
        inst = ext.load_deroussi_norre(path)
        assert inst.synthetic_tau, 'expected synthetic tau for Deroussi-Norre'
        print(f'{stem}: J={inst.n_jobs} M={inst.n_machines} '
              f'ops={inst.n_ops} V={inst.n_vehicles} '
              f'cells={inst.tau_cells.shape[0]} '
              f'(ws={inst.notes["n_workstations"]}, station_cell='
              f'{inst.station_cell.tolist()})')
        try:
            results = run_all_pdr(inst)
            _print_table(stem, results)
            n_instances += 1
        except AssertionError as e:
            print(f'  FAIL: {e}')
            n_fail += 1

    print()
    print('=' * 74)
    print('TRACK 2  Berterottiere  (REAL shipped travel layout + best-known '
          'Cmax)')
    print('=' * 74)
    # 01a/02a: 10x5 (layout5);  07a: 15x8 (layout8).  2-vehicle variant.
    for stem in ['01a', '02a', '07a']:
        n_veh = 2
        inst, best_known = ext.load_berterottiere_by_stem(stem, n_veh)
        assert not inst.synthetic_tau, 'Berterottiere tau must be real'
        bk = f'{best_known:.1f}' if best_known is not None else 'n/a'
        print(f'{stem}: J={inst.n_jobs} M={inst.n_machines} '
              f'ops={inst.n_ops} V={inst.n_vehicles} '
              f'cells={inst.tau_cells.shape[0]} '
              f'layout={inst.notes["layout"]} | published best-known Cmax={bk}')
        try:
            results = run_all_pdr(inst)
            best_pdr = min(results.values())
            if best_known is not None:
                ratio = best_pdr / best_known
                rel = ('>=' if best_pdr >= best_known - 1e-6 else '<')
                extra = (f'  | best_PDR/best_known={ratio:.3f} '
                         f'(PDR {rel} best-known)')
            else:
                extra = ''
            _print_table(stem, results, extra)
            n_instances += 1
        except AssertionError as e:
            print(f'  FAIL: {e}')
            n_fail += 1

    print()
    print('=' * 74)
    if n_fail == 0:
        print(f'SMOKE TEST PASSED: {n_instances} instances loaded, all 9 PDR '
              f'schedules per instance validated feasible.')
    else:
        print(f'SMOKE TEST FAILED: {n_fail} instance(s) had an infeasible '
              f'schedule or load error.')
    print('=' * 74)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
