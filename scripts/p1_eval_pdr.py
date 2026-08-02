"""PDR-pair baselines over the PPVCT test sets (T2 baseline rows).

9 rule pairs x 9 regime cells x 100 test instances, via the reference
simulator, validated per schedule, multiprocessing over instances.
Output: results/pdr/{cell}.json with per-pair per-instance makespans.
"""

import sys, os, glob, json
from multiprocessing import Pool

sys.argv = [sys.argv[0]]
sys.path.insert(0, '.')

import numpy as np
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair


def one_instance(task):
    stem, = task
    jl, pt, meta = load_instance(stem)
    tr = meta['transport']
    out = {}
    for mn in MCH_RULES:
        for vn in VEH_RULES:
            sim = TransportSim(jl, pt, meta['time_lag'], tr['station_cell'],
                               tr['tau_cells'], int(tr['n_vehicles']),
                               int(tr['veh_start_cell']))
            ms = run_pdr_pair(sim, mn, vn)
            res = validate_transport_schedule(
                jl, pt, meta['time_lag'], np.array(tr['station_cell']),
                np.array(tr['tau_cells']), int(tr['n_vehicles']),
                int(tr['veh_start_cell']), sim.schedule_record())
            assert res['feasible'], f'{stem} {mn}+{vn}: {res["violations"][:2]}'
            out[f'{mn}+{vn}'] = ms
    return os.path.basename(stem), out


def main():
    os.makedirs('results/pdr', exist_ok=True)
    # 10x25 = trained size; 15x25 = scale-transfer cells (E6)
    cells = sorted(glob.glob('data/PPVCT/10x25+ppvct-mixed+v*+t*/test') +
                   glob.glob('data/PPVCT/15x25+ppvct-mixed+v*+t*/test'))
    for ds in cells:
        cell = ds.split('/')[2]
        out_path = f'results/pdr/{cell}.json'
        if os.path.exists(out_path):
            print(f'skip {cell} (exists)'); continue
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        with Pool(min(30, os.cpu_count() - 2)) as pool:
            rows = pool.map(one_instance, [(s,) for s in stems])
        data = {name: ms for name, ms in rows}
        with open(out_path, 'w') as f:
            json.dump(data, f)
        pairs = sorted(next(iter(data.values())).keys())
        means = {p: float(np.mean([d[p] for d in data.values()])) for p in pairs}
        best = min(means, key=means.get)
        print(f'{cell}: n={len(data)} best={best} {means[best]:.1f} '
              f'(FIFO+FIFO {means["FIFO+FIFO"]:.1f})', flush=True)


if __name__ == '__main__':
    main()
