"""the nine dispatching-rule pairs on the heterogeneity cells.

Same protocol as scripts/p1_eval_pdr.py (reference simulator, every schedule
re-validated with validator_t), restricted to the heterogeneity cells and levels and
writing under results/hetero/ so nothing lands in results/pdr/.

CPU only, pinned by the caller; the pool size is a parameter so the job can
share the box with another project's training.

Usage:
  python scripts/x2_hetero_pdr.py --cells v1+t0.6 --levels base,2,5,10,20 --procs 8
Outputs: results/hetero/pdr/{cell}+R{level}.json
"""

import argparse
import json
import os
import sys

cli = argparse.ArgumentParser()
cli.add_argument('--cells', type=str, default='v1+t0.6,v2+t0.6,v1+t1.0')
cli.add_argument('--levels', type=str, default='base,2,5,10,20')
cli.add_argument('--split', type=str, default='test')
cli.add_argument('--procs', type=int, default=8)
A = cli.parse_args()
sys.argv = [sys.argv[0]]

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import glob
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, '.')
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair


def one_instance(stem):
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
            out[f'{mn}+{vn}'] = float(ms)
    return os.path.basename(stem), out


def dataset_dir(cell, level, split):
    base = cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'
    if level == 'base':
        return f'data/PPVCT/{base}/{split}'
    return f'data/PPVCT_HET/{base}+hetR{level}/{split}'


def main():
    os.makedirs('results/hetero/pdr', exist_ok=True)
    for cell in A.cells.split(','):
        for level in A.levels.split(','):
            out_path = f'results/hetero/pdr/{cell}+R{level}.json'
            if os.path.exists(out_path):
                print(f'skip {cell} R={level} (exists)', flush=True)
                continue
            ds = dataset_dir(cell, level, A.split)
            stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
            assert stems, f'no instances under {ds}'
            with Pool(A.procs) as pool:
                rows = pool.map(one_instance, stems)
            data = {name: ms for name, ms in rows}
            with open(out_path, 'w') as f:
                json.dump(dict(cell=cell, level=level, dataset=ds,
                               n=len(data), makespan=data), f, indent=1)
            pairs = sorted(next(iter(data.values())).keys())
            means = {p: float(np.mean([d[p] for d in data.values()]))
                     for p in pairs}
            best = min(means, key=means.get)
            print(f'{cell} R={level:>4s}: n={len(data)} best={best} '
                  f'{means[best]:.2f} (FIFO+FIFO {means["FIFO+FIFO"]:.2f})',
                  flush=True)


if __name__ == '__main__':
    main()
