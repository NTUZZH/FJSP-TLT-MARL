"""PDR-pair baselines on the scale-up pilot cells (Phase A).

Identical protocol to scripts/p1_eval_pdr.py -- the same nine
machine x vehicle rule pairs, the same reference simulator, every schedule
independently re-checked by validator_t -- but over the 20x25 / 30x25 pilot
cells and writing to results/scaleup/pdr/ so nothing can be swept into the
manuscript's results/pdr/ analysis.

Output: {outdir}/{cell}.json = {instance: {pair: makespan}}, outdir
        defaulting to results/scaleup/pdr

Usage: python scripts/x2_scale_pdr.py [--workers 10] [--outdir DIR]
"""

import sys, os, glob, json, argparse, time

ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=10)
ap.add_argument('--cells', type=str, default='')
ap.add_argument('--outdir', type=str, default='results/scaleup/pdr')
_A = ap.parse_args()
sys.argv = [sys.argv[0]]
sys.path.insert(0, '.')

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'

import numpy as np
from multiprocessing import Pool
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair

CELLS = ['20x25+ppvct-mixed+v1+t0.6', '20x25+ppvct-mixed+v1+t1.0',
         '30x25+ppvct-mixed+v1+t0.6', '30x25+ppvct-mixed+v2+t1.0']


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
    os.makedirs(_A.outdir, exist_ok=True)
    cells = _A.cells.split(',') if _A.cells else CELLS
    for cell in cells:
        ds = f'data/PPVCT/{cell}/test'
        out_path = f'{_A.outdir}/{cell}.json'
        if os.path.exists(out_path):
            print(f'skip {cell} (exists)', flush=True); continue
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        t0 = time.time()
        with Pool(_A.workers) as pool:
            rows = pool.map(one_instance, [(s,) for s in stems])
        data = {name: ms for name, ms in rows}
        with open(out_path, 'w') as f:
            json.dump(data, f)
        pairs = sorted(next(iter(data.values())).keys())
        means = {p: float(np.mean([d[p] for d in data.values()])) for p in pairs}
        best = min(means, key=means.get)
        b9 = float(np.mean([min(d.values()) for d in data.values()]))
        print(f'{cell}: n={len(data)} best-fixed={best} {means[best]:.1f} '
              f'best-of-9 {b9:.1f} (FIFO+FIFO {means["FIFO+FIFO"]:.1f}) '
              f'[{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
