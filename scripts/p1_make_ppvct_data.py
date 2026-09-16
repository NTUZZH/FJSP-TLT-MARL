"""Generate PPVC-T datasets: PPVC instances + transport layout side-car.

Extends the public generator's meta.json with the layout fields pinned in
layout conventions (station_cell, cell_xy, speed_const, tau_cells,
n_vehicles, veh_start_cell, tau_over_p) under meta['transport']. Old
loaders ignore the new key; transport code requires it.

Datasets: data/PPVCT/10x25+ppvct-mixed+v{V}+t{r}/{split}/instance_NNN.*
Grid (Appendix B): V in {1,2,3} x r in {0.1,0.3,0.6}; 100 test (seed0=20000)
+ 100 vali (seed0=10000) per cell. Training instances are generated on the
fly by the trainer (not here).

Usage: python scripts/p1_make_ppvct_data.py [--smoke]
"""

import sys, os, json

SMOKE = '--smoke' in sys.argv[1:]
sys.argv = [sys.argv[0]]   # scrub CLI before params.py's import-time argparse


def main():
    import numpy as np
    sys.path.insert(0, '.')
    from ppvc_instance_generator import ppvc_instance_generator, save_instance
    from transport_marl.layout import build_transport_layout

    fleets = [1, 2, 3]
    ratios = [0.1, 0.3, 0.6]
    splits = [('test', 20000, 100), ('vali', 10000, 100)]
    if SMOKE:
        fleets, ratios = [2], [0.3]
        splits = [('test', 20000, 3)]

    for V in fleets:
        for r in ratios:
            for split, seed0, n_inst in splits:
                ds = f'data/PPVCT/10x25+ppvct-mixed+v{V}+t{r}/{split}'
                os.makedirs(ds, exist_ok=True)
                for i in range(n_inst):
                    jl, pt, meta = ppvc_instance_generator(
                        n_modules=10, class_mix='mixed', seed=seed0 + i)
                    layout = build_transport_layout(
                        jl, pt, np.asarray(meta['mch_type']), r, V)
                    meta = dict(meta)
                    meta['transport'] = layout
                    save_instance(f'{ds}/instance_{i:03d}', jl, pt, meta)
                dm = dict(seed0=seed0, n_instances=n_inst, n_modules=10,
                          ppvc_mix='mixed', ppvc_factory='default',
                          n_vehicles=V, tau_over_p=r,
                          generator='p1_make_ppvct_data.py v1')
                with open(f'{ds}/dataset_meta.json', 'w') as f:
                    json.dump(dm, f, indent=1)
                print(f'wrote {ds}: {n_inst} instances')


if __name__ == '__main__':
    main()
