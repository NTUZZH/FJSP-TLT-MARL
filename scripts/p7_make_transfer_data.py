"""Held-out regime-transfer datasets (E6: deployment claim, novelty item B).

Two transfer axes, TEST split only (zero-shot evaluation; no training or
validation ever touches these):
  fleet transfer : 10 modules, V=4 (outside the trained fleet grid {1,2,3}),
                   tau/p in {0.6, 1.0}
  scale transfer : 15 modules (trained size is 10), V=2,
                   tau/p in {0.6, 1.0}

Same protocol as p1_make_ppvct_data.py (seed0=20000, 100 instances/cell,
mixed class, default factory); dirs follow the house naming so eval and
fill_macros glob them uniformly.

Usage: python scripts/p7_make_transfer_data.py
"""

import sys, os, json

sys.argv = [sys.argv[0]]   # scrub CLI before params.py's import-time argparse

CELLS = [  # (n_modules, V, ratio)
    (10, 4, 0.6), (10, 4, 1.0),
    (15, 2, 0.6), (15, 2, 1.0),
]


def main():
    import numpy as np
    sys.path.insert(0, '.')
    from ppvc_instance_generator import ppvc_instance_generator, save_instance
    from transport_marl.layout import build_transport_layout

    seed0, n_inst = 20000, 100
    for (J, V, r) in CELLS:
        ds = f'data/PPVCT/{J}x25+ppvct-mixed+v{V}+t{r}/test'
        os.makedirs(ds, exist_ok=True)
        for i in range(n_inst):
            jl, pt, meta = ppvc_instance_generator(
                n_modules=J, class_mix='mixed', seed=seed0 + i)
            layout = build_transport_layout(
                jl, pt, np.asarray(meta['mch_type']), r, V)
            meta = dict(meta)
            meta['transport'] = layout
            save_instance(f'{ds}/instance_{i:03d}', jl, pt, meta)
        dm = dict(seed0=seed0, n_instances=n_inst, n_modules=J,
                  ppvc_mix='mixed', ppvc_factory='default',
                  n_vehicles=V, tau_over_p=r, split='test-only (zero-shot)',
                  generator='p7_make_transfer_data.py v1')
        with open(f'{ds}/dataset_meta.json', 'w') as f:
            json.dump(dm, f, indent=1)
        print(f'wrote {ds}: {n_inst} instances', flush=True)


if __name__ == '__main__':
    main()
