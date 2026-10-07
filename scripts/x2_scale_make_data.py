"""Scale-up test datasets: Phase A (20/30 modules) and Phase B (50/80 modules).

Zero-shot only. Training and model selection happen at 10 modules
(scripts/p2_train_mappo.py: --n_modules default 10, validation on
data/PPVCT/10x25+.../vali), and training instance seeds are
seed*1_000_000 + upd*num_envs, i.e. >= 301,000,000. The test seeds used
here (20000..) therefore cannot collide with any training instance.

Pipeline is byte-for-byte the one used by scripts/p1_make_ppvct_data.py and
scripts/p7_make_transfer_data.py: ppvc_instance_generator(class_mix='mixed',
default factory = 25 machines) + build_transport_layout(..., target_ratio, V),
so the per-instance travel-intensity calibration (speed_const snapped to a
multiple of 1/128 so tau_bar/p_bar hits the target) is identical. The Phase B
cells add no new code path: only (n_modules, V, target_ratio) change.

Cells
  Phase A: 20x25 v1 t0.6 / v1 t1.0, 30x25 v1 t0.6 / v2 t1.0
  Phase B: 50x25 v2 t0.6 / v2 t1.0, 80x25 v3 t1.0 (stretch)
30 instances each, seeds 20000..20029 in every cell.

An existing cell is never regenerated (dataset_meta.json is the marker), so a
rerun cannot disturb results already computed on it; pass --force to override.

Usage:
  python scripts/x2_scale_make_data.py --phase B
  python scripts/x2_scale_make_data.py --cells 50x25+ppvct-mixed+v2+t0.6
"""

import sys, os, json, re, argparse

_ap = argparse.ArgumentParser()
_ap.add_argument('--phase', type=str, default='A', choices=['A', 'B', 'AB'])
_ap.add_argument('--cells', type=str, default='',
                 help='comma-separated cell names, e.g. 50x25+ppvct-mixed+v2+t0.6')
_ap.add_argument('--force', action='store_true')
_A = _ap.parse_args()

sys.argv = [sys.argv[0]]   # scrub CLI before params.py's import-time argparse

CELLS_A = [  # (n_modules, V, ratio)
    (20, 1, 0.6), (20, 1, 1.0),
    (30, 1, 0.6), (30, 2, 1.0),
]
CELLS_B = [
    (50, 2, 0.6), (50, 2, 1.0),
    (80, 3, 1.0),
]
SEED0, N_INST = 20000, 30
# Production batches (50 and 80 modules) use fixed base seeds of their own
# (seed0 100000 at 50 modules, 110000 at 80, 20 instances each), so any study
# that calls ppvc_instance_generator with these seeds gets the same module
# routings, processing times and lags; only the transport layout is added
# here. Other sizes keep the 20000.. block.
PRODUCTION_SEEDS = {50: (100000, 20), 80: (110000, 20)}

_CELL_RE = re.compile(r'^(\d+)x25\+ppvct-mixed\+v(\d+)\+t([0-9.]+)$')


def parse_cell(name):
    m = _CELL_RE.match(name.strip())
    assert m, f'cannot parse cell name {name!r}'
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def select_cells():
    if _A.cells:
        return [parse_cell(c) for c in _A.cells.split(',') if c.strip()]
    return {'A': CELLS_A, 'B': CELLS_B, 'AB': CELLS_A + CELLS_B}[_A.phase]


def main():
    import numpy as np
    sys.path.insert(0, '.')
    from ppvc_instance_generator import ppvc_instance_generator, save_instance
    from transport_marl.layout import (build_transport_layout, UNIT_TAU,
                                       min_unit_travel_per_pair,
                                       station_cells_from_mch_type)

    for (J, V, r) in select_cells():
        cell = f'{J}x25+ppvct-mixed+v{V}+t{r}'
        ds = f'data/PPVCT/{cell}/test'
        if os.path.exists(f'{ds}/dataset_meta.json') and not _A.force:
            print(f'skip {ds} (exists)', flush=True)
            continue
        os.makedirs(ds, exist_ok=True)
        seed0, n_inst = PRODUCTION_SEEDS.get(J, (SEED0, N_INST))
        n_ops, ratios, speeds = [], [], []
        for i in range(n_inst):
            jl, pt, meta = ppvc_instance_generator(
                n_modules=J, class_mix='mixed', seed=seed0 + i)
            layout = build_transport_layout(
                jl, pt, np.asarray(meta['mch_type']), r, V)
            meta = dict(meta)
            meta['transport'] = layout
            save_instance(f'{ds}/instance_{i:03d}', jl, pt, meta)
            # realized travel intensity: speed * tau_bar(unit) / p_bar, the
            # quantity layout.calibrate_speed targets before the 1/128 snap
            sc = station_cells_from_mch_type(np.asarray(meta['mch_type']))
            _, min_tau_unit, mand = min_unit_travel_per_pair(jl, pt, sc)
            ptm = np.asarray(pt, dtype=float)
            p_bar = float(np.mean([ptm[o][ptm[o] > 0].mean()
                                   for o in range(ptm.shape[0])]))
            tau_bar_unit = float(min_tau_unit[mand].mean())
            ratios.append(layout['speed_const'] * tau_bar_unit / p_bar)
            speeds.append(layout['speed_const'])
            n_ops.append(int(np.sum(jl)))
        dm = dict(seed0=seed0, n_instances=n_inst, n_modules=J,
                  ppvc_mix='mixed', ppvc_factory='default',
                  n_vehicles=V, tau_over_p=r, split='test-only (zero-shot)',
                  generator=('x2_scale_make_data.py v3 (fixed base seeds)'
                             if J in PRODUCTION_SEEDS else 'x2_scale_make_data.py v2'),
                  ops_mean=float(np.mean(n_ops)),
                  ops_min=int(np.min(n_ops)), ops_max=int(np.max(n_ops)),
                  realized_tau_over_p_mean=float(np.mean(ratios)),
                  realized_tau_over_p_min=float(np.min(ratios)),
                  realized_tau_over_p_max=float(np.max(ratios)))
        with open(f'{ds}/dataset_meta.json.tmp', 'w') as f:
            json.dump(dm, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(f'{ds}/dataset_meta.json.tmp', f'{ds}/dataset_meta.json')
        # header check: the .fjs first line must read "<J> 25 <flex>"
        with open(f'{ds}/instance_000.fjs') as f:
            hdr = f.readline().split()
        assert int(hdr[0]) == J and int(hdr[1]) == 25, \
            f'{ds}: bad header {hdr[:2]}, expected {J} 25'
        print(f'wrote {ds}: n={n_inst} seed0={seed0} header {hdr[0]}x{hdr[1]} '
              f'ops {np.mean(n_ops):.1f} [{np.min(n_ops)}, {np.max(n_ops)}] '
              f'speed_const {np.mean(speeds):.4f} '
              f'realized tau/p {np.mean(ratios):.4f} '
              f'[{np.min(ratios):.4f}, {np.max(ratios):.4f}] '
              f'(target {r})', flush=True)


if __name__ == '__main__':
    main()
