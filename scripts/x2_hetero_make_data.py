"""build processing-time-heterogeneity copies of the test cells.

Mean-preserving within-operation dispersion, paired on the SAME base
instances (Section VI-C4 of the manuscript).

For every operation o with eligible machine set g = {m : p(o,m) > 0}:
    p_bar   = mean_{m in g} p(o,m)                     (current nominal)
    q_m     ~ LogUniform(1/sqrt(R), sqrt(R)),  m in g
    q_m    <- q_m / mean_{g}(q)                        (mean exactly 1)
    p'(o,m) = max(1, round(p_bar * q_m))

Everything else is carried through unchanged: routes, op types/stations,
lags, machine types, routing classes, module weights, station counts, the
transport layout (cells, tau matrix, fleet size, start cell) and the
eligibility pattern itself. Only the numbers inside the eligible cells move.

The RNG is seeded per (cell, R, instance index, operation index) with a
stable CRC32 of the cell name, so the datasets are reproducible from this
file alone and independent of generation order.

Output lives under data/PPVCT_HET/ and NEVER under data/PPVCT/: every
analysis glob in the repository is anchored at the literal path component
'data/PPVCT/' (eval_ppvct.py, p1_eval_pdr.py, p8_certificate.py,
x2_fleet_util.py, x2_bound_terms.py, x2_scale_cert.py), so a separate top
component cannot be swept in by an existing script.

Usage:
  python scripts/x2_hetero_make_data.py                # all cells, R in 2,5,10,20
  python scripts/x2_hetero_make_data.py --levels 5,20
"""

import argparse
import os
import sys
import zlib

cli = argparse.ArgumentParser()
cli.add_argument('--cells', type=str, default='v1+t0.6,v2+t0.6,v1+t1.0')
cli.add_argument('--levels', type=str, default='2,5,10,20')
cli.add_argument('--split', type=str, default='test')
A = cli.parse_args()
sys.argv = [sys.argv[0]]

import glob
import numpy as np

sys.path.insert(0, '.')
from ppvc_instance_generator import load_instance, save_instance

BASE_ROOT = 'data/PPVCT'
HET_ROOT = 'data/PPVCT_HET'
ENTROPY = 20260901          # fixed study-level entropy for the heterogeneity study


def cell_dir(cell):
    return cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'


def het_dir(cell, R):
    return f'{cell_dir(cell)}+hetR{R}'


def transform(op_pt, cell, R, inst_idx):
    """Mean-preserving within-op dispersion at level R. Returns int array."""
    ck = zlib.crc32(cell.encode()) & 0xFFFFFFFF
    half = 0.5 * np.log(R)
    out = np.zeros_like(op_pt)
    n_ops = op_pt.shape[0]
    for o in range(n_ops):
        g = np.nonzero(op_pt[o] > 0)[0]
        if g.size == 0:
            continue
        p_bar = float(op_pt[o, g].mean())
        rng = np.random.default_rng([ENTROPY, ck, int(R), int(inst_idx), int(o)])
        q = np.exp(rng.uniform(-half, half, size=g.size))
        q = q / q.mean()                       # mean exactly 1 over the group
        out[o, g] = np.maximum(1, np.rint(p_bar * q)).astype(op_pt.dtype)
    return out


def main():
    cells = A.cells.split(',')
    levels = [int(x) for x in A.levels.split(',')]
    for cell in cells:
        src = f'{BASE_ROOT}/{cell_dir(cell)}/{A.split}'
        stems = sorted(g[:-4] for g in glob.glob(f'{src}/instance_*.fjs'))
        assert stems, f'no instances under {src}'
        for R in levels:
            dst = f'{HET_ROOT}/{het_dir(cell, R)}/{A.split}'
            os.makedirs(dst, exist_ok=True)
            for i, stem in enumerate(stems):
                jl, pt, meta = load_instance(stem)
                pt = np.asarray(pt)
                new_pt = transform(pt, cell, R, i)
                assert np.array_equal(new_pt > 0, pt > 0), \
                    f'{stem}: eligibility pattern changed'
                name = os.path.basename(stem)
                save_instance(f'{dst}/{name}', jl, new_pt, meta)
            # carry the dataset card over, annotated
            card = f'{src}/dataset_meta.json'
            if os.path.exists(card):
                import json
                dm = json.load(open(card))
                dm['hetero_R'] = R
                dm['hetero_base'] = f'{BASE_ROOT}/{cell_dir(cell)}/{A.split}'
                dm['generator'] = 'scripts/x2_hetero_make_data.py (heterogeneity study)'
                dm['entropy'] = ENTROPY
                with open(f'{dst}/dataset_meta.json', 'w') as f:
                    json.dump(dm, f, indent=1)
            print(f'wrote {dst}: {len(stems)} instances', flush=True)


if __name__ == '__main__':
    main()
