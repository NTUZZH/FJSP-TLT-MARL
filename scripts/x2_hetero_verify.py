"""verify the heterogeneity datasets before any evaluation.

Three questions, answered from the files themselves:

1. Is the transform mean-preserving? Per operation, the drift
   |mean_g p'(o,m) / mean_g p(o,m) - 1|. Rounding to integer hours makes
   exact preservation impossible, so the answer is a rounding-level
   distribution, and the maximum is reported.
2. What dispersion did we actually realize? The within-operation max/min
   ratio over the eligible set, per level. This sits below the nominal R by
   construction (R is the support width of the factor distribution, not the
   expected range of a small sample, and rounding compresses it further),
   so both the nominal and the realized number are reported.
3. Is anything else touched? The side-car metadata is compared field by
   field against the base instance (op types, stations, lags, machine types,
   routing classes, module weights, station counts, and every transport
   layout field), and the eligibility pattern p(o,m) > 0 is compared cell by
   cell.

Writes results/hetero/verification.json and prints the tables.

Usage: python scripts/x2_hetero_verify.py
"""

import argparse
import json
import os
import sys

cli = argparse.ArgumentParser()
cli.add_argument('--cells', type=str, default='v1+t0.6,v2+t0.6,v1+t1.0')
cli.add_argument('--levels', type=str, default='2,5,10,20')
A = cli.parse_args()
sys.argv = [sys.argv[0]]

import glob
import numpy as np

sys.path.insert(0, '.')
from ppvc_instance_generator import load_instance

BASE_ROOT = 'data/PPVCT'
HET_ROOT = 'data/PPVCT_HET'
META_FIELDS = ['op_type', 'op_station', 'time_lag', 'mch_type',
               'routing_class', 'module_weight_t', 'op_name',
               'station_counts']
TRANSPORT_FIELDS = ['station_cell', 'cell_xy', 'speed_const', 'tau_cells',
                    'n_vehicles', 'veh_start_cell', 'tau_over_p']


def cell_dir(cell):
    return cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'


def ratios_and_drift(base_pt, new_pt):
    """Per-op mean drift, within-op max/min ratio (|g| >= 2), and the two
    instance-level aggregates that the bound actually reads: the total of the
    per-op eligible means (the machine term's work) and the total of the
    per-op eligible minima (the chain term's work)."""
    drift, ratio = [], []
    sum_mean_b = sum_mean_n = sum_min_b = sum_min_n = 0.0
    for o in range(base_pt.shape[0]):
        g = np.nonzero(base_pt[o] > 0)[0]
        if g.size == 0:
            continue
        b = base_pt[o, g].astype(float)
        n = new_pt[o, g].astype(float)
        drift.append(abs(n.mean() / b.mean() - 1.0))
        sum_mean_b += b.mean(); sum_mean_n += n.mean()
        sum_min_b += b.min(); sum_min_n += n.min()
        if g.size >= 2:
            ratio.append(n.max() / n.min())
    return (np.array(drift), np.array(ratio),
            sum_mean_n / sum_mean_b - 1.0, sum_min_n / sum_min_b - 1.0)


def main():
    cells = A.cells.split(',')
    levels = [int(x) for x in A.levels.split(',')]
    os.makedirs('results/hetero', exist_ok=True)
    rows = []
    print(f'{"cell":16s} {"R":>3s} {"n":>4s} {"drift_max":>10s} '
          f'{"drift_mean":>11s} {"Rp_mean":>8s} {"Rp_med":>7s} {"Rp_p95":>7s} '
          f'{"meta_ok":>8s} {"elig_ok":>8s} {"singleton%":>11s}')
    for cell in cells:
        src = f'{BASE_ROOT}/{cell_dir(cell)}/test'
        stems = sorted(g[:-4] for g in glob.glob(f'{src}/instance_*.fjs'))
        # baseline row (level 1 = untouched)
        base_ratio = []
        singleton = 0
        total_ops = 0
        for s in stems:
            _, pt, _ = load_instance(s)
            pt = np.asarray(pt)
            for o in range(pt.shape[0]):
                g = np.nonzero(pt[o] > 0)[0]
                total_ops += 1
                if g.size < 2:
                    singleton += 1
                    continue
                v = pt[o, g].astype(float)
                base_ratio.append(v.max() / v.min())
        base_ratio = np.array(base_ratio)
        rows.append(dict(cell=cell, R=1, n=len(stems),
                         drift_max=0.0, drift_mean=0.0,
                         realized_R_mean=float(base_ratio.mean()),
                         realized_R_median=float(np.median(base_ratio)),
                         realized_R_p95=float(np.percentile(base_ratio, 95)),
                         realized_R_max=float(base_ratio.max()),
                         meta_identical=True, eligibility_identical=True,
                         singleton_op_frac=singleton / total_ops))
        print(f'{cell:16s} {1:3d} {len(stems):4d} {0.0:10.5f} {0.0:11.5f} '
              f'{base_ratio.mean():8.3f} {np.median(base_ratio):7.3f} '
              f'{np.percentile(base_ratio, 95):7.3f} {"n/a":>8s} {"n/a":>8s} '
              f'{100*singleton/total_ops:10.1f}%')

        for R in levels:
            dst = f'{HET_ROOT}/{cell_dir(cell)}+hetR{R}/test'
            d_all, r_all, agg_mean, agg_min = [], [], [], []
            meta_ok, elig_ok = True, True
            bad = []
            for s in stems:
                name = os.path.basename(s)
                jl_b, pt_b, mb = load_instance(s)
                jl_n, pt_n, mn = load_instance(f'{dst}/{name}')
                pt_b, pt_n = np.asarray(pt_b), np.asarray(pt_n)
                if not np.array_equal(np.asarray(jl_b), np.asarray(jl_n)):
                    meta_ok = False; bad.append(f'{name}:job_length')
                for f in META_FIELDS:
                    a, b = mb[f], mn[f]
                    same = (a == b if isinstance(a, dict)
                            else np.array_equal(np.asarray(a), np.asarray(b)))
                    if not same:
                        meta_ok = False; bad.append(f'{name}:{f}')
                for f in TRANSPORT_FIELDS:
                    a = np.asarray(mb['transport'][f])
                    b = np.asarray(mn['transport'][f])
                    if not np.array_equal(a, b):
                        meta_ok = False; bad.append(f'{name}:transport.{f}')
                if not np.array_equal(pt_b > 0, pt_n > 0):
                    elig_ok = False; bad.append(f'{name}:eligibility')
                d, r, am, ami = ratios_and_drift(pt_b, pt_n)
                d_all.append(d); r_all.append(r)
                agg_mean.append(am); agg_min.append(ami)
            d_all = np.concatenate(d_all); r_all = np.concatenate(r_all)
            agg_mean = np.array(agg_mean); agg_min = np.array(agg_min)
            rows.append(dict(cell=cell, R=R, n=len(stems),
                             drift_max=float(d_all.max()),
                             drift_mean=float(d_all.mean()),
                             drift_p99=float(np.percentile(d_all, 99)),
                             drift_frac_gt5pct=float((d_all > 0.05).mean()),
                             drift_frac_gt10pct=float((d_all > 0.10).mean()),
                             inst_mean_work_drift_mean=float(agg_mean.mean()),
                             inst_mean_work_drift_absmax=float(
                                 np.abs(agg_mean).max()),
                             inst_min_work_drift_mean=float(agg_min.mean()),
                             realized_R_mean=float(r_all.mean()),
                             realized_R_median=float(np.median(r_all)),
                             realized_R_p95=float(np.percentile(r_all, 95)),
                             realized_R_max=float(r_all.max()),
                             meta_identical=bool(meta_ok),
                             eligibility_identical=bool(elig_ok),
                             offenders=bad[:5]))
            print(f'{cell:16s} {R:3d} {len(stems):4d} {d_all.max():10.5f} '
                  f'{d_all.mean():11.5f} {r_all.mean():8.3f} '
                  f'{np.median(r_all):7.3f} {np.percentile(r_all, 95):7.3f} '
                  f'{str(meta_ok):>8s} {str(elig_ok):>8s} '
                  f'  work {100*agg_mean.mean():+.2f}%  minwork '
                  f'{100*agg_min.mean():+.2f}%')
    with open('results/hetero/verification.json', 'w') as f:
        json.dump(rows, f, indent=1)
    ok = all(r['meta_identical'] and r['eligibility_identical'] for r in rows)
    print(f'\nall metadata/eligibility checks passed: {ok}')
    print('wrote results/hetero/verification.json')
    assert ok, 'verification FAILED'


if __name__ == '__main__':
    main()
