"""Phase A scale-up: assemble the per-cell table from the arm outputs.

Reads results/scaleup/{pdr,ga,policy,certificate}/ and prints a markdown table
plus the per-cell detail the verdict paragraphs need. Nothing is written into
the manuscript's result directories.

Conventions, matched to the manuscript's own:
  * policy value on an instance = mean over the three seeds (fill_macros
    seed_mean), so W/T/L is a per-instance comparison of that mean;
  * best-fixed PDR pair = the pair with the lowest CELL mean (fill_macros
    pdr_best), best-of-9 = per-instance minimum over the nine pairs;
  * certificate = (C - B0) / B0 with C the policy's 3-seed-mean makespan.

Usage: python scripts/x2_scale_table.py
"""

import glob
import json
import os
import re
import sys

sys.argv = [sys.argv[0]]
import numpy as np
from scipy.stats import wilcoxon

CELLS = ['20x25+ppvct-mixed+v1+t0.6', '20x25+ppvct-mixed+v1+t1.0',
         '30x25+ppvct-mixed+v1+t0.6', '30x25+ppvct-mixed+v2+t1.0']
ARM = '10x25+ppvct-mixed+m1-bcb-guide'
TOL = 1e-6


def wtl(a, b):
    """W/T/L of a against b (win = a strictly lower makespan)."""
    d = np.asarray(a) - np.asarray(b)
    return int((d < -TOL).sum()), int((np.abs(d) <= TOL).sum()), int((d > TOL).sum())


def cell_stats(cell):
    names = sorted(json.load(open(f'results/scaleup/pdr/{cell}.json')).keys())
    pdr = json.load(open(f'results/scaleup/pdr/{cell}.json'))
    pairs = sorted(next(iter(pdr.values())).keys())
    pdr_arr = {q: np.array([pdr[n][q] for n in names]) for q in pairs}
    best_pair = min(pdr_arr, key=lambda q: pdr_arr[q].mean())
    pdr_fixed = pdr_arr[best_pair]
    pdr_b9 = np.array([min(pdr[n].values()) for n in names])

    ga = json.load(open(f'results/scaleup/ga/{cell}.json'))
    ga_arr = np.array([ga[n]['ga'] for n in names])
    ga_improved = int(sum(ga[n]['improved_on_seed'] for n in names))
    ga_cpu = float(np.mean([ga[n]['cpu'] for n in names]))

    seed_ms, seed_dec, lat = {}, {}, {}
    for p in sorted(glob.glob(f'results/scaleup/policy/{ARM}-s*_{cell}.json')):
        if p.endswith('+cpu4.json'):
            continue
        s = int(re.search(r'-s(\d+)_', os.path.basename(p)).group(1))
        d = json.load(open(p))
        seed_ms[s] = np.array([d['rows'][n]['ms'] for n in names])
        seed_dec[s] = np.array([d['rows'][n]['decisions'] for n in names])
        lat[s] = float(np.mean([d['rows'][n]['lat_batch_s'] for n in names]))
    assert len(seed_ms) == 3, f'{cell}: {sorted(seed_ms)} seeds, expected 3'
    pol = np.mean(list(seed_ms.values()), axis=0)
    dec = np.mean(list(seed_dec.values()), axis=0)

    B0 = json.load(open(f'results/scaleup/certificate/{cell}.json'))
    b0 = np.array([B0[n] for n in names])
    cert = (pol - b0) / b0

    w9, t9, l9 = wtl(pol, pdr_b9)
    wf, tf, lf = wtl(pol, pdr_fixed)
    wg, tg, lg = wtl(pol, ga_arr)
    p9 = wilcoxon(pol, pdr_b9).pvalue if (w9 + l9) else float('nan')
    pg = wilcoxon(pol, ga_arr).pvalue if (wg + lg) else float('nan')
    return dict(
        cell=cell, n=len(names), best_pair=best_pair,
        pdr_fixed=pdr_fixed.mean(), pdr_b9=pdr_b9.mean(),
        ga=ga_arr.mean(), ga_improved=ga_improved, ga_cpu=ga_cpu,
        pol=pol.mean(), pol_sd_inst=pol.std(ddof=1),
        seed_means={s: float(v.mean()) for s, v in seed_ms.items()},
        pol_sd_seed=float(np.std([v.mean() for v in seed_ms.values()], ddof=1)),
        wtl9=(w9, t9, l9), wtlf=(wf, tf, lf), wtlg=(wg, tg, lg),
        p9=p9, pg=pg, cert=cert.mean(), cert_min=cert.min(),
        cert_max=cert.max(), dec=dec.mean(), b0=b0.mean(),
        lat_ms=float(np.mean(list(lat.values())) * 1000),
        d9=float(np.mean((pol - pdr_b9) / pdr_b9)),
        df=float(np.mean((pol - pdr_fixed) / pdr_fixed)),
        dg=float(np.mean((pol - ga_arr) / ga_arr)))


def main():
    rows = [cell_stats(c) for c in CELLS]
    hdr = ('| cell | n | best fixed PDR pair (mean) | best-of-9 PDR (mean) | '
           'GA v2 60 CPU-s (mean, improved/n) | policy 3-seed (mean +- sd) | '
           'policy vs best-of-9 W/T/L | policy vs GA W/T/L | mean certificate | '
           'mean decisions |')
    print(hdr)
    print('|' + '---|' * 10)
    for r in rows:
        print(f'| {r["cell"]} | {r["n"]} | {r["best_pair"]} {r["pdr_fixed"]:.1f} '
              f'| {r["pdr_b9"]:.1f} | {r["ga"]:.1f} ({r["ga_improved"]}/{r["n"]}) '
              f'| {r["pol"]:.1f} +- {r["pol_sd_inst"]:.1f} '
              f'| {r["wtl9"][0]}/{r["wtl9"][1]}/{r["wtl9"][2]} '
              f'| {r["wtlg"][0]}/{r["wtlg"][1]}/{r["wtlg"][2]} '
              f'| {r["cert"]:.1%} | {r["dec"]:.0f} |')
    print()
    for r in rows:
        print(f'{r["cell"]}: policy {r["pol"]:.1f} '
              f'(seed means {", ".join(f"{s}:{v:.1f}" for s, v in sorted(r["seed_means"].items()))}, '
              f'sd across seeds {r["pol_sd_seed"]:.2f}) | '
              f'vs best-fixed {r["df"]:+.2%} W/T/L {r["wtlf"]} | '
              f'vs best-of-9 {r["d9"]:+.2%} W/T/L {r["wtl9"]} p={r["p9"]:.2e} | '
              f'vs GA {r["dg"]:+.2%} W/T/L {r["wtlg"]} p={r["pg"]:.2e} | '
              f'B0 {r["b0"]:.1f} cert {r["cert"]:.2%} '
              f'[{r["cert_min"]:.2%}, {r["cert_max"]:.2%}] | '
              f'GA cpu/inst {r["ga_cpu"]:.0f}s | '
              f'lat/batched-fwd {r["lat_ms"]:.1f} ms CONTENDED')
    with open('results/scaleup/summary.json', 'w') as f:
        json.dump(rows, f, indent=1, default=float)


if __name__ == '__main__':
    main()
