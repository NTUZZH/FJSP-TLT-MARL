"""Paired equivalence testing (TOST) for the E4 parity claim.

WHY THIS EXISTS
---------------
Paper X2's multi-agent claim is *quality parity at decentralized execution*,
not quality superiority. A non-significant difference test does NOT establish
parity: failing to reject "they differ" is not evidence that "they are the
same" (absence of evidence / evidence of absence). To claim equivalence we must
test for it directly, against an equivalence margin fixed BEFORE seeing the
data.

PROTOCOL (pre-specified on 2026-07-11, before the
single-agent arm was trained):
  - unit of replication: the TEST INSTANCE (n=100 per cell). Per-instance
    makespan is first averaged over the training seeds, so seeds are a nuisance
    factor averaged out and are NOT treated as independent replicates
    (no pseudo-replication).
  - statistic: the per-instance relative difference
        d_i = (ours_i - single_i) / single_i
    and its mean dbar (negative = our policy is better).
  - equivalence margin: delta = 2% of the single-agent makespan (EQUIV_DELTA).
  - decision: TOST at alpha = 0.05, implemented as the assumption-light
    equivalent: EQUIVALENT iff the two-sided 90% bootstrap CI of dbar lies
    entirely inside (-delta, +delta).  (A 90% CI corresponds to two one-sided
    tests at 5%: the standard TOST/CI duality.)
  - we ALWAYS report the CI itself, so a reader who prefers a different margin
    can apply it.
  - a difference test (two-sided Wilcoxon) is reported ALONGSIDE, never as the
    parity evidence.

Outcomes are reported honestly and can be any of:
  EQUIVALENT            CI inside +-delta
  OURS BETTER           CI entirely below -delta
  SINGLE BETTER         CI entirely above +delta
  INCONCLUSIVE          CI straddles a margin (not enough evidence either way)
"""

import numpy as np
from scipy.stats import wilcoxon

EQUIV_DELTA = 0.02       # 2% of makespan; pre-specified, see module docstring
ALPHA = 0.05
N_BOOT = 10000
BOOT_SEED = 20260711     # fixed: the CI must be reproducible


def paired_tost(ours, single, delta=EQUIV_DELTA, alpha=ALPHA,
                n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired equivalence test of `ours` against `single` (lower = better).

    `ours` and `single` are arrays of per-instance makespans, ALREADY averaged
    over training seeds, aligned instance-by-instance.
    """
    ours = np.asarray(ours, dtype=float)
    single = np.asarray(single, dtype=float)
    if ours.shape != single.shape:
        raise ValueError(f'unpaired arrays: {ours.shape} vs {single.shape}')

    d = (ours - single) / single           # relative difference, per instance
    dbar = float(d.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])  # 90% CI

    if hi < -delta:
        verdict = 'OURS BETTER'
    elif lo > delta:
        verdict = 'SINGLE BETTER'
    elif -delta < lo and hi < delta:
        verdict = 'EQUIVALENT'
    else:
        verdict = 'INCONCLUSIVE'

    # difference test, reported alongside and never used as parity evidence
    try:
        p_diff = float(wilcoxon(ours, single, alternative='two-sided').pvalue)
    except ValueError:                      # all-zero differences
        p_diff = 1.0

    w = int((ours < single - 1e-6).sum())
    t = int((np.abs(ours - single) <= 1e-6).sum())
    lo_ = int((ours > single + 1e-6).sum())

    return dict(n=len(d), delta=delta, mean_rel_diff_pct=100 * dbar,
                ci90_lo_pct=100 * float(lo), ci90_hi_pct=100 * float(hi),
                verdict=verdict, p_wilcoxon_two_sided=p_diff,
                wtl=f'{w}/{t}/{lo_}',
                ours_mean=float(ours.mean()), single_mean=float(single.mean()))


def format_row(r):
    return (f"n={r['n']}  ours={r['ours_mean']:.2f}  single={r['single_mean']:.2f}  "
            f"rel_diff={r['mean_rel_diff_pct']:+.2f}%  "
            f"90% CI [{r['ci90_lo_pct']:+.2f}%, {r['ci90_hi_pct']:+.2f}%]  "
            f"margin=+-{100 * r['delta']:.0f}%  -> {r['verdict']}  "
            f"(W/T/L {r['wtl']}, two-sided Wilcoxon p={r['p_wilcoxon_two_sided']:.3g})")


if __name__ == '__main__':
    # self-check on synthetic data: the procedure must be able to FAIL.
    rng = np.random.default_rng(0)
    base = rng.uniform(200, 600, 100)
    print('truly equivalent (0.2% worse):')
    print(' ', format_row(paired_tost(base * 1.002 + rng.normal(0, 1, 100), base)))
    print('truly different (6% worse):')
    print(' ', format_row(paired_tost(base * 1.06 + rng.normal(0, 1, 100), base)))
    print('truly better (5% better):')
    print(' ', format_row(paired_tost(base * 0.95 + rng.normal(0, 1, 100), base)))
    print('borderline (1.9% worse, near the margin):')
    print(' ', format_row(paired_tost(base * 1.019 + rng.normal(0, 8, 100), base)))
