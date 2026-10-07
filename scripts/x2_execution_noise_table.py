"""Realized makespan under duration noise: supplement table and macros.

Reads the ledgers written by scripts/x2_execution_noise.py
(results/execution_noise/{cell}.jsonl) and produces, for the two production
cells and every noise level, the realized makespan of the seven arms: the
policy's plan kept (right shift), re-planned at every checkpoint, or
re-planned only above the lateness threshold; best-of-nine rule re-dispatch;
the GA-60 plan kept, re-planned by the GA, or re-planned by residual CP-SAT.

Conventions (those of scripts/x2_disruption_search_table.py):
  * a value per instance, the mean over the noise replicates r0-r2, and over
    the policy's training seeds where the key carries one (the design runs
    seed 301 only, see MODELS);
  * makespan in hours as mean +- sample standard deviation (ddof=1) over the
    20 instances;
  * relative differences are 100 * (mean(rival) - mean(ours)) / mean(rival),
    so the rival is the denominator and a positive value means the policy
    re-plan is shorter;
  * win/tie/loss of the policy re-plan, ties at 1e-6; paired two-sided
    Wilcoxon over the 20 instances, Holm-adjusted within each (cell,
    comparison) family across the four noise levels.

The ledgers may still be growing. The script prints, per (cell, arm, sigma),
how many of the expected records exist; the table and macros printed from an
incomplete ledger are marked partial, and --write refuses to run until every
expected key is present.

The table environment is printed between the marker comments
'% >>> tab:execnoise' and '% <<< tab:execnoise' for manual placement in
supplementary.tex. With --write the macros are patched into macros.tex in
place (or appended as one block the first time); without it they are printed.

Usage:
  python scripts/x2_execution_noise_table.py           # print only
  python scripts/x2_execution_noise_table.py --write   # patch macros.tex
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

WRITE = '--write' in sys.argv
MACROS = 'paper/main_manuscript_bundle/macros.tex'
LEDGER = 'results/execution_noise/{}.jsonl'
TAG = 'tab:execnoise'
TOL = 1e-6
CELLS = (('50x25+ppvct-mixed+v2+t1.0', 'Fifty', '$50$, $2$, $1.0$'),
         ('80x25+ppvct-mixed+v3+t1.0', 'Eighty', '$80$, $3$, $1.0$'))
SIGMAS = (0.0, 0.1, 0.2, 0.3)
REPS = (0, 1, 2)
MODELS = ('mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-s301',)
ARMS = (('policy_rs', 'Policy', 'Right shift'),
        ('policy_replan', 'Policy', 'Re-plan'),
        ('policy_threshold', 'Policy', 'Re-plan when late'),
        ('pdr_replan', 'Best of nine rules', 'Rule re-dispatch'),
        ('ga60_rs', 'GA, 60 CPU-s', 'Right shift'),
        ('ga_replan', 'GA, 60 CPU-s', 'GA re-plan'),
        ('cpsat_replan', 'GA, 60 CPU-s', 'CP-SAT re-plan'))
POLICY_ARMS = ('policy_rs', 'policy_replan', 'policy_threshold')
REPLAN_ARMS = ('policy_replan', 'pdr_replan', 'ga_replan', 'cpsat_replan')
CPSAT_DESIGN = ((0.0, 0), (0.2, 0))          # (sigma, replicate) actually run
RIVALS = (('policy_rs', 'policy right shift'),
          ('pdr_replan', 'rule re-dispatch'),
          ('ga_replan', 'GA re-plan'))
CORES = set(range(2, 8))                     # performance cores 2-7
SETTINGS = dict(handoff='commit', checkpoints_k=10, ga_replan_budget=10.0,
                cpsat_budget=10.0, ga_nominal_budget=60.0, ga_pop=100)


def instances(cell):
    names = sorted(os.path.basename(p)[:-4] for p in
                   glob.glob(f'data/PPVCT/{cell}/test/instance_*.fjs'))
    assert len(names) == 20, (cell, len(names))
    return names


def key(cell, name, rep, sigma, arm, model=None):
    tag = f'|{model}' if model else ''
    return f'{cell}|{name}|r{rep}|s{sigma:g}|{arm}{tag}'


def expected(cell):
    """{(arm, sigma): set of keys} of the design."""
    out = defaultdict(set)
    for name in instances(cell):
        for arm, _, _ in ARMS:
            models = MODELS if arm in POLICY_ARMS else (None,)
            for sigma in SIGMAS:
                for rep in REPS:
                    if arm == 'cpsat_replan' and (sigma, rep) not in CPSAT_DESIGN:
                        continue
                    for m in models:
                        out[(arm, sigma)].add(key(cell, name, rep, sigma, arm, m))
    return out


def cores_of(s):
    lo, hi = (int(x) for x in s.split('-'))
    return set(range(lo, hi + 1))


def load(cell):
    recs = {}
    lines = open(LEDGER.format(cell)).read().split('\n')
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            # a line still being appended; anything earlier is an error
            assert i == len(lines) - 1, f'{cell}: bad line {i + 1}'
            continue
        assert r['key'] not in recs, f'duplicate key {r["key"]}'
        assert r['validated'], f'unvalidated record {r["key"]}'
        for k, v in SETTINGS.items():
            assert r[k] == v, f'{r["key"]}: {k}={r[k]}, expected {v}'
        assert cores_of(r['cores']) <= CORES, f'{r["key"]}: cores {r["cores"]}'
        if r['arm'] == 'cpsat_replan':
            assert r['cpsat_workers'] == 3, r['key']
        if r['arm'] == 'policy_threshold':
            assert r['replan_threshold'] == 0.05, r['key']
        if r['arm'] in ('policy_rs', 'ga60_rs') and r['sigma'] == 0.0:
            assert abs(r['realized_makespan'] - r['zero_noise_makespan']) < TOL, \
                f'{r["key"]}: zero-noise right shift differs from its execution'
        recs[r['key']] = r
    return recs


def coverage(cell, recs):
    exp = expected(cell)
    want = set().union(*exp.values())
    extra = sorted(set(recs) - want)
    grid = {}
    for arm, _, _ in ARMS:
        for sigma in SIGMAS:
            ks = exp.get((arm, sigma), set())
            grid[(arm, sigma)] = (sum(k in recs for k in ks), len(ks))
    return grid, extra


def per_instance(recs, cell, arm, sigma, reps=REPS):
    """{instance: mean realized makespan over replicates (and seeds)}."""
    acc = defaultdict(list)
    for r in recs.values():
        if (r['arm'] == arm and r['sigma'] == sigma
                and r['replicate'] in reps):
            acc[r['instance']].append(r['realized_makespan'])
    return {i: float(np.mean(v)) for i, v in acc.items()}


def paired(recs, arm_a, arm_b, sigma, reps=REPS):
    """Instance values of two arms on the (instance, replicate) pairs both
    have; with a complete ledger these are all 20 x len(reps) pairs."""
    acc = {arm_a: defaultdict(list), arm_b: defaultdict(list)}
    for r in recs.values():
        if r['arm'] in acc and r['sigma'] == sigma and r['replicate'] in reps:
            acc[r['arm']][(r['instance'], r['replicate'])].append(
                r['realized_makespan'])
    both = sorted(set(acc[arm_a]) & set(acc[arm_b]))
    names = sorted({n for n, _ in both})

    def inst(arm):
        return np.array([np.mean([np.mean(acc[arm][k]) for k in both
                                  if k[0] == n]) for n in names])
    return inst(arm_a), inst(arm_b)


def rel(ours, rival):
    return 100.0 * (np.mean(rival) - np.mean(ours)) / np.mean(rival)


def wtl(ours, rival):
    d = np.asarray(ours) - np.asarray(rival)
    w, l = int((d < -TOL).sum()), int((d > TOL).sum())
    return f'{w}/{len(d) - w - l}/{l}'


def wilcoxon_p(ours, rival):
    d = np.asarray(ours) - np.asarray(rival)
    d[np.abs(d) <= TOL] = 0.0
    if not np.any(d):
        return 1.0
    return float(wilcoxon(d).pvalue)


def holm(ps):
    order = np.argsort(ps)
    m, run, adj = len(ps), 0.0, [0.0] * len(ps)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def fmt_p(p):
    if p >= 0.01:
        return f'{p:.2f}'
    m_, e_ = f'{p:.1e}'.split('e')
    return f'${m_}\\times10^{{{int(e_)}}}$'


def main():
    recs, complete = {}, True
    print('coverage: records present / expected, per arm and sigma')
    for cell, _, _ in CELLS:
        recs[cell] = load(cell)
        grid, extra = coverage(cell, recs[cell])
        print(f'  {cell}')
        print('    ' + f'{"arm":18s}' + ''.join(f'{"s=" + format(s, "g"):>10s}'
                                               for s in SIGMAS))
        for arm, _, _ in ARMS:
            row = ''.join(f'{f"{n}/{e}" if e else "--":>10s}'
                          for n, e in (grid[(arm, s)] for s in SIGMAS))
            print(f'    {arm:18s}{row}')
            complete &= all(n == e for n, e in (grid[(arm, s)] for s in SIGMAS))
        if extra:
            complete = False
            print(f'    {len(extra)} records outside the design, e.g. {extra[0]}')
    if not complete:
        print('\nPARTIAL: the ledgers are incomplete; every number below is '
              'provisional and --write is refused.')

    table, tests, cpu, wall, nrep = {}, {}, {}, {}, {}
    growth, vs, zero_cost, cp_vs = {}, defaultdict(list), {}, []
    for cell, word, _ in CELLS:
        R = recs[cell]
        for arm, _, _ in ARMS:
            for sigma in SIGMAS:
                v = per_instance(R, cell, arm, sigma)
                if v:
                    a = np.array(list(v.values()))
                    sd = a.std(ddof=1) if len(a) > 1 else float('nan')
                    table[(cell, arm, sigma)] = (a.mean(), sd, len(a))
            rs = [r for r in R.values() if r['arm'] == arm]
            if arm in REPLAN_ARMS and rs:
                nrep[(cell, arm)] = [r['n_replans'] for r in rs]
                c = [x for r in rs for x in r['plan_cpu_s']]
                w = [x for r in rs for x in r['plan_wall_s']]
                cpu[(cell, arm)] = float(np.mean(c)) if c else float('nan')
                wall[(cell, arm)] = float(np.mean(w)) if w else float('nan')
        for rival, _ in RIVALS:
            ps, rows = [], []
            for sigma in SIGMAS:
                ours, riv = paired(R, 'policy_replan', rival, sigma)
                if len(ours) == 0:
                    rows.append(None)
                    ps.append(1.0)
                    continue
                g = rel(ours, riv)
                rows.append((wtl(ours, riv), g, len(ours)))
                ps.append(wilcoxon_p(ours, riv))
                if sigma > 0:
                    vs[rival].append(g)
                elif rival == 'policy_rs':
                    zero_cost[word] = g
            for sigma, row, p, padj in zip(SIGMAS, rows, ps, holm(ps)):
                tests[(cell, rival, sigma)] = row and row + (padj,)
                if row:
                    print(f'{cell[:2]} s={sigma:g} replan vs {rival:10s} '
                          f'{row[1]:+6.2f}%  W/T/L {row[0]}  n={row[2]}  '
                          f'p={p:.1e}  Holm p={padj:.1e}')
        for sigma, rep in CPSAT_DESIGN:
            ours, riv = paired(R, 'policy_replan', 'cpsat_replan', sigma,
                               (rep,))
            if len(ours):
                cp_vs.append(rel(ours, riv))
                print(f'{cell[:2]} s={sigma:g} r{rep} replan vs cpsat_replan '
                      f'{cp_vs[-1]:+6.2f}%  W/T/L {wtl(ours, riv)}  '
                      f'n={len(ours)}')
        for arm in ('policy_rs', 'ga60_rs'):
            if (cell, arm, 0.3) in table and (cell, arm, 0.0) in table:
                growth[(word, arm)] = 100.0 * (table[(cell, arm, 0.3)][0]
                                               / table[(cell, arm, 0.0)][0] - 1)

    thr = [r for c, _, _ in CELLS for r in recs[c].values()
           if r['arm'] == 'policy_threshold']
    late = [max(x[1] for x in r['lateness']) / (0.05 * r['nominal_makespan'])
            for r in thr if r['lateness']]

    def mm(vals, name, note):
        if vals:
            macro(f'{name}Min', f'{min(vals):.1f}', note)
            macro(f'{name}Max', f'{max(vals):.1f}', 'same, max')

    mm(vs['policy_rs'], 'NoisePolReplanVsRs',
       '(policy right shift - policy re-plan)/policy right shift in %, '
       'min over sigma 0.1/0.2/0.3 and both cells')
    mm(vs['pdr_replan'], 'NoisePolVsPdr',
       '(rule re-dispatch - policy re-plan)/rule re-dispatch in %, '
       'min over sigma 0.1/0.2/0.3 and both cells')
    mm(vs['ga_replan'], 'NoisePolVsGa',
       '(GA re-plan - policy re-plan)/GA re-plan in %, min over sigma '
       '0.1/0.2/0.3 and both cells; negative = GA re-plan shorter')
    mm(cp_vs, 'NoisePolVsCp',
       '(CP-SAT re-plan - policy re-plan)/CP-SAT re-plan in %, replicate 0 '
       'only for both arms, min over sigma 0/0.2 and both cells')
    # the same two search comparisons as positive gaps, for prose that says
    # "the policy re-plan is X-Y% longer than the GA (CP-SAT) re-plan"
    if vs['ga_replan']:
        macro('NoisePolBehindGaMin', f'{-max(vs["ga_replan"]):.1f}',
              '-NoisePolVsGaMax: smallest gap by which the policy re-plan is '
              'longer than the GA re-plan, in % of the GA re-plan')
        macro('NoisePolBehindGaMax', f'{-min(vs["ga_replan"]):.1f}',
              '-NoisePolVsGaMin: same, largest')
    if cp_vs:
        macro('NoisePolBehindCpMin', f'{-max(cp_vs):.1f}',
              '-NoisePolVsCpMax: smallest gap by which the policy re-plan is '
              'longer than the CP-SAT re-plan, in % of the CP-SAT re-plan')
        macro('NoisePolBehindCpMax', f'{-min(cp_vs):.1f}',
              '-NoisePolVsCpMin: same, largest')
    prs = [tests[(c, 'policy_rs', s)][3] for c, _, _ in CELLS for s in SIGMAS
           if s > 0 and tests.get((c, 'policy_rs', s))]
    if prs:
        macro('NoisePolReplanVsRsPMin', f'{min(prs):.2f}',
              'smallest Holm-adjusted p, policy re-plan vs policy right shift, '
              'sigma 0.1/0.2/0.3, both cells')
    for word in ('Fifty', 'Eighty'):
        cell = next(c for c, w, _ in CELLS if w == word)
        t0 = tests.get((cell, 'policy_rs', 0.0))
        if t0:
            macro(f'NoisePolReplanVsRsZeroP{word}', f'{t0[3]:.2f}',
                  'Holm-adjusted p, policy re-plan vs policy right shift, sigma 0')
        if word in zero_cost:
            macro(f'NoisePolReplanVsRsZero{word}', f'{zero_cost[word]:.1f}',
                  '(policy right shift - policy re-plan)/policy right shift '
                  'in %, sigma 0: the cost of re-planning without noise')
        if (word, 'policy_rs') in growth:
            macro(f'NoiseGrowthPolRs{word}', f'{growth[(word, "policy_rs")]:.1f}',
                  'policy right shift, realized makespan at sigma 0.3 '
                  'relative to sigma 0, in %')
        if (word, 'ga60_rs') in growth:
            macro(f'NoiseGrowthGaRs{word}', f'{growth[(word, "ga60_rs")]:.1f}',
                  'GA-60 right shift, same')
        for arm, tag in (('policy_replan', 'Pol'), ('pdr_replan', 'Pdr'),
                         ('ga_replan', 'Ga'), ('cpsat_replan', 'Cp')):
            if (cell, arm) in cpu:
                macro(f'NoiseReplanCpu{tag}{word}', f'{cpu[(cell, arm)]:.2f}'
                      if cpu[(cell, arm)] < 10 else f'{cpu[(cell, arm)]:.1f}',
                      f'{arm}: mean planning CPU s per re-plan, all sigma'
                      + (', three workers' if arm == 'cpsat_replan' else ''))
        if (cell, 'cpsat_replan') in wall:
            macro(f'NoiseReplanWallCp{word}', f'{wall[(cell, "cpsat_replan")]:.1f}',
                  'cpsat_replan: mean wall s per re-plan')
    pol = [r['n_replans'] for c, _, _ in CELLS for r in recs[c].values()
           if r['arm'] == 'policy_replan']
    if pol:
        macro('NoiseReplansPol', f'{np.mean(pol):.1f}',
              f'policy re-plan: mean re-plans per run, both cells, all sigma '
              f'(min {min(pol)}, max {max(pol)})')
    macro('NoiseThresholdTriggers', f'{sum(r["n_replans"] for r in thr)}',
          'threshold arm: total re-plans triggered, both cells, all sigma')
    macro('NoiseThresholdRuns', f'{len(thr)}', 'threshold arm: runs')
    if late:
        macro('NoiseThresholdLateMax', f'{100 * max(late):.0f}',
              'largest checkpoint lateness as % of the 5% trigger limit, '
              'threshold arm, all runs')

    tex = render(table, tests, cpu, wall, nrep, thr)
    print(('\n% PARTIAL DATA: not for the paper' if not complete else '')
          + f'\n% >>> {TAG}\n' + tex + f'% <<< {TAG}')
    print('% ---- macros')
    for name, value, note in OUT:
        print(f'\\newcommand{{\\{name}}}{{{value}}}  % {note}')
    if WRITE:
        if not complete:
            sys.exit('refusing --write: the ledgers are incomplete')
        patch_macros()


OUT = []


def macro(name, value, note):
    OUT.append((name, value, note))


def patch_macros():
    s = open(MACROS).read()
    block = []
    for name, value, note in OUT:
        new = f'\\newcommand{{\\{name}}}{{{value}}}  % {note}'
        pat = re.compile(r'^\\newcommand\{\\' + name + r'\}\{[^\n]*$', re.M)
        if pat.search(s):
            s = pat.sub(lambda m: new, s, count=1)
        else:
            block.append(new)
    if block:
        s = s.rstrip('\n') + '\n\n' + '\n'.join([
            '% ---- Realized makespan under duration noise, with and without re-planning',
            '% (scripts/x2_execution_noise_table.py -> results/execution_noise/*.jsonl;',
            '% 20 instances x 3 noise replicates x sigma 0/0.1/0.2/0.3, CP-SAT at sigma',
            '% 0/0.2 on replicate 0; policy seed 301. Relative differences use the rival',
            '% as denominator, positive = policy re-plan shorter.)'] + block) + '\n'
    open(MACROS, 'w').write(s)
    print(f'patched {MACROS}: {len(OUT) - len(block)} rewritten, {len(block)} appended')


def render(table, tests, cpu, wall, nrep, thr):
    ns = len(SIGMAS)
    best = {}
    for cell, _, _ in CELLS:
        for sigma in SIGMAS:
            have = [a for a, _, _ in ARMS if (cell, a, sigma) in table]
            if have:
                best[(cell, sigma)] = min(
                    have, key=lambda a: table[(cell, a, sigma)][0])
    c1, c2 = 3, 3 + ns
    lines = [
        '\\begin{table*}[!t]',
        '\\centering',
        '\\caption{Realized makespan under duration noise, by noise level '
        '$\\sigma$. Bold: shortest makespan in each column.}',
        '\\label{tab:execnoise}',
        '\\begin{threeparttable}',
        '\\footnotesize',
        '\\setlength{\\tabcolsep}{2.5pt}',
        '\\begin{tabular}{@{}ll' + 'r' * 2 * ns + '@{}}',
        '\\toprule',
        ' & & ' + ' & '.join(f'\\multicolumn{{{ns}}}{{c}}{{Cell {lab}}}'
                             for _, _, lab in CELLS) + ' \\\\',
        f'\\cmidrule(lr){{{c1}-{c1 + ns - 1}}}\\cmidrule(l){{{c2}-{c2 + ns - 1}}}',
        'Initial plan & Online response & '
        + ' & '.join(f'$\\sigma={s:g}$' for _ in CELLS for s in SIGMAS)
        + ' \\\\',
        '\\midrule']
    for k, (arm, plan, resp) in enumerate(ARMS):
        cells = []
        for cell, _, _ in CELLS:
            for sigma in SIGMAS:
                if (cell, arm, sigma) not in table:
                    cells.append('\\multicolumn{1}{c}{--}')
                    continue
                mu, sd, _ = table[(cell, arm, sigma)]
                ms = f'{mu:.1f}$\\pm${sd:.1f}'
                if best[(cell, sigma)] == arm:
                    ms = f'\\textbf{{{ms}}}'
                cells.append(ms)
        first = k == 0 or ARMS[k - 1][1] != plan
        lines.append(f'{plan if first else ""} & {resp} & '
                     + ' & '.join(cells) + ' \\\\')
        if k in (2, 3):
            lines.append('\\addlinespace')
    lines += ['\\midrule',
              f'\\multicolumn{{{2 + 2 * ns}}}{{@{{}}l}}{{Policy re-plan against '
              'each rival: win/tie/loss and Holm-adjusted $p$} \\\\']
    for j, (rival, label) in enumerate(RIVALS):
        if j:
            lines.append('\\addlinespace')
        w_cells, p_cells = [], []
        for cell, _, _ in CELLS:
            for sigma in SIGMAS:
                t = tests.get((cell, rival, sigma))
                w_cells.append(t[0] if t else '\\multicolumn{1}{c}{--}')
                p_cells.append(fmt_p(t[3]) if t else '\\multicolumn{1}{c}{--}')
        lines.append(f'vs {label} & W/T/L & ' + ' & '.join(w_cells) + ' \\\\')
        lines.append(f' & $p$ & ' + ' & '.join(p_cells) + ' \\\\')
    lines += [
        '\\bottomrule',
        '\\end{tabular}',
        '\\begin{tablenotes}\\footnotesize',
        '\\item ' + note(cpu, wall, nrep, thr),
        '\\end{tablenotes}',
        '\\end{threeparttable}',
        '\\end{table*}']
    return '\n'.join(lines) + '\n'


def note(cpu, wall, nrep, thr):
    def pair(d, arm, f):
        return '/'.join(f.format(d[(c, arm)]) if (c, arm) in d else '--'
                        for c, _, _ in CELLS)
    if all(set(v) == {9} for v in nrep.values()):
        count = ('Every re-planning arm re-planned at all nine checkpoints of '
                 'every run. ')
    else:
        mean = {k: float(np.mean(v)) for k, v in nrep.items()}
        loads = [f'{pair(mean, a, "{:.1f}")} for the {l}' for a, l in
                 (('policy_replan', 'policy'), ('pdr_replan', 'rules'),
                  ('ga_replan', 'GA'), ('cpsat_replan', 'CP-SAT'))]
        count = ('The mean number of re-plans per run at 50/80 modules is '
                 + ', '.join(loads[:-1]) + ', and ' + loads[-1] + '. ')
    cpu_s = [f'{pair(cpu, a, "{:.2f}")}~s for the {l}' for a, l in
             (('policy_replan', 'policy'), ('pdr_replan', 'rules'),
              ('ga_replan', 'GA'))]
    cpu_s.append(f'{pair(cpu, "cpsat_replan", "{:.2f}")}~s for CP-SAT '
                 f'({pair(wall, "cpsat_replan", "{:.1f}")}~s wall)')
    trig = sum(r['n_replans'] for r in thr)
    fired = (f'it never re-planned in {len(thr)} runs' if trig == 0 else
             f'it re-planned {trig} times in {len(thr)} runs')
    return (
        'Cell: modules, $|V|$, $\\bar\\tau/\\bar p$ (machines: 25). Makespan in '
        'hours, mean $\\pm$ standard deviation over the 20 instances. An '
        "instance's value is the mean over three noise replicates. The "
        'policy is the size-mixture policy of training seed 301. Each '
        'processing time, machine-free lag and loaded travel time is '
        'multiplied by $1+\\varepsilon$, with $\\varepsilon$ uniform on '
        '$[-\\sigma,\\sigma]$. The draw is made once per instance and '
        'replicate and is shared by every arm. A re-planning arm re-plans '
        'each time a further 10\\% of the operations has completed, which '
        'gives nine checkpoints. The threshold arm re-plans at a checkpoint '
        'only when the lateness, the largest delay of a finished operation '
        'against the plan in force, exceeds 5\\% of the nominal makespan, '
        f'and {fired}. At '
        'each re-plan, an operation already committed to a machine keeps '
        'that machine. The GA re-plans with 10~s of CPU time, and CP-SAT '
        'with 10~s of wall time on three workers. Both keep the current plan '
        'when it is shorter. ' + count + 'The planning CPU time per re-plan '
        'at 50/80 modules is ' + ', '.join(cpu_s[:-1]) + ', and '
        + cpu_s[-1] + '. The CP-SAT arm ran at $\\sigma=0$ and $0.2$ only, '
        'on one replicate. All arms ran on the same cores, cores 2--7 of a '
        'Core Ultra 9 285K at 5.5~GHz. The lower panel counts the instances '
        'on which the policy re-plan is shorter (W), equal within '
        '$10^{-6}$~h (T) or longer (L). Its $p$ is the two-sided paired '
        'Wilcoxon test over the 20 instances, Holm-adjusted across the four '
        'noise levels of each cell and rival.')


if __name__ == '__main__':
    main()
