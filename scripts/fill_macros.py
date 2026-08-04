"""Regenerate the AUTO-FILLED block of paper macros from result files.

Single-source-of-truth pipeline (proposal §10.4): every manuscript number is
a macro; this script maps result files -> macro values and rewrites the
block between the AUTOGEN markers in macros.tex. Idempotent; NEVER fabricates:
a macro whose result file is missing stays \\prelim.

Headline policy arm: m1-bcb-guide (bound-guided; adoption gate PASS,
notes/gate_GUIDE.md + decisions 2026-07-29). All multi-seed arms are
aggregated per the pre-registered nuisance-factor treatment: per-instance
makespans averaged over training seeds, then instance-level statistics.

Run: python scripts/fill_macros.py   (from repo root; then recompile paper)
"""

import sys, os, json, glob, re, math
import numpy as np
from scipy.stats import wilcoxon

MACROS = 'paper/main_manuscript_bundle/macros.tex'
BEGIN = '% ==== AUTOGEN RESULTS BEGIN (scripts/fill_macros.py; do not edit) ===='
END = '% ==== AUTOGEN RESULTS END ===='

HEADLINE = 'm1-bcb-guide'
SINGLE = 'single-joint'
E0B = 'e0b-uncontended'

# (cell dir under test_results/PPVCT, macro suffix). First 12 = training grid;
# last 4 = zero-shot transfer (fleet v4, scale 15x25).
GRID = [(f'v{v}+t{t}', f'V{fw}T{rw}')
        for v, fw in ((1, 'One'), (2, 'Two'), (3, 'Three'))
        for t, rw in ((0.1, 'One'), (0.3, 'Three'), (0.6, 'Six'), (1.0, 'Ten'))]
TRANSFER = [('v4+t0.6', 'VFourTSix'), ('v4+t1.0', 'VFourTTen'),
            ('15x25+ppvct-mixed+v2+t0.6', 'JFifteenTSix'),
            ('15x25+ppvct-mixed+v2+t1.0', 'JFifteenTTen')]
GATE_CELLS = [('v1+t0.3', 'VOneTThree'), ('v2+t0.3', 'VTwoTThree'),
              ('v1+t0.6', 'VOneTSix'), ('v2+t0.6', 'VTwoTSix')]
LADDER = [('joint-v1', 'Joint'), ('m1-bcb', 'Credit'), ('m1-bcb-guide', 'Guide'),
          ('m2-shaped', 'MTwo'), ('coma-critic', 'Coma'), ('single-joint', 'Single')]


def pdr_file(cell):
    return cell if cell.startswith('15x25') else f'10x25+ppvct-mixed+{cell}'


def seed_arrays(cell, prefix):
    """Per-seed per-instance makespans for a policy arm on a cell."""
    pat = f'test_results/PPVCT/{cell}/Result_greedy+10x25+ppvct-mixed+{prefix}-s*_{cell}.npy'
    out = {}
    for p in sorted(glob.glob(pat)):
        m = re.search(r'-s(\d+)_', os.path.basename(p))
        if m:
            out[int(m.group(1))] = np.load(p)[:, 0]
    return out


def seed_mean(cell, prefix):
    """Per-instance makespans averaged over available seeds (None if absent)."""
    arrs = seed_arrays(cell, prefix)
    if not arrs:
        return None, 0
    return np.mean(list(arrs.values()), axis=0), len(arrs)


def pdr_best(cell):
    p = f'results/pdr/{pdr_file(cell)}.json'
    if not os.path.exists(p):
        return None
    data = json.load(open(p))
    pairs = sorted(next(iter(data.values())).keys())
    arr = {q: np.array([d[q] for d in data.values()]) for q in pairs}
    best = min(arr, key=lambda q: arr[q].mean())
    return arr[best], best, len(data)


def cpsat_path(cell):
    """Ledger of the strengthened (v2) CP-SAT references for a cell.

    Source switch, decisions 2026-08-03: the manuscript's CP-SAT reference is
    the strengthened model (redundant fleet-capacity cumulative, tightened
    horizon, symmetry breaking; 300 s, 4 search workers), written by
    scripts/p2_cpsat_refs_v2.py. The unstrengthened v1 references in
    or_solution/PPVCT/ are kept on disk but are no longer a macro source.
    """
    return f'results/cpsat_v2/{pdr_file(cell)}.jsonl'


def cpsat_rows(cell):
    p = cpsat_path(cell)
    if not os.path.exists(p):
        return None
    seen = {}
    for l in open(p):
        if not l.strip():
            continue
        r = json.loads(l)
        if r['ub'] is None or r['lb'] is None:
            continue          # no incumbent within the budget: not a reference
        seen[r['instance']] = dict(instance=r['instance'], ms=r['ub'],
                                   lb=r['lb'], status=r['status'])
    if len(seen) < 100:
        return None      # incomplete cell: do not report partial means
    return [seen[k] for k in sorted(seen)]


def ga_result(cell):
    p = f'results/ga_v2/{pdr_file(cell)}.json'
    if not os.path.exists(p):
        return None
    data = json.load(open(p))
    if len(data) < 100:
        return None
    return np.array([d['ga'] for d in data.values()])


def e0b_eval(cell):
    p = (f'test_results/PPVCT/{cell}/Result_greedy-NVF+10x25+ppvct-mixed+'
         f'{E0B}-s301_{cell}.npy')
    return np.load(p)[:, 0] if os.path.exists(p) else None


def trained_seeds(model_stem=HEADLINE):
    """Seeds ACTUALLY trained for the headline arm, read off the checkpoints."""
    seeds = set()
    for p in glob.glob(f'trained_network/PPVCT/*{model_stem}-s*.pth'):
        m = re.search(r'-s(\d+)(?:-last)?\.pth$', os.path.basename(p))
        if m:
            seeds.add(int(m.group(1)))
    return sorted(seeds)


def gate_rows(path):
    """Parse the JSON verdict block scripts/gate_eval.py appends to notes/*.md."""
    if not os.path.exists(path):
        return {}
    txt = open(path).read()
    if '```json' not in txt:
        return {}
    blob = txt.split('```json', 1)[1].split('```', 1)[0]
    try:
        rows = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    if isinstance(rows, dict):
        return rows
    return {r['cell']: r for r in rows}


def texpval(p):
    """A p-value as a LaTeX scientific-notation upper bound (never rounded down)."""
    if p <= 0:
        return '10^{-300}'
    e = math.floor(math.log10(p))
    m = math.ceil(p / 10 ** e * 10) / 10        # round the mantissa UP: honest bound
    if m >= 10:
        m, e = 1.0, e + 1
    return f'{m:g}\\times10^{{{e}}}'


def ms(arr):
    return f'{arr.mean():.1f}$\\pm${arr.std(ddof=1):.1f}'


def new(lines, name, val, comment=''):
    c = f'  % {comment}' if comment else ''
    lines.append(f'\\newcommand{{\\{name}}}{{{val}}}{c}')


def _seed_macros():
    s = trained_seeds()
    if not s:
        return ('\\renewcommand{\\Nseeds}{\\prelim}'
                '\\newcommand{\\seedset}{\\prelim}')
    word = {1: 'one', 2: 'two', 3: 'three'}.get(len(s), str(len(s)))
    return (f'\\renewcommand{{\\Nseeds}}{{{len(s)}}}  % DERIVED from '
            f'trained_network/PPVCT: {word} {HEADLINE} seed(s) actually trained\n'
            f'\\newcommand{{\\seedset}}{{$\\{{{",".join(str(x) for x in s)}\\}}$}}')


# ---------------------------------------------------------------- main table
def cell_macros(lines, cell, suff, with_ga=True):
    pb = pdr_best(cell)
    if pb:
        arr, best, n = pb
        new(lines, f'qPdr{suff}', ms(arr), f'{best}, n={n}, results/pdr/{pdr_file(cell)}.json')
    else:
        new(lines, f'qPdr{suff}', '\\prelim')
    cp = cpsat_rows(cell)
    if cp:
        ub = np.array([r['ms'] for r in cp])
        new(lines, f'qCp{suff}', ms(ub),
            f'strengthened warm-started 300s UB, n={len(cp)}, {cpsat_path(cell)}')
        lb = np.array([r['lb'] for r in cp])
        new(lines, f'CpGap{suff}', f'{(100*(ub-lb)/lb).mean():.1f}\\%',
            'mean (UB-LB)/LB of the strengthened 300s run')
    else:
        new(lines, f'qCp{suff}', '\\prelim')
        new(lines, f'CpGap{suff}', '\\prelim')
    if with_ga:
        ga = ga_result(cell)
        if ga is not None:
            new(lines, f'qGa{suff}', ms(ga),
                f'PDR-seeded GA v2 60 CPU-s, n={len(ga)}, results/ga_v2/{pdr_file(cell)}.json')
        else:
            new(lines, f'qGa{suff}', '\\prelim')
    ours, k = seed_mean(cell, HEADLINE)
    if ours is not None and cp:
        ub = np.array([r['ms'] for r in cp])
        new(lines, f'qOurs{suff}', ms(ours),
            f'{HEADLINE} greedy, {k}-seed per-instance mean, test_results/PPVCT/{cell}')
        gap = 100 * (ours.mean() - ub.mean()) / ub.mean()
        new(lines, f'qGap{suff}', f'{gap:+.1f}\\%', 'vs warm CP-SAT UB')
        ga = ga_result(cell) if with_ga else None
        base, bname = (pb[0], f'best-PDR ({pb[1]})') if pb else (None, '')
        if base is not None and len(base) == len(ours):
            w = int((ours < base - 1e-6).sum())
            tie = int((np.abs(ours - base) <= 1e-6).sum())
            new(lines, f'qWtl{suff}', f'{w}/{tie}/{len(ours)-w-tie}',
                f'vs best PDR pair ({bname}), paired, seed-mean')
        else:
            new(lines, f'qWtl{suff}', '\\prelim')
        if ga is not None and len(ga) == len(ours):
            w = int((ours < ga - 1e-6).sum())
            tie = int((np.abs(ours - ga) <= 1e-6).sum())
            new(lines, f'qWtlGa{suff}', f'{w}/{tie}/{len(ours)-w-tie}',
                'vs GA v2 60 CPU-s, paired, seed-mean')
    else:
        for fam in ('qOurs', 'qGap', 'qWtl'):
            new(lines, f'{fam}{suff}', '\\prelim')
    sa, ks = seed_mean(cell, SINGLE)
    if sa is not None:
        new(lines, f'qSa{suff}', ms(sa), f'{SINGLE} greedy, {ks}-seed per-instance mean')
    else:
        new(lines, f'qSa{suff}', '\\prelim')


WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
         'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
         'fifteen', 'sixteen']


def cpsat_summary_macros(lines):
    """Grid-wide closure of the strengthened CP-SAT references."""
    lines.append('% Strengthened CP-SAT references, grid-wide closure '
                 '(results/cpsat_v2/*.jsonl)')
    nopt = tot = ncells = nzero = 0
    complete = True
    for cell, _ in GRID + TRANSFER:
        cp = cpsat_rows(cell)
        if cp is None:
            complete = False
            continue
        k = sum(1 for r in cp if r['status'] == 'OPTIMAL')
        nopt += k
        tot += len(cp)
        ncells += 1
        nzero += (k == 0)
    if not complete:
        for nme in ('CpNopt', 'CpNref', 'CpNcellsNoOpt'):
            new(lines, nme, '\\prelim')
        return
    new(lines, 'CpNopt', nopt, f'instances proved optimal, {ncells} cells')
    new(lines, 'CpNref', tot, 'instances with a strengthened CP-SAT reference')
    new(lines, 'CpNcellsNoOpt', WORDS[nzero],
        'cells in which no instance was closed')


# ------------------------------------------------------------- G1 (as before)
def g1_macros(lines):
    # Source switch 2026-08-05: the retry (\Gb*) family now reads the 3-seed
    # POOLED verdict, notes/gate_G1retry_s3.md (per-instance makespans averaged
    # over seeds 301/302/303 before the test, Option A of gpu_arm_designs 1.7).
    # The single-seed record notes/gate_G1retry.md is kept unmodified on disk
    # but is no longer a macro source. Attempt one (\GaOne*) is single-seed and
    # unchanged.
    lines.append('% Premise gate G1 (notes/gate_G1.md, '
                 'notes/gate_G1retry_s3.md = 3-seed pooled retry)')
    for tag, path, cells in (
            ('GaOne', 'notes/gate_G1.md', ('v1+t0.6', 'v2+t0.6')),
            ('Gb', 'notes/gate_G1retry_s3.md', ('v1+t1.0', 'v2+t1.0'))):
        rows = gate_rows(path)
        pmax = 0.0
        for cell, word in zip(cells, ('VOne', 'VTwo')):
            r = rows.get(cell)
            if not r:
                for fam in ('Imp', 'Wtl'):
                    new(lines, f'{tag}{fam}{word}', '\\prelim')
                continue
            new(lines, f'{tag}Imp{word}', f'{r["mean_improvement_pct"]:+.2f}',
                f'{cell}, n={r["n"]}')
            new(lines, f'{tag}Wtl{word}', r['wtl'], cell)
            pmax = max(pmax, r['p_holm'])
        new(lines, f'{tag}PvalMax',
            texpval(pmax) if rows else '\\prelim',
            'largest Holm-corrected p in the gate family' if rows else '')
    r = gate_rows('notes/gate_G1.md').get('v2+t0.6')
    new(lines, 'GaOnePvVTwo', f'{r["p_holm"]:.2f}' if r else '\\prelim',
        'n.s.' if r else '')


# ------------------------------------------- E2 ladder + guide/G2 gate stats
def holm(ps):
    order = np.argsort(ps)
    out = [0.0] * len(ps)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[idx]))
        out[idx] = running
    return out


def ladder_macros(lines):
    lines.append('% E2 ladder (per gate cell, per arm: seed-mean makespan; '
                 'seeds actually evaluated in the comment)')
    for cell, csuff in GATE_CELLS:
        for prefix, asuff in LADDER:
            arr, k = seed_mean(cell, prefix)
            if arr is None:
                new(lines, f'Lad{asuff}{csuff}', '\\prelim')
            else:
                seeds = sorted(seed_arrays(cell, prefix))
                new(lines, f'Lad{asuff}{csuff}', ms(arr),
                    f'{prefix}, seeds {seeds}, n={len(arr)}')
    # M2 (credit in the return) vs M1 (credit as baseline): the lemma's teeth;
    # COMA (learned counterfactual critic) vs joint (plain shared reward)
    for cell, csuff in GATE_CELLS:
        m1, _ = seed_mean(cell, 'm1-bcb')
        m2, _ = seed_mean(cell, 'm2-shaped')
        jt, _ = seed_mean(cell, 'joint-v1')
        cm, _ = seed_mean(cell, 'coma-critic')
        if m1 is not None and m2 is not None:
            new(lines, f'MTwoD{csuff}',
                f'{100*(m2.mean()-m1.mean())/m1.mean():+.0f}\\%',
                f'{cell}, M2 vs M1 seed-mean')
        else:
            new(lines, f'MTwoD{csuff}', '\\prelim')
        if jt is not None and cm is not None:
            new(lines, f'ComaD{csuff}',
                f'{100*(cm.mean()-jt.mean())/jt.mean():+.1f}\\%',
                f'{cell}, COMA-critic vs shared-reward seed-mean')
        else:
            new(lines, f'ComaD{csuff}', '\\prelim')
    # guide adoption, 3-seed pooled (guide < m1, one-sided Wilcoxon + Holm)
    lines.append('% Guide adoption, pooled seeds (guide < m1-bcb): '
                 'per decisions 2026-07-29')
    stats = []
    for cell, csuff in GATE_CELLS:
        g, kg = seed_mean(cell, HEADLINE)
        m, km = seed_mean(cell, 'm1-bcb')
        if g is None or m is None or kg < 3 or km < 3:
            stats.append(None)
            continue
        p = float(wilcoxon(g, m, alternative='less').pvalue)
        imp = 100 * (m.mean() - g.mean()) / m.mean()
        w = int((g < m - 1e-6).sum())
        t = int((np.abs(g - m) <= 1e-6).sum())
        stats.append((cell, csuff, imp, f'{w}/{t}/{len(g)-w-t}', p))
    ps = holm([s[4] for s in stats if s]) if all(stats) else []
    j = 0
    pmax = 0.0
    for s in stats:
        if s is None:
            continue
        cell, csuff, imp, wtl, _ = s
        new(lines, f'GuImp{csuff}', f'{imp:+.2f}', f'{cell}, guide vs m1, 3-seed pooled')
        new(lines, f'GuWtl{csuff}', wtl, cell)
        pmax = max(pmax, ps[j]); j += 1
    new(lines, 'GuPvalMax', texpval(pmax) if all(stats) and stats else '\\prelim',
        'largest Holm-corrected p, guide vs m1 family')
    # G2 final (from the pre-registered verdict file; FAIL -> credit is neutral)
    g2 = gate_rows('notes/gate_G2_final.md')
    lines.append('% G2 final, 3 seeds (notes/gate_G2_final.md): credit-only vs '
                 'shared reward, all n.s.')
    for cell, csuff in GATE_CELLS:
        r = g2.get(cell)
        if r:
            new(lines, f'GtwoImp{csuff}', f'{r["improv_pct"]:+.2f}', f'{cell}, n.s.')
        else:
            new(lines, f'GtwoImp{csuff}', '\\prelim')
    if g2:
        pmin = min(r['p_holm'] for r in g2.values())
        new(lines, 'GtwoPvalMin', f'{pmin:.2f}',
            'smallest Holm-corrected p in the G2 family (all n.s.)')
    else:
        new(lines, 'GtwoPvalMin', '\\prelim')


# ---------------------------------------------------------------- E4 / TOST
def tost_macros(lines):
    rows = gate_rows('notes/e4_tost_final.md')
    lines.append('% E4/G3 final TOST (notes/e4_tost_final.md), guide vs single, '
                 '3-seed per-instance means')
    if not rows:
        for nme in ('TostCells', 'TostEq', 'TostBetter', 'TostBetterDiff',
                    'TostBetterCi'):
            new(lines, nme, '\\prelim')
        return
    verd = {c: r['verdict'] for c, r in rows.items()}
    n_eq = sum(1 for v in verd.values() if v == 'EQUIVALENT')
    n_b = sum(1 for v in verd.values() if v == 'OURS BETTER')
    new(lines, 'TostCells', len(rows))
    new(lines, 'TostEq', n_eq)
    new(lines, 'TostBetter', n_b)
    better = [c for c, v in verd.items() if v == 'OURS BETTER']
    if better:
        r = rows[better[0]]
        new(lines, 'TostBetterDiff', f'{r["mean_rel_diff_pct"]:+.2f}\\%', better[0])
        new(lines, 'TostBetterCi',
            f'[{r["ci90_lo_pct"]:+.2f}\\%, {r["ci90_hi_pct"]:+.2f}\\%]',
            '90\\% bootstrap CI')
    else:
        new(lines, 'TostBetterDiff', '\\prelim')
        new(lines, 'TostBetterCi', '\\prelim')
    # largest |rel diff| across EQUIVALENT cells: the parity envelope
    eqd = [abs(r['mean_rel_diff_pct']) for c, r in rows.items()
           if verd[c] == 'EQUIVALENT']
    if eqd:
        new(lines, 'TostEqMaxAbs', f'{max(eqd):.2f}\\%',
            'largest |rel diff| among EQUIVALENT cells')


# ----------------------------------------------------------------- latency
def latency_macros(lines):
    """Certified per-event latencies from the exclusive-slot rerun logs."""
    lines.append('% Certified latency (exclusive queue slot; '
                 'train_log/latency_uncontended2.log, latency_cpu4_2.log)')

    def parse(path):
        if not os.path.exists(path):
            return {}
        out, model = {}, None
        for ln in open(path):
            mm = re.search(r'diagnostics -> results/diagnostics/(\S+?)_(v\d\+t[\d.]+)_', ln)
            lat = re.search(r'^(\S+) greedy: .*lat/event=([\d.]+)ms', ln)
            if lat:
                pend = (lat.group(1), float(lat.group(2)))
            elif mm:
                out.setdefault(mm.group(1), {})[mm.group(2)] = pend[1]
        return out

    gpu = parse('train_log/latency_uncontended2.log')
    cpu = parse('train_log/latency_cpu4_2.log')
    g = gpu.get(f'10x25+ppvct-mixed+{HEADLINE}-s301', {})
    s = gpu.get(f'10x25+ppvct-mixed+{SINGLE}-s301', {})
    c = cpu.get(f'10x25+ppvct-mixed+{HEADLINE}-s301', {})

    def rng(d):
        v = sorted(d.values())
        if not v:
            return None
        return f'{v[0]:.1f}' if len(v) == 1 or v[0] == v[-1] else f'{v[0]:.1f}--{v[-1]:.1f}'

    new(lines, 'LatMarlDec', rng(g) or '\\prelim',
        f'{HEADLINE} GPU, cells {sorted(g)}' if g else '')
    new(lines, 'LatSingle', rng(s) or '\\prelim',
        f'{SINGLE} GPU, cells {sorted(s)}' if s else '')
    new(lines, 'LatCpuFour', rng(c) or '\\prelim',
        'headline arm, 4 pinned CPU cores, OMP=4' if c else '')


# ---------------------------------------------------------------- L1 external
def link_seed_scores(stem):
    """Per-seed released-simulator score records for one L1 arm.

    `stem` is the model name without its -s### suffix; every seed directory
    present on disk is picked up, so adding a seed repoints the macros with no
    code edit. The ablation stem link-m1 does not match the guide stem's
    directories, because the seed suffix must follow the stem immediately.
    """
    out = {}
    pat = f'../external/replays_l1/{stem}-s*/score_vs_released.json'
    for p in sorted(glob.glob(pat)):
        m = re.search(r'-s(\d+)/score_vs_released\.json$', p)
        if m:
            out[int(m.group(1))] = json.load(open(p))
    return out


def link_macros(lines):
    # Source switch 2026-08-05 (notes/harvest_2026-08-05.md sec 5): the
    # headline external arm now pools training seeds 301/302/303 instead of
    # reporting seed 301 alone. Per-seed statistics are averaged over seeds;
    # because the released anchor rows are identical across seeds, the seed
    # mean of each relative difference equals the relative difference of the
    # per-instance seed means. The Mann-Whitney p is reported as the WORST
    # seed's, which is the conservative reading of "every seed agrees".
    lines.append('% L1 external (Link et al. benchmark, released simulator; '
                 'external/replays_l1/*/score_vs_released.json); per-seed '
                 'statistics averaged over training seeds, MWU p = worst seed')
    VW = {3: 'Three', 6: 'Six', 9: 'Nine', 12: 'Twelve', 15: 'Fifteen', 18: 'Eighteen'}
    ANCH = (('joint', 'Joint'), ('best-modular', 'Mod'), ('best-heuristic', 'Heur'))
    # headline external arm is link-m1-guide (3 seeds); ablation arm link-m1
    for tag, stem in (('Lk', '15x10+link+link-m1-guide'),
                      ('LkAbl', '15x10+link+link-m1')):
        S = link_seed_scores(stem)
        if not S:
            new(lines, f'{tag}Missing', '\\prelim',
                f'../external/replays_l1/{stem}-s*/score_vs_released.json missing')
            continue
        seeds = sorted(S)
        cells = sorted(S[seeds[0]], key=lambda k: int(k[1:]))
        for vk in cells:
            V = int(vk[1:])
            if tag == 'LkAbl' and V != 3:
                continue           # ablation row: scarce-fleet cell only
            w = VW[V]
            recs = [S[s][vk] for s in seeds]
            n = recs[0]['n']
            # Mean over seeds of the per-seed instance mean; the +- keeps its
            # instance-to-instance meaning (n per seed), averaged over seeds.
            om = float(np.mean([r['ours_mean'] for r in recs]))
            osd = float(np.mean([r['ours_std'] for r in recs]))
            new(lines, f'{tag}OursV{w}', f'{om:.0f}$\\pm${osd:.0f}',
                f'{stem}, seeds {seeds}, their simulator units, n={n}/seed')
            for akey, asuff in ANCH:
                a = [r['anchors'][akey] for r in recs]
                assert len({x['name'] for x in a}) == 1, (stem, vk, akey)
                d = float(np.mean([x['rel_diff_pct'] for x in a]))
                pw = max(x['mwu_p'] for x in a)
                new(lines, f'{tag}{asuff}DV{w}', f'{d:+.1f}\\%',
                    f'vs {a[0]["name"]}, worst-seed MWU p={pw:.2g}')
        if tag == 'Lk':
            v3 = [S[s]['v3'] for s in seeds if 'v3' in S[s]]
            if v3:
                pmax = max(x['mwu_p'] for r in v3 for x in r['anchors'].values())
                new(lines, 'LkVThreePvalMax', texpval(pmax),
                    'largest MWU p among the three V=3 anchors, over all seeds')
            # anticipatory-semantics wedge, per V: the released simulator's
            # anticipatory vehicle rule re-times our decision sequences;
            # wedge = (their-timing mean - our-env-timing mean)/our-env mean,
            # computed per seed and then averaged over seeds
            for vk in cells:
                V = int(vk[1:])
                ws, n = [], 0
                for s in seeds:
                    rec = S[s][vk]
                    env_ms = [json.load(open(fp))['ours_makespan_link_units']
                              for fp in sorted(glob.glob(
                                  f'../external/replays_l1/{stem}-s{s}/'
                                  f'v{V}/replay_*.json'))]
                    if len(env_ms) != rec['n']:
                        ws = []
                        break
                    e = float(np.mean(env_ms))
                    ws.append(100 * (rec['ours_mean'] - e) / e)
                    n = rec['n']
                if ws:
                    new(lines, f'LkWedgeV{VW[V]}', f'{np.mean(ws):+.1f}\\%',
                        'their timing vs our timing of the same schedules, '
                        f'seeds {seeds}, n={n}/seed')


# -------------------------------------------------------------- certificates
def certificate_macros(lines):
    lines.append('% Shipped certificate (C - B0)/B0 with C = headline seed-mean '
                 'schedule, B0 = analytic root bound (results/certificate/)')
    have_cp = False
    for cell, suff in GRID + TRANSFER:
        stem = cell if cell.startswith('15x25') else f'10x25+ppvct-mixed+{cell}'
        p = f'results/certificate/{stem}.json'
        cp = cpsat_rows(cell)
        ours, k = seed_mean(cell, HEADLINE)
        if not os.path.exists(p) or ours is None:
            new(lines, f'Cert{suff}', '\\prelim')
            continue
        b0 = json.load(open(p))
        b0v = np.array([b0[f'instance_{i:03d}'] for i in range(len(ours))])
        cert = 100 * (ours - b0v) / b0v
        new(lines, f'Cert{suff}', f'{cert.mean():.1f}\\%',
            f'mean per-schedule certificate, {k}-seed C, n={len(ours)}')
        have_cp = have_cp or bool(cp)
    # \CertBeatLb* retired 2026-08-03 (decisions): under the strengthened
    # reference the analytic root bound never exceeds the solver's proven
    # lower bound, so the claim it carried was withdrawn.
    if have_cp:
        # admissibility check: B0 <= CP-SAT UB everywhere a reference exists
        viol = tot = 0
        for cell, suff in GRID + TRANSFER:
            stem = cell if cell.startswith('15x25') else f'10x25+ppvct-mixed+{cell}'
            p = f'results/certificate/{stem}.json'
            cp = cpsat_rows(cell)
            if not os.path.exists(p) or not cp:
                continue
            b0 = json.load(open(p))
            ub = {r['instance']: r['ms'] for r in cp}
            for inst, v in b0.items():
                if inst in ub:
                    tot += 1
                    viol += v > ub[inst] + 1e-6
        new(lines, 'CertNadm', tot, 'instances with a CP-SAT UB to check against')
        new(lines, 'CertViol', viol, 'admissibility violations (B0 > UB)')


# ------------------------------------------------------------- E3 regret map
def regret_macros(lines):
    lines.append('% E3 coupling-regret map: penalty-trained E0b re-evaluated in '
                 'the explicit environment vs headline arm, per cell')
    vals = {}
    for cell, suff in GRID:
        e0 = e0b_eval(cell)
        ours, k = seed_mean(cell, HEADLINE)
        if e0 is None or ours is None:
            new(lines, f'Rg{suff}', '\\prelim')
            continue
        r = 100 * (e0.mean() - ours.mean()) / ours.mean()
        vals[suff] = r
        new(lines, f'Rg{suff}', f'{r:+.1f}\\%', f'{cell}, n={len(e0)}')
    if vals:
        new(lines, 'RgMax', f'{max(vals.values()):.1f}',
            f'max over grid ({max(vals, key=vals.get)})')
        new(lines, 'RgMin', f'{min(vals.values()):.1f}',
            f'min over grid ({min(vals, key=vals.get)})')
        new(lines, 'RgMinAbs', f'{abs(min(vals.values())):.1f}',
            'absolute value of RgMin')
    else:
        new(lines, 'RgMax', '\\prelim')
        new(lines, 'RgMin', '\\prelim')


def main():
    lines = [BEGIN,
             '% Protocol constants (Appendix B)',
             '\\providecommand{\\prelim}{\\textsuperscript{\\dag}}',
             '\\renewcommand{\\Ninst}{100}',
             _seed_macros(),
             '\\newcommand{\\gridfleetset}{\\{1,2,3\\}}',
             '\\newcommand{\\gridtaupset}{\\{0.1,0.3,0.6,1.0\\}}',
             '\\newcommand{\\ncells}{twelve}',
             '\\newcommand{\\primarysize}{$10\\times25$}',
             '\\newcommand{\\siglevel}{0.05}',
             '\\newcommand{\\sizemixset}{\\{10,15\\}}',
             '\\newcommand{\\cpbudget}{300}',
             '\\newcommand{\\gawall}{60}',
             '\\newcommand{\\gapopsize}{100}',
             '% Per-cell results (real values only; \\prelim = not yet landed)']
    for cell, suff in GRID:
        cell_macros(lines, cell, suff, with_ga=True)
    lines.append('% Zero-shot transfer cells (no GA; policies trained on the '
                 '10x25 v{1,2,3} grid only)')
    for cell, suff in TRANSFER:
        cell_macros(lines, cell, suff, with_ga=False)
    cpsat_summary_macros(lines)
    g1_macros(lines)
    ladder_macros(lines)
    tost_macros(lines)
    latency_macros(lines)
    link_macros(lines)
    certificate_macros(lines)
    regret_macros(lines)
    lines.append(END)

    txt = open(MACROS).read()
    if BEGIN in txt:
        pre = txt[:txt.index(BEGIN)]
        post = txt[txt.index(END) + len(END):]
        txt = pre + '\n'.join(lines) + post
    else:
        txt = txt.rstrip() + '\n\n' + '\n'.join(lines) + '\n'
    open(MACROS, 'w').write(txt)
    filled = sum(1 for l in lines if '\\prelim' not in l and l.startswith('\\newcommand'))
    print(f'wrote {len(lines)} lines; real-valued macros: {filled}')


if __name__ == '__main__':
    sys.exit(main())
