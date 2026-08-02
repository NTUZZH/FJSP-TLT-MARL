"""
PPVC (Prefabricated Prefinished Volumetric Construction) instance generator.

Generates FJSP instances modeling a PPVC module factory. Operation sets,
precedence and module metadata are grounded in the BCA "Design for
Manufacturing and Assembly (DfMA): PPVC" Guidebook (Singapore Building and
Construction Authority); durations are literature/industry estimates (the
guidebook prescribes sequences, not durations).

PARAMETER PROVENANCE
  [G]  extracted from the BCA PPVC Guidebook (section cited per line)
  [E]  estimated (durations in hours; guidebook gives no durations)
  [N]  industry-norm range (lags: curing / ponding / paint drying)

Key facts taken from the guidebook:
  - Module weights: RC 20-35 t, Steel 15-20 t                        [G 2.6]
  - RC production: mould setup -> rebar cage -> MEP cast-in ->
    pre-pour inspection -> concreting -> curing -> demould +
    post-pour inspection -> trial assembly + hydrostatic tests       [G 4.1]
  - Steel production: material validation -> cutting/engraving ->
    2D frame welding -> 3D shell welding -> weld QC -> trial stack
    (factory joints are WELDED; bolting is the on-site method)       [G 4.2, 3.2.3]
  - Steel members carry fire/corrosion protection coating            [G 8.2]
  - Finishing: lightweight panels/drywall, waterproofing + water
    ponding test (wet areas), screed/tiling, plaster/skim, floor
    finishes, two factory paint coats (final coat is allowed
    on-site and is out of factory scope)                             [G 4.1.3, 4.2.2, 10.2]
  - QC gates: pre-pour, weld, MEP tests, finishing, final            [G 8]
  - A typical 2-bedroom unit = LRDIN(dry) + KIT(wet) + B2-IBB(wet)
    + MB-IBB(wet)  ->  wet:dry ~ 3:1                                 [G 2.4]
  - Real projects use a SINGLE shell system each (concrete or steel
    PPVC, never mixed within a project); 252-4,436 modules/project   [G Appendix]

Modeling assumptions (stated for the paper):
  - Each operation is one station visit; trade micro-ordering inside the
    finishing phase is aggregated into the 4-phase structure (the guidebook
    itself notes per-fabricator variation).
  - All four routing classes have EXACTLY 22 operations, so a fixed module
    count gives a fixed total op count (keeps FJSPEnvForSameOpNums usable
    with mixed-class instances, no padding).
  - Time unit: integer HOURS; all durations >= 1 (op_pt == 0 is the
    "incompatible" sentinel in DANIEL).
  - PT model: per-op nominal duration from the range, then +/-20% across
    compatible stations (SD1-style correlated model).
  - time_lag[i] = mandatory wait AFTER op i completes before its job
    successor becomes eligible. The machine is free during the lag.

Output contract (mirrors SD2_instance_generator + PPVC metadata):
    job_length     [J]   int    ops per module (always 22)
    op_pt          [N,M] int    processing time, 0 = incompatible
    meta dict:
        op_type         [N] int  0..4 {structural, MEP, finishing, assembly, QC}
        op_station      [N] int  0..8 index into STATION_TYPES (A..Q)
        time_lag        [N] int  hours of post-completion lag (0 = none)
        mch_type        [M] int  0..8 station type of each machine
        routing_class   [J] int  0..3 {RC-wet, RC-dry, Steel-wet, Steel-dry}
        module_weight_t [J] float module weight in tonnes  [G 2.6]
        op_name         [N] str  human-readable op names
        station_counts  dict     factory configuration

Self-test:  python ppvc_instance_generator.py
"""
import json
import numpy as np

# ---------------------------------------------------------------------------
# Station & type taxonomies
# ---------------------------------------------------------------------------

STATION_TYPES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Q']
STATION_NAMES = {
    'A': 'formwork', 'B': 'pour', 'C': 'MEP', 'D': 'finishing',
    'E': 'paint', 'F': 'assembly', 'G': 'dispatch', 'H': 'steel_fab',
    'Q': 'QC',
}

OP_TYPE_NAMES = ['structural', 'MEP', 'finishing', 'assembly', 'QC']
ROUTING_CLASSES = ['RC-wet', 'RC-dry', 'Steel-wet', 'Steel-dry']

# Module weight ranges in tonnes  [G 2.6]
MODULE_WEIGHT_T = {'RC': (20.0, 35.0), 'Steel': (15.0, 20.0)}

# Default station counts per type (blueprint 3.7 mid-range; factory layouts
# are fabricator-specific [G 2.1] so counts stay a generator parameter).
DEFAULT_FACTORY = {'A': 3, 'B': 2, 'C': 4, 'D': 4, 'E': 3, 'F': 3,
                   'G': 1, 'H': 2, 'Q': 3}
SMALL_FACTORY = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1,
                 'G': 1, 'H': 1, 'Q': 1}
# Capacity-tight variant (DEFAULT halved, min 1): machine contention
# interacts with lags, amplifying differences between scheduling methods
# (the default factory is generous enough that optima hug the lag-extended
# critical path and method gaps compress).
TIGHT_FACTORY = {'A': 2, 'B': 1, 'C': 2, 'D': 2, 'E': 2, 'F': 2,
                 'G': 1, 'H': 1, 'Q': 2}

# ---------------------------------------------------------------------------
# Routes — one step: (name, station_type, op_type, (dur_lo, dur_hi), (lag_lo, lag_hi))
# Durations in hours [E]; lags in hours [N]; sequence/source cited per line [G].
# ---------------------------------------------------------------------------

_RC_SHELL = [  # [G 4.1.1-4.1.2, 8.1.1]
    ('mould_setup',        'A', 'structural', (4, 6),  (0, 0)),   # G 4.1.1 step3 + 8.1.1 mould QC
    ('rebar_cage',         'A', 'structural', (4, 6),  (0, 0)),   # G 4.1.2 step1
    ('mep_cast_in',        'A', 'MEP',        (3, 5),  (0, 0)),   # G 4.1.2 step2 (cast-in MEP items)
    ('pre_pour_QC',        'Q', 'QC',         (1, 2),  (0, 0)),   # G 4.1.2 step3 / 8.1.1
    ('concrete_pour',      'B', 'structural', (2, 3),  (24, 48)), # G 4.1.2 step4; curing lag [N]
    ('demould_inspect',    'A', 'structural', (2, 3),  (0, 0)),   # G 4.1.2 steps5-6 (post-pour inspection)
    ('trial_assembly',     'F', 'QC',         (2, 3),  (0, 0)),   # G 4.1.2 end-note (trial assembly + hydro tests)
]
_STEEL_SHELL = [  # [G 4.2.2, 8.1.2, 8.2; factory joints WELDED, bolting = site method G 3.2.3]
    ('material_prep',      'H', 'structural', (2, 3),  (0, 0)),   # G 4.2.2 steps1,3 (validation+engraving)
    ('steel_cutting',      'H', 'structural', (3, 4),  (0, 0)),   # G 4.2.2 step2
    ('frame_welding_2d',   'H', 'structural', (6, 8),  (0, 0)),   # G 4.2.2 step4
    ('shell_welding_3d',   'H', 'structural', (4, 6),  (0, 0)),   # G 4.2.2 step5
    ('weld_QC',            'Q', 'QC',         (2, 2),  (0, 0)),   # G 4.2.2 step6 / 8.1.2
    ('fire_corrosion_protect', 'E', 'finishing', (3, 4), (0, 0)), # G 8.2 (passive fire+corrosion layer)
    ('trial_stack',        'F', 'QC',         (2, 3),  (0, 0)),   # G 4.2.2 step7
]
_MEP = [  # shared in-module MEP fit-out  [G 4.1.3 steps3-4 / 4.2.2 step8, 8.3]
    ('plumbing',           'C', 'MEP',        (6, 8),  (0, 0)),
    ('electrical',         'C', 'MEP',        (4, 6),  (0, 0)),
    ('HVAC_ducting',       'C', 'MEP',        (3, 4),  (0, 0)),
    ('mep_test_QC',        'Q', 'QC',         (2, 2),  (0, 0)),   # G 8.3 (pressure/tightness/continuity)
]
_WET_FINISH = [  # bathroom/kitchen modules  [G 4.1.3, 4.2.2 steps9-13, 8.4]
    ('internal_partitions', 'D', 'finishing', (4, 6),  (0, 0)),   # G 4.1.3 step2 / 4.2.2 step9 (panels/drywall)
    ('waterproofing',      'D', 'finishing',  (4, 6),  (24, 48)), # G 4.1.3 step5 / 4.2.2 step11; ponding test lag [N]
    ('screed_tiling',      'D', 'finishing',  (8, 12), (0, 0)),   # G 4.1.3 steps6,9 / 4.2.2 step12
    ('plaster_skim',       'D', 'finishing',  (3, 5),  (0, 0)),   # G 4.1.3 step10 / 8.4
    ('paint_coat_1',       'E', 'finishing',  (3, 4),  (12, 24)), # G 4.2.2 step17; drying lag [N]
    ('paint_coat_2',       'E', 'finishing',  (3, 4),  (12, 24)), # final factory coat (final coat on-site allowed, G 10.2)
    ('finishing_QC',       'Q', 'QC',         (2, 2),  (0, 0)),   # G 8.4
]
_DRY_FINISH = [  # bedroom/living modules — NO waterproofing/ponding  [G 4.1.3, 8.4]
    ('internal_partitions', 'D', 'finishing', (4, 6),  (0, 0)),
    ('plaster_skim',       'D', 'finishing',  (3, 5),  (0, 0)),
    ('floor_screed',       'D', 'finishing',  (3, 5),  (0, 0)),   # G 4.1.3 step6
    ('floor_finishes',     'D', 'finishing',  (4, 6),  (0, 0)),   # G 4.1.3 steps11-12 (vinyl/timber)
    ('paint_coat_1',       'E', 'finishing',  (3, 4),  (12, 24)),
    ('paint_coat_2',       'E', 'finishing',  (3, 4),  (12, 24)),
    ('finishing_QC',       'Q', 'QC',         (2, 2),  (0, 0)),
]
_ASSEMBLY = [  # shared fit-out & dispatch  [G 4.1.3 steps7-8,13-18, 5.2-5.3]
    ('doors_windows',      'F', 'assembly',   (4, 6),  (0, 0)),   # G 4.1.3 steps7-8 (frames+glazing)
    ('fitout_fixtures',    'F', 'assembly',   (6, 10), (0, 0)),   # G 4.1.3 steps13-18 (sanitary/wardrobe/cabinets/railing/ceiling)
    ('final_QC',           'Q', 'QC',         (3, 4),  (0, 0)),   # G 8 final checks
    ('wrap_protect_ship',  'G', 'assembly',   (2, 3),  (0, 0)),   # G 4.1.3 step20 + 5.2-5.3 (protection/labelling)
]

ROUTES = {
    'RC-wet':    _RC_SHELL + _MEP + _WET_FINISH + _ASSEMBLY,
    'RC-dry':    _RC_SHELL + _MEP + _DRY_FINISH + _ASSEMBLY,
    'Steel-wet': _STEEL_SHELL + _MEP + _WET_FINISH + _ASSEMBLY,
    'Steel-dry': _STEEL_SHELL + _MEP + _DRY_FINISH + _ASSEMBLY,
}
OPS_PER_MODULE = 22
assert all(len(r) == OPS_PER_MODULE for r in ROUTES.values())

PT_DEVIATION = 0.2   # +/-20% across compatible stations (SD1-style)

# Class-mix presets. Real projects use ONE shell system each [G Appendix];
# wet:dry ~ 3:1 from the typical 2BR unit (LRDIN dry + KIT/B2-IBB/MB-IBB wet)
# [G 2.4]. 'mixed' models a multi-project (XL) factory load.
PRESET_MIXES = {
    'rc_project':    (0.75, 0.25, 0.0, 0.0),
    'steel_project': (0.0, 0.0, 0.75, 0.25),
    'mixed':         (0.375, 0.125, 0.375, 0.125),
}


def build_factory(station_counts):
    """station_counts: dict type->count  ->  (mch_type [M], machine id ranges)."""
    mch_type, ranges, mid = [], {}, 0
    for t_idx, t in enumerate(STATION_TYPES):
        cnt = int(station_counts.get(t, 0))
        ranges[t] = list(range(mid, mid + cnt))
        mch_type.extend([t_idx] * cnt)
        mid += cnt
    return np.array(mch_type, dtype=int), ranges


def ppvc_instance_generator(n_modules=10,
                            class_mix='mixed',
                            station_counts=None,
                            seed=None):
    """
    Generate one PPVC-FJSP instance.

    :param n_modules: number of modules (jobs), J
    :param class_mix: one of
            - a preset name: 'rc_project' / 'steel_project' / 'mixed'
            - probabilities over ROUTING_CLASSES (len 4, sums to 1)
            - explicit per-module class indices (len n_modules)
    :param station_counts: dict station type -> count (DEFAULT_FACTORY)
    :param seed: RNG seed for reproducibility
    :return: (job_length [J], op_pt [N,M], meta dict)
    """
    rng = np.random.default_rng(seed)
    station_counts = dict(station_counts or DEFAULT_FACTORY)
    mch_type, type_ranges = build_factory(station_counts)
    n_machines = len(mch_type)

    if isinstance(class_mix, str):
        class_mix = PRESET_MIXES[class_mix]

    if len(class_mix) == len(ROUTING_CLASSES) \
            and all(isinstance(x, (int, float)) for x in class_mix) \
            and abs(sum(class_mix) - 1.0) < 1e-6 and len(class_mix) != n_modules:
        classes = rng.choice(len(ROUTING_CLASSES), size=n_modules, p=list(class_mix))
    else:   # explicit per-module class indices
        classes = np.asarray(class_mix, dtype=int)
        assert len(classes) == n_modules

    # every station type used by the sampled classes must exist in the factory
    used_types = {s for j in range(n_modules)
                  for (_, s, _, _, _) in ROUTES[ROUTING_CLASSES[classes[j]]]}
    missing = [t for t in used_types if not type_ranges[t]]
    if missing:
        raise ValueError(f'factory has no stations of required type(s): {missing}')

    job_length = np.full(n_modules, OPS_PER_MODULE, dtype=int)
    n_ops = OPS_PER_MODULE * n_modules

    op_pt = np.zeros((n_ops, n_machines), dtype=int)
    op_type = np.zeros(n_ops, dtype=int)
    op_station = np.zeros(n_ops, dtype=int)
    time_lag = np.zeros(n_ops, dtype=int)
    module_weight = np.zeros(n_modules)
    op_name = []

    row = 0
    for j in range(n_modules):
        cls = ROUTING_CLASSES[classes[j]]
        shell = 'RC' if cls.startswith('RC') else 'Steel'
        w_lo, w_hi = MODULE_WEIGHT_T[shell]
        module_weight[j] = round(rng.uniform(w_lo, w_hi), 1)
        for (name, s_type, o_type, (d_lo, d_hi), (l_lo, l_hi)) in ROUTES[cls]:
            nominal = rng.integers(d_lo, d_hi + 1)
            for m in type_ranges[s_type]:
                pt = int(round(nominal * rng.uniform(1 - PT_DEVIATION,
                                                     1 + PT_DEVIATION)))
                op_pt[row, m] = max(1, pt)
            op_type[row] = OP_TYPE_NAMES.index(o_type)
            op_station[row] = STATION_TYPES.index(s_type)
            time_lag[row] = rng.integers(l_lo, l_hi + 1) if l_hi > 0 else 0
            op_name.append(f'M{j}:{name}')
            row += 1

    meta = {
        'op_type': op_type, 'op_station': op_station, 'time_lag': time_lag,
        'mch_type': mch_type, 'routing_class': classes.astype(int),
        'module_weight_t': module_weight,
        'op_name': op_name, 'station_counts': station_counts,
    }
    return job_length, op_pt, meta


# ---------------------------------------------------------------------------
# Serialization: standard .fjs (backward compatible) + side-car meta .json
# ---------------------------------------------------------------------------

def save_instance(path_stem, job_length, op_pt, meta):
    """Write <stem>.fjs (standard FJSP text) + <stem>.meta.json (PPVC data)."""
    from data_utils import matrix_to_text
    flex = float((op_pt != 0).sum(axis=1).mean())
    lines = matrix_to_text(job_length, op_pt, round(flex, 2))
    with open(path_stem + '.fjs', 'w') as f:
        for line in lines:
            print(line, file=f)
    js = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
          for k, v in meta.items()}
    with open(path_stem + '.meta.json', 'w') as f:
        json.dump(js, f, indent=1)


def load_instance(path_stem):
    """Read back (job_length, op_pt, meta) written by save_instance."""
    from data_utils import text_to_matrix
    with open(path_stem + '.fjs') as f:
        job_length, op_pt = text_to_matrix(f.readlines())
    with open(path_stem + '.meta.json') as f:
        js = json.load(f)
    meta = {k: (np.array(v) if isinstance(v, list) and k != 'op_name' else v)
            for k, v in js.items()}
    return job_length, op_pt, meta


# ---------------------------------------------------------------------------
# Self-test: round-trip + schedule in the (lag-aware) env
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== PPVC instance generator self-test (BCA-grounded routes) ===')
    J, SEED = 5, 42
    job_length, op_pt, meta = ppvc_instance_generator(
        n_modules=J, class_mix='mixed', station_counts=DEFAULT_FACTORY, seed=SEED)
    M = op_pt.shape[1]
    print(f'modules={J}  ops={op_pt.shape[0]}  machines={M}')
    print('routing classes:', [ROUTING_CLASSES[c] for c in meta['routing_class']])
    print('module weights (t):', meta['module_weight_t'].tolist())
    print(f'per-op flexibility: mean={(op_pt != 0).sum(1).mean():.2f} '
          f'min={(op_pt != 0).sum(1).min()} max={(op_pt != 0).sum(1).max()}')
    lag_total = meta['time_lag'].sum()
    pt_proxy = np.ma.masked_equal(op_pt, 0).mean(axis=1).sum()
    print(f'total lag={lag_total}h vs total mean-processing={pt_proxy:.0f}h '
          f'(lag/processing ratio={lag_total / pt_proxy:.2f})')

    # 1) .fjs round-trip
    save_instance('/tmp/ppvc_test', job_length, op_pt, meta)
    jl2, pt2, meta2 = load_instance('/tmp/ppvc_test')
    assert np.array_equal(job_length, jl2), 'job_length round-trip mismatch'
    assert np.array_equal(op_pt, pt2), 'op_pt round-trip mismatch'
    assert np.array_equal(meta['time_lag'], meta2['time_lag'])
    print('round-trip via .fjs + meta.json: OK')

    # 2) schedule end-to-end with FIFO, WITHOUT lags (baseline)
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from common_utils import heuristic_select_action
    np.random.seed(SEED)
    env = FJSPEnvForSameOpNums(n_j=J, n_m=M)
    env.set_initial_data([job_length], [op_pt])
    while not env.done().all():
        env.step(np.array([heuristic_select_action('FIFO', env)]))
    ms_nolag = env.current_makespan[0]
    print(f'FIFO makespan WITHOUT lags = {ms_nolag:.1f}h')

    # 3) schedule WITH lags (lag-aware env, Adaptation 2a)
    np.random.seed(SEED)
    env2 = FJSPEnvForSameOpNums(n_j=J, n_m=M)
    env2.set_initial_data([job_length], [op_pt], time_lag_list=[meta['time_lag']])
    while not env2.done().all():
        env2.step(np.array([heuristic_select_action('FIFO', env2)]))
    ms_lag = env2.current_makespan[0]
    print(f'FIFO makespan WITH lags    = {ms_lag:.1f}h '
          f'(+{ms_lag - ms_nolag:.0f}h from cascaded lags)')
    assert ms_lag > ms_nolag, 'lag-aware makespan must exceed lag-free makespan'
    print('=== self-test PASSED ===')
