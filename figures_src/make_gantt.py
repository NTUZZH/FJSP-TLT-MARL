"""Render the supplement Gantt comparison (s_gantt): one test instance of the
10-module cell with one vehicle and travel intensity 0.6, scheduled by BOLT
(top) and by the transport-as-penalty anchor (bottom).

Both schedules come from the evaluation protocol of scripts/eval_ppvct.py:
greedy two-head rollout on the first test batch (instances 000-019) of the
cell; BOLT with its vehicle head and the full bound attached for the price
channel; the anchor with the nearest-vehicle-first rule at vehicle events.
Every schedule is re-checked by validator_t. The rollout uses the device
rule of eval_ppvct.py (GPU when present), because near-tied greedy choices
can resolve differently on CPU; the script asserts that the selected
instance reproduces the stored makespan of both arms exactly.

Instance selection (fixed rule, nothing else tuned): among the first 20 test
instances, the one whose two makespans differ most in absolute value.

Each arm is rolled out in its own subprocess, because params.configs is a
module-level object and a key absent from one arm's config snapshot would
otherwise keep the other arm's value.

Run:  python figures_src/make_gantt.py          (from repo root)
Outputs: paper/figures/s_gantt.{pdf,png}
         figures_src/data/s_gantt_schedules.json (the two plotted schedules)
"""

import argparse
import glob
import json
import os
import subprocess
import sys

CELL = 'v1+t0.6'
DATASET = f'data/PPVCT/10x25+ppvct-mixed+{CELL}/test'
ARMS = [  # (key, model name, vehicle rule, stored result tag)
    ('bolt', '10x25+ppvct-mixed+m1-bcb-guide-s301', 'policy', 'greedy'),
    ('anchor', '10x25+ppvct-mixed+e0b-uncontended-s301', 'NVF', 'greedy-NVF'),
]
N_FIRST = 20
CACHE = 'figures_src/data/s_gantt_schedules.json'
OUT = 'paper/figures/s_gantt'

ARCH_KEYS = ['fea_j_input_dim', 'fea_m_input_dim', 'n_op_types', 'n_mch_types',
             'type_emb_dim', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor', 'hidden_dim_actor',
             'num_mlp_layers_critic', 'hidden_dim_critic', 'dropout_prob',
             'guide']


# ---------------------------------------------------------------- rollout
def rollout(model_name, veh_rule, out_path):
    """Greedy rollout of one arm on the first N_FIRST test instances; writes
    every schedule record to out_path (runs in a fresh interpreter)."""
    sys.argv = [sys.argv[0]]
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, '.')
    from params import configs
    with open(f'train_log/PPVCT/config_{model_name}.json') as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            v = snap[k]
            if isinstance(v, str) and v.startswith('['):
                v = json.loads(v)
            setattr(configs, k, v)
    # same device rule as scripts/eval_ppvct.py, which wrote the stored
    # results; near-tied greedy choices can resolve differently on CPU
    configs.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    from ppvc_instance_generator import load_instance
    from transport_marl.fjsp_env_transport import FJSPEnvTransport
    from transport_marl.validator_t import validate_transport_schedule
    from transport_marl.model_transport import DANIELTransport
    from transport_marl.mappo import select_actions

    algo = snap.get('algo', 'mappo')
    assert algo == 'mappo', algo
    policy = DANIELTransport(configs)
    sd = torch.load(f'trained_network/PPVCT/{model_name}.pth',
                    map_location=configs.device)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    policy.eval()

    stems = sorted(g[:-4] for g in glob.glob(f'{DATASET}/instance_*.fjs'))
    stems = stems[:N_FIRST]
    insts = [load_instance(s) for s in stems]
    jls = [np.asarray(x[0]) for x in insts]
    pts = [np.asarray(x[1], dtype=float) for x in insts]
    lags = [np.asarray(x[2]['time_lag'], dtype=float) for x in insts]
    opt = [np.asarray(x[2]['op_type']) for x in insts]
    mct = [np.asarray(x[2]['mch_type']) for x in insts]
    lay = [x[2]['transport'] for x in insts]
    guide = bool(getattr(configs, 'guide', False))
    env = FJSPEnvTransport(
        len(jls[0]), pts[0].shape[1], use_lag_features=True, use_guide=guide,
        guide_price=snap.get('guide_price', 'certified'),
        guide_price_scale=float(snap.get('guide_price_scale', 1.0)))
    env.set_initial_data(jls, pts, lags, opt, mct, lay)
    if guide:
        from transport_marl.bound import TransportBound
        env.attach_bound(TransportBound(
            env, use_mch=True, use_veh=bool(int(snap.get('bound_veh', 1)))))

    def fixed_veh_action(e):          # scripts/eval_ppvct.py, rule NVF
        v = env.event_veh[e]
        loc = env.veh_cell[e, v]
        keys = [((env.tau_cells[e, loc, env.task_from[e, j]], j), j)
                for j in np.nonzero(env.task_active[e])[0]]
        return min(keys)[1]

    state = env.state
    while env.done().min() < 1:
        with torch.no_grad():
            pi_m, pi_v, _ = policy(
                fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                mch_type=state.mch_type_tensor, fea_v=state.fea_v_tensor,
                fea_veh_pairs=state.fea_veh_pairs_tensor,
                veh_action_mask=state.veh_action_mask_tensor,
                event_veh=state.event_veh_tensor,
                task_dest_op=state.task_dest_op_tensor)
            a, _ = select_actions(pi_m, pi_v, state.event_type_tensor,
                                  greedy=True)
        a = a.cpu().numpy()
        if veh_rule == 'NVF':
            for e in range(env.number_of_envs):
                if (env.event_type[e] == 1
                        and env.n_realized[e] < env.number_of_ops):
                    a[e] = fixed_veh_action(e)
        state, _, _ = env.step(a)

    out = []
    for e, stem in enumerate(stems):
        rec = env.schedule_record(e)
        res = validate_transport_schedule(
            jls[e], pts[e], lags[e], np.array(lay[e]['station_cell']),
            np.array(lay[e]['tau_cells']), int(lay[e]['n_vehicles']),
            int(lay[e]['veh_start_cell']), rec)
        assert res['feasible'], f'{stem}: {res["violations"][:3]}'
        out.append(dict(
            instance=os.path.basename(stem), makespan=rec['makespan'],
            assigned_mch=rec['assigned_mch'].tolist(),
            op_start=rec['op_start'].tolist(), op_ct=rec['op_ct'].tolist(),
            transports=rec['transports'],
            job_length=jls[e].tolist(), time_lag=lags[e].tolist(),
            mch_type=mct[e].tolist(), n_vehicles=int(lay[e]['n_vehicles'])))
    with open(out_path, 'w') as f:
        json.dump(out, f)


def collect():
    """Select the instance from the stored test results (the numbers the
    paper reports), roll both arms out again to recover the full schedules, and require the selected instance to reproduce the stored
    makespan exactly for both arms."""
    import numpy as np
    runs, stored, agree = {}, {}, {}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    for key, model, rule, tag in ARMS:
        stored[key] = np.load(f'test_results/PPVCT/{CELL}/Result_{tag}+'
                              f'{model}_{CELL}.npy')[:N_FIRST, 0]
        tmp = f'{CACHE}.{key}.tmp'
        subprocess.run([sys.executable, __file__, '--rollout', model, rule,
                        tmp], check=True)
        with open(tmp) as f:
            runs[key] = json.load(f)
        os.remove(tmp)
        got = np.array([r['makespan'] for r in runs[key]])
        agree[key] = int(np.isclose(got, stored[key], atol=1e-4).sum())
    idx = int(np.argmax(np.abs(stored['anchor'] - stored['bolt'])))
    for key in stored:
        assert abs(runs[key][idx]['makespan'] - stored[key][idx]) < 1e-4, \
            f'{key}: instance {idx} does not reproduce the stored makespan'
    sel = dict(cell=CELL, index=idx, rule='argmax |anchor - BOLT| stored '
               f'test makespan over the first {N_FIRST} test instances',
               arms={k: dict(model=m, veh_rule=r, schedule=runs[k][idx])
                     for k, m, r, _ in ARMS},
               stored_first_makespans={k: v.tolist()
                                       for k, v in stored.items()},
               rollout_agrees_with_stored=agree)
    with open(CACHE, 'w') as f:
        json.dump(sel, f)
    return sel


# ---------------------------------------------------------------- figure
STATION_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8]           # types A-H, Q
STATION_LABEL = ['Formwork', 'Pour', 'MEP fit-out', 'Finishing', 'Paint',
                 'Assembly', 'Dispatch', 'Steel fabrication', 'Quality check']
# pale fills, one per station type (no amber or yellow)
# rose is reserved for the waiting outline, so no processing fill uses it
TYPE_FILL = ['#bfe3ec', '#c6d6ec', '#dad3ec', '#d5dde5', '#cfe6d1',
             '#dddddd', '#bfe0da', '#d7e4c8', '#e6e1d8']
EDGE = '#5a6470'
BLUE_F, BLUE_M, ROSE_M = '#a8c6e3', '#4f81ad', '#c25b6a'


def draw(sel):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from x2_style import apply
    apply(8)
    plt.rcParams.update({'axes.linewidth': 0.6, 'axes.edgecolor': 'black',
                         'hatch.linewidth': 0.45, 'hatch.color': BLUE_M})

    s0 = sel['arms']['bolt']['schedule']
    mtype = np.array(s0['mch_type'])
    rows = []                                  # (label, machine or None)
    for t in STATION_ORDER:
        ms_t = np.nonzero(mtype == t)[0]
        for k, m in enumerate(ms_t):
            rows.append((f'{STATION_LABEL[t]} {k + 1}', int(m)))
    n_veh = s0['n_vehicles']
    for v in range(n_veh):
        rows.append(('Vehicle' if n_veh == 1 else f'Vehicle {v + 1}', None))
    nrow = len(rows)
    ypos = {m: i for i, (_, m) in enumerate(rows) if m is not None}
    yveh = {v: nrow - n_veh + v for v in range(n_veh)}
    xmax = max(a['schedule']['makespan'] for a in sel['arms'].values())
    xlim = 10 * np.ceil(xmax * 1.02 / 10)

    pitch = 0.098                              # inch per row
    h_ax = nrow * pitch
    top_m, gap, bot_m, leg_h = 0.42, 0.42, 0.40, 0.0
    H = top_m + 2 * h_ax + gap + bot_m
    W = 7.16
    fig = plt.figure(figsize=(W, H))
    left, right = 1.02 / W, 1 - 0.14 / W
    axes = []
    for p in range(2):
        y0 = bot_m + (1 - p) * (h_ax + gap)
        axes.append(fig.add_axes([left, y0 / H, right - left, h_ax / H]))

    bh = 0.72
    for p, (ax, key) in enumerate(zip(axes, ('bolt', 'anchor'))):
        s = sel['arms'][key]['schedule']
        amch = np.array(s['assigned_mch'])
        st, ct = np.array(s['op_start']), np.array(s['op_ct'])
        lag = np.array(s['time_lag'])
        jl = np.array(s['job_length'])
        first = np.concatenate([[0], np.cumsum(jl)[:-1]])
        is_first = np.zeros(len(amch), bool)
        is_first[first] = True
        # station group shading separators
        for i, (lab, m) in enumerate(rows):
            if i > 0 and m is not None and rows[i - 1][1] is not None and \
                    mtype[m] != mtype[rows[i - 1][1]]:
                ax.axhline(i - 0.5, color='#c3cbd3', lw=0.4, zorder=0)
        ax.axhline(nrow - n_veh - 0.5, color='#6b7480', lw=0.6, zorder=0)
        # wait for a vehicle: from the time the module is ready to move
        # (completion + lag of its previous operation) to its pickup
        for tr in s['transports']:
            o = tr['op']
            prev = o - 1
            ready = ct[prev] + lag[prev]
            w = tr['pickup'] - ready
            if w > 1e-6:
                ax.add_patch(Rectangle(
                    (ready, ypos[int(amch[prev])] - bh / 2), w, bh,
                    facecolor='none', edgecolor=ROSE_M, lw=0.55, zorder=1))
        # processing
        for o in range(len(amch)):
            m = int(amch[o])
            ax.add_patch(Rectangle(
                (st[o], ypos[m] - bh / 2), ct[o] - st[o], bh,
                facecolor=TYPE_FILL[mtype[m]], edgecolor=EDGE, lw=0.3,
                zorder=2))
        # vehicle legs
        for tr in s['transports']:
            y = yveh[tr['veh']]
            if tr['pickup'] - tr['depart'] > 1e-6:
                ax.add_patch(Rectangle(
                    (tr['depart'], y - bh / 2), tr['pickup'] - tr['depart'],
                    bh, facecolor='white', edgecolor=BLUE_M, lw=0.3,
                    hatch='//////', zorder=2))
            ax.add_patch(Rectangle(
                (tr['pickup'], y - bh / 2), tr['arrival'] - tr['pickup'], bh,
                facecolor=BLUE_F, edgecolor=BLUE_M, lw=0.3, zorder=2))
        cmax = s['makespan']
        ax.axvline(cmax, color='black', lw=0.8, ls=(0, (3, 2)), zorder=3)
        ax.text(cmax - 2, -0.5 - 0.3, f'Makespan {cmax:.1f} h',
                ha='right', va='bottom', fontsize=7.5, clip_on=False)
        ax.set_xlim(0, xlim)
        ax.set_ylim(nrow - 0.5, -0.5)
        ax.set_yticks(range(nrow))
        ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
        ax.tick_params(axis='y', length=0, pad=2)
        ax.tick_params(axis='x', length=2.5, width=0.6)
        ax.set_xticks(np.arange(0, xlim + 1, 50))
        if p == 0:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel('Time (h)')
        fig.text(0.06 / W, (bot_m + (1 - p) * (h_ax + gap) + h_ax) / H,
                 f'({"ab"[p]})', ha='left', va='bottom', fontsize=8)

    handles = [
        Patch(facecolor=TYPE_FILL[0], edgecolor=EDGE, lw=0.3,
              label='Processing'),
        Patch(facecolor='none', edgecolor=ROSE_M, lw=0.55,
              label='Ready module waiting for the vehicle'),
        Patch(facecolor='white', edgecolor=BLUE_M, lw=0.3, hatch='//////',
              label='Empty vehicle move'),
        Patch(facecolor=BLUE_F, edgecolor=BLUE_M, lw=0.3,
              label='Loaded vehicle move'),
        Line2D([], [], color='black', lw=0.8, ls=(0, (3, 2)),
               label='Makespan'),
    ]
    leg = fig.legend(handles=handles, loc='upper center', ncol=5,
                     bbox_to_anchor=((left + right) / 2, 1 - 0.02 / H),
                     frameon=True, fancybox=False, fontsize=7,
                     handlelength=1.8, handleheight=0.9, columnspacing=1.2,
                     borderpad=0.4)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor('black')
    fig.savefig(OUT + '.pdf')
    fig.savefig(OUT + '.png', dpi=300)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--rollout', nargs=3, metavar=('MODEL', 'RULE', 'OUT'))
    a = cli.parse_args()
    if a.rollout:
        rollout(*a.rollout)
        return
    sel = collect()
    b = sel['arms']['bolt']['schedule']['makespan']
    n = sel['arms']['anchor']['schedule']['makespan']
    print(f'instance index {sel["index"]} '
          f'({sel["arms"]["bolt"]["schedule"]["instance"]}): '
          f'BOLT {b:.2f} h, penalty anchor {n:.2f} h, difference '
          f'{n - b:+.2f} h ({100 * (n - b) / b:+.1f}%)')
    draw(sel)
    print('rollouts equal to the stored results (of %d): %s'
          % (N_FIRST, sel['rollout_agrees_with_stored']))
    print(f'wrote {OUT}.pdf, {OUT}.png, {CACHE}')


if __name__ == '__main__':
    main()
