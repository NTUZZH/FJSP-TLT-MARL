"""Evaluation protocol for transport checkpoints on PPVCT test cells.

Per instance: greedy rollout (batched E<=20), independent validation
(validator_t), makespan + mean per-event policy latency; summary vs CP-SAT
reference (or_solution/PPVCT/{cell}.jsonl, warm-started UB) when present.

--veh_rule policy  : vehicle head decides (joint policy, default)
--veh_rule NVF|STT|FIFO : fixed rule at vehicle events (E0b protocol pin)

Usage:
  python -u scripts/eval_ppvct.py --model_name 10x25+ppvct-mixed+joint-v1-s301 \
      --cells v1+t0.6,v2+t0.6 [--veh_rule policy] [--split test]
Outputs: test_results/PPVCT/{cell}/Result_greedy[{veh_rule}]+{model}_{cell}.npy
"""

import sys, os, json, glob, time, argparse

cli = argparse.ArgumentParser()
cli.add_argument('--model_name', type=str, required=True)
cli.add_argument('--cells', type=str, default='all')
cli.add_argument('--veh_rule', type=str, default='policy',
                 choices=['policy', 'NVF', 'STT', 'FIFO'])
cli.add_argument('--split', type=str, default='test')
cli.add_argument('--batch', type=int, default=20)
args_cli = cli.parse_args()
sys.argv = [sys.argv[0]]

import numpy as np
import torch

sys.path.insert(0, '.')
from params import configs

ARCH_KEYS = ['fea_j_input_dim', 'fea_m_input_dim', 'n_op_types', 'n_mch_types',
             'type_emb_dim', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor', 'hidden_dim_actor',
             'num_mlp_layers_critic', 'hidden_dim_critic', 'dropout_prob',
             'guide']


def load_snapshot(model_name):
    with open(f'train_log/PPVCT/config_{model_name}.json') as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            v = snap[k]
            if isinstance(v, str) and v.startswith('['):
                v = json.loads(v)
            setattr(configs, k, v)
    return snap


snap = load_snapshot(args_cli.model_name)
configs.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.model_transport import DANIELTransport
from transport_marl.mappo import select_actions
from transport_marl.diagnostics import agent_diagnostics


def fixed_veh_action(env, e, rule):
    v = env.event_veh[e]
    loc = env.veh_cell[e, v]
    tasks = np.nonzero(env.task_active[e])[0]
    keys = []
    for j in tasks:
        frm, to = env.task_from[e, j], env.task_to[e, j]
        if rule == 'STT':
            key = (env.tau_cells[e, loc, frm] + env.tau_cells[e, frm, to], j)
        elif rule == 'NVF':
            key = (env.tau_cells[e, loc, frm], j)
        else:
            key = (env.task_release[e, j], j)
        keys.append((key, j))
    return min(keys)[1]


def greedy_rollout(env, policy, veh_rule):
    device = torch.device(configs.device)
    state = env.state
    nn_time, nn_calls = 0.0, 0
    while env.done().min() < 1:
        t0 = time.perf_counter()
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
            a, _ = select_actions(pi_m, pi_v, state.event_type_tensor, greedy=True)
        nn_time += time.perf_counter() - t0
        nn_calls += 1
        a = a.cpu().numpy()
        if veh_rule != 'policy':
            for e in range(env.number_of_envs):
                if env.event_type[e] == 1 and env.n_realized[e] < env.number_of_ops:
                    a[e] = fixed_veh_action(env, e, veh_rule)
        state, _, _ = env.step(a)
    return env.current_makespan.copy(), nn_time / max(nn_calls, 1)


def build_policy(snap):
    """Instantiate the network class the checkpoint was TRAINED with.

    The arm identity lives in the training config snapshot ('algo'). Snapshots
    written before that field existed default to 'mappo', which is what every
    such run in fact was. Guessing wrong here does not always raise: a
    mismatched state_dict can load partially and silently produce garbage
    makespans, so we dispatch explicitly and refuse unknown values.
    """
    algo = snap.get('algo', 'mappo')
    if algo == 'single':
        from transport_marl.single_agent import DANIELSingle
        return DANIELSingle(configs), algo
    if algo in ('mappo', 'coma'):
        # COMA differs only in the critic, which evaluation never uses.
        return DANIELTransport(configs), algo
    raise ValueError(f"unknown algo '{algo}' in the config snapshot of "
                     f"{args_cli.model_name}; cannot pick a network class")


def main():
    device = torch.device(configs.device)
    policy, algo = build_policy(snap)
    ckpt = f'trained_network/PPVCT/{args_cli.model_name}.pth'
    sd = torch.load(ckpt, map_location=device)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/architecture mismatch for {args_cli.model_name} "
            f"(algo='{algo}'): {len(missing)} missing and {len(unexpected)} "
            f"unexpected parameters. Refusing to evaluate a partially loaded "
            f"policy.")
    policy.eval()

    if args_cli.cells == 'all':
        cells = sorted(d.split('+ppvct-mixed+')[1].split('/')[0]
                       for d in glob.glob('data/PPVCT/10x25+ppvct-mixed+v*+t*'))
    else:
        cells = args_cli.cells.split(',')
    tag = 'greedy' if args_cli.veh_rule == 'policy' else f'greedy-{args_cli.veh_rule}'

    for cell in cells:
        # transfer cells may carry a size prefix ('15x25+ppvct-mixed+v2+t0.6')
        dirname = cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'
        ds = f'data/PPVCT/{dirname}/{args_cli.split}'
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        results = []
        diag_rows = []
        for i0 in range(0, len(stems), args_cli.batch):
            chunk = stems[i0:i0 + args_cli.batch]
            insts = [load_instance(s) for s in chunk]
            jls = [np.asarray(x[0]) for x in insts]
            pts = [np.asarray(x[1], dtype=float) for x in insts]
            lags = [np.asarray(x[2]['time_lag'], dtype=float) for x in insts]
            opt = [np.asarray(x[2]['op_type']) for x in insts]
            mct = [np.asarray(x[2]['mch_type']) for x in insts]
            lay = [x[2]['transport'] for x in insts]
            guide = bool(getattr(configs, 'guide', False))
            env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1],
                                   use_lag_features=True, use_guide=guide)
            env.set_initial_data(jls, pts, lags, opt, mct, lay)
            if guide:
                # the guide channel prices actions against the FULL bound at
                # training time; eval must attach the same bound or the
                # feature distribution silently shifts to chain-only
                from transport_marl.bound import TransportBound
                env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
            ms, lat = greedy_rollout(env, policy, args_cli.veh_rule)
            for e, stem in enumerate(chunk):
                res = validate_transport_schedule(
                    jls[e], pts[e], lags[e], np.array(lay[e]['station_cell']),
                    np.array(lay[e]['tau_cells']), int(lay[e]['n_vehicles']),
                    int(lay[e]['veh_start_cell']), env.schedule_record(e))
                assert res['feasible'], f'{stem}: {res["violations"][:3]}'
                results.append((os.path.basename(stem), ms[e], lat))
                diag_rows.append(agent_diagnostics(
                    env.schedule_record(e), pts[e],
                    int(lay[e]['n_vehicles']), pts[e].shape[1]))
            del env
            import gc
            gc.collect()
        out_dir = f'test_results/PPVCT/{cell}'
        os.makedirs(out_dir, exist_ok=True)
        arr = np.array([(m, l) for _, m, l in results])
        np.save(f'{out_dir}/Result_{tag}+{args_cli.model_name}_{cell}.npy', arr)

        # reference gap if available
        ref_path = f'or_solution/PPVCT/{cell}.jsonl'
        gap_str = ''
        if os.path.exists(ref_path):
            refs = {}
            with open(ref_path) as f:
                for line in f:
                    r = json.loads(line)
                    refs[r['instance']] = r['ms']
            gaps = [(m - refs[n]) / refs[n] for n, m, _ in results if n in refs]
            if gaps:
                gap_str = f'  gap-vs-ref {np.mean(gaps):.2%} (cov {len(gaps)})'
        print(f'{cell} {tag}: n={len(results)} mean={arr[:,0].mean():.2f} '
              f'std(1)={arr[:,0].std(ddof=1):.2f} '
              f'lat/event={arr[:,1].mean()*1000:.1f}ms{gap_str}', flush=True)
        if diag_rows:
            import numpy as _np
            agg = dict(
                veh_gini=float(_np.mean([d['veh_gini'] for d in diag_rows])),
                veh_idle_mean=float(_np.mean([_np.mean(d['veh_idle_rate'])
                                              for d in diag_rows if d['veh_idle_rate']])),
                mch_gini=float(_np.mean([d['mch_gini'] for d in diag_rows])),
                n_transports_mean=float(_np.mean([d['n_transports'] for d in diag_rows])),
                n=len(diag_rows))
            os.makedirs('results/diagnostics', exist_ok=True)
            dp = f'results/diagnostics/{args_cli.model_name}_{cell}_{tag}.json'
            with open(dp, 'w') as f:
                json.dump(agg, f, indent=1)
            print(f'  diagnostics -> {dp}', flush=True)


if __name__ == '__main__':
    main()
