"""Matched-compute control: best-of-N sampled decoding of a trained policy.

Answers the reviewer question "is the deficit to per-instance search a
property of the policy or of one-shot greedy decoding?" by spending a
stated per-instance compute budget on the policy itself: candidate 0 is
the greedy rollout, candidates 1..N-1 are multinomial samples from the
same policy (torch.manual_seed(1000+p) per pass, reproducible), and the
per-instance result is the best FEASIBLE makespan (every improving
schedule is checked by the independent validator).

Outputs results/sample_decode/{model}_{cell}_N{n}.json:
  {"rows": {instance: best_ms}, "meta": {...timings, N, batch}}
Distinct prefix and directory from Result_greedy*, so no existing
analysis or fill_macros regex can sweep these in silently.

Usage:
  python -u scripts/x2_eval_sample.py --model_name 10x25+ppvct-mixed+m1-bcb-guide-s301 \
      --cells v1+t0.6,v1+t1.0 --n_samples 64 [--limit 0] [--batch 20]
"""
import sys, os, json, glob, time, argparse

cli = argparse.ArgumentParser()
cli.add_argument('--model_name', type=str, required=True)
cli.add_argument('--cells', type=str, required=True)
cli.add_argument('--n_samples', type=int, default=64)
cli.add_argument('--batch', type=int, default=20)
cli.add_argument('--split', type=str, default='test')
cli.add_argument('--limit', type=int, default=0, help='smoke: cap instances')
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
from transport_marl.bound import TransportBound


def build_policy(snap):
    algo = snap.get('algo', 'mappo')
    if algo == 'single':
        from transport_marl.single_agent import DANIELSingle
        return DANIELSingle(configs)
    if algo in ('mappo', 'coma'):
        return DANIELTransport(configs)
    raise ValueError(f"unknown algo '{algo}'")


def rollout(env, policy, greedy):
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
                                  greedy=greedy)
        state, _, _ = env.step(a.cpu().numpy())
    return env.current_makespan.copy()


def make_env(insts):
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
        env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
    return env, jls, pts, lags, lay


def main():
    device = torch.device(configs.device)
    policy = build_policy(snap)
    sd = torch.load(f'trained_network/PPVCT/{args_cli.model_name}.pth',
                    map_location=device)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    policy.eval()

    os.makedirs('results/sample_decode', exist_ok=True)
    for cell in args_cli.cells.split(','):
        dirname = cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'
        ds = f'data/PPVCT/{dirname}/{args_cli.split}'
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        if args_cli.limit:
            stems = stems[:args_cli.limit]
        best = {os.path.basename(s): np.inf for s in stems}
        t_cell = time.time()
        pass_times = []
        for i0 in range(0, len(stems), args_cli.batch):
            chunk = stems[i0:i0 + args_cli.batch]
            insts = [load_instance(s) for s in chunk]
            for p in range(args_cli.n_samples):
                torch.manual_seed(1000 + p)
                t0 = time.time()
                env, jls, pts, lags, lay = make_env(insts)
                ms = rollout(env, policy, greedy=(p == 0))
                pass_times.append(time.time() - t0)
                for e, stem in enumerate(chunk):
                    name = os.path.basename(stem)
                    if ms[e] < best[name] - 1e-9:
                        res = validate_transport_schedule(
                            jls[e], pts[e], lags[e],
                            np.array(lay[e]['station_cell']),
                            np.array(lay[e]['tau_cells']),
                            int(lay[e]['n_vehicles']),
                            int(lay[e]['veh_start_cell']),
                            env.schedule_record(e))
                        assert res['feasible'], (name, res['violations'][:3])
                        best[name] = float(ms[e])
                del env
        wall = time.time() - t_cell
        out = {'rows': best,
               'meta': {'model': args_cli.model_name, 'cell': cell,
                        'n_samples': args_cli.n_samples,
                        'greedy_is_candidate_0': True,
                        'n_instances': len(stems),
                        'wall_s': round(wall, 1),
                        'wall_per_instance_s': round(wall / len(stems), 2),
                        'mean_pass_s_per_chunk': round(float(np.mean(pass_times)), 2),
                        'batch': args_cli.batch,
                        'note': 'batched GPU, machine shared (contended); '
                                'not a certified latency'}}
        path = (f'results/sample_decode/{args_cli.model_name}_{cell}'
                f'_N{args_cli.n_samples}.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        vals = np.array(list(best.values()))
        print(f'{cell}: n={len(stems)} best-of-{args_cli.n_samples} '
              f'mean={vals.mean():.1f} wall={wall:.0f}s '
              f'({wall/len(stems):.1f}s/inst)', flush=True)


if __name__ == '__main__':
    main()
