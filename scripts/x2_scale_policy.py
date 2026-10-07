"""Zero-shot policy evaluation on the scale-up pilot cells (Phase A).

Same rollout protocol as scripts/eval_ppvct.py (greedy two-head rollout in
the batched env, guide channel attached to the FULL bound when the checkpoint
was trained with it, every schedule independently re-checked by validator_t),
with two additions the scaling exhibit needs:
  * per-instance decision count (number of events the policy acted on), and
  * per-instance makespan written to JSON rather than a bare .npy,
so nothing lands in test_results/PPVCT and nothing can be swept into the
manuscript's existing analyses.

Latency: nn_time / nn_calls is the mean wall time of one batched forward
pass, exactly the quantity eval_ppvct.py reports. It is only a certified
figure when the device has an exclusive slot; this pilot runs beside a CP-SAT
queue, so every latency number produced here is CONTENDED and marked as such.

Output: results/scaleup/policy/{model_name}_{cell}.json

Usage:
  python scripts/x2_scale_policy.py --model_name 10x25+ppvct-mixed+m1-bcb-guide-s301 \
      --cells 20x25+ppvct-mixed+v1+t0.6 --device cuda
"""

import sys, os, json, glob, time, argparse

cli = argparse.ArgumentParser()
cli.add_argument('--model_name', type=str, required=True)
cli.add_argument('--cells', type=str, required=True)
cli.add_argument('--device', type=str, default='auto',
                 choices=['auto', 'cuda', 'cpu'])
cli.add_argument('--batch', type=int, default=15)
cli.add_argument('--outdir', type=str, default='results/scaleup/policy')
cli.add_argument('--tag', type=str, default='')
cli.add_argument('--n', type=int, default=0,
                 help='evaluate only the first N instances of each cell; 0 = '
                      'all. Only for the batch-1 latency probe, which must '
                      'carry a --tag so its file cannot be read as a cell '
                      'result.')
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
if args_cli.device == 'auto':
    configs.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
else:
    configs.device = args_cli.device

from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.model_transport import DANIELTransport
from transport_marl.mappo import select_actions


def greedy_rollout(env, policy):
    """Greedy rollout; returns makespans, per-env decision counts, mean
    wall time of one batched forward pass."""
    nn_time, nn_calls = 0.0, 0
    decisions = np.zeros(env.number_of_envs, dtype=int)
    state = env.state
    while env.done().min() < 1:
        alive = env.n_realized < env.number_of_ops
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
            a, _ = select_actions(pi_m, pi_v, state.event_type_tensor,
                                  greedy=True)
        if configs.device == 'cuda':
            torch.cuda.synchronize()
        nn_time += time.perf_counter() - t0
        nn_calls += 1
        decisions += alive.astype(int)
        state, _, _ = env.step(a.cpu().numpy())
    return (env.current_makespan.copy(), decisions,
            nn_time / max(nn_calls, 1))


def build_policy(snap):
    algo = snap.get('algo', 'mappo')
    if algo == 'single':
        from transport_marl.single_agent import DANIELSingle
        return DANIELSingle(configs), algo
    if algo in ('mappo', 'coma'):
        return DANIELTransport(configs), algo
    raise ValueError(f"unknown algo '{algo}'")


def main():
    device = torch.device(configs.device)
    policy, algo = build_policy(snap)
    n_par = sum(p.numel() for p in policy.parameters())
    ckpt = f'trained_network/PPVCT/{args_cli.model_name}.pth'
    sd = torch.load(ckpt, map_location=device)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'checkpoint/architecture mismatch for {args_cli.model_name} '
            f'(algo={algo}): {len(missing)} missing, {len(unexpected)} '
            f'unexpected. Refusing to evaluate a partially loaded policy.')
    policy.eval()
    print(f'[x2_scale_policy] {args_cli.model_name} algo={algo} '
          f'params={n_par} device={configs.device} '
          f'guide={bool(getattr(configs, "guide", False))} '
          f'bound_veh={int(snap.get("bound_veh", 1))}', flush=True)

    os.makedirs(args_cli.outdir, exist_ok=True)
    for cell in args_cli.cells.split(','):
        ds = f'data/PPVCT/{cell}/test'
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        assert stems, f'no instances in {ds}'
        if args_cli.n:
            assert args_cli.tag, '--n requires --tag (partial cell, not a result)'
            stems = stems[:args_cli.n]
        rows = {}
        t0 = time.time()
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
            env = FJSPEnvTransport(
                len(jls[0]), pts[0].shape[1], use_lag_features=True,
                use_guide=guide,
                guide_price=snap.get('guide_price', 'certified'),
                guide_price_scale=float(snap.get('guide_price_scale', 1.0)))
            env.set_initial_data(jls, pts, lags, opt, mct, lay)
            if guide:
                from transport_marl.bound import TransportBound
                # same bound as at training time: the fleet-term ablation
                # (snapshot bound_veh=0) prices actions without B_veh;
                # snapshots without the key were all trained with it
                env.attach_bound(TransportBound(
                    env, use_mch=True,
                    use_veh=bool(int(snap.get('bound_veh', 1)))))
            ms, dec, lat = greedy_rollout(env, policy)
            for e, stem in enumerate(chunk):
                res = validate_transport_schedule(
                    jls[e], pts[e], lags[e],
                    np.array(lay[e]['station_cell']),
                    np.array(lay[e]['tau_cells']), int(lay[e]['n_vehicles']),
                    int(lay[e]['veh_start_cell']), env.schedule_record(e))
                assert res['feasible'], f'{stem}: {res["violations"][:3]}'
                assert abs(res['makespan'] - ms[e]) < 1e-6, \
                    f'{stem}: validator {res["makespan"]} != env {ms[e]}'
                rows[os.path.basename(stem)] = dict(
                    ms=float(ms[e]), decisions=int(dec[e]),
                    lat_batch_s=float(lat), validated=True)
            del env
            import gc
            gc.collect()
        out = dict(
            model=args_cli.model_name, cell=cell, n=len(rows),
            device=configs.device, batch=args_cli.batch, params=n_par,
            latency_contended=True,
            latency_note='mean wall time of ONE BATCHED forward pass; the '
                         'box also ran a CP-SAT queue, so this is CONTENDED '
                         'and is not a certified per-event latency',
            rows=rows)
        tag = args_cli.tag
        path = f'{args_cli.outdir}/{args_cli.model_name}_{cell}{tag}.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        arr = np.array([r['ms'] for r in rows.values()])
        dd = np.array([r['decisions'] for r in rows.values()])
        print(f'{cell}: n={len(rows)} mean={arr.mean():.2f} '
              f'std(1)={arr.std(ddof=1):.2f} decisions={dd.mean():.0f} '
              f'lat/batched-fwd={np.mean([r["lat_batch_s"] for r in rows.values()])*1000:.1f}ms '
              f'(CONTENDED) [{time.time() - t0:.0f}s] -> {path}', flush=True)


if __name__ == '__main__':
    main()
