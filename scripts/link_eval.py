"""L1 external evaluation, our side: greedy rollouts on the certified Link
test set for each fleet size, decision sequences exported for scoring in the
RELEASED simulator (the pre-specified protocol: our policy decides, their
executor times).

Usage:
  python scripts/link_eval.py --model_name 15x10+link+link-m1-s301 \
      [--fleets 3,6,9,12,15,18] [--batch 20]
Outputs: ../external/replays_l1/{model}/v{V}/replay_{i:03d}.json
         (+ our-env makespans in summary.json; feasibility validator-checked)
"""

import argparse
import glob
import json
import os
import sys

cli = argparse.ArgumentParser()
cli.add_argument('--model_name', required=True)
cli.add_argument('--fleets', default='3,6,9,12,15,18')
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

with open(f'train_log/PPVCT/config_{args_cli.model_name}.json') as f:
    snap = json.load(f)
for k in ARCH_KEYS:
    if k in snap:
        v = snap[k]
        if isinstance(v, str) and v.startswith('['):
            v = json.loads(v)
        setattr(configs, k, v)
configs.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'

from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.model_transport import DANIELTransport
from transport_marl.mappo import select_actions

PSEUDO_PT = 1.0
DS = 'data/LINK/15x10/test'


def rollout(env, policy):
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
            a, _ = select_actions(pi_m, pi_v, state.event_type_tensor, greedy=True)
        state, _, _ = env.step(a.cpu().numpy())
    return env.current_makespan.copy()


def main():
    device = torch.device(configs.device)
    guide = bool(snap.get('guide', False))
    policy = DANIELTransport(configs)
    sd = torch.load(f'trained_network/PPVCT/{args_cli.model_name}.pth',
                    map_location=device)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'checkpoint mismatch: {len(missing)} missing, '
                           f'{len(unexpected)} unexpected')
    policy.eval()

    stems = sorted(g[:-4] for g in glob.glob(f'{DS}/instance_*.fjs'))
    out_root = f'../external/replays_l1/{args_cli.model_name}'
    summary = {}
    for V in [int(x) for x in args_cli.fleets.split(',')]:
        out_dir = f'{out_root}/v{V}'
        os.makedirs(out_dir, exist_ok=True)
        ms_all = []
        for i0 in range(0, len(stems), args_cli.batch):
            chunk = stems[i0:i0 + args_cli.batch]
            jls, pts, lags, opt, mct, lay = [], [], [], [], [], []
            for s in chunk:
                jl, pt, meta = load_instance(s)
                tr = dict(meta['transport'], n_vehicles=int(V))
                jls.append(np.asarray(jl))
                pts.append(np.asarray(pt, dtype=float))
                lags.append(np.asarray(meta['time_lag'], dtype=float))
                opt.append(np.asarray(meta['op_type']))
                mct.append(np.asarray(meta['mch_type']))
                lay.append(tr)
            env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1],
                                   use_lag_features=True, use_guide=guide)
            env.set_initial_data(jls, pts, lags, opt, mct, lay)
            env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
            ms = rollout(env, policy)
            for e, stem in enumerate(chunk):
                rec = env.schedule_record(e)
                res = validate_transport_schedule(
                    jls[e], pts[e], lags[e],
                    np.array(lay[e]['station_cell']),
                    np.array(lay[e]['tau_cells']), V,
                    int(lay[e]['veh_start_cell']), rec,
                    job_start_cell=int(lay[e]['job_start_cell']))
                assert res['feasible'], f'{stem} V={V}: {res["violations"][:3]}'
                idx = int(os.path.basename(stem).split('_')[1])
                order = np.argsort(np.asarray(rec['op_start']))
                veh_of_op = {t['op']: t['veh'] for t in rec['transports']}
                cum = np.cumsum(jls[e])
                seq = [[int(np.searchsorted(cum, int(o), side='right')),
                        int(veh_of_op[int(o)])] for o in order]
                with open(f'{out_dir}/replay_{idx:03d}.json', 'w') as f:
                    json.dump(dict(seed=910_000 + idx, num_jobs=15,
                                   num_machines=10, n_veh=V,
                                   ours_makespan_link_units=float(ms[e] - PSEUDO_PT),
                                   sequence=seq), f)
                ms_all.append(float(ms[e] - PSEUDO_PT))
            del env
        summary[f'v{V}'] = dict(n=len(ms_all), mean=float(np.mean(ms_all)),
                                std=float(np.std(ms_all, ddof=1)))
        print(f'V={V}: our-env mean {np.mean(ms_all):.1f} '
              f'std {np.std(ms_all, ddof=1):.1f} (link units)', flush=True)
    with open(f'{out_root}/summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'replays -> {out_root}')


if __name__ == '__main__':
    main()
