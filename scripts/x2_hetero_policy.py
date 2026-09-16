"""policy evaluation on the heterogeneity cells, CPU only.

Reproduces the protocol of scripts/eval_ppvct.py exactly (greedy decoding,
policy vehicle head, batch of 20 instances per environment build, every
schedule re-validated with validator_t) and adds one thing that script does
not record: the certified action prices the bound-guided policy actually
sees along the rollout.

Price instrumentation. With the guide channel on, the last channel of
fea_pairs is the certified price Delta_a of a machine assignment and the
last channel of fea_veh_pairs is the price of a transport task, both scaled
by env.inv_slope. At every decision event, for every alive environment, we
take the prices of the LEGAL candidates only (machine events: pairs not in
dynamic_pair_mask; vehicle events: active tasks) and record how many are
strictly positive and how large the positive ones are. A price of zero says
the commitment does not move the certified floor at all, so the channel
carries no ranking information for that candidate; the fraction of positive
prices is therefore the channel's informativeness. Prices are reported both
in the scaled units the network reads and in hours (channel / inv_slope).
The instrumentation is a read-only wrapper around the unmodified
environment: no environment or model file is touched.

The GPU belongs to another project. This script forces CPU and one torch
thread (a batch-of-20 forward pass on a small instance does not parallelize;
more threads collapse it on a contended box). No timing is reported.

Usage:
  python scripts/x2_hetero_policy.py --model_name 10x25+ppvct-mixed+m1-bcb-guide-s301 \
      --cells v1+t0.6 --levels base,2,5,10,20
Outputs: results/hetero/policy/{cell}+R{level}+{model}.json
"""

import argparse
import json
import os
import sys

cli = argparse.ArgumentParser()
cli.add_argument('--model_name', type=str, required=True)
cli.add_argument('--cells', type=str, default='v1+t0.6,v2+t0.6,v1+t1.0')
cli.add_argument('--levels', type=str, default='base,2,5,10,20')
cli.add_argument('--split', type=str, default='test')
cli.add_argument('--batch', type=int, default=20)
cli.add_argument('--limit', type=int, default=0, help='first N instances only')
cli.add_argument('--out_dir', type=str, default='results/hetero/policy')
A = cli.parse_args()
sys.argv = [sys.argv[0]]

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import glob
import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass                      # already initialized in this process

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


snap = load_snapshot(A.model_name)
configs.device = 'cpu'

from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.validator_t import validate_transport_schedule
from transport_marl.model_transport import DANIELTransport
from transport_marl.mappo import select_actions


class PriceRecorder:
    """Read-only accumulator over the guide channel of the live features."""

    def __init__(self):
        self.n_cand = 0
        self.n_pos = 0
        self.pos = []            # positive prices, scaled units
        self.n_events = 0
        self.n_events_any_pos = 0
        self.inv_slope = None

    def observe(self, env):
        self.inv_slope = float(env.inv_slope)
        gp = env.fea_pairs[:, :, :, -1]
        gt = env.fea_veh_pairs[:, :, -1]
        M = env.number_of_machines
        for e in range(env.number_of_envs):
            if env.n_realized[e] >= env.number_of_ops:
                continue          # done envs keep frozen features by design
            if env.event_type[e] == 0:
                legal = ~env.dynamic_pair_mask[e]
                vals = gp[e][legal]
            else:
                legal = env.task_active[e]
                vals = gt[e][legal]
            if vals.size == 0:
                continue
            self.n_events += 1
            self.n_cand += int(vals.size)
            p = vals[vals > 1e-12]
            self.n_pos += int(p.size)
            if p.size:
                self.n_events_any_pos += 1
                self.pos.append(p.astype(float))

    def summary(self):
        pos = np.concatenate(self.pos) if self.pos else np.zeros(0)
        s = self.inv_slope or 1.0
        out = dict(n_candidates=self.n_cand, n_positive=self.n_pos,
                   frac_positive=(self.n_pos / self.n_cand
                                  if self.n_cand else None),
                   n_events=self.n_events,
                   frac_events_with_any_positive=(
                       self.n_events_any_pos / self.n_events
                       if self.n_events else None),
                   inv_slope=s)
        if pos.size:
            out.update(
                mean_positive_scaled=float(pos.mean()),
                median_positive_scaled=float(np.median(pos)),
                p90_positive_scaled=float(np.percentile(pos, 90)),
                mean_positive_hours=float(pos.mean() / s),
                median_positive_hours=float(np.median(pos) / s),
                p90_positive_hours=float(np.percentile(pos, 90) / s))
        else:
            for k in ('mean_positive_scaled', 'median_positive_scaled',
                      'p90_positive_scaled', 'mean_positive_hours',
                      'median_positive_hours', 'p90_positive_hours'):
                out[k] = None
        return out


def greedy_rollout(env, policy, rec):
    state = env.state
    while env.done().min() < 1:
        if rec is not None:
            rec.observe(env)
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
        state, _, _ = env.step(a.cpu().numpy())
    return env.current_makespan.copy()


def build_policy(snap):
    algo = snap.get('algo', 'mappo')
    if algo == 'single':
        from transport_marl.single_agent import DANIELSingle
        return DANIELSingle(configs), algo
    if algo in ('mappo', 'coma'):
        return DANIELTransport(configs), algo
    raise ValueError(f"unknown algo '{algo}' for {A.model_name}")


def dataset_dir(cell, level):
    base = cell if 'x25+' in cell else f'10x25+ppvct-mixed+{cell}'
    if level == 'base':
        return f'data/PPVCT/{base}/{A.split}'
    return f'data/PPVCT_HET/{base}+hetR{level}/{A.split}'


def main():
    policy, algo = build_policy(snap)
    ckpt = f'trained_network/PPVCT/{A.model_name}.pth'
    sd = torch.load(ckpt, map_location='cpu')
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'checkpoint/architecture mismatch for '
                           f'{A.model_name} (algo={algo}): {len(missing)} '
                           f'missing, {len(unexpected)} unexpected')
    policy.eval()
    guide = bool(getattr(configs, 'guide', False))
    os.makedirs(A.out_dir, exist_ok=True)

    for cell in A.cells.split(','):
        for level in A.levels.split(','):
            out_path = f'{A.out_dir}/{cell}+R{level}+{A.model_name}.json'
            if os.path.exists(out_path):
                print(f'skip {cell} R={level} (exists)', flush=True)
                continue
            ds = dataset_dir(cell, level)
            stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
            if A.limit:
                stems = stems[:A.limit]
            assert stems, f'no instances under {ds}'
            rec = PriceRecorder() if guide else None
            names, ms_all = [], []
            for i0 in range(0, len(stems), A.batch):
                chunk = stems[i0:i0 + A.batch]
                insts = [load_instance(s) for s in chunk]
                jls = [np.asarray(x[0]) for x in insts]
                pts = [np.asarray(x[1], dtype=float) for x in insts]
                lags = [np.asarray(x[2]['time_lag'], dtype=float) for x in insts]
                opt = [np.asarray(x[2]['op_type']) for x in insts]
                mct = [np.asarray(x[2]['mch_type']) for x in insts]
                lay = [x[2]['transport'] for x in insts]
                env = FJSPEnvTransport(
                    len(jls[0]), pts[0].shape[1], use_lag_features=True,
                    use_guide=guide,
                    guide_price=snap.get('guide_price') or 'certified',
                    guide_price_scale=float(snap.get('guide_price_scale') or 1.0))
                env.set_initial_data(jls, pts, lags, opt, mct, lay)
                if guide:
                    from transport_marl.bound import TransportBound
                    env.attach_bound(TransportBound(env, use_mch=True,
                                                    use_veh=True))
                ms = greedy_rollout(env, policy, rec)
                for e, stem in enumerate(chunk):
                    res = validate_transport_schedule(
                        jls[e], pts[e], lags[e],
                        np.array(lay[e]['station_cell']),
                        np.array(lay[e]['tau_cells']),
                        int(lay[e]['n_vehicles']),
                        int(lay[e]['veh_start_cell']), env.schedule_record(e))
                    assert res['feasible'], \
                        f'{stem}: {res["violations"][:3]}'
                    names.append(os.path.basename(stem))
                    ms_all.append(float(ms[e]))
                del env
                import gc
                gc.collect()
            out = dict(model=A.model_name, algo=algo, cell=cell, level=level,
                       dataset=ds, n=len(names), veh_rule='policy',
                       decoding='greedy', batch=A.batch, guide=guide,
                       makespan={n: m for n, m in zip(names, ms_all)},
                       price=(rec.summary() if rec is not None else None))
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=1)
            arr = np.array(ms_all)
            pr = ''
            if rec is not None:
                s = rec.summary()
                pr = (f'  price frac+={s["frac_positive"]:.3f} '
                      f'mean+={s["mean_positive_hours"]:.2f}h')
            print(f'{cell} R={level:>4s} {A.model_name}: n={len(arr)} '
                  f'mean={arr.mean():.2f} std={arr.std(ddof=1):.2f}{pr}',
                  flush=True)


if __name__ == '__main__':
    main()
