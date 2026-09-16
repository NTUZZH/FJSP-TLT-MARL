"""Magnitude calibration for the non-admissible price control (Paper X2 arm c).

The control arm replaces the certified
Theorem-1 action price by a myopic duration price (guide.naive_price_features).
The arm only tests ADMISSIBILITY if the two channels have comparable magnitude;
otherwise it is a feature-scale ablation wearing an admissibility label. This
script measures both channels on the same rollout and reports the constant

    C = mean(certified) / mean(naive)

to be passed as --guide_price_scale C on the arm's training line.

Protocol (design 2.5):
  1. one 20-instance batch at the mid training cell (v2+t0.3), built exactly as
     scripts/p2_train_mappo.py:make_env does, with the full Theorem-1 bound
     attached (TransportBound(use_mch=True, use_veh=True));
  2. greedy rollout with the headline guide checkpoint
     trained_network/PPVCT/10x25+ppvct-mixed+m1-bcb-guide-s301.pth; at every
     step, record the certified channel actually fed to the network
     (fea_pairs[..., -1] / fea_veh_pairs[..., -1]) and the naive channel
     computed on the same state with guide_price_scale = 1.0;
  3. statistics are pooled over UNMASKED entries only, i.e. exactly the
     entries the two actor heads' softmaxes see: ~dynamic_pair_mask on the
     machine grid, ~veh_action_mask on the vehicle grid, alive envs only;
  4. acceptance: with C applied, the two channel means agree within 10% and
     the two standard deviations within a factor of 2.

CPU only; the GPU is reserved for training. Run pinned, e.g.
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 12-23 \
      python -u scripts/x2_calibrate_naive_price.py
"""

import argparse
import glob
import os
import sys

cli = argparse.ArgumentParser()
cli.add_argument('--cell', type=str, default='v2+t0.3',
                 help='training cell whose fixed instances are rolled out')
cli.add_argument('--split', type=str, default='vali')
cli.add_argument('--n_inst', type=int, default=20)
cli.add_argument('--model_name', type=str,
                 default='10x25+ppvct-mixed+m1-bcb-guide-s301')
args_cli = cli.parse_args()
sys.argv = [sys.argv[0]]

import json
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
configs.device = 'cpu'          # calibration never touches the GPU

from ppvc_instance_generator import load_instance
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound
from transport_marl.guide import naive_price_features
from transport_marl.model_transport import DANIELTransport
from transport_marl.mappo import select_actions


class Acc:
    """Exact running mean/std over an arbitrarily long stream."""

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.nz = 0
        self.hi = 0.0

    def add(self, x):
        x = np.asarray(x, dtype=float).ravel()
        self.n += x.size
        self.s += float(x.sum())
        self.ss += float((x * x).sum())
        self.nz += int((x > 0).sum())
        if x.size:
            self.hi = max(self.hi, float(x.max()))

    @property
    def mean(self):
        return self.s / self.n if self.n else float('nan')

    @property
    def std(self):
        if self.n < 2:
            return float('nan')
        return float(np.sqrt(max(self.ss / self.n - self.mean ** 2, 0.0)))

    @property
    def frac_zero(self):
        return 1.0 - self.nz / self.n if self.n else float('nan')

    @property
    def mean_nonzero(self):
        return self.s / self.nz if self.nz else 0.0


def build_env():
    ds = f'data/PPVCT/10x25+ppvct-mixed+{args_cli.cell}/{args_cli.split}'
    stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
    stems = stems[:args_cli.n_inst]
    assert stems, f'no instances at {ds}'
    insts = [load_instance(s) for s in stems]
    jls = [np.asarray(x[0]) for x in insts]
    pts = [np.asarray(x[1], dtype=float) for x in insts]
    lags = [np.asarray(x[2]['time_lag'], dtype=float) for x in insts]
    opt = [np.asarray(x[2]['op_type']) for x in insts]
    mct = [np.asarray(x[2]['mch_type']) for x in insts]
    lay = [x[2]['transport'] for x in insts]
    # same construction as p2_train_mappo.make_env for a --guide arm
    env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1], use_lag_features=True,
                           use_guide=True, guide_price='certified',
                           guide_price_scale=1.0)
    env.set_initial_data(jls, pts, lags, opt, mct, lay)
    env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
    return env, len(stems)


def main():
    env, n_inst = build_env()
    policy = DANIELTransport(configs)
    ckpt = f'trained_network/PPVCT/{args_cli.model_name}.pth'
    sd = torch.load(ckpt, map_location='cpu')
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'checkpoint/architecture mismatch for {ckpt}: '
                           f'{len(missing)} missing, {len(unexpected)} unexpected')
    policy.eval()
    n_par = sum(p.numel() for p in policy.parameters())
    print(f'checkpoint {os.path.basename(ckpt)}: {n_par} parameters, '
          f'guide={getattr(configs, "guide", False)}', flush=True)

    cert_all, naive_all = Acc(), Acc()
    cert_pair, naive_pair = Acc(), Acc()
    cert_veh, naive_veh = Acc(), Acc()
    state = env.state
    steps = 0
    while env.done().min() < 1:
        alive = env.n_realized < env.number_of_ops
        # certified channel: read the value the network is actually fed
        c_pair = env.fea_pairs[:, :, :, -1]
        c_veh = env.fea_veh_pairs[:, :, -1]
        # naive channel on the same state, unscaled (guide_price_scale = 1.0)
        n_pair, n_veh = naive_price_features(env)
        pm = (~env.dynamic_pair_mask) & alive[:, None, None]
        vm = (~env.veh_action_mask) & alive[:, None]
        if pm.any():
            cert_pair.add(c_pair[pm]); naive_pair.add(n_pair[pm])
            cert_all.add(c_pair[pm]); naive_all.add(n_pair[pm])
        if vm.any():
            cert_veh.add(c_veh[vm]); naive_veh.add(n_veh[vm])
            cert_all.add(c_veh[vm]); naive_all.add(n_veh[vm])
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
        steps += 1

    def sig3(x):
        if x == 0 or not np.isfinite(x):
            return x
        from math import floor, log10
        return round(x, -int(floor(log10(abs(x)))) + 2)

    c = sig3(cert_all.mean / naive_all.mean)
    print(f'\ncell={args_cli.cell} split={args_cli.split} instances={n_inst} '
          f'greedy steps={steps} makespan mean={env.current_makespan.mean():.2f}')
    print('\nchannel statistics over UNMASKED entries (scale 1.0)')
    print(f'{"grid":<10}{"n":>12}{"mean cert":>14}{"mean naive":>14}'
          f'{"std cert":>14}{"std naive":>14}')
    for name, a1, a2 in (('pair', cert_pair, naive_pair),
                         ('vehicle', cert_veh, naive_veh),
                         ('pooled', cert_all, naive_all)):
        print(f'{name:<10}{a1.n:>12d}{a1.mean:>14.6f}{a2.mean:>14.6f}'
              f'{a1.std:>14.6f}{a2.std:>14.6f}')
    print('\nshape of the two channels (pooled, scale 1.0)')
    print(f'{"":<10}{"frac == 0":>14}{"mean if > 0":>14}{"max":>14}')
    for name, a in (('certified', cert_all), ('naive', naive_all)):
        print(f'{name:<10}{a.frac_zero:>14.4f}{a.mean_nonzero:>14.6f}'
              f'{a.hi:>14.6f}')

    m_after = naive_all.mean * c
    s_after = naive_all.std * c
    rel = abs(m_after - cert_all.mean) / cert_all.mean
    ratio = s_after / cert_all.std if cert_all.std > 0 else float('inf')
    print(f'\nC = mean(certified) / mean(naive) = {cert_all.mean:.6f} / '
          f'{naive_all.mean:.6f} = {cert_all.mean / naive_all.mean:.6f} '
          f'-> {c} (3 s.f.)')
    print(f'after applying C: mean naive = {m_after:.6f} vs certified '
          f'{cert_all.mean:.6f}  (relative gap {100 * rel:.2f}%, limit 10%)')
    print(f'                  std  naive = {s_after:.6f} vs certified '
          f'{cert_all.std:.6f}  (ratio {ratio:.3f}, limit 0.5-2.0)')
    ok_mean = rel <= 0.10
    ok_std = 0.5 <= ratio <= 2.0
    print(f'\nACCEPTANCE: mean {"PASS" if ok_mean else "FAIL"}, '
          f'std {"PASS" if ok_std else "FAIL"}')
    print(f'queue flag: --guide_price naive --guide_price_scale {c}')
    if not ok_mean:
        raise SystemExit('calibration failed the mean-magnitude acceptance test')
    if not ok_std:
        print('[warn] the two channels differ in spread by more than a factor '
              'of 2; report this next to the arm rather than hiding it.')


if __name__ == '__main__':
    main()
