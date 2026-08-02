"""P2 joint MAPPO v1 training (shared Theorem-1 telescoped team reward).

Regime-mixed per Appendix B / §7: each update draws one cell of the
fleet x travel-intensity grid (round-robin), builds num_envs fresh on-the-fly
instances there, rolls out with the two-head policy, and does one PPO update.
Greedy validation on the fixed PPVCT vali split every `validate_timestep`
updates; best mean-of-cell-means checkpoint kept.

Usage (inside tmux):
  python -u scripts/p2_train_mappo.py --model_suffix joint-v1 --seed 301 \
      --max_updates 2000
"""

import sys, os, json, time, argparse, glob

cli = argparse.ArgumentParser()
cli.add_argument('--fleet_grid', type=str, default='1,2,3')
cli.add_argument('--ratio_grid', type=str, default='0.1,0.3,0.6')
cli.add_argument('--max_updates', type=int, default=2000)
cli.add_argument('--seed', type=int, default=301)
cli.add_argument('--model_suffix', type=str, default='joint-v1')
cli.add_argument('--vali_subset', type=int, default=20)
cli.add_argument('--reward', type=str, default='full', choices=['full', 'chain'])
cli.add_argument('--credit', type=str, default='shared',
                 choices=['shared', 'm1', 'm2'])
cli.add_argument('--algo', type=str, default='mappo',
                 choices=['mappo', 'coma', 'single'])
cli.add_argument('--fixed_veh_rule', type=str, default='none',
                 choices=['none', 'NVF', 'STT', 'FIFO'],
                 help='E0b arm: vehicle events follow this rule; those steps '
                      'stay in GAE/value but are masked from the policy loss')
cli.add_argument('--guide', action='store_true',
                 help='bound-guided action prior: per-candidate certified '
                      'price tags as an extra channel on both pair grids')
cli.add_argument('--dist', type=str, default='ppvc', choices=['ppvc', 'link'],
                 help='instance distribution: ppvc (default) or the certified '
                      'Link JSSPT port (external anchor L1); with link, '
                      'fleet_grid lists AGV counts and ratio_grid is ignored')
cli.add_argument('--link_jobs', type=int, default=15)
cli.add_argument('--link_machines', type=int, default=10)
cli.add_argument('--vali_cells', type=str, default='',
                 help='comma list like v1+t0.6,v2+t0.6; default = grid cells')
cli.add_argument('--n_modules', type=int, default=10)
cli.add_argument('--vali_every', type=int, default=20)
args_cli = cli.parse_args()
sys.argv = [sys.argv[0]]

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')   # link_external import (L1 external arm)
from params import configs
# device must be pinned BEFORE the env modules import: EnvState.device is a
# class attribute bound at import time from configs.device
configs.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
from ppvc_instance_generator import ppvc_instance_generator, load_instance
from transport_marl.layout import build_transport_layout
from transport_marl.fjsp_env_transport import FJSPEnvTransport
from transport_marl.bound import TransportBound
from transport_marl.mappo import TransportMemory, TransportPPO, select_actions
from transport_marl.bcb import r_pass as bcb_r_pass
from transport_marl.pdr_pairs import env_veh_rule_action


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def make_env(instances, n_veh, ratio):
    jls, pts, lags, opt, mct, lay = [], [], [], [], [], []
    for jl, pt, meta in instances:
        layout = meta['transport'] if 'transport' in meta else \
            build_transport_layout(jl, pt, np.asarray(meta['mch_type']), ratio, n_veh)
        jls.append(np.asarray(jl)); pts.append(np.asarray(pt, dtype=float))
        lags.append(np.asarray(meta['time_lag'], dtype=float))
        opt.append(np.asarray(meta['op_type'])); mct.append(np.asarray(meta['mch_type']))
        lay.append(layout)
    env = FJSPEnvTransport(len(jls[0]), pts[0].shape[1], use_lag_features=True,
                           use_guide=args_cli.guide)
    env.set_initial_data(jls, pts, lags, opt, mct, lay)
    if args_cli.reward == 'full':
        env.attach_bound(TransportBound(env, use_mch=True, use_veh=True))
    return env


def policy_forward(policy, state):
    return policy(
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


def rollout(env, ppo, mem, device, greedy=False, hard_cap_mult=8):
    state = env.reset()
    cap = hard_cap_mult * env.number_of_ops + 1000
    steps = 0
    while env.done().min() < 1:
        alive = torch.from_numpy(env.n_realized < env.number_of_ops).to(device)
        with torch.no_grad():
            pi_m, pi_v, v = policy_forward(
                ppo.policy_old if not greedy else ppo.policy, state)
            a, logp = select_actions(pi_m, pi_v, state.event_type_tensor,
                                     greedy=greedy)
        a_np = a.cpu().numpy()
        polmask = np.ones(env.number_of_envs, dtype=bool)
        if args_cli.fixed_veh_rule != 'none':
            from transport_marl.mappo import eval_per_row
            for e in range(env.number_of_envs):
                if env.event_type[e] == 1 and env.n_realized[e] < env.number_of_ops:
                    a_np[e] = env_veh_rule_action(env, e, args_cli.fixed_veh_rule)
                    polmask[e] = False
            a = torch.from_numpy(a_np).to(a.device)
            # logp must match the EXECUTED action (rule-overridden rows would
            # otherwise blow up the PPO ratio: exp(logp_new - wrong_logp_old)
            # can overflow before the policy mask zeroes it)
            logp, _ = eval_per_row(pi_m, pi_v, state.event_type_tensor, a)
        if mem is not None:
            mem.push(state, alive)
            mem.action_seq.append(a)
            mem.log_probs.append(logp)
            mem.val_seq.append(v.squeeze(1))
            mem.polmask_seq.append(torch.from_numpy(polmask).to(device))
            if args_cli.credit in ('m1', 'm2'):
                mem.rpass_seq.append(
                    torch.from_numpy(bcb_r_pass(env)).float().to(device))
        state, r, done = env.step(a_np)
        if mem is not None:
            mem.reward_seq.append(torch.from_numpy(r).float().to(device))
            mem.done_seq.append(torch.from_numpy(done).float().to(device))
        steps += 1
        if steps > cap:
            raise RuntimeError('rollout exceeded step cap')
    return env.current_makespan.copy()


def main():
    fleets = [int(x) for x in args_cli.fleet_grid.split(',')]
    ratios = [float(x) for x in args_cli.ratio_grid.split(',')]
    if args_cli.dist == 'link':
        cells = [(v, None) for v in fleets]     # AGV-count cells only
        model_name = (f'{args_cli.link_jobs}x{args_cli.link_machines}'
                      f'+link+{args_cli.model_suffix}-s{args_cli.seed}')
    else:
        cells = [(v, r) for v in fleets for r in ratios]
        model_name = f'{args_cli.n_modules}x25+ppvct-mixed+{args_cli.model_suffix}-s{args_cli.seed}'

    configs.fea_j_input_dim = 14
    configs.fea_m_input_dim = 10
    if args_cli.dist == 'link':
        configs.n_op_types = 2      # real / out-buf delivery
        configs.n_mch_types = 2     # real / per-job pseudo
    else:
        configs.n_op_types = 5
        configs.n_mch_types = 9
    configs.guide = args_cli.guide   # model pair-head dims read this
    device = torch.device(configs.device)
    setup_seed(args_cli.seed)

    os.makedirs('train_log/PPVCT', exist_ok=True)
    os.makedirs('trained_network/PPVCT', exist_ok=True)
    snap = {k: v for k, v in vars(configs).items()}
    # The snapshot must carry the ARM IDENTITY, not just the architecture
    # hyper-parameters: eval_ppvct.py rebuilds the policy from this file and
    # cannot otherwise know which network class to instantiate (a single-agent
    # checkpoint loaded into the MARL class fails or, worse, loads silently).
    snap.update(dict(model_name=model_name, transport=True,
                     veh_fea_dim=4, veh_pair_dim=6,
                     fleet_grid=fleets, ratio_grid=ratios,
                     reward=args_cli.reward, seed_train=args_cli.seed,
                     max_updates=args_cli.max_updates,
                     algo=args_cli.algo, credit=args_cli.credit,
                     fixed_veh_rule=args_cli.fixed_veh_rule,
                     guide=args_cli.guide, dist=args_cli.dist,
                     link_jobs=args_cli.link_jobs,
                     link_machines=args_cli.link_machines))
    with open(f'train_log/PPVCT/config_{model_name}.json', 'w') as f:
        json.dump(snap, f, indent=1, default=str)

    if args_cli.algo == 'coma':
        from transport_marl.coma_baseline import COMAPPO
        ppo = COMAPPO(configs)
    elif args_cli.algo == 'single':
        from transport_marl.single_agent import SinglePPO
        ppo = SinglePPO(configs)
    else:
        ppo = TransportPPO(configs)

    # fixed vali envs (subset per cell), built once and reset each pass
    vali_envs = []
    if args_cli.dist == 'link':
        # one shared vali instance set; fleet size varied per cell by
        # overriding the recorded n_vehicles
        ds = f'data/LINK/{args_cli.link_jobs}x{args_cli.link_machines}/vali'
        stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
        stems = stems[:args_cli.vali_subset]
        if not stems:
            print(f'[warn] no vali data at {ds} (generate first)')
        else:
            for (v, _) in cells:
                insts = []
                for s in stems:
                    jl, pt, meta = load_instance(s)
                    meta = dict(meta)
                    meta['transport'] = dict(meta['transport'], n_vehicles=int(v))
                    insts.append((jl, pt, meta))
                vali_envs.append((f'v{v}', make_env(insts, None, None)))
    else:
        if args_cli.vali_cells:
            vali_cell_dirs = [(c, f'data/PPVCT/10x25+ppvct-mixed+{c}/vali')
                              for c in args_cli.vali_cells.split(',')]
        else:
            vali_cell_dirs = [((v, r), f'data/PPVCT/10x25+ppvct-mixed+v{v}+t{r}/vali')
                              for (v, r) in cells]
        for tag, ds in vali_cell_dirs:
            stems = sorted(g[:-4] for g in glob.glob(f'{ds}/instance_*.fjs'))
            stems = stems[:args_cli.vali_subset]
            if not stems:
                print(f'[warn] no vali data at {ds} (generate first)'); continue
            insts = [load_instance(s) for s in stems]
            vali_envs.append((tag, make_env(insts, None, None)))

    best_score = np.inf
    log_path = f'train_log/PPVCT/train_{model_name}.log.jsonl'
    t0 = time.time()
    for upd in range(args_cli.max_updates):
        v, r = cells[upd % len(cells)]
        seed0 = args_cli.seed * 1_000_000 + upd * configs.num_envs
        insts = []
        for e in range(configs.num_envs):
            if args_cli.dist == 'link':
                from link_external import gen_link_instance, link_to_env_inputs
                jobs, tt = gen_link_instance(
                    args_cli.link_jobs, args_cli.link_machines, seed0 + e)
                jl, pt, meta = link_to_env_inputs(jobs, tt, v)
            else:
                jl, pt, meta = ppvc_instance_generator(
                    n_modules=args_cli.n_modules, class_mix='mixed', seed=seed0 + e)
            insts.append((jl, pt, meta))
        env = make_env(insts, v, r)
        mem = TransportMemory(configs.gamma, configs.gae_lambda)
        ms = rollout(env, ppo, mem, device, greedy=False)
        loss, vloss = ppo.update(mem, credit=args_cli.credit)
        rec = dict(update=upd, cell=[v, r], train_ms=float(ms.mean()),
                   loss=loss, vloss=vloss, wall=round(time.time() - t0, 1))

        if (upd + 1) % args_cli.vali_every == 0 and vali_envs:
            per_cell = {}
            for tag, venv in vali_envs:
                vms = rollout(venv, ppo, None, device, greedy=True)
                key = f'v{tag[0]}t{tag[1]}' if isinstance(tag, tuple) else tag
                per_cell[key] = float(vms.mean())
            score = float(np.mean(list(per_cell.values())))
            rec['vali'] = per_cell
            rec['vali_score'] = score
            if score < best_score:
                best_score = score
                torch.save(ppo.policy.state_dict(),
                           f'trained_network/PPVCT/{model_name}.pth')
                rec['saved'] = 'best'
            torch.save(ppo.policy.state_dict(),
                       f'trained_network/PPVCT/{model_name}-last.pth')
        with open(log_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        if upd % 10 == 0:
            print(f'[{upd}/{args_cli.max_updates}] cell=({v},{r}) '
                  f'train_ms={ms.mean():.1f} loss={loss:.4f} '
                  f'best_vali={best_score if best_score < np.inf else -1:.1f} '
                  f'wall={rec["wall"]}s', flush=True)

    print(f'done. best vali score {best_score:.2f}')


if __name__ == '__main__':
    main()
