"""Batch-1 decision latency and memory of the deployed size-mixture policy.

Measures, for the size-mixture checkpoints (seeds 301-303) on one cell per
instance size (10, 20, 50 and 80 modules), with ONE instance per rollout:
  (a) per-event decision latency, mean / median / p95 / max over every event
      of every rollout, in two timing conventions:
        fwd   policy forward pass plus greedy action selection, the same timed
              region as scripts/eval_ppvct.py (which produced LatCpuFour and
              LatMarlDec), followed by torch.cuda.synchronize() on the GPU;
        event fwd plus the action copy to the host and env.step(), i.e. the
              wall time from receiving an event to having the next state;
  (b) schedule time per instance: rollout only, and end to end (instance file
      read, environment and bound construction, rollout);
  (c) peak resident memory: ru_maxrss of a process that ran only that cell,
      plus VmHWM from /proc/self/status reset before every instance (and on the
      GPU, torch peak allocated / reserved device memory);
  (d) checkpoint file size and parameter count.

Every schedule is re-checked by validator_t, and its makespan is compared with
the stored batched greedy result of the same checkpoint and instance
(test_results/PPVCT for the 10-module cell, results/scaleup/policy for the
others). The stored rows come from batched GPU rollouts. Some states hold two
actions with exactly equal probability, and a change of batch size or device
can break such a tie the other way (50-module instance_000, seed 301: equal
top-2 probabilities at decision 21), after which the batch-1 schedule is a
different, equally valid greedy schedule. The comparison is therefore recorded
per instance, and --strict turns a mismatch into an error. The stored rows keep
no action sequence, so the exact decision where a stored run diverged cannot be
recovered without a lockstep replay of the stored batched rollout on its own
device; matched rollouts also contain exact ties, so a tie count is no
substitute and is not recorded.

Each cell runs in its own child process, so ru_maxrss is the peak of that
cell alone. Cores are pinned with os.sched_setaffinity and the thread count is
fixed before torch is imported.

Output: results/latency/x2_latency_memory_{device}{tag}.json

Usage:
  python scripts/x2_latency_memory.py --cores 4 --threads 1 --device cpu
  python scripts/x2_latency_memory.py --cores 4 --threads 1 --device cuda
  python scripts/x2_latency_memory.py --cores 22,23 --sizes 10,50 --n 1 \
      --strict --tag +sanity
"""

import os
import sys
import json
import time
import argparse
import platform
import resource
import subprocess

MODEL_STEM = 'mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix'
SEEDS = (301, 302, 303)
CELLS = {
    10: '10x25+ppvct-mixed+v1+t0.6',
    20: '20x25+ppvct-mixed+v1+t0.6',
    50: '50x25+ppvct-mixed+v2+t1.0',
    80: '80x25+ppvct-mixed+v3+t1.0',
}
WARMUP_FORWARDS = 10
MS_TOL = 1e-6

cli = argparse.ArgumentParser()
cli.add_argument('--cores', type=str, default='',
                 help='comma list or a-b range of CPU cores to pin to; '
                      'empty keeps the inherited affinity')
cli.add_argument('--threads', type=int, default=1,
                 help='torch intra-op threads; also sets OMP/MKL_NUM_THREADS')
cli.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
cli.add_argument('--n', type=int, default=3, help='instances per cell')
cli.add_argument('--sizes', type=str, default='10,20,50,80')
cli.add_argument('--seeds', type=str, default=','.join(map(str, SEEDS)))
cli.add_argument('--out', type=str, default='',
                 help='output JSON; default results/latency/'
                      'x2_latency_memory_{device}{tag}.json')
cli.add_argument('--tag', type=str, default='')
cli.add_argument('--strict', action='store_true',
                 help='fail when a makespan differs from the stored result')
cli.add_argument('--child', type=str, default='',
                 help='internal: run one cell and write its JSON here')
cli.add_argument('--size', type=int, default=0, help='internal: cell size')
args_cli = cli.parse_args()


def parse_cores(spec):
    cores = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-')
            cores.extend(range(int(lo), int(hi) + 1))
        else:
            cores.append(int(part))
    return sorted(set(cores))


def proc_status_kb(key):
    with open('/proc/self/status') as f:
        for ln in f:
            if ln.startswith(key + ':'):
                return int(ln.split()[1])
    return None


def reset_hwm():
    """Reset VmHWM to the current RSS (Linux >= 4.0, own process)."""
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
        return True
    except OSError:
        return False


def git_info():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    head = run(['git', 'rev-parse', 'HEAD'])
    dirty = run(['git', 'status', '--porcelain', '--untracked-files=no'])
    script_dirty = run(['git', 'status', '--porcelain', '--',
                        'scripts/x2_latency_memory.py'])
    return dict(commit=head, tree_dirty=bool(dirty),
                script_modified_or_untracked=bool(script_dirty))


# ------------------------------------------------------------------ child
def run_cell():
    cores = parse_cores(args_cli.cores)
    if cores:
        os.sched_setaffinity(0, cores)
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[k] = str(args_cli.threads)
    if args_cli.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.argv = [sys.argv[0]]

    import gc
    import numpy as np
    import torch
    torch.set_num_threads(args_cli.threads)
    torch.set_num_interop_threads(1)

    sys.path.insert(0, '.')
    from params import configs
    configs.device = args_cli.device
    # env state tensors take their device from configs at import time
    from ppvc_instance_generator import load_instance
    from transport_marl.fjsp_env_transport import FJSPEnvTransport
    from transport_marl.validator_t import validate_transport_schedule
    from transport_marl.model_transport import DANIELTransport
    from transport_marl.mappo import select_actions
    from transport_marl.bound import TransportBound

    arch_keys = ['fea_j_input_dim', 'fea_m_input_dim', 'n_op_types',
                 'n_mch_types', 'type_emb_dim', 'num_heads_OAB',
                 'num_heads_MAB', 'layer_fea_output_dim',
                 'num_mlp_layers_actor', 'hidden_dim_actor',
                 'num_mlp_layers_critic', 'hidden_dim_critic',
                 'dropout_prob', 'guide']
    device = torch.device(args_cli.device)
    cuda = args_cli.device == 'cuda'
    size = args_cli.size
    cell = CELLS[size]

    def load_policy(model_name):
        with open(f'train_log/PPVCT/config_{model_name}.json') as f:
            snap = json.load(f)
        assert snap.get('algo', 'mappo') == 'mappo', snap.get('algo')
        for k in arch_keys:
            if k in snap:
                v = snap[k]
                if isinstance(v, str) and v.startswith('['):
                    v = json.loads(v)
                setattr(configs, k, v)
        policy = DANIELTransport(configs)
        ckpt = f'trained_network/PPVCT/{model_name}.pth'
        sd = torch.load(ckpt, map_location=device)
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f'checkpoint/architecture mismatch for {model_name}: '
                f'{len(missing)} missing, {len(unexpected)} unexpected')
        policy.eval()
        info = dict(
            checkpoint=ckpt, checkpoint_bytes=os.path.getsize(ckpt),
            params=int(sum(p.numel() for p in policy.parameters())),
            param_bytes=int(sum(p.numel() * p.element_size()
                                for p in policy.parameters())),
            guide=bool(getattr(configs, 'guide', False)))
        return policy, snap, info

    def build_env(stem, snap):
        x = load_instance(stem)
        jl = np.asarray(x[0])
        pt = np.asarray(x[1], dtype=float)
        lag = np.asarray(x[2]['time_lag'], dtype=float)
        lay = x[2]['transport']
        guide = bool(getattr(configs, 'guide', False))
        env = FJSPEnvTransport(
            len(jl), pt.shape[1], use_lag_features=True, use_guide=guide,
            guide_price=snap.get('guide_price', 'certified'),
            guide_price_scale=float(snap.get('guide_price_scale', 1.0)))
        env.set_initial_data([jl], [pt], [lag],
                             [np.asarray(x[2]['op_type'])],
                             [np.asarray(x[2]['mch_type'])], [lay])
        if guide:
            env.attach_bound(TransportBound(
                env, use_mch=True,
                use_veh=bool(int(snap.get('bound_veh', 1)))))
        return env, (jl, pt, lag, lay)

    def act(policy, state):
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
        return a

    def stored_makespans(model_name):
        if size == 10:
            short = cell.split('+ppvct-mixed+')[1]
            p = (f'test_results/PPVCT/{short}/'
                 f'Result_greedy+{model_name}_{short}.npy')
            if not os.path.exists(p):
                return p, {}, {}
            arr = np.load(p)
            return p, {f'instance_{i:03d}': float(arr[i, 0])
                       for i in range(len(arr))}, {}
        p = f'results/scaleup/policy/{model_name}_{cell}.json'
        if not os.path.exists(p):
            return p, {}, {}
        rows = json.load(open(p))['rows']
        return (p, {k: float(v['ms']) for k, v in rows.items()},
                {k: int(v['decisions']) for k, v in rows.items()})

    def summarize(x):
        x = np.asarray(x) * 1000.0
        return dict(n=int(x.size), mean_ms=float(x.mean()),
                    median_ms=float(np.median(x)),
                    p95_ms=float(np.percentile(x, 95)),
                    max_ms=float(x.max()))

    ds = f'data/PPVCT/{cell}/test'
    stems = sorted(os.path.join(ds, f[:-4]) for f in os.listdir(ds)
                   if f.startswith('instance_') and f.endswith('.fjs'))
    assert len(stems) >= args_cli.n, f'{ds}: {len(stems)} instances'
    stems = stems[:args_cli.n]

    rss_after_import_kb = proc_status_kb('VmRSS')
    seeds_out = {}
    pooled_fwd, pooled_evt = [], []
    for seed in [int(s) for s in args_cli.seeds.split(',')]:
        model_name = f'{MODEL_STEM}-s{seed}'
        policy, snap, info = load_policy(model_name)
        ref_path, ref, ref_dec = stored_makespans(model_name)

        # untimed warm-up on the first instance's initial state
        env, _ = build_env(stems[0], snap)
        for _ in range(WARMUP_FORWARDS):
            act(policy, env.state)
        if cuda:
            torch.cuda.synchronize()
        del env
        gc.collect()

        rows = {}
        for stem in stems:
            name = os.path.basename(stem)
            gc.collect()
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            hwm_reset = reset_hwm()
            rss_before_kb = proc_status_kb('VmRSS')

            t_start = time.perf_counter()
            env, (jl, pt, lag, lay) = build_env(stem, snap)
            t_ready = time.perf_counter()
            state = env.state
            fwd, evt = [], []
            idle = 0
            while env.done().min() < 1:
                idle += int(env.n_realized[0] >= env.number_of_ops)
                t0 = time.perf_counter()
                a = act(policy, state)
                if cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                state, _, _ = env.step(a.cpu().numpy())
                t2 = time.perf_counter()
                fwd.append(t1 - t0)
                evt.append(t2 - t0)
            t_end = time.perf_counter()

            hwm_kb = proc_status_kb('VmHWM')
            ms = float(env.current_makespan[0])
            assert idle == 0, f'{stem}: {idle} steps after completion'
            res = validate_transport_schedule(
                jl, pt, lag, np.array(lay['station_cell']),
                np.array(lay['tau_cells']), int(lay['n_vehicles']),
                int(lay['veh_start_cell']), env.schedule_record(0))
            assert res['feasible'], f'{stem}: {res["violations"][:3]}'
            assert abs(res['makespan'] - ms) < MS_TOL, \
                f'{stem}: validator {res["makespan"]} != env {ms}'
            stored = ref.get(name)
            match = stored is not None and abs(stored - ms) < MS_TOL
            if args_cli.strict:
                assert stored is not None, f'no stored result in {ref_path}'
                assert match, (f'{model_name} {name}: batch-1 makespan {ms} '
                               f'!= stored {stored} ({ref_path})')
            row = dict(
                makespan=ms, stored_makespan=stored, matches_stored=match,
                events=len(fwd), stored_events=ref_dec.get(name),
                fwd=summarize(fwd), event=summarize(evt),
                rollout_s=float(t_end - t_ready),
                end_to_end_s=float(t_end - t_start),
                setup_s=float(t_ready - t_start),
                rss_before_kb=rss_before_kb, vm_hwm_kb=hwm_kb,
                vm_hwm_reset=hwm_reset)
            if cuda:
                row['cuda_max_allocated_bytes'] = int(
                    torch.cuda.max_memory_allocated())
                row['cuda_max_reserved_bytes'] = int(
                    torch.cuda.max_memory_reserved())
            rows[name] = row
            pooled_fwd.extend(fwd)
            pooled_evt.extend(evt)
            print(f'  s{seed} {name}: ms={ms:.3f} stored={stored} '
                  f'match={match} events={len(fwd)} '
                  f'fwd mean={row["fwd"]["mean_ms"]:.2f} '
                  f'p95={row["fwd"]["p95_ms"]:.2f} ms  '
                  f'event mean={row["event"]["mean_ms"]:.2f} '
                  f'p95={row["event"]["p95_ms"]:.2f} ms  '
                  f'e2e={row["end_to_end_s"]:.2f}s '
                  f'VmHWM={hwm_kb / 1024:.0f}MB', flush=True)
            del env
        seeds_out[str(seed)] = dict(model=model_name, stored_source=ref_path,
                                    **info, rows=rows)
        del policy
        gc.collect()

    all_rows = [r for s in seeds_out.values() for r in s['rows'].values()]
    out = dict(
        size=size, cell=cell, instances=[os.path.basename(s) for s in stems],
        torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        affinity=sorted(os.sched_getaffinity(0)),
        cuda_device=(torch.cuda.get_device_name(0) if cuda else None),
        warmup_forwards_per_seed=WARMUP_FORWARDS,
        fwd=summarize(pooled_fwd), event=summarize(pooled_evt),
        end_to_end_s_mean=float(np.mean([r['end_to_end_s']
                                         for r in all_rows])),
        rollout_s_mean=float(np.mean([r['rollout_s'] for r in all_rows])),
        events_mean=float(np.mean([r['events'] for r in all_rows])),
        all_match_stored=all(r['matches_stored'] for r in all_rows),
        n_match_stored=sum(bool(r['matches_stored']) for r in all_rows),
        n_rollouts=len(all_rows),
        rss_after_import_kb=rss_after_import_kb,
        vm_hwm_kb_max=max(r['vm_hwm_kb'] for r in all_rows),
        ru_maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        seeds=seeds_out)
    if cuda:
        out['cuda_max_allocated_bytes'] = max(
            r['cuda_max_allocated_bytes'] for r in all_rows)
        out['cuda_max_reserved_bytes'] = max(
            r['cuda_max_reserved_bytes'] for r in all_rows)
    with open(args_cli.child, 'w') as f:
        json.dump(out, f, indent=1)


# ----------------------------------------------------------------- parent
def main():
    cores = parse_cores(args_cli.cores)
    sizes = [int(s) for s in args_cli.sizes.split(',')]
    for s in sizes:
        assert s in CELLS, f'no cell for size {s}; known: {sorted(CELLS)}'
    out_path = args_cli.out or (f'results/latency/x2_latency_memory_'
                                f'{args_cli.device}{args_cli.tag}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(out_path), '.cells')
    os.makedirs(tmp_dir, exist_ok=True)

    meta = dict(
        script='scripts/x2_latency_memory.py', model_stem=MODEL_STEM,
        seeds=[int(s) for s in args_cli.seeds.split(',')],
        hostname=platform.node(), device=args_cli.device, batch=1,
        cores=cores or sorted(os.sched_getaffinity(0)),
        cores_pinned=bool(cores), torch_threads=args_cli.threads,
        omp_mkl_threads=args_cli.threads, n_per_cell=args_cli.n,
        strict=args_cli.strict, git=git_info(),
        started=time.strftime('%Y-%m-%d %H:%M:%S'),
        loadavg_start=list(os.getloadavg()),
        conventions=dict(
            fwd='policy forward + greedy action selection (the timed region '
                'of scripts/eval_ppvct.py), plus cuda synchronize on GPU',
            event='fwd + action copy to host + env.step (next state ready)',
            end_to_end_s='instance file read + env and bound construction + '
                         'rollout; excludes validation',
            rollout_s='first forward to final env.step',
            memory='ru_maxrss of a child process that ran only this cell '
                   '(model load, warm-up, all seeds); vm_hwm_kb is VmHWM '
                   'reset before each instance'),
        cells={})
    print(f'[x2_latency_memory] device={args_cli.device} cores={meta["cores"]} '
          f'threads={args_cli.threads} n={args_cli.n} sizes={sizes} '
          f'load={meta["loadavg_start"]}', flush=True)

    for size in sizes:
        cell_json = os.path.join(tmp_dir, f'{args_cli.device}{args_cli.tag}'
                                          f'_{size}.json')
        if os.path.exists(cell_json):
            os.remove(cell_json)
        cmd = [sys.executable, '-u', os.path.abspath(__file__),
               '--cores', args_cli.cores, '--threads', str(args_cli.threads),
               '--device', args_cli.device, '--n', str(args_cli.n),
               '--seeds', args_cli.seeds, '--child', cell_json,
               '--size', str(size)]
        if args_cli.strict:
            cmd.append('--strict')
        print(f'{CELLS[size]}:', flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(f'cell {CELLS[size]} failed (rc={rc})')
        c = json.load(open(cell_json))
        os.remove(cell_json)
        meta['cells'][str(size)] = c
        print(f'  -> fwd mean={c["fwd"]["mean_ms"]:.2f} '
              f'p95={c["fwd"]["p95_ms"]:.2f} ms, event '
              f'mean={c["event"]["mean_ms"]:.2f} '
              f'p95={c["event"]["p95_ms"]:.2f} ms, '
              f'e2e={c["end_to_end_s_mean"]:.2f} s, '
              f'ru_maxrss={c["ru_maxrss_kb"] / 1024:.0f} MB, '
              f'VmHWM={c["vm_hwm_kb_max"] / 1024:.0f} MB, '
              f'match_stored={c["n_match_stored"]}/{c["n_rollouts"]} '
              f'[{time.time() - t0:.0f}s]', flush=True)

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    meta['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    meta['loadavg_end'] = list(os.getloadavg())
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'-> {out_path}  load end={meta["loadavg_end"]}', flush=True)


if __name__ == '__main__':
    if args_cli.child:
        run_cell()
    else:
        main()
