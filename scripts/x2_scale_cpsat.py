"""Strengthened warm-started CP-SAT references for the scale-up pilot cells.

The per-instance procedure is copied verbatim from scripts/p2_cpsat_refs_v2.py,
which is what the main reference queue (scripts/run_cpsat_v2_queue.sh) runs:
nine PDR pairs are simulated, the best of the nine becomes an AddHint warm
start, and solve_transport_instance is called with strengthen=True (horizon =
warm-start makespan, redundant fleet cumulative, vehicle symmetry breaking),
energy_cut=True, fleet=True, time_limit=300 s, num_search_workers=4. Only the
dataset list and the output directory differ, so results/cpsat_v2/ is never
written to and nothing can be swept into the manuscript's existing analysis.

Output: results/scaleup/cpsat/{cell}.jsonl, one JSON record per instance,
appended and fsynced as each solve returns. The ledger is keyed by instance
name and read back on start, so this script is idempotent: killing and
restarting it loses at most the solves in flight and never redoes finished
work. A companion {cell}.json (dict keyed by instance) is rewritten from the
ledger via tmp + atomic rename after every solve.

Wall-clock warning: the 300 s budget is wall time, so a contended box weakens
this arm. Run only on cores the main CP-SAT queue has yielded; the supervisor
scripts/run_x2_scale_cpsat.sh enforces that gate.

ANYTIME MODE (--anytime, Phase B). With the flag the solve additionally records
its whole upper-bound trajectory: a CpSolverSolutionCallback appends
[wall_s, ub, lb] at every improving incumbent, and the list lands in the ledger
record as 'anytime'. One 3600 s solve then yields the entire curve, so the
question "how long does exact search need to reach policy quality" is answered
without a ladder of separate budgets. The model, the parameters, the warm start
and the worker count are untouched; only a callback is attached. The flag is
OFF by default, so a Phase A invocation of this script is unchanged.

Usage:
  python -u scripts/x2_scale_cpsat.py CELL [CELL ...] [--par 3] [--workers 4]
         [--time 300] [--cores 12-23] [--chunk 12]
  python -u scripts/x2_scale_cpsat.py CELL --time_limit 3600 --anytime \
         --out_dir results/scaleup/cpsat_b
"""
import sys, os, json, glob, time
from multiprocessing import Pool

ARGS = [a for a in sys.argv[1:]]
sys.argv = [sys.argv[0]]
sys.path.insert(0, '.')

PAR, WORKERS, TLIM, CORES = 3, 4, 300.0, None
ENERGY, FLEET, CHUNK, LIMIT_N = True, True, None, None
ANYTIME = False
OUT_DIR = 'results/scaleup/cpsat'
cells = []
i = 0
while i < len(ARGS):
    if ARGS[i] == '--chunk':
        CHUNK = int(ARGS[i + 1]); i += 2
    elif ARGS[i] == '--n':
        LIMIT_N = int(ARGS[i + 1]); i += 2
    elif ARGS[i] == '--par':
        PAR = int(ARGS[i + 1]); i += 2
    elif ARGS[i] == '--workers':
        WORKERS = int(ARGS[i + 1]); i += 2
    elif ARGS[i] in ('--time', '--time_limit'):
        TLIM = float(ARGS[i + 1]); i += 2
    elif ARGS[i] == '--cores':
        CORES = ARGS[i + 1]; i += 2
    elif ARGS[i] == '--out_dir':
        OUT_DIR = ARGS[i + 1]; i += 2
    elif ARGS[i] == '--anytime':
        ANYTIME = True; i += 1
    else:
        cells.append(ARGS[i]); i += 1

# thread caps must be set before the numerical runtimes are imported: an
# affinity mask alone does not stop MKL/OMP sizing their pool from nproc.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = str(WORKERS)

import numpy as np
from ppvc_instance_generator import load_instance
from transport_marl.sim_single import TransportSim
from transport_marl.pdr_pairs import MCH_RULES, VEH_RULES, run_pdr_pair
from transport_marl.cpsat_transport import solve_transport_instance

# --------------------------------------------------------------------------- #
# Anytime instrumentation.
#
# solve_transport_instance takes no callback argument, and transport_marl/ is
# frozen for this campaign, so the callback is attached by shimming the
# cp_model handle that module resolves at solve time. The shim forwards every
# attribute to the real ortools module and overrides exactly one: CpSolver,
# whose Solve() attaches the recording callback. Nothing else about the model
# or the parameters changes, and with --anytime absent the shim is never
# installed.
# --------------------------------------------------------------------------- #
TRACE = []


def _install_anytime_shim():
    import transport_marl.cpsat_transport as _ct
    from ortools.sat.python import cp_model as _cp

    class _Cb(_cp.CpSolverSolutionCallback):
        def on_solution_callback(self):
            try:
                lb = self.BestObjectiveBound() / _ct.SCALE
            except Exception:
                lb = None
            TRACE.append([round(self.WallTime(), 3),
                          self.ObjectiveValue() / _ct.SCALE, lb])

    class _TracingCpSolver(_cp.CpSolver):
        def Solve(self, model, solution_callback=None):
            return _cp.CpSolver.Solve(self, model, _Cb())

    class _Shim:
        CpSolver = _TracingCpSolver

        def __getattr__(self, name):
            return getattr(_cp, name)

    _ct.cp_model = _Shim()


if ANYTIME:
    _install_anytime_shim()


def _pin(core_list):
    try:
        os.sched_setaffinity(0, core_list)
    except OSError:
        pass


def _worker_init(cfg):
    """Spawn-mode worker bootstrap.

    Under the 'spawn' start method the child re-imports this module with its
    own argv, so every CLI-derived global silently falls back to its default
    (TLIM 300, ANYTIME False, ...). That would be an invisible protocol
    drift: children would solve at the wrong budget and drop the anytime
    trace. This initializer overwrites the child's globals from the parent's
    resolved configuration and installs the anytime shim explicitly.
    Spawn (rather than fork) is used because forking a parent that holds
    ortools/absl locks intermittently deadlocks workers at birth, which is
    the 0%-CPU hang observed on the 80-module campaign.
    """
    global PAR, WORKERS, TLIM, CORES, ENERGY, FLEET, ANYTIME, OUT_DIR
    PAR = cfg['par']; WORKERS = cfg['workers']; TLIM = cfg['tlim']
    CORES = cfg['cores']; ENERGY = cfg['energy']; FLEET = cfg['fleet']
    ANYTIME = cfg['anytime']; OUT_DIR = cfg['out_dir']
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[_v] = str(WORKERS)
    if ANYTIME:
        _install_anytime_shim()


def solve_one(task):
    cell, stem, slot = task
    t0_all = time.time()
    if CORES:
        lo, hi = (int(x) for x in CORES.split('-'))
        pool_cores = list(range(lo, hi + 1))
        mine = pool_cores[slot * WORKERS:(slot + 1) * WORKERS]
        if mine:
            _pin(mine)
    jl, pt, meta = load_instance(stem)
    tr = meta['transport']
    best_ms, best_rec = np.inf, None
    for mn in MCH_RULES:
        for vn in VEH_RULES:
            sim = TransportSim(jl, pt, meta['time_lag'], tr['station_cell'],
                               tr['tau_cells'], int(tr['n_vehicles']),
                               int(tr['veh_start_cell']))
            ms = run_pdr_pair(sim, mn, vn)
            if ms < best_ms:
                best_ms, best_rec = ms, sim.schedule_record()
    t_seed = time.time() - t0_all
    del TRACE[:]                      # per-solve trace, worker is sequential
    t0 = time.time()
    sol = solve_transport_instance(jl, pt, meta, time_limit=TLIM,
                                   n_workers=WORKERS, warmstart=best_rec,
                                   strengthen=True, energy_cut=ENERGY,
                                   fleet=FLEET)
    rec = dict(instance=os.path.basename(stem), cell=cell, dataset=cell,
               pdr_seed=round(float(best_ms), 6), status=sol['status'],
               ub=sol['makespan'], lb=sol.get('objective_bound'),
               horizon=sol.get('horizon'),
               walltime=round(sol.get('walltime', time.time() - t0), 2),
               warmstart_cpu_s=round(t_seed, 2),
               n_workers=WORKERS, time_limit_s=TLIM, strengthened=True,
               energy_cut=ENERGY, fleet=FLEET)
    if ANYTIME:
        trace = [list(x) for x in TRACE]
        # the last incumbent must be the returned upper bound
        if trace and sol.get('makespan') is not None:
            assert abs(trace[-1][1] - sol['makespan']) < 1e-6, \
                f'{stem}: anytime tail {trace[-1][1]} != ub {sol["makespan"]}'
        rec['anytime'] = trace
        rec['anytime_note'] = ('[wall_s, ub, lb] at every improving '
                               'incumbent; wall_s is solver time from the '
                               'start of Solve(), so it excludes the '
                               'warm-start PDR rollouts (warmstart_cpu_s)')
    return rec


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for cell in cells:
        jsonl = f'{OUT_DIR}/{cell}.jsonl'
        js = f'{OUT_DIR}/{cell}.json'
        done = set()
        if os.path.exists(jsonl):
            with open(jsonl) as f:
                done = {json.loads(l)['instance'] for l in f if l.strip()}
        stems = sorted(g[:-4] for g in glob.glob(
            f'data/PPVCT/{cell}/test/instance_*.fjs'))
        if LIMIT_N is not None:
            stems = stems[:LIMIT_N]
        todo = [s for s in stems if os.path.basename(s) not in done]
        if CHUNK is not None:
            todo = todo[:CHUNK]
        print(f'{cell}: {len(todo)} to solve ({len(done)} done), par={PAR} '
              f'workers={WORKERS} cores={CORES}', flush=True)
        tasks = [(cell, s, i % PAR) for i, s in enumerate(todo)]
        if tasks:
            from multiprocessing import get_context
            _cfg = dict(par=PAR, workers=WORKERS, tlim=TLIM, cores=CORES,
                        energy=ENERGY, fleet=FLEET, anytime=ANYTIME,
                        out_dir=OUT_DIR)
            # maxtasksperchild=1: a pool worker that solves several large
            # instances in a row does not return their memory to the OS, so
            # its resident size ratchets up (43.8 GB after one 80-module
            # solve, 51.7 GB after three) until the kernel reaps it. One
            # task per child makes each solve start from a clean process;
            # the extra spawn costs seconds against a 3600 s solve.
            with get_context('spawn').Pool(PAR, initializer=_worker_init,
                                           initargs=(_cfg,),
                                           maxtasksperchild=1) as pool:
                for rec in pool.imap_unordered(solve_one, tasks):
                    with open(jsonl, 'a') as f:
                        f.write(json.dumps(rec) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    print(f'{rec["instance"]}: seed {rec["pdr_seed"]} -> '
                          f'ub {rec["ub"]} lb {rec["lb"]} ({rec["status"]}, '
                          f'{rec["walltime"]}s)', flush=True)
                    recs = {}
                    with open(jsonl) as f:
                        for l in f:
                            if l.strip():
                                r = json.loads(l)
                                recs[r['instance']] = r
                    with open(js + '.tmp', 'w') as f:
                        json.dump({k: recs[k] for k in sorted(recs)}, f,
                                  indent=1)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(js + '.tmp', js)
        print(f'{cell}: done', flush=True)


if __name__ == '__main__':
    main()
