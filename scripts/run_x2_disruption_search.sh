#!/usr/bin/env bash
# Disruption recovery, the budgeted search arms: GA and warm-started residual
# CP-SAT at wall-clock budgets of 1, 10 and 60 s, on the two production cells.
# Protocol, seeds, instances and ledgers are those of the phase-1 run
# (scripts/x2_disruption.py --methods right_shift,pdr,policy): t = 0.30 C0, d = 0.20 C0, the
# mixture policy at seeds 301-303, both disrupted-plan generators, all 20
# instances. Records append to results/disruption/{cell}.jsonl, which
# scripts/x2_disruption_phase1.py reads without change (it only tabulates
# right_shift, pdr and policy) and scripts/x2_disruption_report.py tabulates
# per budget. Under --baseline pdr the search arms do not depend on the
# checkpoint, so they are solved once (seed 301) and the 302/303 passes skip.
#
# The budgets are wall clock, so the run needs its cores to itself: before
# every invocation it refuses to go on if another process uses more than
# BUSY_PCT % CPU on CORES, or if MemAvailable is below MIN_MEM_GB. Resumable:
# x2_disruption.py skips every (instance, method, budget) key already in the
# ledger, and an invocation with nothing left is not launched at all.
#
# Parameters (environment): SEEDS ("301 302 303"), METHODS (ga,cpsat),
# BUDGETS (1,10,60), CORES (18-21), OMP_NUM_THREADS (4, the cap for
# the numerical libraries), CPSAT_WORKERS (4), BUSY_PCT (20), MIN_MEM_GB (16).
# Torch runs one thread: the baseline plan under --baseline policy is a
# batch-1 forward pass per event, which slows down with more threads.
#
# Launch:  setsid nohup bash scripts/run_x2_disruption_search.sh >/dev/null 2>&1 &
# Check:   bash scripts/run_x2_disruption_search.sh --dry_run
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=logs/disruption_search.log
CORES=${CORES:-18-21}
THREADS=${OMP_NUM_THREADS:-4}
CPSAT_WORKERS=${CPSAT_WORKERS:-4}
BUSY_PCT=${BUSY_PCT:-20}
MIN_MEM_GB=${MIN_MEM_GB:-16}
CELLS=(50x25+ppvct-mixed+v2+t1.0 80x25+ppvct-mixed+v3+t1.0)
SEEDS=(${SEEDS:-301 302 303})
BASELINES=(pdr policy)
METHODS=${METHODS:-ga,cpsat}
BUDGETS=${BUDGETS:-1,10,60}
MODEL=mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix
DRY=0
[ "${1:-}" = "--dry_run" ] && DRY=1
mkdir -p logs
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

# a dry run prints instead of writing the queue log
log(){ if [ "$DRY" -eq 1 ]; then echo "$*"; else echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; fi; }

# Processes other than this script's own tree that used more than BUSY_PCT %
# CPU on CORES over a 3 s window, summed over the threads that ran there.
# Prints one line per offender; prints nothing when the cores are free.
busy_on_cores(){
  "$PY" - "$CORES" "$BUSY_PCT" "$$" <<'EOF'
import os, sys, time
lo, hi = (int(x) for x in sys.argv[1].split('-'))
cores, limit, root = set(range(lo, hi + 1)), float(sys.argv[2]), int(sys.argv[3])
hz = os.sysconf('SC_CLK_TCK')

def sample():
    out = {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            for tid in os.listdir(f'/proc/{pid}/task'):
                with open(f'/proc/{pid}/task/{tid}/stat') as f:
                    rest = f.read().rsplit(')', 1)[1].split()
                out[(int(pid), int(tid))] = (int(rest[11]) + int(rest[12]),
                                             int(rest[36]), int(rest[1]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return out

a = sample(); t0 = time.time(); time.sleep(3.0); b = sample()
dt = time.time() - t0
own = {root, os.getpid(), os.getppid()}
parent = {pid: v[2] for (pid, tid), v in b.items() if pid == tid}
use = {}
for (pid, tid), (ticks, cpu, _) in b.items():
    if cpu in cores and (pid, tid) in a:
        use[pid] = use.get(pid, 0) + ticks - a[(pid, tid)][0]
for pid, ticks in sorted(use.items()):
    pct = 100.0 * ticks / hz / dt
    p, mine = pid, False
    while p > 1:
        if p in own:
            mine = True
            break
        p = parent.get(p, 1)
    if pct > limit and not mine:
        try:
            cmd = open(f'/proc/{pid}/cmdline').read().replace('\0', ' ')[:120]
        except OSError:
            cmd = '?'
        print(f'pid {pid} {pct:.0f}% {cmd}')
EOF
}

mem_gb(){ awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }

# Keys of one invocation that are not yet in the ledger, written exactly as
# x2_disruption.py writes them (rec_key).
missing_keys(){  # cell seed baseline
  "$PY" - "$1" "$2" "$3" "$MODEL" "$METHODS" "$BUDGETS" <<'EOF'
import glob, json, os, sys
cell, seed, base, model, methods, budgets = sys.argv[1:]
model = f'{model}-s{seed}'
p = f'results/disruption/{cell}.jsonl'
done = set()
if os.path.exists(p):
    done = {json.loads(l)['key'] for l in open(p) if l.strip()}
names = sorted(os.path.basename(g)[:-4] for g in
               glob.glob(f'data/PPVCT/{cell}/test/instance_*.fjs'))
tag = f'|{model}' if base == 'policy' else ''
n = 0
for name in names:
    for b in budgets.split(','):
        for m in methods.split(','):
            if f'{cell}|{name}|{base}|{m}|B{float(b):g}{tag}' not in done:
                n += 1
print(n)
EOF
}

gate(){
  local busy a
  busy=$(busy_on_cores)
  if [ -n "$busy" ]; then
    log "REFUSED: cores $CORES are busy (> ${BUSY_PCT}% CPU):"
    while IFS= read -r l; do log "   $l"; done <<< "$busy"
    return 1
  fi
  a=$(mem_gb)
  if [ "$a" -lt "$MIN_MEM_GB" ]; then
    log "REFUSED: MemAvailable ${a} GB < ${MIN_MEM_GB} GB"
    return 1
  fi
  return 0
}

N_BUDGETS=$(awk -F, '{print NF}' <<< "$BUDGETS")
SUM_BUDGET=$(awk -F, '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}' <<< "$BUDGETS")
N_METHODS=$(awk -F, '{print NF}' <<< "$METHODS")
log "=== disruption search start: methods $METHODS budgets $BUDGETS, cores $CORES, OMP $THREADS, CP-SAT workers $CPSAT_WORKERS, torch 1 ==="
if ! gate; then
  [ "$DRY" -eq 1 ] || exit 2
fi
total_missing=0
declare -A pdr_counted   # dry run: the pdr-baseline keys are shared by seeds
for c in "${CELLS[@]}"; do
  log "$c: start"
  for s in "${SEEDS[@]}"; do
    for b in "${BASELINES[@]}"; do
      miss=$(missing_keys "$c" "$s" "$b")
      if [ "$miss" -eq 0 ]; then
        log "$c s$s baseline=$b: nothing left, skipped"
        continue
      fi
      if [ "$DRY" -eq 1 ] && [ "$b" = pdr ] && [ -n "${pdr_counted[$c]:-}" ]; then
        echo "  [same keys as the first seed; skipped once it has run]"
        continue
      fi
      [ "$b" = pdr ] && pdr_counted[$c]=1
      total_missing=$((total_missing + miss))
      cmd=(taskset -c "$CORES" "$PY" -u scripts/x2_disruption.py --cell "$c"
           --model_name "$MODEL-s$s" --methods "$METHODS" --budgets "$BUDGETS"
           --baseline "$b" --device cpu --cores "$CORES" --threads "$THREADS"
           --torch_threads 1 --cpsat_workers "$CPSAT_WORKERS")
      if [ "$DRY" -eq 1 ]; then
        echo "  [$miss keys missing] ${cmd[*]} >> $LOG 2>&1"
        continue
      fi
      gate || exit 3
      log "$c s$s baseline=$b: $miss keys to solve"
      "${cmd[@]}" >> "$LOG" 2>&1
      rc=$?
      log "$c s$s baseline=$b: rc=$rc, $(missing_keys "$c" "$s" "$b") keys still missing"
    done
  done
  log "$c: end"
done
if [ "$DRY" -eq 1 ]; then
  # each missing key is one search of BUDGET seconds; keys per instance-plan
  # are N_METHODS x N_BUDGETS, and their budgets sum to N_METHODS x SUM_BUDGET
  plans=$((total_missing / (N_METHODS * N_BUDGETS)))
  echo "missing keys: $total_missing (= $plans instance-plans x $N_METHODS methods x $N_BUDGETS budgets)"
  echo "search budget alone: $((plans * N_METHODS * SUM_BUDGET)) s = $(awk -v s=$((plans * N_METHODS * SUM_BUDGET)) 'BEGIN{printf "%.1f", s/3600}') h"
  echo "dry run: nothing executed"
  exit 0
fi
log "=== disruption search COMPLETE ==="
