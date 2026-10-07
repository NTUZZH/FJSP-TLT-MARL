#!/usr/bin/env bash
# Execution-noise study (scripts/x2_execution_noise.py) on the two production
# cells, 20 instances x 3 replicates x sigma {0, 0.1, 0.2, 0.3}; sigma 0 is the
# control that prices re-planning without noise. Resumable: every record is
# keyed, so a rerun skips finished work.
#   pass 1  policy_rs, policy_replan, policy_threshold, pdr_replan, ga60_rs,
#           ga_replan: SHARDS processes (default 4), each pinned to one core
#           of CORES with one thread. The policy runs on the CPU at one torch
#           thread, where a batch-1 forward pass is fastest, and the GPU stays
#           free for the timed arms.
#   pass 2  cpsat_replan: one process per CPSAT_WORKERS cores of CORES
#           (default 4), one CP-SAT worker per core, on the reduced design
#           fixed before any run: sigma CPSAT_SIGMAS (0, 0.2) and replicate 0
#           only (CPSAT_REPS=1), both cells, all 20 instances. It reuses the
#           GA-60 plans cached by pass 1.
# Example for twelve cores: CORES=12-23 SHARDS=12 bash scripts/run_x2_execution_noise.sh
# Refuses to start when any core of CORES is already busy.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
CORES=${CORES:-18-21}
MODEL=${MODEL:-mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-s301}
SIGMAS=${SIGMAS:-0,0.1,0.2,0.3}
CELLS=${CELLS:-"50x25+ppvct-mixed+v2+t1.0 80x25+ppvct-mixed+v3+t1.0"}
REPS=${REPS:-3}
CPSAT_SIGMAS=${CPSAT_SIGMAS:-0,0.2}
CPSAT_REPS=${CPSAT_REPS:-1}
N=${N:-0}
OUTDIR=${OUTDIR:-results/execution_noise}
LOG=${LOG:-logs/execution_noise.log}
mkdir -p "$(dirname "$LOG")"
COMMON="--model_name $MODEL --n $N --outdir $OUTDIR --device cpu"
DESIGN1="--sigmas $SIGMAS --replicates $REPS"
DESIGN2="--sigmas $CPSAT_SIGMAS --replicates $CPSAT_REPS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

LO=${CORES%-*}; HI=${CORES#*-}
NCORES=$((HI - LO + 1))
SHARDS=${SHARDS:-4}
CPSAT_WORKERS=${CPSAT_WORKERS:-4}
if [ "$SHARDS" -gt "$NCORES" ] || [ "$CPSAT_WORKERS" -gt "$NCORES" ]; then
  echo "SHARDS=$SHARDS and CPSAT_WORKERS=$CPSAT_WORKERS must not exceed the $NCORES cores of $CORES"
  exit 1
fi
W2=$((NCORES / CPSAT_WORKERS))

# busy check: per-core utilization over three seconds from /proc/stat
busy=$($PY - "$LO" "$HI" <<'EOF'
import sys, time
lo, hi = int(sys.argv[1]), int(sys.argv[2])
def snap():
    out = {}
    for line in open('/proc/stat'):
        f = line.split()
        if f[0].startswith('cpu') and f[0] != 'cpu':
            v = [int(x) for x in f[1:]]
            out[int(f[0][3:])] = (sum(v), v[3] + v[4])
    return out
a = snap(); time.sleep(3); b = snap()
for c in range(lo, hi + 1):
    tot, idle = b[c][0] - a[c][0], b[c][1] - a[c][1]
    u = 1.0 - idle / max(tot, 1)
    if u > 0.25:
        print(f'core {c} {u:.0%}')
EOF
)
if [ -n "$busy" ]; then
  echo "refusing to start: cores $CORES are busy: $busy" | tee -a "$LOG"
  ps -eo pid,psr,pcpu,etime,args --sort=-pcpu | awk -v lo="$LO" -v hi="$HI" \
    'NR==1 || ($2>=lo && $2<=hi && $3>5)' | head -8 | tee -a "$LOG"
  exit 1
fi

log "=== execution-noise queue start: cores=$CORES shards=$SHARDS cpsat=${W2}x$CPSAT_WORKERS model=$MODEL sigmas=$SIGMAS reps=$REPS cpsat_sigmas=$CPSAT_SIGMAS cpsat_reps=$CPSAT_REPS out=$OUTDIR ==="
for c in $CELLS; do
  for i in $(seq 0 $((SHARDS - 1))); do
    core=$((LO + i))
    taskset -c "$core" $PY -u scripts/x2_execution_noise.py --cell "$c" $COMMON $DESIGN1 \
      --arms policy_rs,policy_replan,policy_threshold,pdr_replan,ga60_rs,ga_replan \
      --shard "$i/$SHARDS" --cores "$core-$core" --threads 1 --torch_threads 1 \
      >> "$LOG" 2>&1 &
  done
  rc=0
  for p in $(jobs -p); do wait "$p" || rc=$?; done
  log "pass 1 $c rc=$rc"
done
for c in $CELLS; do
  for i in $(seq 0 $((W2 - 1))); do
    lo=$((LO + i * CPSAT_WORKERS)); hi=$((lo + CPSAT_WORKERS - 1))
    taskset -c "$lo-$hi" $PY -u scripts/x2_execution_noise.py --cell "$c" $COMMON $DESIGN2 \
      --arms cpsat_replan --shard "$i/$W2" --cores "$lo-$hi" \
      --threads "$CPSAT_WORKERS" --cpsat_workers "$CPSAT_WORKERS" >> "$LOG" 2>&1 &
  done
  rc=0
  for p in $(jobs -p); do wait "$p" || rc=$?; done
  log "pass 2 $c rc=$rc"
done
log "=== execution-noise queue COMPLETE ==="
