#!/usr/bin/env bash
# Short-travel, one-vehicle cells (tau/p = 0.3, V = 1) at 50 and 80 modules:
# the policy arms on the GPU, with the arms, seeds and flags of the
# production-cell evaluation (batched greedy rollouts, default batch 15). Run it
# only when the GPU is granted: the latency it records is otherwise contended.
#
# Results go to results/shorttravel/policy/, not results/scaleup/policy/, so
# the scale figures cannot pick these cells up. Resumable per (model, cell):
# a cell whose output file exists is not evaluated again.
#
# Parameters (environment): CORES (0-7), the host cores feeding the GPU.
# Launch:  setsid nohup bash scripts/run_x2_shorttravel_gpu.sh >/dev/null 2>&1 &
# Check:   bash scripts/run_x2_shorttravel_gpu.sh --dry_run
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=logs/shorttravel.log
OUT=results/shorttravel/policy
CORES=${CORES:-0-7}
C50=50x25+ppvct-mixed+v1+t0.3; C80=80x25+ppvct-mixed+v1+t0.3
DRY=0
[ "${1:-}" = "--dry_run" ] && DRY=1
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
log(){ if [ "$DRY" -eq 1 ]; then echo "$*"; else echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; fi; }
log "=== short-travel GPU queue start: cores $CORES ==="
for arm in 10x25+ppvct-mixed+m1-bcb-guide 10x25+ppvct-mixed+m1-bcb 10x25+ppvct-mixed+single-joint \
           mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix mix10-15-20x25+ppvct-mixed+single-joint-mix; do
  for s in 301 302 303; do
    m=${arm}-s$s
    todo=""
    for c in $C50 $C80; do
      [ -f "$OUT/${m}_$c.json" ] || todo="${todo:+$todo,}$c"
    done
    if [ -z "$todo" ]; then log "policy $m: done, skipped"; continue; fi
    cmd=(taskset -c "$CORES" "$PY" -u scripts/x2_scale_policy.py --model_name "$m" --cells "$todo" --device cuda --outdir "$OUT")
    if [ "$DRY" -eq 1 ]; then echo "  ${cmd[*]} >> $LOG 2>&1"; continue; fi
    log "policy $m on $todo start"
    "${cmd[@]}" >> "$LOG" 2>&1
    log "policy $m end rc=$?"
  done
done
log "=== short-travel GPU queue COMPLETE ==="
