#!/usr/bin/env bash
# Short-travel, one-vehicle cells (tau/p = 0.3, V = 1) at 50 and 80 modules:
# the CPU arms. The module routings, processing times and lags are those of
# the production cells (base seeds 100000 and 110000, 20 instances); only the
# transport layout differs. Order: dispatching rules (nine pairs), then GA at
# 60 CPU-s with the flags of the production-cell GA60 arm. The GA
# meters CPU time, so co-tenancy cannot weaken it. The policy arm needs the
# GPU and lives in run_x2_shorttravel_gpu.sh.
#
# Results go to results/shorttravel/{pdr,ga}/, not results/scaleup/: the
# scale figures glob results/scaleup/pdr/*.json and would pick these cells up.
# Resumable per cell: each tool skips a cell whose output file exists.
#
# Parameters (environment): CORES (18-21); the worker count is the core count.
# Launch:  setsid nohup bash scripts/run_x2_shorttravel.sh >/dev/null 2>&1 &
# Check:   bash scripts/run_x2_shorttravel.sh --dry_run
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=logs/shorttravel.log
OUT=results/shorttravel
CORES=${CORES:-18-21}
NW=$(( ${CORES#*-} - ${CORES%-*} + 1 ))
C50=50x25+ppvct-mixed+v1+t0.3; C80=80x25+ppvct-mixed+v1+t0.3
DRY=0
[ "${1:-}" = "--dry_run" ] && DRY=1
mkdir -p logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log(){ if [ "$DRY" -eq 1 ]; then echo "$*"; else echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; fi; }
run(){  # name cmd...
  local name=$1; shift
  if [ "$DRY" -eq 1 ]; then echo "  $* >> $LOG 2>&1"; return 0; fi
  log "$name start"
  "$@" >> "$LOG" 2>&1
  log "$name end rc=$?"
}
for c in $C50 $C80; do
  [ -f data/PPVCT/$c/test/dataset_meta.json ] || { log "missing data/PPVCT/$c; generate it with scripts/x2_scale_make_data.py --cells $c"; exit 1; }
done
log "=== short-travel CPU queue start: cores $CORES, $NW workers ==="
run pdr taskset -c "$CORES" "$PY" -u scripts/x2_scale_pdr.py --workers "$NW" --cells $C50,$C80 --outdir $OUT/pdr
run ga60 taskset -c "$CORES" "$PY" -u scripts/x2_scale_ga.py --cells $C50,$C80 --budget 60 --workers "$NW" --outdir $OUT/ga
log "=== short-travel CPU queue COMPLETE ==="
