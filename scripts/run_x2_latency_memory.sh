#!/usr/bin/env bash
# Batch-1 latency and memory of the size-mixture policy (seeds 301-303) at 10,
# 20, 50 and 80 modules, for the quiet-machine slot:
#   (1) CPU pass: one pinned core, one torch thread, OMP/MKL = 1;
#   (2) GPU pass: same core for the host side, one torch thread.
# Each pass refuses to start when the 1-minute load average exceeds LOAD_MAX
# (default 2.0; the busy check on the pinned core is the stricter test), when
# the chosen core is busy with another process, or (GPU pass) when another
# process holds the GPU device open. A refused pass is logged and skipped.
# Outputs: results/latency/x2_latency_memory_{cpu,cuda}.json
#
# Usage: CORE=4 N=3 [LOAD_MAX=2.0] bash scripts/run_x2_latency_memory.sh
#        FORCE=1 overwrites existing output files.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
CORE=${CORE:-4}
N=${N:-3}
FORCE=${FORCE:-0}
MAX_LOAD=${LOAD_MAX:-2.0}   # 1-minute load average ceiling
MAX_CORE_BUSY=5      # percent of the core used by others over the sample
SAMPLE_S=5
mkdir -p logs/latency results/latency
LOG=logs/latency/run.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

core_busy_pct(){
  # busy share of core $1 over $SAMPLE_S seconds, from /proc/stat
  local a b
  a=$(grep "^cpu$1 " /proc/stat)
  sleep "$SAMPLE_S"
  b=$(grep "^cpu$1 " /proc/stat)
  awk -v a="$a" -v b="$b" 'BEGIN{
    n=split(a,x," "); split(b,y," "); tot=0; idle=0;
    for(i=2;i<=n;i++){ d=y[i]-x[i]; tot+=d; if(i==5||i==6) idle+=d }
    if(tot<=0){ print 0; exit } printf "%.1f", 100*(tot-idle)/tot }'
}

gpu_holders(){
  # pids (other than this shell) holding a /dev/nvidiaN device open
  local p
  for p in /proc/[0-9]*; do
    [ "${p#/proc/}" = "$$" ] && continue
    if ls -l "$p/fd" 2>/dev/null | grep -q '/dev/nvidia[0-9]'; then
      echo "${p#/proc/} $(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-60)"
    fi
  done
}

quiet_or_refuse(){
  # $1 = pass name; returns 0 when the machine is quiet enough to time
  local load busy holders
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
    log "$1 REFUSED: 1-min load $load > $MAX_LOAD"
    return 1
  fi
  busy=$(core_busy_pct "$CORE")
  if awk -v b="$busy" -v m="$MAX_CORE_BUSY" 'BEGIN{exit !(b>m)}'; then
    log "$1 REFUSED: core $CORE busy ${busy}% over ${SAMPLE_S}s; running there:"
    ps -eLo stat=,psr=,pid=,comm= | awk -v c="$CORE" '$1 ~ /^R/ && $2==c' \
      | tee -a "$LOG"
    return 1
  fi
  if [ "$1" = cuda ]; then
    holders=$(gpu_holders)
    if [ -n "$holders" ]; then
      log "$1 REFUSED: GPU held open by: $holders"
      return 1
    fi
  fi
  log "$1 quiet: 1-min load $load (max $MAX_LOAD), core $CORE busy ${busy}%"
  return 0
}

run_pass(){
  # $1 = cpu | cuda
  local out=results/latency/x2_latency_memory_$1.json rc
  if [ -e "$out" ] && [ "$FORCE" != 1 ]; then
    log "$1 SKIPPED: $out exists (FORCE=1 to overwrite)"
    return 1
  fi
  quiet_or_refuse "$1" || return 1
  local cuda_env=""
  [ "$1" = cpu ] && cuda_env="CUDA_VISIBLE_DEVICES="
  env $cuda_env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c "$CORE" $PY -u scripts/x2_latency_memory.py \
    --cores "$CORE" --threads 1 --device "$1" --n "$N" --out "$out" \
    >> "logs/latency/$1.log" 2>&1
  rc=$?
  log "$1 rc=$rc -> $out"
  return $rc
}

log "START core=$CORE n=$N load_max=$MAX_LOAD commit=$(git rev-parse --short HEAD)"
status=0
run_pass cpu || status=1
run_pass cuda || status=1
log "COMPLETE status=$status"
exit $status
