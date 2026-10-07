#!/usr/bin/env bash
# Fleet-term ablation on the size mixture. The deployed size-mixture arm's
# exact command (m1-bcb-guide-mix: --credit m1 --guide --size_mix 10,15,20
# --vali_size_mix 10,20 --max_updates 2000, seeds 301-303) plus --bound_veh 0,
# under the model suffix m1-bcb-guide-mix-noveh. Precedent: the 10-module
# ablation, scripts/run_x2_ablation_noveh.sh. Evaluation is a separate step:
# scripts/run_x2_ablation_noveh_mix_eval.sh. Analysis:
# scripts/x2_ablation_noveh_mix_report.py.
#
# Before launch, per seed and on CPU with the GPU hidden:
#   (a) p2_train_mappo.py --dry_run resolves the configuration and builds the
#       network without training;
#   (b) x2_preflight_config.py diffs it against the comparator's
#       config_*.json of the same seed and aborts unless the only differences
#       are bound_veh, model_name/model_suffix and seed_train;
#   (c) the same check asserts the parameter count and every state-dict shape
#       equal the comparator checkpoint, the CLI-only settings that the
#       snapshot does not record (vali_subset, vali_every) are the defaults the
#       comparator ran with, and torch uses 2 intra-op threads.
# Then: refuses if any file of the new run already exists (a stale jsonl would
# be appended to, a checkpoint overwritten), if MemAvailable < 12 GB, or if
# another process holds more than 2 GB of the card (override ALLOW_SHARED=1).
# After launch, each run's own config_*.json is diffed again with the device
# no longer excused; any failure kills all three runs.
#
# Three processes share one GPU, two pinned cores each, OMP/MKL threads 2
# (torch's intra-op pool follows OMP_NUM_THREADS, as in the training arm).
# Comparator wall time per seed, alone on the card: 9.3, 11.6, 7.3 h.
# GPU memory per process: not recorded by the comparator logs (unknown).
#
# Usage: bash scripts/run_x2_ablation_noveh_mix.sh ["4-5 6-7 8-9"]
#   PREFLIGHT_ONLY=1  run (a)-(c) and the RAM gate, then exit; no GPU access
#   ALLOW_SHARED=1    launch although another process holds >2 GB of the card
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
CORES=${1:-"4-5 6-7 8-9"}
SEEDS=(${SEEDS:-301 302 303})
COMP=mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix
SUFFIX=m1-bcb-guide-mix-noveh
NEW=mix10-15-20x25+ppvct-mixed+$SUFFIX
ARGS=(--model_suffix "$SUFFIX" --credit m1 --guide --size_mix 10,15,20
      --vali_size_mix 10,20 --max_updates 2000 --bound_veh 0)
PRE=logs/ablation/preflight_noveh_mix
QLOG=logs/ablation/queue_mix.log
mkdir -p logs/ablation "$PRE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$QLOG"; }
die(){ log "ABORT: $*"; exit 1; }

read -ra CARR <<< "$CORES"
[ ${#CARR[@]} -eq ${#SEEDS[@]} ] || die "need ${#SEEDS[@]} core sets, got '$CORES'"

# (e) host memory
mem_kb=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
(( mem_kb >= 12 * 1024 * 1024 )) || die "MemAvailable $((mem_kb / 1024 / 1024)) GB < 12 GB"
log "MemAvailable $((mem_kb / 1024 / 1024)) GB (>= 12 GB)"

# contamination: nothing of the new run may exist yet
for s in "${SEEDS[@]}"; do
  for f in train_log/PPVCT/config_$NEW-s$s.json train_log/PPVCT/train_$NEW-s$s.log.jsonl \
           trained_network/PPVCT/$NEW-s$s.pth trained_network/PPVCT/$NEW-s$s-last.pth \
           logs/ablation/noveh_mix_s$s.log; do
    [ -e "$f" ] && die "$f exists (would be appended to or overwritten); move it first"
  done
  for f in train_log/PPVCT/config_$COMP-s$s.json trained_network/PPVCT/$COMP-s$s.pth; do
    [ -f "$f" ] || die "comparator file missing: $f"
  done
done

# (a)-(c) per seed, GPU hidden
for i in "${!SEEDS[@]}"; do
  s=${SEEDS[$i]}; c=${CARR[$i]}
  CUDA_VISIBLE_DEVICES= taskset -c "$c" $PY -u scripts/p2_train_mappo.py "${ARGS[@]}" \
    --seed "$s" --dry_run "$PRE/dry_s$s.json" > "$PRE/dry_s$s.log" 2>&1 \
    || die "dry run s$s failed (see $PRE/dry_s$s.log)"
  $PY scripts/x2_preflight_config.py dry "$PRE/dry_s$s.json" \
    train_log/PPVCT/config_$COMP-s$s.json --ckpt trained_network/PPVCT/$COMP-s$s.pth \
    --allow bound_veh,model_name,model_suffix,seed_train \
    --expect "bound_veh=0,model_name=$NEW-s$s" \
    --cli_expect vali_subset=20,vali_every=20 --threads 2 > "$PRE/diff_s$s.txt" 2>&1
  rc=$?; cat "$PRE/diff_s$s.txt"
  [ $rc -eq 0 ] || die "preflight s$s failed (see $PRE/diff_s$s.txt)"
  log "preflight s$s OK on cores $c"
done
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { log "PREFLIGHT_ONLY: stop before the GPU gate"; exit 0; }

# (d) the card must be ours
command -v nvidia-smi > /dev/null || die "nvidia-smi not found"
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits \
       | awk -F', *' '$3 > 2048')
if [ -n "$busy" ]; then
  if [ "${ALLOW_SHARED:-0}" = 1 ]; then
    log "ALLOW_SHARED=1: launching beside [$busy]; wall times will be contended"
  else
    die "another process holds >2 GB of the card: [$busy] (set ALLOW_SHARED=1 to override)"
  fi
fi

# launch, one detached process per seed
PIDS=()
for i in "${!SEEDS[@]}"; do
  s=${SEEDS[$i]}; c=${CARR[$i]}
  setsid nohup taskset -c "$c" $PY -u scripts/p2_train_mappo.py "${ARGS[@]}" --seed "$s" \
    >> logs/ablation/noveh_mix_s$s.log 2>&1 < /dev/null &
  PIDS+=($!)
  log "train $NEW-s$s pid ${PIDS[-1]} cores $c"
done

# post-launch: the config each run actually wrote, device included
kill_all(){ kill "${PIDS[@]}" 2>/dev/null; die "$*; all three runs killed"; }
for i in "${!SEEDS[@]}"; do
  s=${SEEDS[$i]}; cfg=train_log/PPVCT/config_$NEW-s$s.json
  for _ in $(seq 1 120); do [ -s "$cfg" ] && break; sleep 1; done
  [ -s "$cfg" ] || kill_all "s$s wrote no config within 120 s"
  $PY scripts/x2_preflight_config.py post "$cfg" train_log/PPVCT/config_$COMP-s$s.json \
    --allow bound_veh,model_name,model_suffix,seed_train \
    --expect "bound_veh=0,model_name=$NEW-s$s" > "$PRE/post_s$s.txt" 2>&1 \
    || { cat "$PRE/post_s$s.txt"; kill_all "post-launch config check s$s failed"; }
  log "post-launch config s$s OK (device $(grep -o '"device": "[a-z]*"' "$cfg"))"
done
log "all three runs launched and verified; logs/ablation/noveh_mix_s{301,302,303}.log"
