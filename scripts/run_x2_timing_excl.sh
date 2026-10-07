#!/usr/bin/env bash
# Timing reruns on dedicated resources: (1) batch-1 single-core CPU probe of
# the 10-module policy (one torch thread, one pinned core) on the three
# production-batch cells; (2) best-of-64 sampling of the size-mixture policy,
# one lane on an otherwise idle GPU with four pinned cores. Outputs carry the
# tags +b1lat and +excl, so no cell-result reader sweeps them in.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=logs/timing/queue.log
mkdir -p logs/timing
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
CELLS=50x25+ppvct-mixed+v2+t0.6,50x25+ppvct-mixed+v2+t1.0,80x25+ppvct-mixed+v3+t1.0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 11 $PY -u scripts/x2_scale_policy.py \
  --model_name 10x25+ppvct-mixed+m1-bcb-guide-s301 --cells $CELLS \
  --device cpu --batch 1 --n 3 --tag +b1lat >> logs/timing/b1lat.log 2>&1
log "b1lat rc=$?"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for cell in ${CELLS//,/ }; do
  b=20; case "$cell" in 80x25*) b=10;; esac
  taskset -c 20-23 $PY -u scripts/x2_eval_sample.py \
    --model_name mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-s301 --cells $cell \
    --n_samples 64 --batch $b --tag +excl \
    --note 'batched GPU, exclusive GPU and four pinned cores' >> logs/timing/sample.log 2>&1
  log "sample $cell rc=$?"
done
log "COMPLETE"
