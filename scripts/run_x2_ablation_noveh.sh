#!/usr/bin/env bash
# Fleet-term ablation: the headline arm's exact
# command (m1-bcb-guide: --credit m1 --guide --max_updates 2000, default grids)
# plus --bound_veh 0, one lane per seed, then greedy evaluation on the 12 grid
# cells on the GPU (the headline grid evaluations are GPU; no device mixing).
# Usage: bash scripts/run_x2_ablation_noveh.sh SEED CORES
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
S=$1; C=$2
LOG=logs/ablation/noveh_s$S.log
mkdir -p logs/ablation
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "[$(date '+%m-%d %H:%M:%S')] train s$S on cores $C" >> logs/ablation/queue.log
taskset -c $C $PY -u scripts/p2_train_mappo.py --model_suffix m1-bcb-guide-noveh --credit m1 --guide \
  --bound_veh 0 --seed $S --max_updates 2000 >> $LOG 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] train s$S rc=$?" >> logs/ablation/queue.log
taskset -c $C $PY -u scripts/eval_ppvct.py --model_name 10x25+ppvct-mixed+m1-bcb-guide-noveh-s$S \
  --cells v1+t0.1,v1+t0.3,v1+t0.6,v1+t1.0,v2+t0.1,v2+t0.3,v2+t0.6,v2+t1.0,v3+t0.1,v3+t0.3,v3+t0.6,v3+t1.0 \
  --split test >> $LOG 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] eval s$S rc=$?" >> logs/ablation/queue.log
