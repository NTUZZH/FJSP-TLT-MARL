#!/usr/bin/env bash
# Greedy evaluation of the size-mixture fleet-term ablation
# (mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-noveh, seeds 301-303), with the
# comparator's own commands:
#   * production cells: scripts/x2_scale_policy.py --device cuda, default batch
#     15, taskset 0-7, OMP/MKL 4;
#   * 10- and 15-module cells: scripts/eval_ppvct.py on the six cells the
#     mixture was evaluated on (files in
#     test_results/PPVCT/{cell}/Result_greedy+mix...-guide-mix-s30x_{cell}.npy).
# Both evaluators attach the bound the checkpoint was trained with (snapshot
# key bound_veh), so the guide prices match training.
#
# Every output goes under results/ablation/noveh_mix/, never into
# results/scaleup/policy/, test_results/PPVCT/ or results/diagnostics/: those
# folders are globbed with wildcards by x2_util_extended.py (best-known mean),
# x2_scale_cert.py and
# figures_src/make_s5_tightness.py panel (a), which would otherwise sweep the
# ablation in. The analysis is scripts/x2_ablation_noveh_mix_report.py.
#
# Usage: bash scripts/run_x2_ablation_noveh_mix_eval.sh [CORES]   (default 0-7)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
C=${1:-0-7}
ARM=mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-noveh
OUT=results/ablation/noveh_mix
LOG=logs/ablation/noveh_mix_eval.log
C50a=50x25+ppvct-mixed+v2+t0.6; C50b=50x25+ppvct-mixed+v2+t1.0; C80=80x25+ppvct-mixed+v3+t1.0
PROD=$C50a,$C50b,$C80
GRID=v1+t0.6,v2+t0.6,v1+t1.0,v2+t1.0,15x25+ppvct-mixed+v2+t0.6,15x25+ppvct-mixed+v2+t1.0
mkdir -p "$OUT/policy" "$OUT/test_results" logs/ablation
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for s in ${SEEDS:-301 302 303}; do
  m=$ARM-s$s
  [ -f trained_network/PPVCT/$m.pth ] || { log "missing checkpoint $m"; exit 1; }
  # the evaluators read bound_veh from this snapshot; refuse anything else
  $PY -c "import json,sys; c=json.load(open('train_log/PPVCT/config_$m.json')); sys.exit(0 if c.get('bound_veh')==0 and c.get('guide') is True else 1)" \
    || { log "$m snapshot is not bound_veh=0 with guide"; exit 1; }
  taskset -c "$C" $PY -u scripts/x2_scale_policy.py --model_name "$m" --cells "$PROD" \
    --device cuda --outdir "$OUT/policy" >> "$LOG" 2>&1
  log "policy $m production cells rc=$?"
  taskset -c "$C" $PY -u scripts/eval_ppvct.py --model_name "$m" --cells "$GRID" \
    --split test --out_root "$OUT/test_results" >> "$LOG" 2>&1
  log "eval_ppvct $m six cells rc=$?"
done
log "noveh-mix evaluation complete; run: python scripts/x2_ablation_noveh_mix_report.py"
