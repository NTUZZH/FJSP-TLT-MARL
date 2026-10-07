# BOLT: Bound-Oriented Learning for Transport-Coupled Flexible Job-Shop Scheduling with Time Lags (FJSP-TL-T)

This repository contains the environment, learning code, benchmark datasets,
trained policies and result files behind a study of production scheduling in
which a scarce vehicle fleet moves work between stations. In the flexible job
shop with time lags and transport (FJSP-TL-T), a minimum waiting time may
separate consecutive operations of a job, a limited fleet carries jobs between
station cells, and the wait for a vehicle depends on the schedule itself, so
machine assignment and vehicle assignment must be decided together.

The method, BOLT (bound-oriented learning for transport-coupled scheduling),
is built around one certified multi-bottleneck lower bound on the makespan
attainable from a partial schedule: the maximum of a chain relaxation, a
machine-capacity relaxation and a fleet-capacity relaxation
(`transport_marl/bound.py`). The bound is admissible at every reachable state
of the constructive scheduler and tight at termination, and it is reused four
ways. Its one-step change is the training reward, so the undiscounted return
equals a constant minus the makespan. Its increase under each candidate
action is a certified action price appended to the policy's input. Its root
value gives a solver-free upper bound on the optimality gap of any completed
schedule. Its fleet term, added to the CP-SAT reference model as a redundant
constraint, raises the lower bound the solver proves. The policy runs either
as one merged decision head or as factorized machine and vehicle heads (the
deployed form); an analytic credit baseline for the factorized form is
included as a diagnostic.

**Reference:** anonymized manuscript under double-anonymous review at *IEEE
Transactions on Automation Science and Engineering*.

Per-cell CP-SAT, dispatching-rule and genetic-algorithm reference solutions are
included, along with the per-instance evaluation arrays for every released
checkpoint. Every comparison and every statistic reported in the manuscript can
therefore be recomputed from this repository without re-solving or re-training
anything.

Every result file in this repository is final.

### Artifact inventory

The manuscript's reproducibility footnote names the artifact families below. Each one is here:

| Family | Where it lives |
| --- | --- |
| Generator | `ppvc_instance_generator.py` and `transport_marl/layout.py` (the station-cell layout and travel-time model), driven by `scripts/p1_make_ppvct_data.py`, `scripts/p7_make_transfer_data.py` and `scripts/x2_scale_make_data.py`; `transport_marl/external_adapter.py` ports the external distribution |
| Instance data | `data/PPVCT/` (training grid, held-out transfer cells, scale-up cells), `data/PPVCT_HET/` (processing-time heterogeneity copies of the test cells) and `data/LINK/15x10/` |
| CP-SAT references | `results/cpsat_v2/` (reported), `results/scaleup/{cpsat,cpsat_b}/` (scale-up, including the anytime ledgers), `or_solution/PPVCT/` (archival); model in `transport_marl/cpsat_transport.py` |
| Dispatching-rule references | `results/pdr/` and `results/scaleup/pdr/`; rules in `transport_marl/pdr_pairs.py` |
| GA references | `results/ga_v2/` (reported), `results/ga_v2_budget/` and `results/scaleup/{ga,ga_long,ga_budget}/` (budget sweeps), `results/ga/` (archival); solvers in `transport_marl/ga_transport_v2.py` and `ga_transport.py` |
| Trained models | `trained_network/PPVCT/` (96 checkpoint files: 48 arm-seed pairs, best plus final) with the hyperparameter snapshots in `train_log/PPVCT/` that the loaders read to rebuild each network |
| Evaluation code | `scripts/` (rollouts, baselines, statistics, acceptance tests, scale-up pipeline), `transport_marl/validator_t.py` (independent feasibility checking), `figures_src/` (figures), with the per-instance outputs in `test_results/PPVCT/`, `results/diagnostics/`, `results/certificate/` and `results/scaleup/` |
| External-benchmark scores | `results/external_l1/{arm}/score_vs_released.json` and `summary.json`, one directory per training seed: our schedules re-timed in the benchmark authors' own simulator, against their released anchors |
| Regime and mechanism diagnostics | `results/fleet_util.json` and `results/fleet_util_extended.json` (minimum loaded fleet utilization per cell), `results/bound_terms.json` and `results/bound_terms_grid.json` (which bound component is active at the root, per cell), `results/hetero/` (processing-time heterogeneity study), `results/disruption/` (breakdown-recovery ledgers), `results/sample_decode/` (best-of-N decoding) |
| Decision-latency evidence | `train_log/latency_uncontended2.log` (GPU) and `train_log/latency_cpu4_2.log` (four pinned CPU cores), the console logs of the runs that produced the reported per-decision times; `results/latency/` (batch-1 time per event and memory by instance size) |
| Supplement studies | `results/disruption/` (breakdown recovery, including the GA and CP-SAT search arms), `results/execution_noise/` (execution under duration noise), `results/shorttravel/` (one-vehicle production batches with short moves), `results/ablation/noveh_mix/` (fleet-term ablation of the size-mixture policy), `results/scaleup/cpsat_cold/` (CP-SAT without a warm start); see "Supplement studies" below |

---

## Installation

Python 3.11, PyTorch 2.11 with CUDA 12.8. A GPU is needed only for training;
evaluation, CP-SAT, dispatching rules, the genetic algorithm and all statistics
run on CPU.

```bash
conda create -n fjsptlt python=3.11 -y
conda activate fjsptlt

# PyTorch from the CUDA 12.8 wheel index (not plain PyPI)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

All commands below are run **from the repository root**, which must be on
`PYTHONPATH` (the scripts insert it themselves).

---

## Datasets

The released datasets are already under `data/`; the generators are included so
they can be rebuilt bit-for-bit from their pinned seeds.

| Dataset | Location | Content |
| --- | --- | --- |
| PPVC-T training grid | `data/PPVCT/10x25+ppvct-mixed+v{V}+t{r}/{test,vali}`, `V` in {1,2,3}, `r` in {0.1,0.3,0.6} | 10 modules; 100 test + 100 validation instances per cell |
| PPVC-T held-out intensity | `data/PPVCT/10x25+ppvct-mixed+v{1,2,3}+t1.0/{test,vali}` | the travel intensity the policy never trains on; 100 test instances per cell |
| PPVC-T transfer cells | `data/PPVCT/10x25+ppvct-mixed+v4+t{0.6,1.0}`, `data/PPVCT/15x25+ppvct-mixed+v2+t{0.6,1.0}` | held-out fleet size and held-out problem scale, test split only |
| PPVC-T scale-up cells | `data/PPVCT/{20,30}x25+ppvct-mixed+...`, `data/PPVCT/{50,80}x25+ppvct-mixed+...` (7 cells) | 20, 30, 50 and 80 modules against the 10-module training size; 30 test instances per cell at 20 and 30 modules and 20 per production batch at 50 and 80, test split only |
| PPVC-T short-move production batches | `data/PPVCT/{50,80}x25+ppvct-mixed+v1+t0.3/test` | one vehicle and travel intensity 0.3; the module routings, processing times and lags of the 50- and 80-module production batches (same base seeds), with only the transport layout changed; 20 instances per cell |
| Link JSSPT port | `data/LINK/15x10/{test,vali}` | 15 jobs x 10 machines, external anchor distribution, with `PROVENANCE.json` per split |
| PPVC-T heterogeneity copies | `data/PPVCT_HET/` | the test cells of the heterogeneity study with processing times redrawn at dispersion levels R in {2, 5, 10, 20, 100}, mean preserved per operation before one-hour discretization |

Regenerate them:

```bash
# PPVC-T training grid: V x r cells, 100 test (seed0=20000) + 100 vali (seed0=10000)
python scripts/p1_make_ppvct_data.py          # add --smoke for a 3-instance dry run

# held-out transfer cells (test split only; never seen in training or validation)
python scripts/p7_make_transfer_data.py

# scale-up cells: phase A holds 20 and 30 modules, 30 instances each; phase B holds the
# 50- and 80-module production batches, 20 instances each on fixed base seeds 100000 and 110000
python scripts/x2_scale_make_data.py --phase A
python scripts/x2_scale_make_data.py --phase B

# one-vehicle production batches with short moves (same base seeds as phase B)
python scripts/x2_scale_make_data.py --cells 50x25+ppvct-mixed+v1+t0.3,80x25+ppvct-mixed+v1+t0.3

# heterogeneity copies of the test cells (paired on the same base instances)
python scripts/x2_hetero_make_data.py
python scripts/x2_hetero_verify.py       # checks mean drift and realized dispersion
```

The seven scale-up cells are zero-shot only: training and model selection
happen at 10 modules, and their test seeds (20000 onward, and 100000 and
110000 for the production batches) cannot collide with a training seed.

Training instances are drawn on the fly by the trainer and are not stored.

The travel intensity `r` is the ratio of mean travel time to mean processing
time; the layout module calibrates vehicle speed per instance so that the
realized ratio matches the cell. `V` is the number of vehicles.

---

## Training

One command trains one arm. `--model_suffix` names the checkpoint; the file
written is `trained_network/PPVCT/10x25+ppvct-mixed+{suffix}-s{seed}.pth` (best
validation score) alongside `...-s{seed}-last.pth` (final update). The three
mechanisms are switched independently:

- `--credit {shared,m1,m2}` selects the credit assignment: `shared` gives both
  heads the undivided bound-difference reward, `m1` is the analytic credit baseline (BCB),
  `m2` is the shaped-difference variant.
- `--guide` appends the certified action-price channel to both action grids.
- `--algo {mappo,coma,single}` selects the learner: factorized heads trained
  with MAPPO, the COMA learned-critic control, or the merged-head control
  (one decision head for both event classes).
- `--dist {ppvc,link}` selects the instance distribution; with `link`,
  `--fleet_grid` lists vehicle counts and `--ratio_grid` is ignored.
- `--guide_price {certified,naive}` selects what the action-price channel
  carries: the admissible bound-based price (default) or the non-admissible
  myopic duration control, whose `--guide_price_scale` matches the two
  channels' mean magnitude so the arm tests admissibility and not feature
  scale. The channel count, and therefore the parameter count, is the same
  either way.
- `--size_mix` round-robins the module count across updates (a batch stays
  size-uniform), and `--vali_size_mix` sets the sizes of the size-mixed
  validation set. Checkpoints trained this way are named
  `mix{sizes}x25+ppvct-mixed+{suffix}-s{seed}.pth`.

```bash
# BOLT headline arm: analytic credit baseline + certified action prices
python -u scripts/p2_train_mappo.py --model_suffix m1-bcb-guide --credit m1 --guide \
    --seed 301 --max_updates 2000

# credit-assignment ladder (ablations of the headline arm)
python -u scripts/p2_train_mappo.py --model_suffix joint-v1    --seed 301 --max_updates 2000
python -u scripts/p2_train_mappo.py --model_suffix m1-bcb      --credit m1 --seed 301 --max_updates 2000
python -u scripts/p2_train_mappo.py --model_suffix m2-shaped   --credit m2 --seed 301 --max_updates 2000
python -u scripts/p2_train_mappo.py --model_suffix coma-critic --algo coma --seed 301 --max_updates 2000
python -u scripts/p2_train_mappo.py --model_suffix single-joint --algo single --seed 301 --max_updates 2000

# fleet-term ablation: the headline arm with the fleet-capacity term removed from the
# bound, so reward, action prices and credit all read the chain and machine terms only
python -u scripts/p2_train_mappo.py --model_suffix m1-bcb-guide-noveh --credit m1 --guide \
    --bound_veh 0 --seed 301 --max_updates 2000

# ample-fleet ("uncontended") arm, deployed later under a fixed vehicle rule
python -u scripts/p2_train_mappo.py --model_suffix e0b-uncontended --fleet_grid 10 \
    --seed 301 --max_updates 1000

# scarce-cell reruns used for the explicit-coupling comparison (seeds 301, 302, 303).
# Both arms share the undivided bound-difference reward and no price channel; only the fleet
# the trainer sees, and whether a fixed vehicle rule replaces the vehicle head, differ.
python -u scripts/p2_train_mappo.py --model_suffix g1r-explicit --fleet_grid 1,2 \
    --ratio_grid 0.6,1.0 --vali_cells v1+t1.0,v2+t1.0 --vali_every 20 \
    --seed 301 --max_updates 1000
python -u scripts/p2_train_mappo.py --model_suffix g1r-e0b --fleet_grid 10 \
    --ratio_grid 0.6,1.0 --fixed_veh_rule NVF --vali_cells v1+t1.0,v2+t1.0 \
    --vali_every 20 --seed 301 --max_updates 1000

# non-admissible price control: same channel shape, myopic duration content,
# mean-matched by the scale calibrated with scripts/x2_calibrate_naive_price.py
python -u scripts/p2_train_mappo.py --model_suffix m1-bcb-guide-naive --credit m1 \
    --guide --guide_price naive --guide_price_scale 0.0185 --seed 301 --max_updates 2000

# train-in-regime anchor for the zero-shot travel-intensity column
python -u scripts/p2_train_mappo.py --model_suffix m1-bcb-guide-t10 --credit m1 --guide \
    --seed 301 --max_updates 2000 --ratio_grid 1.0

# size-mixture arm used by the scale-up exhibit (10, 15 and 20 modules in training)
python -u scripts/x2_gate_mix_smoke.py     # 20-update calibration smoke, run first
python -u scripts/p2_train_mappo.py --model_suffix m1-bcb-guide-mix --credit m1 --guide \
    --seed 301 --max_updates 2000 --size_mix 10,15,20 --vali_size_mix 10,20

# fleet-term ablation of the size-mixture arm; scripts/run_x2_ablation_noveh_mix.sh wraps
# this command and first checks the resolved configuration against the comparator
# (p2_train_mappo.py --dry_run, then scripts/x2_preflight_config.py)
python -u scripts/p2_train_mappo.py --model_suffix m1-bcb-guide-mix-noveh --credit m1 --guide \
    --seed 301 --max_updates 2000 --size_mix 10,15,20 --vali_size_mix 10,20 --bound_veh 0

# external anchor (Link JSSPT distribution); checkpoints named 15x10+link+{suffix}-s{seed}
python -u scripts/p2_train_mappo.py --dist link --model_suffix link-m1 --credit m1 \
    --seed 301 --max_updates 2000 --fleet_grid 3,6,9,12,15,18
python -u scripts/p2_train_mappo.py --dist link --model_suffix link-m1-guide --credit m1 --guide \
    --seed 301 --max_updates 2000 --fleet_grid 3,6,9,12,15,18
```

Seed coverage, per arm: the ample-fleet anchor and the external `link-m1`
ablation are single runs on seed 301. Every other arm ships three seeds (301,
302, 303), including the scarce-cell reruns, the non-admissible price control,
the train-in-regime anchor, both fleet-term ablations (10 modules and
size mixture), the external `link-m1-guide` arm and the size-mixture arms. Run one training job
at a time: two concurrent jobs on one GPU roughly double each other's
wall-clock, which corrupts any timing comparison.

`trained_network/PPVCT/` already holds the 96 checkpoint files (48 arm-seed
pairs, best plus final) these commands produce, so training can be skipped
entirely.

---

## Evaluation

```bash
# greedy rollouts on the PPVC-T test cells, with independent feasibility checking
python -u scripts/eval_ppvct.py --model_name 10x25+ppvct-mixed+m1-bcb-guide-s301 --cells all

# a fixed vehicle rule at vehicle events instead of the policy head
python -u scripts/eval_ppvct.py --model_name 10x25+ppvct-mixed+e0b-uncontended-s301 \
    --cells all --veh_rule NVF

# external anchor: our policy decides, decision sequences exported for their executor
python scripts/link_eval.py --model_name 15x10+link+link-m1-s301 --fleets 3,6,9,12,15,18
```

`link_eval.py` writes one replay per instance plus a scored summary. The scored
files are released under `results/external_l1/{arm}/`
(`score_vs_released.json`, holding our mean, the three released anchors and the
Mann-Whitney p per fleet size, and `summary.json`); the raw per-instance replay
JSONs are not, because the command above regenerates them from the released
checkpoint and the released instances. `scripts/fill_macros.py` reads the
scored files from `../external/replays_l1/{arm}/`, next to the repository,
which is where the external benchmark's working copy lives; point it at
`results/external_l1/` or copy the directory across.

Decision latency is reported from a run that had the device to itself, because
a contended measurement is not a measurement. The two console logs behind the
reported times are released as `train_log/latency_uncontended2.log` (GPU) and
`train_log/latency_cpu4_2.log` (four pinned cores, `OMP_NUM_THREADS=4`); each
line carries the cell, the instance count, the mean makespan and the mean wall
time of one batched forward pass. The only edit made to them is that the
absolute interpreter path in a PyTorch warning line was rewritten to
`<site-packages>/`; no measured line was touched.

`eval_ppvct.py` writes one array per cell to
`test_results/PPVCT/{cell}/Result_greedy[-{veh_rule}]+{model}_{cell}.npy`,
holding the per-instance makespan and per-event decision latency. Every
schedule is re-checked by `transport_marl/validator_t.py`, an independent
validator that shares no code with the training environment or the simulator.

Baselines and references (CPU; all are resume-safe per cell):

```bash
python scripts/p1_eval_pdr.py                             # 9 dispatching-rule pairs  -> results/pdr/
python scripts/p8_certificate.py                          # solver-free root bound    -> results/certificate/

# the two reference families the manuscript reports (see the note below)
python scripts/p3_eval_ga_v2.py all --budget 60 --workers 12    # GA v2, 60 CPU-s/inst -> results/ga_v2/
python -u scripts/p2_cpsat_refs_v2.py v1+t0.6 v2+t0.6 \
    --par 3 --workers 4 --time 300                              # strengthened CP-SAT  -> results/cpsat_v2/

# GA search-budget sweep behind the "how much search does the GA need" claim
python scripts/p3_eval_ga_v2.py v1+t0.6 v1+t1.0 v2+t0.6 v2+t1.0 --budget 5 \
    --tag '+b5' --outdir results/ga_v2_budget                   # repeat for 1, 15
python scripts/ga_budget_table.py                               # budget table

# archival first-generation references, kept for provenance, no longer a reported number
python scripts/p3_eval_ga.py all --budget 60 --workers 20 # PDR-seeded GA v1  -> results/ga/
python -u scripts/p2_cpsat_refs.py v1+t0.6 v2+t0.6        # CP-SAT v1         -> or_solution/PPVCT/
```

Two reference families were superseded during the study, and both generations
are released. The reported CP-SAT column comes from the **strengthened** model
in `results/cpsat_v2/` (redundant fleet-capacity cumulative carrying the
vehicle relaxation, warm-start-tightened horizon, vehicle symmetry breaking;
300 s, four search workers); `or_solution/PPVCT/` holds the unstrengthened
first-generation references, which are kept on disk unchanged and are read by
nothing. The reported GA column comes from `results/ga_v2/`, whose vehicle-side
encoding contains the three dispatching vehicle rules exactly; `results/ga/`
holds the first-generation GA. `transport_marl/cpsat_transport.py` reproduces
either model: `strengthen=False` is the v1 semantics, byte-for-byte, so the
archival ledgers stay checkable.

Scale-up (zero-shot at 20, 30, 50 and 80 modules; 30 instances per cell at 20
and 30 modules, 20 per production batch at 50 and 80). The
solver jobs are the expensive part and are core-pinned so they cannot starve a
concurrent trainer:

```bash
python scripts/x2_scale_pdr.py --workers 10                        # -> results/scaleup/pdr/
python scripts/x2_scale_cert.py                                    # -> results/scaleup/certificate/
python scripts/x2_scale_policy.py --model_name 10x25+ppvct-mixed+m1-bcb-guide-s301 \
    --cells 20x25+ppvct-mixed+v1+t0.6 --device cuda                # -> results/scaleup/policy/
python scripts/x2_scale_ga.py --cells 20x25+ppvct-mixed+v1+t0.6 --budget 60 --workers 8
python -u scripts/x2_scale_cpsat.py 20x25+ppvct-mixed+v1+t0.6 \
    --par 3 --workers 4 --time 300 --cores 8-23                    # -> results/scaleup/cpsat/
python -u scripts/x2_scale_cpsat.py 50x25+ppvct-mixed+v2+t0.6 --time_limit 3600 \
    --anytime --out_dir results/scaleup/cpsat_b                    # anytime ledger
python scripts/x2_scale_report.py                                  # every quoted scale number
python scripts/x2_scale_report_b.py                                # phase-B cost accounting
python scripts/x2_scale_table.py                                   # scale table
python scripts/x2_cpsat_halftime.py                                # 1800-s incumbents read off the 3600-s ledgers
python scripts/x2_eval_sample.py --model_name mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-s301 \
    --cells 80x25+ppvct-mixed+v3+t1.0 --n_samples 64                 # best-of-64 decoding -> results/sample_decode/
python scripts/x2_single_vs_marl_mix.py                            # merged-head vs factorized, size-mixture arms
python scripts/x2_production_macros.py                             # every production-batch number the manuscript quotes
```

Regime and mechanism diagnostics (CPU):

```bash
python scripts/x2_fleet_util.py            # minimum loaded fleet utilization, primary grid -> results/fleet_util.json
python scripts/x2_util_extended.py         # the same for the scale-up cells and the external benchmark
python scripts/x2_bound_terms.py           # active bound component per cell            -> results/bound_terms.json
python scripts/x2_bound_terms_grid.py      # the same over the 12 primary-grid cells      -> results/bound_terms_grid.json
python scripts/x2_root_tightness.py        # root bound against the proven optimum where CP-SAT closed the instance
python scripts/e3_regret_map.py --explicit 10x25+ppvct-mixed+joint-v1-s301 \
    --penalty 10x25+ppvct-mixed+e0b-uncontended-s301 --penalty_veh_rule NVF   # coupling-regret map
```

Processing-time heterogeneity (zero-shot, CPU; every level measured by one harness on one device):

```bash
python scripts/x2_hetero_crosscheck.py     # the harness reproduces the stored baseline evaluation
python scripts/x2_hetero_policy.py --model_name 10x25+ppvct-mixed+m1-bcb-guide-s301   # repeat for the shared-reward arm and seeds
python scripts/x2_hetero_pdr.py            # the nine dispatching-rule pairs
python scripts/x2_hetero_bound.py          # root bound and active component per level
python scripts/x2_hetero_report.py         # -> results/hetero/summary.json and the tables
```

Breakdown recovery (one machine fails mid-execution; the policy replans from the updated state):

```bash
python scripts/x2_disruption.py --cell 50x25+ppvct-mixed+v2+t1.0 \
    --model_name mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-s301   # -> results/disruption/
python scripts/x2_disruption_report.py     # recovered makespans per repair method and budget
```

The GA on the large cells records `startup_cpu` and `search_cpu` separately,
because at 20 modules and above the fixed cost of starting the search already
exceeds the budget being varied, and a bare "60 CPU-s" label would understate
what the baseline costs a user.

Statistical comparisons:

```bash
python -u scripts/gate_eval.py --gate G1 \
    --explicit 10x25+ppvct-mixed+joint-v1-s301 \
    --baseline 10x25+ppvct-mixed+e0b-uncontended-s301 --baseline_veh_rule NVF \
    --cells v1+t0.6,v2+t0.6 --alpha 0.05      # paired Wilcoxon with Holm correction
python scripts/g2_final.py --seeds 301,302,303   # credit-assignment comparison, 3 seeds
python scripts/e4_final.py --seeds 301,302,303   # equivalence test (TOST) against the merged-head control
```

The unit of replication is the test instance. Per-instance makespans are first
averaged over training seeds, so seeds are treated as a nuisance factor rather
than as independent replicates; the equivalence margin was fixed before the
merged-head arm was trained.

Acceptance tests for each component:

```bash
python scripts/p1_sim_tests.py            # simulator + validator on real instances
python scripts/p1_env_equiv.py            # batched environment == reference simulator
python scripts/p1_cpsat_sanity.py         # CP-SAT model against known-optimal special cases
python scripts/p2_bound_and_nn_tests.py   # bound admissibility and tightness at termination
python scripts/p3_bcb_tests.py            # analytic credit baseline properties
python scripts/p4_guide_tests.py          # action-price channel shapes and masking
```

---

## Supplement studies

Each study below has its own result folder and one script that turns the
result files into the supplement table. The table scripts print by default;
`--write` also patches the manuscript's macro file and supplement table, which
are not part of this release (see "Reproducing the reported numbers"). The
`run_x2_*.sh` wrappers hold the full launch settings (pinned cores, thread
caps, resume and busy-machine checks); run them from the repository root, with
`PY` pointing at the Python interpreter if `python` is not the right one.

| Supplement section or table | Result files | Produced by | Table or numbers from |
| --- | --- | --- | --- |
| Scale-up compute sweep: CP-SAT without a warm start | `results/scaleup/cpsat_cold/{cell}+cold.{json,jsonl}` | `scripts/x2_scale_cpsat.py --cold` | read directly from the ledgers; the warm-started runs on the same instances are in `results/cpsat_v2/` (10 modules) and `results/scaleup/cpsat_b/` (50 modules) |
| Scale-up: 3600-s anytime CP-SAT at 80 modules (20 of 20 instances) | `results/scaleup/cpsat_b/80x25+ppvct-mixed+v3+t1.0.{json,jsonl}` | `scripts/x2_scale_cpsat.py --anytime` | `scripts/x2_production_macros.py`, `scripts/x2_cpsat_halftime.py` |
| Wall times on dedicated resources (best-of-64 decoding on an exclusive GPU; one schedule on one CPU core) | `results/sample_decode/*+excl.json`, `results/scaleup/policy/*+b1lat.json` | `scripts/run_x2_timing_excl.sh` | `scripts/x2_production_macros.py` |
| One-vehicle production batches with short moves | `results/shorttravel/{pdr,ga,policy,policy_cpu}/` | `scripts/run_x2_shorttravel.sh` (rules and GA), `scripts/run_x2_shorttravel_gpu.sh` (policies) | `scripts/x2_shorttravel_report.py` |
| Fleet-term ablation, 10 modules | `test_results/PPVCT/{cell}/Result_greedy+10x25+ppvct-mixed+m1-bcb-guide-noveh-s*.npy` | `scripts/run_x2_ablation_noveh.sh` | `scripts/x2_ablation_noveh_report.py` |
| Fleet-term ablation, size mixture | `results/ablation/noveh_mix/{policy,test_results}/` | `scripts/run_x2_ablation_noveh_mix.sh` (training), `scripts/run_x2_ablation_noveh_mix_eval.sh` (evaluation) | `scripts/x2_ablation_noveh_mix_report.py` |
| Schedules of BOLT and the transport-as-penalty anchor | `figures_src/data/s_gantt_schedules.json` (the two plotted schedules) | `figures_src/make_gantt.py` | the figure itself |
| Recovery from a machine breakdown, with the search arms | `results/disruption/{cell}.jsonl` | `scripts/x2_disruption.py`; `scripts/run_x2_disruption_search.sh` for the GA and CP-SAT arms | `scripts/x2_disruption_report.py`, `scripts/x2_disruption_search_table.py` |
| Execution under duration noise | `results/execution_noise/{cell}.jsonl`, with the initial plans in `results/execution_noise/plans/{cell}/` | `scripts/x2_execution_noise.py` via `scripts/run_x2_execution_noise.sh` | `scripts/x2_execution_noise_table.py` |
| Deployment cost by instance size | `results/latency/x2_latency_memory_{cpu,cuda}.json` | `scripts/x2_latency_memory.py` via `scripts/run_x2_latency_memory.sh` | `scripts/x2_latency_macros.py` |

```bash
# CP-SAT without a warm start, one cell per size
python -u scripts/x2_scale_cpsat.py 10x25+ppvct-mixed+v1+t0.6 --par 1 --workers 4 --time 600 \
    --n 5 --anytime --out_dir results/scaleup/cpsat_cold --cold
python -u scripts/x2_scale_cpsat.py 50x25+ppvct-mixed+v2+t1.0 --par 1 --workers 4 --time 3600 \
    --n 3 --anytime --out_dir results/scaleup/cpsat_cold --cold

# breakdown recovery: the arms without search budget, then the GA and CP-SAT arms
python scripts/x2_disruption.py --cell 50x25+ppvct-mixed+v2+t1.0 \
    --model_name mix10-15-20x25+ppvct-mixed+m1-bcb-guide-mix-s301 \
    --methods right_shift,pdr,policy --baseline policy
bash scripts/run_x2_disruption_search.sh --dry_run    # lists the remaining runs
python scripts/x2_disruption_search_table.py

# execution under duration noise (seed-301 size-mixture policy, 3 replicates per instance)
CORES=0-3 bash scripts/run_x2_execution_noise.sh
python scripts/x2_execution_noise_table.py

# short moves with one vehicle, then the report
bash scripts/run_x2_shorttravel.sh && bash scripts/run_x2_shorttravel_gpu.sh
python scripts/x2_shorttravel_report.py

# deployment cost by instance size, on a quiet machine
CORE=4 N=3 bash scripts/run_x2_latency_memory.sh
python scripts/x2_latency_macros.py

# fleet-term ablations
python scripts/x2_ablation_noveh_report.py
python scripts/x2_ablation_noveh_mix_report.py

# schedule figure (reads the stored schedules; no rollout needed)
python figures_src/make_gantt.py
```

The execution-noise study caches each instance's initial plans (the policy
rollout, the best dispatching pair and the 60-CPU-s GA schedule) in
`results/execution_noise/plans/`. The GA budget is CPU time, so a GA run on
another machine does not return the same schedule; with the cached plans
present, a rerun executes exactly the plans behind the reported ledgers. The
two latency files keep every measured field; the machine's host name and the
commit identifier of the unreleased working tree are replaced by `<masked>`.

---

## Reproducing the reported numbers

Every quantity in the manuscript is a macro produced by `scripts/fill_macros.py`
from the released result files. Nothing is typed by hand, and a macro whose
result file is missing stays marked as preliminary rather than being invented.

```bash
python scripts/fill_macros.py
```

The mapping from files to reported numbers:

| Reported quantity | Read from |
| --- | --- |
| Policy makespans, per cell and per instance | `test_results/PPVCT/{cell}/Result_greedy*.npy` |
| Dispatching-rule baselines (9 rule pairs) | `results/pdr/{cell}.json` |
| Genetic-algorithm baseline (60 CPU-s per instance) | `results/ga_v2/{dataset}.json` |
| Genetic-algorithm budget sweep (0.25, 1, 5, 15 CPU-s) | `results/ga_v2_budget/{dataset}+b{budget}.json` |
| CP-SAT reference and its optimality gap `(UB-LB)/LB` | `results/cpsat_v2/{dataset}.jsonl` |
| Optimality certificates (solver-free root bound) | `results/certificate/{cell}.json` |
| Coordination and lazy-agent diagnostics | `results/diagnostics/{model}_{cell}_greedy.json` |
| Scale boundary: policy, GA, CP-SAT and rules at 20 to 80 modules | `results/scaleup/{policy,ga,ga_long,ga_budget,cpsat,cpsat_b,pdr,certificate}/` |
| External benchmark, scored in its authors' simulator | `results/external_l1/{arm}/score_vs_released.json` |
| Minimum loaded fleet utilization per cell (the regime variable) | `results/fleet_util.json`, `results/fleet_util_extended.json` |
| Active bound component at the root, per cell | `results/bound_terms.json`, `results/bound_terms_grid.json` |
| CP-SAT incumbents at 1800 s (production cells) | `results/scaleup/cpsat_b/*.jsonl`, read by `scripts/x2_cpsat_halftime.py` |
| Best-of-64 sampled decoding | `results/sample_decode/`; files tagged `+excl` repeat the timing on an exclusive GPU |
| Time for one 50-module schedule on one CPU core | `results/scaleup/policy/*+b1lat.json` |
| Fleet-term ablation (headline arm without the fleet-capacity term) | `test_results/PPVCT/{cell}/Result_greedy+10x25+ppvct-mixed+m1-bcb-guide-noveh-s*.npy`, read by `scripts/x2_ablation_noveh_report.py` |
| Processing-time heterogeneity study | `results/hetero/summary.json` (from `results/hetero/{policy,pdr,bound}/`) |
| Breakdown-recovery comparison | `results/disruption/` |
| Merged-head against factorized heads, size-mixture arms | `test_results/PPVCT/` and `results/scaleup/policy/`, read by `scripts/x2_single_vs_marl_mix.py` |
| Per-decision latency, GPU and four CPU cores | `train_log/latency_uncontended2.log`, `train_log/latency_cpu4_2.log` |
| Seed count actually trained per arm | `trained_network/PPVCT/*.pth` |
| Statistical verdicts (paired tests, equivalence tests) | `reports/*.md`, written by `gate_eval.py`, `g2_final.py` and `e4_final.py` |
| Archival first-generation references (read by nothing) | `or_solution/PPVCT/{cell}.jsonl`, `results/ga/{cell}.json` |

The statistical comparisons run before `fill_macros.py`: `gate_eval.py`,
`g2_final.py`, `e4_final.py` and `e4_tost_run.py` each write a verdict block to
`reports/`, creating the directory if it does not exist, and `fill_macros.py`
parses those blocks back into macros. Run them in that order on a fresh clone.

`fill_macros.py` rewrites a marked block inside the manuscript's macro file;
the manuscript source is not part of this release, so point the `MACROS`
constant at your own file or read the values from the console output.

Figures are rendered from the same result files:

```bash
python figures_src/make_f3_f4.py           # credit-assignment figure and coupling-regret map
python figures_src/make_f5_scale.py        # scale boundary, budget sweep and anytime panels
python figures_src/make_f4b_s1main.py      # active-component panel and the three-arm training curves
python figures_src/make_s5_tightness.py    # root-bound tightness by regime
python scripts/f3_credit_figure.py         # credit-assignment figure, standalone
python scripts/e3_regret_map.py --explicit 10x25+ppvct-mixed+joint-v1-s301 \
    --penalty 10x25+ppvct-mixed+e0b-uncontended-s301 --penalty_veh_rule NVF
```

---

## Repository layout

| Path | Contents |
| --- | --- |
| `transport_marl/` | The method. Batched environment (`fjsp_env_transport.py`), single-instance reference simulator (`sim_single.py`), certified multi-bottleneck bound (`bound.py`), analytic credit baseline (`bcb.py`), action-price channel (`guide.py`, certified and non-admissible content), two-headed network (`model_transport.py`), MAPPO trainer (`mappo.py`), station layout and travel-time model (`layout.py`), independent validator (`validator_t.py`), CP-SAT model (`cpsat_transport.py`, plain and strengthened), genetic algorithms (`ga_transport.py`, `ga_transport_v2.py`), dispatching rules (`pdr_pairs.py`), comparator learners (`coma_baseline.py`, `single_agent.py`), diagnostics (`diagnostics.py`), external-benchmark adapters (`external_adapter.py`) |
| `scripts/` | Dataset generation, training, evaluation, baselines, statistics, acceptance tests; the `x2_scale_*` family is the scale-up pipeline |
| `figures_src/` | Manuscript figures and the shared plotting style |
| `data/` | PPVC-T, PPVC-T heterogeneity and Link instance files |
| `trained_network/PPVCT/` | 96 checkpoint files (48 arm-seed pairs, best plus final), covering every reported arm and seed |
| `train_log/PPVCT/` | Per-model hyperparameter snapshots (required to rebuild networks for the released checkpoints) |
| `train_log/latency_*2.log` | Console logs of the two uncontended decision-latency runs |
| `results/external_l1/` | External-benchmark scores, one directory per training seed (raw replays regenerable with `scripts/link_eval.py`) |
| `results/cpsat_v2/`, `results/ga_v2/`, `results/ga_v2_budget/` | The CP-SAT and GA references the manuscript reports, and the GA budget sweep |
| `results/scaleup/` | Scale-up evidence at 20 to 80 modules: policy rollouts, GA (60 s, 600 s and the small-budget sweep), CP-SAT (300 s and the 3600 s anytime ledgers), dispatching rules, root bounds |
| `results/pdr`, `results/certificate`, `results/diagnostics` | Dispatching-rule baselines, certificates and diagnostic result files |
| `results/hetero/`, `results/bound_terms*.json`, `results/fleet_util*.json`, `results/disruption/`, `results/sample_decode/` | Heterogeneity study, active bound components, fleet utilization, breakdown recovery, sampled decoding |
| `results/execution_noise/`, `results/shorttravel/`, `results/ablation/noveh_mix/`, `results/scaleup/cpsat_cold/`, `results/latency/` | Duration noise, short-move production batches, size-mixture fleet-term ablation, CP-SAT without a warm start, deployment cost |
| `or_solution/PPVCT/`, `results/ga/` | Archival first-generation CP-SAT and GA references, kept unchanged for provenance |
| `test_results/PPVCT/` | Per-instance evaluation arrays for every released checkpoint |
| `model/`, `fjsp_env_same_op_nums.py`, `ortools_solver.py`, `common_utils.py`, `data_utils.py`, `params.py` | Base flexible job-shop scaffolding adapted from prior work (see `NOTICE`) |
| `ppvc_instance_generator.py` | Instance generator for the production-scheduling distribution |

---

## License

No blanket open-source license is applied to this repository; read `NOTICE` for
the attribution and licensing status before reusing any part of it.
