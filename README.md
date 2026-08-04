# Bound-Guided Multi-Agent Reinforcement Learning for the Transport-Coupled Flexible Job Shop with Time Lags (FJSP-TL-T)

This repository contains the environment, learning code, benchmark datasets,
trained policies and result files behind a study of production scheduling in
which transport is a decision rather than a fixed delay. In the
transport-coupled flexible job shop with time lags (FJSP-TL-T), a limited fleet
of vehicles moves work between stations, a minimum waiting time may separate
consecutive operations of a job, and machine assignment and vehicle assignment
must be decided together, because a good machine choice that no vehicle can
serve in time is not a good choice. The scheduler is a two-headed
parameter-shared policy (one head for machine-class events, one for
vehicle-class events) trained with multi-agent PPO under centralized training
and decentralized execution. Three mechanisms come from a single admissible
lower bound on the makespan of any feasible completion of a partial schedule:
the bound's step-to-step decrease is the *telescoped team reward*, so the
return of an episode equals its makespan improvement exactly; the same bound
evaluated at a virtual "defer" state gives a *closed-form counterfactual
baseline* that splits credit between the two heads without a learned critic per
agent; and the bound evaluated at each candidate successor state gives a
*certified action-price channel* appended to the policy's input, so the network
sees what each legal action would cost before taking it. Because the bound is
computable without a solver, the same quantity also yields a per-instance
optimality certificate for every schedule the policy produces.

**Reference:** anonymized manuscript under double-blind review at *IEEE
Transactions on Industrial Informatics*.

Per-cell CP-SAT, dispatching-rule and genetic-algorithm reference solutions are
included, along with the per-instance evaluation arrays for every released
checkpoint. Every comparison and every statistic reported in the manuscript can
therefore be recomputed from this repository without re-solving or re-training
anything.

One file is still growing at the time of writing: the anytime CP-SAT ledger
`results/scaleup/cpsat_b/50x25+ppvct-mixed+v2+t0.6.jsonl`, whose 3600 s solves
are being extended by a running batch. It is released as-is, and the manuscript
number that reads it (the wall-clock at which CP-SAT first overtakes the policy
at 50 modules) is recomputed and finalized from the completed ledger before
submission. Every other result file here is final.

### Artifact inventory

The manuscript's reproducibility footnote names six families. Each one is here:

| Family | Where it lives |
| --- | --- |
| Generator | `ppvc_instance_generator.py` and `transport_marl/layout.py` (the station-cell layout and travel-time model), driven by `scripts/p1_make_ppvct_data.py`, `scripts/p7_make_transfer_data.py` and `scripts/x2_scale_make_data.py`; `transport_marl/external_adapter.py` ports the external distribution |
| Instance data | `data/PPVCT/` (training grid, held-out transfer cells, scale-up cells) and `data/LINK/15x10/` |
| CP-SAT references | `results/cpsat_v2/` (reported), `results/scaleup/{cpsat,cpsat_b}/` (scale-up, including the anytime ledgers), `or_solution/PPVCT/` (archival); model in `transport_marl/cpsat_transport.py` |
| Dispatching-rule references | `results/pdr/` and `results/scaleup/pdr/`; rules in `transport_marl/pdr_pairs.py` |
| GA references | `results/ga_v2/` (reported), `results/ga_v2_budget/` and `results/scaleup/{ga,ga_long,ga_budget}/` (budget sweeps), `results/ga/` (archival); solvers in `transport_marl/ga_transport_v2.py` and `ga_transport.py` |
| Trained models | `trained_network/PPVCT/` (64 checkpoint files) with the hyperparameter snapshots in `train_log/PPVCT/` that the loaders read to rebuild each network |
| Evaluation code | `scripts/` (rollouts, baselines, statistics, acceptance tests, scale-up pipeline), `transport_marl/validator_t.py` (independent feasibility checking), `figures_src/` (figures), with the per-instance outputs in `test_results/PPVCT/`, `results/diagnostics/`, `results/certificate/` and `results/scaleup/` |
| External-benchmark scores | `results/external_l1/{arm}/score_vs_released.json` and `summary.json`, one directory per training seed: our schedules re-timed in the benchmark authors' own simulator, against their released anchors |
| Decision-latency evidence | `train_log/latency_uncontended2.log` (GPU) and `train_log/latency_cpu4_2.log` (four pinned CPU cores), the console logs of the runs that produced the reported per-decision times |

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
| PPVC-T scale-up cells | `data/PPVCT/{20,30}x25+ppvct-mixed+...`, `data/PPVCT/{50,80}x25+ppvct-mixed+...` (7 cells) | 20, 30, 50 and 80 modules against the 10-module training size; 30 test instances per cell, test split only |
| Link JSSPT port | `data/LINK/15x10/{test,vali}` | 15 jobs x 10 machines, external anchor distribution, with `PROVENANCE.json` per split |

Regenerate them:

```bash
# PPVC-T training grid: V x r cells, 100 test (seed0=20000) + 100 vali (seed0=10000)
python scripts/p1_make_ppvct_data.py          # add --smoke for a 3-instance dry run

# held-out transfer cells (test split only; never seen in training or validation)
python scripts/p7_make_transfer_data.py

# scale-up cells, 30 instances each (phase A = 20 and 30 modules, phase B = 50 and 80)
python scripts/x2_scale_make_data.py --phase A
python scripts/x2_scale_make_data.py --phase B
```

The seven scale-up cells are zero-shot only: training and model selection
happen at 10 modules, and their test seeds (20000 onward) cannot collide with
a training seed.

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
  heads the undivided team reward, `m1` is the bound-counterfactual baseline,
  `m2` is the shaped-difference variant.
- `--guide` appends the certified action-price channel to both action grids.
- `--algo {mappo,coma,single}` selects the learner: multi-agent PPO, the
  learned-critic counterfactual comparator, or the single-agent joint policy.
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
# headline policy: bound-counterfactual credit + certified action prices
python -u scripts/p2_train_mappo.py --model_suffix m1-bcb-guide --credit m1 --guide \
    --seed 301 --max_updates 2000

# credit-assignment ladder (ablations of the headline arm)
python -u scripts/p2_train_mappo.py --model_suffix joint-v1    --seed 301 --max_updates 2000
python -u scripts/p2_train_mappo.py --model_suffix m1-bcb      --credit m1 --seed 301 --max_updates 2000
python -u scripts/p2_train_mappo.py --model_suffix m2-shaped   --credit m2 --seed 301 --max_updates 2000
python -u scripts/p2_train_mappo.py --model_suffix coma-critic --algo coma --seed 301 --max_updates 2000
python -u scripts/p2_train_mappo.py --model_suffix single-joint --algo single --seed 301 --max_updates 2000

# ample-fleet ("uncontended") arm, deployed later under a fixed vehicle rule
python -u scripts/p2_train_mappo.py --model_suffix e0b-uncontended --fleet_grid 10 \
    --seed 301 --max_updates 1000

# scarce-cell reruns used for the explicit-coupling comparison (seeds 301, 302, 303).
# Both arms share the undivided team reward and no price channel; only the fleet
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

# external anchor (Link JSSPT distribution); checkpoints named 15x10+link+{suffix}-s{seed}
python -u scripts/p2_train_mappo.py --dist link --model_suffix link-m1 --credit m1 \
    --seed 301 --max_updates 2000 --fleet_grid 3,6,9,12,15,18
python -u scripts/p2_train_mappo.py --dist link --model_suffix link-m1-guide --credit m1 --guide \
    --seed 301 --max_updates 2000 --fleet_grid 3,6,9,12,15,18
```

Seed coverage, per arm: the primary-grid arms, the scarce-cell reruns and the
external `link-m1-guide` arm are trained on three seeds (301, 302, 303); the
ample-fleet anchor, the external `link-m1` ablation, the non-admissible price
control and the train-in-regime anchor are single runs on seed 301. The
size-mixture arm ships seed 301 here; its seeds 302 and 303 were still training
when this snapshot was taken and are added when they land. Run one training job
at a time: two concurrent jobs on one GPU roughly double each other's
wall-clock, which corrupts any timing comparison.

`trained_network/PPVCT/` already holds the 64 checkpoint files (32 arms, best
plus final) these commands produce, so training can be skipped entirely.

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

Scale-up (zero-shot at 20, 30, 50 and 80 modules; 30 instances per cell). The
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
python scripts/e4_final.py --seeds 301,302,303   # equivalence test (TOST) against the single-agent policy
```

The unit of replication is the test instance. Per-instance makespans are first
averaged over training seeds, so seeds are treated as a nuisance factor rather
than as independent replicates; the equivalence margin was fixed before the
single-agent arm was trained.

Acceptance tests for each component:

```bash
python scripts/p1_sim_tests.py            # simulator + validator on real instances
python scripts/p1_env_equiv.py            # batched environment == reference simulator
python scripts/p1_cpsat_sanity.py         # CP-SAT model against known-optimal special cases
python scripts/p2_bound_and_nn_tests.py   # bound admissibility and tightness at termination
python scripts/p3_bcb_tests.py            # counterfactual baseline properties
python scripts/p4_guide_tests.py          # action-price channel shapes and masking
```

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
| Per-decision latency, GPU and four CPU cores | `train_log/latency_uncontended2.log`, `train_log/latency_cpu4_2.log` |
| Seed count actually trained per arm | `trained_network/PPVCT/*.pth` |
| Statistical verdicts (paired tests, equivalence tests) | `notes/*.md`, written by `gate_eval.py`, `g2_final.py` and `e4_final.py` |
| Archival first-generation references (read by nothing) | `or_solution/PPVCT/{cell}.jsonl`, `results/ga/{cell}.json` |

The statistical comparisons run before `fill_macros.py`: `gate_eval.py`,
`g2_final.py`, `e4_final.py` and `e4_tost_run.py` each write a verdict block to
`notes/`, creating the directory if it does not exist, and `fill_macros.py`
parses those blocks back into macros. Run them in that order on a fresh clone.

`fill_macros.py` rewrites a marked block inside the manuscript's macro file;
the manuscript source is not part of this release, so point the `MACROS`
constant at your own file or read the values from the console output.

Figures are rendered from the same result files:

```bash
python figures_src/make_f3_f4.py           # credit-assignment figure and coupling-regret map
python figures_src/make_f5_scale.py        # scale boundary, budget sweep and anytime panels
python scripts/f3_credit_figure.py         # credit-assignment figure, standalone
python scripts/e3_regret_map.py --explicit 10x25+ppvct-mixed+joint-v1-s301 \
    --penalty 10x25+ppvct-mixed+e0b-uncontended-s301 --penalty_veh_rule NVF
```

---

## Repository layout

| Path | Contents |
| --- | --- |
| `transport_marl/` | The method. Batched environment (`fjsp_env_transport.py`), single-instance reference simulator (`sim_single.py`), admissible bound (`bound.py`), counterfactual credit baseline (`bcb.py`), action-price channel (`guide.py`, certified and non-admissible content), two-headed network (`model_transport.py`), MAPPO trainer (`mappo.py`), station layout and travel-time model (`layout.py`), independent validator (`validator_t.py`), CP-SAT model (`cpsat_transport.py`, plain and strengthened), genetic algorithms (`ga_transport.py`, `ga_transport_v2.py`), dispatching rules (`pdr_pairs.py`), comparator learners (`coma_baseline.py`, `single_agent.py`), diagnostics (`diagnostics.py`), external-benchmark adapters (`external_adapter.py`) |
| `scripts/` | Dataset generation, training, evaluation, baselines, statistics, acceptance tests; the `x2_scale_*` family is the scale-up pipeline |
| `figures_src/` | Manuscript figures and the shared plotting style |
| `data/` | PPVC-T and Link instance files (59 MB) |
| `trained_network/PPVCT/` | 64 checkpoint files (32 arms, best plus final), 12 MB, covering every reported arm and seed |
| `train_log/PPVCT/` | Per-model hyperparameter snapshots (required to rebuild networks for the released checkpoints) |
| `train_log/latency_*2.log` | Console logs of the two uncontended decision-latency runs |
| `results/external_l1/` | External-benchmark scores, one directory per training seed (raw replays regenerable with `scripts/link_eval.py`) |
| `results/cpsat_v2/`, `results/ga_v2/`, `results/ga_v2_budget/` | The CP-SAT and GA references the manuscript reports, and the GA budget sweep |
| `results/scaleup/` | Scale-up evidence at 20 to 80 modules: policy rollouts, GA (60 s, 600 s and the small-budget sweep), CP-SAT (300 s and the 3600 s anytime ledgers), dispatching rules, root bounds |
| `results/pdr`, `results/certificate`, `results/diagnostics` | Dispatching-rule baselines, certificates and diagnostic result files |
| `or_solution/PPVCT/`, `results/ga/` | Archival first-generation CP-SAT and GA references, kept unchanged for provenance |
| `test_results/PPVCT/` | Per-instance evaluation arrays for every released checkpoint |
| `model/`, `fjsp_env_same_op_nums.py`, `ortools_solver.py`, `common_utils.py`, `data_utils.py`, `params.py` | Base flexible job-shop scaffolding adapted from prior work (see `NOTICE`) |
| `ppvc_instance_generator.py` | Instance generator for the production-scheduling distribution |

---

## License

No blanket open-source license is applied to this repository; read `NOTICE` for
the attribution and licensing status before reusing any part of it.
