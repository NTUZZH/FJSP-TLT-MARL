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
| PPVC-T training grid | `data/PPVCT/10x25+ppvct-mixed+v{V}+t{r}/{test,vali}` | 10 modules, fleet size `V` in {1,2,3}, travel intensity `r` in {0.1,0.3,0.6}; 100 test + 100 validation instances per cell |
| PPVC-T transfer cells | `data/PPVCT/10x25+ppvct-mixed+v4+t{0.6,1.0}`, `data/PPVCT/15x25+ppvct-mixed+v2+t{0.6,1.0}` | held-out fleet size and held-out problem scale, test split only |
| Link JSSPT port | `data/LINK/15x10/{test,vali}` | 15 jobs x 10 machines, external anchor distribution, with `PROVENANCE.json` per split |

Regenerate them:

```bash
# PPVC-T training grid: V x r cells, 100 test (seed0=20000) + 100 vali (seed0=10000)
python scripts/p1_make_ppvct_data.py          # add --smoke for a 3-instance dry run

# held-out transfer cells (test split only; never seen in training or validation)
python scripts/p7_make_transfer_data.py
```

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

# scarce-cell reruns used for the explicit-coupling comparison
python -u scripts/p2_train_mappo.py --model_suffix g1r-explicit --fleet_grid 1,2 \
    --ratio_grid 0.6,1.0 --seed 301 --max_updates 1000
python -u scripts/p2_train_mappo.py --model_suffix g1r-e0b --fleet_grid 10 \
    --ratio_grid 0.6,1.0 --seed 301 --max_updates 1000

# external anchor (Link JSSPT distribution); checkpoints named 15x10+link+{suffix}-s{seed}
python -u scripts/p2_train_mappo.py --dist link --model_suffix link-m1 --credit m1 \
    --seed 301 --max_updates 2000 --fleet_grid 3,6,9,12,15,18
python -u scripts/p2_train_mappo.py --dist link --model_suffix link-m1-guide --credit m1 --guide \
    --seed 301 --max_updates 2000 --fleet_grid 3,6,9,12,15,18
```

Every arm reported in the manuscript was trained on three seeds (301, 302, 303)
except the ample-fleet and scarce-cell reruns, which use seed 301 only. Run one
training job at a time: two concurrent jobs on one GPU roughly double each
other's wall-clock, which corrupts any timing comparison.

`trained_network/PPVCT/` already holds the 46 checkpoints these commands
produce, so training can be skipped entirely.

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

`eval_ppvct.py` writes one array per cell to
`test_results/PPVCT/{cell}/Result_greedy[-{veh_rule}]+{model}_{cell}.npy`,
holding the per-instance makespan and per-event decision latency. Every
schedule is re-checked by `transport_marl/validator_t.py`, an independent
validator that shares no code with the training environment or the simulator.

Baselines and references (CPU; all are resume-safe per cell):

```bash
python scripts/p1_eval_pdr.py                             # 9 dispatching-rule pairs  -> results/pdr/
python scripts/p3_eval_ga.py all --budget 60 --workers 20 # PDR-seeded GA, 60 s/inst   -> results/ga/
python -u scripts/p2_cpsat_refs.py v1+t0.6 v2+t0.6        # warm-started CP-SAT, 300 s -> or_solution/PPVCT/
python scripts/p8_certificate.py                          # solver-free root bound     -> results/certificate/
```

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
| Genetic-algorithm baseline (60 s per instance) | `results/ga/{cell}.json` |
| CP-SAT reference (warm-started, 300 s) | `or_solution/PPVCT/{cell}.jsonl` |
| Optimality certificates (solver-free root bound) | `results/certificate/{cell}.json` |
| Coordination and lazy-agent diagnostics | `results/diagnostics/{model}_{cell}_greedy.json` |
| Seed count actually trained per arm | `trained_network/PPVCT/*.pth` |
| Statistical verdicts (paired tests, equivalence tests) | `notes/*.md`, written by `gate_eval.py`, `g2_final.py` and `e4_final.py` |

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
python scripts/f3_credit_figure.py         # credit-assignment figure, standalone
python scripts/e3_regret_map.py --explicit 10x25+ppvct-mixed+joint-v1-s301 \
    --penalty 10x25+ppvct-mixed+e0b-uncontended-s301 --penalty_veh_rule NVF
```

---

## Repository layout

| Path | Contents |
| --- | --- |
| `transport_marl/` | The method. Batched environment (`fjsp_env_transport.py`), single-instance reference simulator (`sim_single.py`), admissible bound (`bound.py`), counterfactual credit baseline (`bcb.py`), action-price channel (`guide.py`), two-headed network (`model_transport.py`), MAPPO trainer (`mappo.py`), station layout and travel-time model (`layout.py`), independent validator (`validator_t.py`), CP-SAT model (`cpsat_transport.py`), genetic algorithm (`ga_transport.py`), dispatching rules (`pdr_pairs.py`), comparator learners (`coma_baseline.py`, `single_agent.py`), diagnostics (`diagnostics.py`), external-benchmark adapters (`external_adapter.py`) |
| `scripts/` | Dataset generation, training, evaluation, baselines, statistics, acceptance tests |
| `figures_src/` | Manuscript figures and the shared plotting style |
| `data/` | PPVC-T and Link instance files (48 MB) |
| `trained_network/PPVCT/` | 46 checkpoints, 8.4 MB, covering every reported arm and seed |
| `train_log/PPVCT/` | Per-model hyperparameter snapshots (required to rebuild networks for the released checkpoints) |
| `or_solution/PPVCT/` | CP-SAT reference solutions, one JSON Lines file per cell |
| `results/pdr`, `results/ga`, `results/certificate`, `results/diagnostics` | Baseline, certificate and diagnostic result files |
| `test_results/PPVCT/` | Per-instance evaluation arrays for every released checkpoint |
| `model/`, `fjsp_env_same_op_nums.py`, `ortools_solver.py`, `common_utils.py`, `data_utils.py`, `params.py` | Base flexible job-shop scaffolding adapted from prior work (see `NOTICE`) |
| `ppvc_instance_generator.py` | Instance generator for the production-scheduling distribution |

---

## License

No blanket open-source license is applied to this repository; read `NOTICE` for
the attribution and licensing status before reusing any part of it.
