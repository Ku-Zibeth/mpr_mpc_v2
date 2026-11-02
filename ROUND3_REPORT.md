# Third-Round Delivery

## Scope

Started from clean `main` at `d2cbefb6af4af5c7cef31dac13fc2651fd57c897`, after
`git fetch origin main` and `git pull --ff-only origin main` confirmed it was
current. Only this V2 repository was edited. No original model was overwritten,
no trained weights were copied, and no formal 50K training was launched.

The architecture remains Lattice -> Residual SAC -> WM evaluation -> MPPI.
Defaults now implement the requested 20K/50K schedule: WM/cost training in A,
frozen WM and SAC training in B. MPPI is unconditionally off for training
actions, even beyond the deployment threshold; it remains available to
evaluation at/after 50K. Named E0--E5 profiles and 527/535 inputs are retained.

## TTC And Road Risk

Swept separating-axis entry is local to each future sample, so TTC is now the
minimum of `sample_time + local_entry`, subject to the absolute lookahead.
Future contact at .4s is not TTC=0. Current overlap is 0, separating/empty cases
are infinity. The footprint, heading and relative-velocity checks are retained.
Tests exercise current contact, crossing .4s, head-on .6s, stationary 1.2s,
separating, empty and multiple obstacles across dt=.05/.1/.2 and substeps=1/2/5.

Signed boundary margin is positive inside the real road union and negative
outside. Soft boundary risk is `max(0, safe_margin-margin)/max(safe_margin,1e-6)`.
It is added to clearance and TTC risks and actually subtracted from MPPI scores;
it does not replace hard road/footprint rejection. Default desired margin .5m
is near the .524m center clearance on a 3.5m lane with the 1.852m car plus .3m
inflation on both sides. Multi-lane center trajectories are not penalized.
Boundary weights, baseline/selected risk, signed margin and reduction are logged.

## Mode Calibration

Actual sampler IDs are 0 nominal, 1 braking, 2 right_lateral, 3 left_lateral,
4 joint_right, 5 joint_left. These are sampling-component labels, not inferred
from realized action signs. Previous-best candidates retain their source label;
expanded emergency/alternative proposals are unknown (-1), so receive no cost
permission. Unsigned lateral/joint readiness requires BOTH signed components.
The coarse Lattice trajectory is unlabelled, not an MPPI nominal sample; it
cannot inherit nominal approval. Its learned-cost diagnostic remains unavailable.
The regenerated SAC sequence is the actual nominal baseline of the proposal
bank, so nominal approval can cover its cost diagnostic/auxiliary label.

`tools/calibrate_cost_modes.py` generates candidates from the actual proposal
bank at a shared anchor. For every candidate it resets the same scene, replays
the exact stored historical controls, checks ego/obstacle fingerprint AND
observation agreement (1e-5 absolute tolerance), then executes controls in the
real simulator. Labels come exclusively from actual environment `info.cost`.
No MPPI optimization, training optimizer, simulator clone or imaginary label is
used. Early termination yields shorter records; insufficient horizons cannot
gain samples. Candidates are not discarded for being unsafe, which would bias
calibration toward zero cost. This is simulation-only intervention collection.

Scene IDs must be disjoint from the recorded training pool except explicitly
reserved held-out maps. Old checkpoints without this provenance require an
explicit, truthful training range. The tool refuses existing output paths.
It saves numeric `mode_replays.npz`, `calibration.json`, and a NEW
`calibrated.pt`, preserving the original checkpoint container and weight file.
The manifest contains cost definition, data checksum, scene provenance, seed,
mode mapping and per-mode/per-horizon audit statistics. Loading this dataset
checks checksum, scene provenance and matching environment cost definition.

For each mode/horizon, statistics include replay count, unique scenes, predicted
and actual mean costs, MAE, RMSE, Pearson, cross-scene pairwise ranking and 95%
scenario-cluster bootstrap intervals for MAE/RMSE/ranking. Full per-replay
controls/labels are retained in the numeric dataset for reproducible predictions.
Bootstrap resamples scene clusters, not overlapping trajectory windows. Ranking
uses actual gaps >.05; low-variance/unidentifiable ranking cannot pass.

Default gates: supervised updates>=1000, candidate records>=128, scenes>=10,
MAE<=.1H, RMSE<=.5H, ranking>=.60, ranking CI lower bound>=.50, plus passing
one-step calibration for that same mode. Each H is independent; H10 approval
does not authorize H20. These thresholds are conservative starting settings,
not proof of safety or universal cost-scale calibration.

New serialized `[6,63]` buffers hold approvals/sample/scene counts and a WM update
stamp. ANY WM optimizer step invalidates approvals because representations may
change. Legacy global flags never grant a mode permission. Missing old buffers
load safely with zero approvals; no cost weights means zero supervised count.
Only approved candidate rows invoke the cost head. Other rows have unavailable
predictions and zero *score contribution*, not a declaration of zero actual
danger. Geometry and reward/value continue unchanged. The runtime logs the
approved fraction and per-mode readiness. Legacy executed-window calibration is
diagnostic only and cannot silently approve MPPI distributions.

Example after training E5 on scenes 10000--11999 (32 x 4 records/mode):

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
/home/kzb/anaconda3/envs/tdmpc2/bin/python tools/calibrate_cost_modes.py \
  --checkpoint logs/metadrive-risk/1/mpr_mpc_v2_E5/models/milestone_50k_world_model_residual_rl.pt \
  --output "logs/calibration/E5_seed1_$(date +%Y%m%d_%H%M%S)" \
  --scenarios $(seq 30000 30031) --samples-per-mode 4 --horizon 20 --seed 1
```

This is an independent validation command, NOT run automatically inside training.
For old files, add `--training-start ACTUAL_START --training-count ACTUAL_COUNT`.
Use the new calibrated checkpoint for inference. Optionally set
`cost.mode_validation_dataset=/absolute/path/mode_replays.npz` to revalidate a
trusted independent dataset as WM weights change. Training itself never runs
MPPI to collect modes. A small/constant-safe dataset correctly leaves cost off.

## Best Checkpoint

Default lexicographic priority: minimize collision, offroad, average safety cost;
then maximize success, route completion, reward. The configurable priority must
contain all six metrics. Missing/NaN/infinite required metrics are ineligible.
Default feasibility floors success>=.05 and mean route completion>=.1 reject
zero-progress stopping policies. An additional default guard disallows more than
.10 absolute success/completion loss from the accepted-best high-water marks,
so successive small safety improvements cannot gradually erase throughput.
References are versioned/saved with the best metrics. Set the corresponding
`max_success_drop`/`max_route_completion_drop` to 1 to disable relative guards;
tune and report these constraints for the task.
If nothing meets them, `best.pt` is intentionally absent, but `latest.pt` and
`final.pt` still exist. A best update logs its first deciding metric/reason.
The end of training always evaluates, so default eval_freq=50K cannot leave
best selection comparing only initialization. This evaluation is not a training
MPPI action or replay source. Versioned selection metadata is saved/resumed;
legacy reward-first best tuples are reset, never compared with new six-metric
keys. Changing priority/floors starts a new comparison schema.

## Benchmark

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
/home/kzb/anaconda3/envs/tdmpc2/bin/python tools/benchmark_mppi.py \
  --checkpoint /absolute/path/full_mpr_mpc.pt \
  --output /new/output/benchmark_after.json \
  --scenarios 1714 539 1525 1084 694 \
  --calls 100 --warmup 5 --horizon 10 --samples 64 --iterations 4 --seed 1 \
  --compare /matching/baseline/benchmark_before.json
```

Before measurements use the SAME harness with `--source-root /path/to/baseline`
and a fresh `--output .../benchmark_before.json`, omitting `--compare`.
An existing output JSON is refused. Baseline source is the detached worktree
`/tmp/mpr_v2_r3_before` at the exact starting revision.

The main pass has no phase wrappers and uses CUDA synchronization before/after
every total-wall timer. Five warmups per scene precede 100 valid planning calls.
Twenty identical executed prefix controls establish live states; timed calls
repeat those states without executing selected controls. This prevents changed
selection from changing the comparison workload. JSON records state fingerprints,
effective parameters, revision/dirty state, hardware, checkpoint and seeds.
Comparisons reject mismatched states and workload. Output includes raw call CSV,
Mean/P50/P95/P99/Max/count JSON and `benchmark_comparison.csv`.

A separate instrumented pass uses CPU wall time and CUDA event stream elapsed
for context, candidate generation, safety, WM, cost, scoring, per-mode distribution
update and total planning. CUDA events are synchronized when read. They measure
stream elapsed including CPU scheduling gaps, NOT isolated GPU kernel activity.
Nested phases are not additive: context includes coarse WM; WM includes cost.
For the old implementation scoring profiling covers penalties only. Formal total
wall comparisons are unaffected. No approved mode means cost-head time is 0,
not a claim that executing a calibrated cost head is free.

Baseline profiling: safety checking ~973ms of ~1115ms instrumented total;
context ~62ms, WM ~64ms, generation ~6ms, distribution updates ~19ms. This
motivated caching prepared/inset road geometry and road boundaries. Obstacle
tensors are cached ONLY inside a freshly generated planning context and rebuilt
at every planning tick. No dynamic state survives into the next context.
Redundant visualization rollouts are skipped only when their output is replaced
by the full global safety rollout. No precision, candidate, substep or hard
safety check was removed. The checker remains dominated by real geometry work.

## Changed Files

| File | Third-Round Change |
| --- | --- |
| `planning/safety.py` | Absolute TTC, signed continuous boundary risk, safe context/road caches |
| `planning/local_mppi.py` | Actual mode propagation, row-gated cost scoring, boundary/readiness diagnostics |
| `planning/multimodal.py` | Explicit signed component-ID names |
| `planning/evaluator.py` | Only approved mode/horizon rows invoke cost head |
| `planning/types.py` | Optional per-row cost approval mask |
| `planning/cost_validation.py` | Mode/horizon stats, clustered intervals, readiness gates, dataset checks |
| `planning/coordinator.py` | Training MPPI guard, source IDs in diagnostics, remove overwritten visualization work |
| `vendor/tdmpc2_runtime/common/world_model.py` | Mode buffers, update stamp, explicit invalidation |
| `vendor/tdmpc2_runtime/tdmpc2.py` | Old-buffer compatibility, optimizer invalidation, missing-head handling |
| `agent.py` | Dataset integration, cost-definition guard, metadata, safe best-state migration |
| `trainer.py` | Safety-first best/latest, honest train-MPPI metric, final evaluation |
| `checkpoint_selection.py` | Versioned finite six-metric comparison and progress guards |
| `config.yaml` | Required training defaults, boundary/calibration/selection parameters |
| `schedule/frozen_20k_50k.yaml` | Update comment to match current default |
| `tools/calibrate_cost_modes.py` | Real counterfactual collection and new calibrated checkpoint |
| `tools/benchmark_mppi.py` | Warmup/multi-scene/synchronized comparison benchmark |
| `tools/planning_profiler.py` | Separate diagnostic CPU/CUDA stream timing |
| `tools/smoke_v2.py` | Real training-mode MPPI-off assertion |
| `tests/test_round3.py` | TTC/boundary/calibration/selection/provenance/default-contract tests |
| `tests/test_v2_safety.py` | Correct future TTC expectation and legacy global-gate assertions |
| `tests/test_v2_cost_training.py` | CUDA mixed-mode permissions and save/load compatibility |
| `tests/test_mpr_mpc.py` | Evaluator fixtures accept real mode IDs |
| `README.md`, `V2_IMPLEMENTATION.md`, `ROUND3_REPORT.md` | Current usage, mechanisms, verification and limitations |
| `reports/round3/benchmark_before.json`, `benchmark_after.json` | Actual synchronized measurements and separate profiles |
| `reports/round3/benchmark_before.csv`, `benchmark_after.csv`, `benchmark_comparison.csv` | Raw call timings and comparison |

## Verification And Limits

### Task Status

| Task | Status |
| --- | --- |
| Absolute TTC | Implemented and regression tested, including future .4s crossing |
| Continuous road risk | Implemented, hard rejection retained, actual MPPI selection tested |
| Per-mode cost calibration | Pipeline/gates implemented, real collection tested; production-model readiness NOT established |
| Safety-first best selection | Implemented, finite/progress checks and best/latest tests pass |
| MPPI benchmark/optimization | Reproducible before/after and phase profiling complete; NOT real-time |

### Final Tests

- `/home/kzb/anaconda3/envs/tdmpc2/bin/python -m pytest -q`: **135 passed,
  0 failed, 0 skipped**, 9.21 seconds, including the real CUDA cost optimizer,
  mixed-mode cost gating, new-buffer roundtrip and legacy-global rejection.
- `tools/smoke_v2.py`: PASS, real CUDA/MetaDrive. Cost loss .730696, readiness
  false after one update, positive steering yaw .005857 rad, one-step position
  error .003499m, maximum one-second gentle-turn error .250574m. Training-mode
  planning at global step 50K has MPPI off; evaluation MPPI selects safe=1.
  All three milestone formats roundtrip. A smoke plan measured ~431ms; this is
  not substituted for the formal sustained benchmark.
- Actual 24-step compressed training: A 0--12, B 12--24, deployment threshold24.
  Four WM cost updates, ten SAC optimizer updates, twelve residual transitions,
  frozen=true. Comparing Stage-A and final state dictionaries confirms WM
  parameters/targets stayed exactly unchanged. Final evaluation made eight
  MPPI calls; training actions made none. Three milestone files, latest.pt and
  final.pt exist. best.pt is correctly absent because two-step/eight-step
  truncations do not meet success/progress floors; unit tests cover actual
  best/latest save decisions for eligible data. Artifact directory:
  `/tmp/mpr_v2_r3_train_smoke/metadrive-risk/1/round3_smoke`.
- Actual independent collector: two scenes (539,694), six signed components,
  one candidate/component/scene, H10. All twelve real reset/replays verified.
  The new dataset checksum/provenance loader succeeds; readiness remains false
  at the unchanged default minimum gates. Results and NEW calibrated checkpoint:
  `/tmp/mpr_v2_r3_modes_verified`.
- Twenty unique random scenes: complete E5 evaluator smoke, two actual steps
  per scene, Lattice/SAC/MPPI on, outputs in `/tmp/mpr_v2_r3_eval20`. The zero
  success rate reflects deliberate two-step truncation and tiny training
  weights, NOT a real success-rate estimate. Fixed-100 unique-order/pairing and
  E0--E5 config/runner contracts remain covered by regression tests, not a new
  six-model formal performance experiment.
- `git diff --check`: PASS. Benchmark evidence copied into `reports/round3`;
  weights and large datasets remain outside the repository.

### Measured Performance

Same RTX 3070, checkpoint, five live simulator states, H10, 64 samples, eight
elites, four iterations, five warmups/scene, 100 timed calls/version:

| Milliseconds | Before | After |
| --- | ---: | ---: |
| Mean | 1125.12 | 1079.87 |
| P50 | 800.42 | 765.25 |
| P95 | 1842.96 | 1748.18 |
| P99 | 1882.47 | 1837.64 |
| Max | 1905.37 | 1840.00 |

Measured mean decreased about 4.0% in this single comparison. This includes
correctness/scoring changes and ordinary runtime variation, not an isolated
causal estimate of caching speedup. Repeated runs are needed for a robust
performance claim. After-change separate profiling: safety ~939ms, total
~1078ms. No production mode was calibrated, so cost evaluation time is zero
in BOTH versions. Raw measurements and effective configs are linked above.

All tiny/truncated runs are interface checks, NOT a task success-rate experiment.
No new 50K training or full fixed-100 task-performance evaluation is run here.
Existing fixed-100 paired entry points remain covered by regression tests.

Mode calibration is implemented and the real collector is smoke-tested, but no
production model has yet passed the enlarged per-mode/horizon held-out gates.
Collect diverse risky interventions; merely expanding zero-cost safe scenes
cannot establish risk ranking. Training-only executed windows cannot do so.
The current E5 coarse-Lattice cumulative-cost state pair stays masked: this
collector does not separately calibrate unlabelled coarse Lattice trajectories.
Its geometric risk pairs remain available and the 535 input layout is retained.
Adding a future dedicated coarse-trajectory calibration would change the actor's
cost-feature availability distribution; evaluate that change separately rather
than assuming it learned to use a feature that was masked in training.
The collector currently samples initial bounded mode proposals at one anchor
per scene. It does not certify every later adaptive MPPI distribution or future
traffic interaction; expand anchor speeds/positions and intervention diversity
before drawing model reliability conclusions. Bootstrap intervals are conditional
on these held-out scenes and the chosen cost definition.

Geometry remains an inflated bicycle/constant-velocity approximation, not a
certificate for Bullet/IDM/tyre interactions. .1s real-time control is not
established. These are starting states on five maps and an unready cost head;
benchmark results do not cover full-episode late intersections, calibrated cost
overhead, H20, all map types, rendering or other GPUs. CPU/CUDA stream profiling
does not replace a kernel-level Nsight trace. GUI rendering remains untested.
