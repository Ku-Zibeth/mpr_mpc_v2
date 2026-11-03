# Round 4 Targeted Repairs

## Version And Scope

Fetched and fast-forward-checked GitHub `main` before editing. Both local and
remote were `9d3d9d8b3d48c41f568366991966fc55f06cd914`; the workspace was clean.
Only this independent V2 repository was changed. Original sibling code,
checkpoints, historical logs and Round 3 measurements are untouched.

The architecture is still Lattice -> Residual SAC -> TD-MPC2 WM -> local MPPI
with an independent geometric checker. Default training remains 50K steps:
0--20K WM and cost-head learning, 20--50K frozen WM and SAC learning, MPPI only
in evaluation. Both default/named schedule use 20 evaluation episodes, no video,
10K evaluation frequency, and WM update budgets A=1, B=C=0. Stage B still records
actual environment costs for the SAC constraint; it never runs a WM optimizer.
The 20K/50K milestone formats and safety-first best/latest/final logic remain.

No formal 50K training or long 100-scene task-performance run is launched here.

## Matched-State Cost Ranking

`anchor_id` hashes the scene ID, simulator seed, executed prefix controls,
prediction step, control dt, observation and a measured state fingerprint.
The fingerprint includes ego XY/velocity/heading/dimensions/previous control and
surrounding actor XY/velocity/heading/rectangle sizes. Each real intervention
resets the same scene, replays the exact prefix and verifies fingerprint AND
observation agreement (absolute tolerance 1e-5). The manifest stores anchor
payloads; loading verifies hashes, observation/scene agreement, file checksum,
held-out-scene provenance and matching environment cost definition.

For each H, the validator evaluates actual recorded controls and discounted
real environment costs from that identical origin. It compares ONLY pairs
with the SAME anchor and horizon. It reports within-mode, cross-mode and total
pairwise accuracy, eligible pair/anchor/scene counts and confidence intervals;
each mode also retains prediction/actual costs, MAE, RMSE and Pearson. Errors
are per mode, while its cross-mode statistic includes matched comparisons
touching that mode. Coarse Lattice is source -1, not a relabelled nominal mode.
Global MPPI ranking compares the six actual sampler components, not coarse rows.

Bootstrap resamples whole scene clusters, retaining all their anchors and
interventions. Candidate rows are not treated as independent samples. Pair
construction is sparse per anchor rather than an all-dataset N x N matrix.
Eligible true gaps must exceed .05; predicted ties count as incorrect. A scene-
offset predictor can have excellent Pearson across scenes but zero action
ranking; the new tests demonstrate that it fails the unchanged error gates plus
the new pairing gates, instead of receiving spurious action-risk authority.

Default readiness requires: >=1000 genuinely cost-supervised WM optimizer
steps, >=128 records, >=10 scenes, MAE<=.1H, RMSE<=.5H, identifiable variation,
overall AND cross-mode ranking>=.60 with scenario-bootstrap 95% lower bound>=.50,
>=128 cross-mode pairs on >=16 anchors and >=10 scenes. Longer H also needs
one-step approval for that source. No eligible matched pairs means no approval.
Thresholds were not lowered. Schema 2 is serialized with approval/update buffers;
Round 3 approvals and legacy datasets without anchors cannot grant permissions.
Any WM optimizer update revokes approvals. Geometry/value remain usable for
uncalibrated modes; unvalidated cost never enters MPPI ranking.

## Training Versus Calibration

Cost-head training already exists in the genuine environment-cost chain:
trainer -> executed-control replay -> shift-aligned WM rollout -> cost MSE ->
the existing WM optimizer. The head predicts one-step cost, and its H-step
evaluation sums those predictions with the same configured gamma (default .99).
It is not trained on its own predictions or on calibration labels.

Round 4 adds `cost_grad_norm`, `cost_updates`, `cost_supervised_samples`,
`cost_step_supervised`, and label-mean metrics; `wm/` aliases expose them in
training logs. Zero cost-loss weight does NOT increment supervised updates or
effective supervised samples. Unit/smoke tests verify real CUDA gradients and
actual head-weight changes. A nonzero gradient proves supervision, not adequate
coverage, calibration success, or reliability for every counterfactual action.

The collector performs NO optimizer update. Its held-out records are used only
for calibration/statistics and readiness validation. Training additionally
rejects any configured mode-calibration scene that overlaps the actual training
pool after its held-out split. Never train on this dataset and validate itself.
An optional independent training set must use different scenes and provenance.

## Continuous Collision

Discrete contact checks remain intact. New continuous hard checks consider each
adjacent predicted interval, without increasing the fixed sample count. Dynamic
actors translate at their measured velocities with measured OBB headings/sizes.
For ego, ordinary kinematic intervals interpolate the turning circular arc and
linear endpoint-speed change. Straight/uniform motion is a special case. An
interval spanning >=180 degrees is rejected as an unresolved kinematic state.

A center-distance broad phase encloses the whole motion. The narrow phase
solves continuous separating-axis entry/exit using four OBB axes. A mid-heading
rectangle is inflated by the maximum rotating-corner displacement, plus arc
sagitta and acceleration-progress bounds. These are conservative superset
checks, not endpoint-only tests. Only potentially intersecting ambiguous
intervals are bisected, reevaluating curved/time-dependent midpoint poses.
Default leaf tolerances are 1e-4 radians and 1e-4 metres, maximum depth 10.
Unresolved leaves are HARD collisions, never optimistic free space. This can
produce tiny conservative false positives near tangency; finite sampling is
not used as a safety proof. Geometry inputs are promoted, never downcast.

New interval contacts zero predicted minimum clearance and participate in the
same hard validity mask before WM scoring. Existing absolute TTC is retained;
continuous contact contributes `interval_start + local_fraction * interval_dt`,
not a future local time mistaken for current TTC. Rotating/accelerating entry
is a conservative interval bound, not an exact Bullet collision timestamp.
Already-discretely-colliding candidates need no additional swept proof.

Tests cover between-sample crossing, head-on/high relative speed, static objects,
dt/substep invariance, rotational corner sweeps, curved-center contact missed
by a chord, acceleration-dependent timing, and near misses. A dense independent
Shapely oracle is a regression aid, not the deployed checking method. Road
footprint checking retains its existing substeps; no continuous road certificate
or guarantee against actual Bullet tyre/contact/IDM prediction error is claimed.

## Emergency Fallback

No finite verified MPPI candidate triggers a separate full-braking bank. Its
steering slews toward neutral/holding/limited alternatives within the same
steering-rate bound. Every trajectory is independently checked for road,
collision, clearance/TTC and kinematics. No WM score can authorize it.

If a brake passes, it is logged as a verified EMERGENCY, with ordinary
`safe_candidate_selected=0`. If none pass (including initial overlap or already
off-road), the system executes predefined full braking/slew-to-neutral, with
`selected_safe=0` and `emergency_verified=0`. It never chooses a least-bad normal
candidate and calls it safe. Ordinary selected controls are rechecked before
return, then the coordinator independently verifies phase geometry.

Installed MetaDrive `BaseVehicle._set_action` maps action[0] to normalized
steering. `_apply_throttle_brake` maps negative action[1] to friction braking
ONLY when `enable_reverse=false`; reverse/discrete configurations are rejected.
Zero throttle still applies a small brake in this interface, not a guaranteed
coast command. Real smoke also verifies full braking actually reduces speed.

Episode summaries include emergency events, steps and seconds, plus reason
counts. Reason codes: 1 all collision, 2 all offroad, 3 all kinematic, 4 nonfinite
learned scores, 5 no verified candidate, 6 selected-control hard violation.
Consecutive fallback calls form one event; reset/normal selection ends a streak.
An unavoidable state has no promised safe action; unverified emergency remains
explicitly unsafe, even though braking is a conservative mitigation.

## E5 Features And Compatibility

E5 retains ordered `[cumulative_cost,min_ttc,min_clearance,boundary_margin]`
value/availability pairs, total 535 dimensions; old 527 policies remain valid.
Geometry features are actually available when measured; infinite TTC/clearance
means no predicted conflict, not zero seconds/metres. Logs now report each
`mpr/sac_state_<feature>_valid` and numeric mask reason.

Mask reasons: 0 available, 1 cost disabled, 2 unlabelled legacy source,
3 calibration not ready, 4 source/horizon unvalidated, 5 nonfinite/partial,
6 geometry unavailable, 7 no residual state (e.g. Stage A).

Coarse Lattice controls cannot inherit nominal MPPI approval. The collector now
also records the ACTUAL coarse sequence at the SAME anchor, separately source
-1, and calibrates it against the mode interventions. It needs its own
one-step/H gates. The unchanged E5 profile keeps coarse cost masked, preserving
old actor availability semantics. New `+experiment=E5_calibrated_cost` is an
explicit independent training profile enabling that feature ONLY after its
separate approval. Layout is still 535, but this semantic flag is saved,
restored for inference and checked when loading SAC. Never relabel an old
trained actor as having learned this previously unavailable feature.

This profile can use an independent pilot dataset for revalidation at the 20K
freeze; it does not train the head on calibration records. If gates never pass,
the feature stays masked. Deployment mode costs still need actual-mode gates.

## Performance Changes

Road checking reuses prepared original/inset geometry and cached boundaries.
Exact original-road containment is only recomputed for footprints not already
inside the safe inset. Vertex-to-road distances are computed ONLY outside the
road; inside margins retain exact footprint-to-boundary distances. A reference
test compares these results against the original formula on rotations, outside
poses and a road with a hole. No safety or soft-risk condition was removed.

Predicted obstacle corners/axes are reused only inside the current immutable
planning context; positions/velocities are refreshed every new planning tick.
Swept checks have a batched broad phase and adaptive work only on possible
contacts; a proven discrete collision needs no extra continuous calculation.

Benchmark main totals use CUDA-synchronized wall time, 100 valid calls across
five fixed live states after five warmups/scene, H10/N64/eight elites/four
iterations. Selected controls are not executed during timing. Both versions
are remeasured this round on clean commits, not substituted with historical
Round 3 numbers. Separate profiling reports CPU wall and CUDA stream elapsed;
these are NOT pure GPU kernel time or additive nested totals. Optional
`--kernel-profile` additionally writes a separate Kineto trace and true summed
CUDA kernel durations on the last scene, outside the 100-call main pass.

Clean source/revision is checked before and after timing. JSON stores revision,
device, effective workload, cost schema/permissions, checkpoint checksum and
phase summaries. CLI workload cannot be overridden by saved planner settings.
Benchmark results and final verification are appended after execution below.

## Formal Commands (Not Executed Here)

Run in this directory. No CUDA visibility is forced. A unique experiment name
avoids overwriting previous models. Training scenes 10000--11999 are disjoint
from test scenes 300--2299; last 20 training-pool scenes are held out of replay.

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
set -o pipefail
run="mpr_mpc_v2_E5_round4_seed1_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/launch
MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -u train.py \
  +experiment=E5 +schedule=frozen_20k_50k \
  exp_name="$run" seed=1 steps=50000 \
  checkpoint=null resume_checkpoint=null \
  mpr_mpc.stages.residual_start_step=20000 \
  mpr_mpc.stages.mppi_start_step=50000 \
  residual_rl.freeze_tdmpc2_on_start=true \
  eval_episodes=20 enable_wandb=false save_video=false \
  metadrive.simulator.start_seed=10000 metadrive.simulator.num_scenarios=2000 \
  2>&1 | tee "logs/launch/${run}.log"
models="$PWD/logs/metadrive-risk/1/$run/models"
```

After training, retain the `models` variable above or set it to the actual run.
These two evaluations use the SAME seed and unique 20-map sampling pool:

```bash
out="$PWD/logs/benchmark/${run}_20k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -u evaluate.py \
  +experiment=E5 checkpoint="$models/milestone_20k_world_model.pt" \
  evaluation_stack=world_model eval_episodes=20 seed=1 \
  metadrive.simulator.start_seed=300 metadrive.simulator.num_scenarios=2000 \
  evaluation_output_dir="$out" enable_wandb=false save_video=false \
  2>&1 | tee "$out/run.log"

out="$PWD/logs/benchmark/${run}_50k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -u evaluate.py \
  +experiment=E5 checkpoint="$models/milestone_50k_full_mpr_mpc.pt" \
  evaluation_stack=full_mpr_mpc eval_episodes=20 seed=1 \
  metadrive.simulator.start_seed=300 metadrive.simulator.num_scenarios=2000 \
  evaluation_output_dir="$out" mpr_mpc.mppi.eval_seed=1 \
  enable_wandb=false save_video=false 2>&1 | tee "$out/run.log"
```

Read each output's `summary.json`, `episodes.csv`, effective config and scenario
IDs. The raw 50K checkpoint normally has no counterfactual approvals: geometry
and value work, but cost scoring is correctly disabled until calibration.

Independent calibration after training (32 scenes x four origins; coarse gets
128 real records, each MPPI mode gets 512; no fake duplicated labels):

```bash
cal="$PWD/logs/calibration/${run}_$(date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python tools/calibrate_cost_modes.py \
  --checkpoint "$models/milestone_50k_full_mpr_mpc.pt" --output "$cal" \
  --scenarios $(seq 30000 30031) --anchor-steps 20 50 80 110 \
  --samples-per-mode 4 --horizon 20 --seed 1
```

Early episodes may terminate before late anchors; the collector fails explicitly,
so choose appropriate shorter prefixes/scenes rather than silently dropping
dangerous origins. Adequate record counts do not guarantee identifiable risk.
Use `$cal/calibrated.pt` for the same full-stack evaluation only after inspecting
its report. It is a NEW file; the source checkpoint is never overwritten.

For the independent coarse-cost actor profile, train a new run using
`+experiment=E5_calibrated_cost` and
`cost.mode_validation_dataset=/absolute/independent_pilot/mode_replays.npz`.
Its pilot scenes must remain disjoint from the training pool. Freeze-time
revalidation uses the current WM, not the pilot model's old approvals. Preserve
separate E5/E5_calibrated_cost names/checkpoints and report feature coverage.

Benchmark example (source tree must be clean):

```bash
/home/kzb/anaconda3/envs/tdmpc2/bin/python tools/benchmark_mppi.py \
  --checkpoint "$models/milestone_50k_full_mpr_mpc.pt" \
  --output /new/output/benchmark_after.json --calls 100 --warmup 5 \
  --scenarios 1714 539 1525 1084 694 --horizon 10 --samples 64 \
  --iterations 4 --seed 1 --kernel-profile \
  --compare /matching/clean_baseline/benchmark_before.json
```

## Modified Files

| File | Change |
| --- | --- |
| `planning/calibration_anchor.py` | Content-addressed scene/state/history/time anchors and verification |
| `planning/cost_validation.py` | Sparse same-anchor within/cross/overall ranking, scene bootstrap, schema-2 gates, coarse source |
| `tools/calibrate_cost_modes.py` | Multi-anchor real interventions, explicit coarse controls, complete anchor manifest |
| `vendor/tdmpc2_runtime/common/world_model.py` | New schema/coarse approval buffers; legacy permissions revoked |
| `vendor/tdmpc2_runtime/tdmpc2.py` | Gradient/effective-supervision diagnostics and zero-weight update guard |
| `planning/swept_collision.py` | Continuous SAT, rotating corners, curved/accelerating center bounds, adaptive leaves |
| `planning/safety.py` | Interval hard rejection/TTC, exact faster road metrics, live-context obstacle reuse |
| `planning/emergency.py` | Independent rate-bounded braking bank and unverified predetermined control |
| `planning/local_mppi.py` | Emergency branch, event metrics, ordinary control recheck, no least-bad normal fallback |
| `planning/coordinator.py` | Real action semantics guards, source-specific cost feature, availability/reason diagnostics |
| `planning/evaluator.py` | Independently calibrated coarse cost, explicit unavailable reasons |
| `planning/types.py` | Non-authoritative cost-unavailability reason |
| `residual_rl/state_builder.py` | Feature availability audit, preserve mask/dimensions/order |
| `experiment/E5_calibrated_cost.yaml` | Separate explicit cost-feature training profile |
| `agent.py` | New training metrics and persisted/checked cost-feature semantics |
| `trainer.py` | Actual training/calibration scene-overlap rejection |
| `evaluate_model_suite.py` | Restore checkpoint's cost-feature semantics before model creation |
| `evaluation_diagnostics.py` | Emergency event/step/duration/reason summaries |
| `tools/benchmark_mppi.py` | Clean commits, effective workload checks, phase/kernel summaries and evidence |
| `tools/planning_profiler.py` | Road/swept phase ranges and optional Kineto annotations |
| `tools/smoke_v2.py` | Real cost gradients, braking semantics and E5 feature status |
| `tests/test_round4.py` | Pairing, leakage, interval/curve/acceleration, oracle, emergency, feature/compatibility cases |
| `tests/test_round3.py` | Correct calibration fixtures to represent actual matched counterfactuals |
| `tests/test_v2_cost_training.py` | CUDA source/schema permissions and effective training diagnostics |
| `config.yaml`, `schedule/frozen_20k_50k.yaml` | Consistent frozen 20K/50K budgets, collision/calibration settings |
| `README.md`, `V2_IMPLEMENTATION.md`, `ROUND3_REPORT.md`, `ROUND4_REPORT.md` | Current workflow/commands; historical results explicitly archived |

## Remaining Validation

Formal 50K training, adequate risky held-out interventions and long paired
success/collision/offroad experiments remain necessary. No real production
model has passed the enlarged paired calibration gates in this short repair.
Calibration does not prove sufficient training or distribution-wide safety.
NPC heading/velocity forecasts, the ego bicycle interpolation, visibility and
road-union geometry do not reproduce every Bullet/IDM/sensor behavior.
Near-tangent unresolved intervals are conservatively rejected. This is not a
certified road-vehicle controller. No 100ms real-time claim is made unless the
actual measured mean meets it; a single fast smoke step is not a benchmark.

## Verification Before Benchmark

- `python -m compileall .`: PASS, exit 0.
- `python -m pytest tests -q`: **176 passed**, 0 failed, 0 skipped, 11.71s.
- `python -m pytest env/tests -q`: **3 passed**, 0 failed, 0 skipped.
- Combined: original **135** regression cases preserved/passing, **44** new
  Round 4 cases passing, **179 total**, no failures/skips. Calibration fixtures
  were corrected to genuine matched counterfactual semantics, not weakened
  thresholds. New cases include sparse-pair bootstrap invariance and independent
  dense geometry oracles for linear, curved and accelerating motion.
- `python tools/smoke_v2.py`: real CUDA/MetaDrive PASS. Cost MSE .730696,
  cost-head gradient norm 26.5066 and 40 effective supervised samples. Readiness
  correctly false after one update. Braking reduced speed 5.2655 -> .08279m/s.
  All three milestone types roundtrip; eval selected_safe=1; training-mode
  MPPI false at a simulated 50K threshold. State dimension 535, geometric risk
  masks available, legacy cumulative-cost mask correctly unavailable/reason2.
  Gentle-turn one-step error .003499m, max one-second error .250574m; these are
  one-scene smoke checks, not certified global dynamics error bounds.
- Real independent collector: scenes539/694, prediction starts10/20, four
  distinct content-hashed anchors, six modes plus separate coarse source,
  28 actual executed trajectories. All reset/replays and the persisted
  checksum/anchor/provenance loader pass. Gates stay closed due to insufficient
  supervised updates/independent evidence. Artifacts:
  `/tmp/mpr_v2_r4_modes_smoke`; original checkpoint unchanged.
- Compressed 24-step E5_calibrated_cost training with the independent dataset:
  four WM cost updates, ten SAC optimizer updates, frozen=true, WM state tensors
  exactly unchanged after Stage A (excluding validation buffers). All three
  milestones/latest/final saved. Its short truncated episodes are ineligible
  for best.pt under the existing progress floors; no best file is fabricated.
  Artifacts: `/tmp/mpr_v2_r4_train_smoke/metadrive-risk/1/round4_calibrated_smoke`.
- Intermediate debugging found and fixed integer diagnostic/empty-pair
  bootstrap dtypes, mixed-precision geometry inputs (promoted, not downcast),
  and a test fixture missing checkpoint configuration. No outstanding test
  failure remains. No production-model readiness or task-success improvement
  is inferred from these tiny tests.
