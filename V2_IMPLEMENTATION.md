# MPR-MPC V2 Delivery

## Scope And Status

Source: `/home/kzb/kzb_code/tdmpc2/mpr_mpc`.
Destination: `/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2`.
All implementation changes are in the destination. The copy excludes `.git`,
`logs`, caches, `.pt`, `.pth`, `.ckpt`, and `.safetensors` files. The original
best checkpoint was READ to test compatibility, not copied or overwritten.

This is a complete runnable training/evaluation implementation, not newly
trained production weights. No claim of improved success/collision rates is
made: those require training and paired, independent-scenario evaluation.

## Commands

The commands below use the existing local environment and do not bind CUDA
visibility. Run from the V2 directory; outputs stay under its own `logs`.

### Train The Full E5 Model

This explicitly opts into the requested frozen 20K/50K schedule. It does NOT
silently change the repository's default one-million-step schedule.
Training seeds 10000--11999 are disjoint from the supplied fixed-100 test set.
The last 20 training-pool maps are reserved for calibration/evaluation and
never enter either training replay.

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
set -o pipefail
mkdir -p logs/launch
MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -u train.py \
  +experiment=E5 +schedule=frozen_20k_50k \
  seed=1 checkpoint=null resume_checkpoint=null \
  enable_wandb=false save_video=false \
  metadrive.simulator.start_seed=10000 \
  metadrive.simulator.num_scenarios=2000 \
  2>&1 | tee logs/launch/train_E5_seed1.log
```

Expected model directory: `logs/metadrive-risk/1/mpr_mpc_v2_E5/models/`.
It contains `milestone_20k_world_model.pt`,
`milestone_50k_world_model_residual_rl.pt`,
`milestone_50k_full_mpr_mpc.pt`, `best.pt`, `latest.pt`, and `final.pt`.
The full 50K artifact packages WM/SAC/planner deployment state; MPPI itself has
no learned weights and does not need a million-step checkpoint to run.

To retain the original actual schedule, omit `+schedule=frozen_20k_50k`:
1M steps, residual start 20K, MPPI start 50K, WM freeze false. Other defaults
including 20 evaluation episodes, no video, and 64 local samples remain.
Do not confuse top-level TD-MPC2 `num_samples=512` with
`mpr_mpc.mppi.num_samples=64`, which is the optimizer used by this stack.

### Evaluate The Same Fixed 100 Scenes

The included JSON preserves the user's exact 100 IDs and their order.

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
set -o pipefail
out="$PWD/logs/benchmark/E5_fixed100_seed1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -u evaluate.py \
  +experiment=E5 seed=1 eval_episodes=100 \
  checkpoint="$PWD/logs/metadrive-risk/1/mpr_mpc_v2_E5/models/milestone_50k_world_model_residual_rl.pt" \
  evaluation_stack=full_mpr_mpc \
  evaluation_scenarios_file="$PWD/scenarios/fixed100.json" \
  evaluation_output_dir="$out" \
  metadrive.simulator.start_seed=300 \
  metadrive.simulator.num_scenarios=2000 \
  mpr_mpc.mppi.eval_seed=1 \
  enable_wandb=false save_video=false \
  2>&1 | tee "$out/run.log"
```

Read `$out/summary.json` and `$out/episodes.csv`. The config and scenario list
are saved alongside them; episode data are refreshed after every episode.
Collision/offroad refer to ANY occurrence during the episode, not just the
last transition. Summary includes scenario-cluster bootstrap 95% intervals.
Pure evaluation does not run optimizers or save agent weights. Cost validation
reported after the run does not retrospectively influence its actions.

For rendering, add `metadrive.simulator.use_render=true` in a graphical session.
To test H=15/20 or 128/256 samples, override `horizon=15`/`horizon=20` and
`mpr_mpc.mppi.num_samples=128`/`256`. Validate cost predictions at the chosen
horizon: a validated H=10 does not automatically enable H=20 cost scoring.

### Frozen WM + SAC From Separate Files

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -u evaluate_frozen_stack.py \
  +experiment=E5 seed=1 eval_episodes=100 \
  +world_model_checkpoint="$PWD/logs/metadrive-risk/1/mpr_mpc_v2_E5/models/milestone_20k_world_model.pt" \
  +residual_checkpoint="$PWD/logs/metadrive-risk/1/mpr_mpc_v2_E5/models/milestone_50k_world_model_residual_rl.pt" \
  evaluation_scenarios_file="$PWD/scenarios/fixed100.json" \
  evaluation_output_dir="$PWD/logs/benchmark/E5_composite_fixed100_seed1" \
  metadrive.simulator.start_seed=300 \
  metadrive.simulator.num_scenarios=2000 \
  mpr_mpc.mppi.eval_seed=1 enable_wandb=false
```

An optional `+mppi_checkpoint=...full_mpr_mpc.pt` still loads planner counters,
without overwriting WM/SAC. Runtime planner settings remain explicit unless
`restore_checkpoint_runtime=true` is requested.
Only combine encoders/policies trained with a compatible latent representation;
matching dimensions alone do not prove semantic compatibility.

### E0--E5 Experiments

| Profile | Changes Relative To Previous Profile |
| --- | --- |
| E0 | Original sampling/scoring/constraint formulation; WM cost off |
| E1 | Align cost discount and episode budget; nonnegative conservative critic |
| E2 | Train, validate, and use WM cost head when validation passes |
| E3 | Live-pose dynamic footprint/road/kinematic hard safety + geometric risk |
| E4 | Nominal, braking, lateral, joint proposal modes |
| E5 | Optional risk-state pairs, separate risk auxiliary learning, Q spread penalty |

The unconditional-baseline and all-invalid numerical fixes are shared across
ALL V2 profiles, including E0. Thus E0 is a guard-repaired baseline, not a
bit-for-bit reproduction of the historical unsafe fallback. For an exact old
baseline use the unchanged sibling repository and label it separately.
Encoder type, residual mapping, reward, environment costs and vehicle settings
are not changed across these profiles. E5 changes the SAC input and must be
trained separately; evaluating an E0 actor with an E5 YAML cannot teach it the
new features or the improved critic.

Train each profile using the training command with `+experiment=E0` through
`E5`; the respective experiment name determines its own checkpoint directory.
Then evaluate each respective checkpoint using the same fixed JSON and the
same evaluation seed/MPPI seed. Repeat with seeds 1, 2, 3 (identical seed list
for each profile). Keep the 100 IDs fixed for seed repeats, and collect a
separate 300--500-ID set for final reporting. Do not rank models solely by the
best 100-scenario batch; report the full-set results and confidence intervals.
Compare mean speed, travelled distance, route completion, return and episode
length along with safety to identify improvement caused only by slowing down.

Existing `evaluate_ablation.py` still supports the A--D Lattice/SAC/MPPI
component ablations, fixed scenario files, repeated seeds and paired intervals.
`evaluate_model_suite.py` also collects the new per-stage metrics.

## Actual Data Flow

1. Observation [259] -> existing encoder -> latent [512]. No new encoder.
2. Existing Lattice generates/filters/selects a global long-horizon proposal.
3. Its H+1 tracked controls are evaluated with the existing latent dynamics,
   reward head, full Q ensemble and (only if validated) cost head.
4. Residual SAC observes detached latent/path/value/feasibility/risk features
   and requests two normalized residuals: target lateral position and speed.
5. The existing linear or latent-tanh parameter mapping regenerates the path.
   In MPPI deployment, a collision-only Lattice rejection is deferred to the
   independent local checker; road/finite geometry errors still fail.
6. Local MPPI samples several control modes around the SAC intention, checks
   real-pose geometry first, then scores only eligible candidate controls.
7. Only the first selected control executes. H controls move the vehicle;
   the H+1-th control is used only for terminal Q and is not integrated.
8. Environment reward/cost and executed control supervise the existing WM
   replay. SAC replay retains the REQUESTED SAC residual, environment cost,
   and separately stores downstream execution/risk diagnostics.
9. Evaluation transitions on reserved maps are held out of BOTH training
   buffers and calibrate cost predictions on recorded executed controls.

## SAC State And Constraints

Default layout remains 527: `[latent 512, path 12, WM value 1, feasibility 2]`.
The twelve path features in order are target_d, target_speed, horizon,
min_obstacle_clearance, obstacle_cost, max_abs_d_d, max_abs_d_dd,
max_abs_d_ddd, max_abs_s_dd, max_abs_s_ddd, curvature_max, path_length.
Default scales are `[4,20,5,20,100,5,5,10,5,10,0.5,100]`, clipped to +/-5.
The predicted WM total value is divided by 50; feasibility is feasible ratio
followed by the fallback flag. Latents are detached and not re-normalized.

`risk_features=[]` preserves the old 527 input. Each enabled risk appends
`[normalized_value, availability]`, with feature order saved in metadata.
E5 enables cumulative_cost, min_ttc, min_clearance, boundary_margin: 535 total.
Scales are 10 cost units, 5 seconds, 20 metres and 4 metres; clipped to +/-5.
Unavailable/unvalidated values become `[0,0]`. Observed infinite TTC/clearance
means no predicted conflict, encoded as the positive clipped value with mask 1.

Original critic used gamma=.99 but the dual update compared undiscounted
episode sums against 5. E1+ defaults use an undiscounted episodic cost critic:
`target_c = environment_cost + (1-done)*max(target_Qc1,target_Qc2)`.
Cost-Q outputs use softplus; the actor minimizes max(Qc1,Qc2). With
`constraint_mode=discounted_episode`, both Bellman bootstrapping and accumulated
episode cost use gamma=.99 instead. Reward SAC discount stays .99.
The projected dual update is `lambda <- max(0, lambda + lr*(episode_cost-5))`
once per complete episode. The first partial episode spanning stage activation
does not update the dual. Actor safety loss really backpropagates through the
critic to policy actions. The unchanged budget of 5 can legitimately leave
lambda at zero if actual episode costs are below budget; this is diagnosed,
not hidden by lowering the budget.

Optional `risk_auxiliary_weight` trains a separate short-horizon risk regressor
on requested SAC residuals with validated SAC-trajectory WM labels. It is NOT
added to the environment cost or cost-critic Bellman target. Its actor term
starts after 1000 labelled updates; default weight 0, E5 experimental weight .1.
Unvalidated or failed-regeneration targets are never used. Its weights and
optimizer are serialized separately. This is not a new world model.

## Cost Head Training And Validation

Both top-level `cost.enabled` and `metadrive.cost.enabled` are true by default.
Real environment costs already travelled through trainer.to_td -> replay ->
shift-aligned training sequences. This implementation reuses that chain and
the existing `_cost` MLP and optimizer group. The supervised MSE participates
in the same optimizer step as consistency/reward/value/termination losses.
Outputs use softplus. There is no second dynamics model or duplicated label.

Trajectory cost and reward use exactly the same z_t and a_t before each
latent transition. Cost is the gamma=.99 discounted sum of H predictions,
not the infinite-horizon SAC cost critic. A hallucinated learned termination
does not suppress future safety predictions. Terminal Q is a value estimate,
not an extra explicit safety step.

Before cost may affect state/scoring/auxiliary labels, validation requires:
at least 1000 cost-supervised updates, 128 held-out windows, one-step MAE<=.1,
RMSE<=.5, cumulative MAE<=H*.1, cumulative RMSE<=H*.5, and pairwise risk ranking
accuracy>=.55 at the requested horizon. Horizons 1/5/10/15/20 and the active H
are evaluated using real recorded controls and matching future costs. Ties
within .05 are excluded from pairwise ranking. Constant-cost validation has
no identifiable ranking and does NOT pass just because its MAE is low.

Predictions are revalidated every 1000 cost updates on cached held-out data,
at evaluation, and before freezing. A validation approval lasts at most the
interval until the next check, not indefinitely while weights change. A
per-horizon mask and update count are checkpoint buffers. Different environment
cost/risk definitions or missing historical validation revoke readiness.
Validation metrics computed after a test run are diagnostics, not proof that
untested counterfactual candidate controls are all calibrated.

## Local Safety And Optimization

The checker uses live world XY/heading, m/s velocity, actual vehicle footprint,
wheelbase and steering range. It reuses Lattice's MetaDrive object extraction
and navigation lane polygons, including predecessor roads needed by the rear
footprint at a road join. It checks the UNION of legal route-lane surfaces,
so adjacent-lane seams are not mistaken for road edges.

Ego rollout is a bicycle approximation at 20ms substeps by default; obstacles
use measured constant velocity/heading and their actual bounding dimensions.
Time-synchronized oriented rectangles use separating axes for collision;
vertex-to-edge distances provide footprint clearance. TTC uses swept relative
velocity intervals on separating axes. Static objects naturally have zero
velocity. The current footprint is included. Hard constraints cover collision,
road surface, finite/bounded controls, steering rate and lateral acceleration.
Clearance<1m and TTC<3s are soft risks unless hard thresholds are configured.
The default road inset is .15m, with an additional .3m ego model-error footprint
margin. These margins are configurable and must be calibrated more broadly.

MPPI nominal samples remain in the original small trust region; braking and
lateral/joint modes have their own configurable centers/limits. 64 samples,
8 elites and 4 iterations remain defaults. Mode weights, braking std/center,
lateral shift/limit, global nominal std and expanded delta are configurable.
Broader modes use `mode_score_delta=[.25,1.5]` for deviation normalization;
they are not penalized as if every braking action violated the tiny nominal
trust region. E5 penalizes absolute control differences, not only differences
of deviations from the baseline.
The baseline and previous best are carried explicitly and MUST pass checks.
Invalid candidates do not enter the learned scoring batch. Finite elites only
are used in weighted updates; no all-minus-infinity softmax is computed.

Score = WM value - validated WM cost - geometric soft risk - deviation penalty
- smoothness penalty - terminal-Q-ensemble standard-deviation penalty.
Default weights are 1,10,5,1,.1 respectively. The old improvement gate can only
return a VALID baseline and does not veto a meaningful geometric risk decrease.
Hard invalidity can never be compensated by a high learned value.

If normal proposals are all unsafe, the optimizer tries wider perturbations,
strong braking (including keeping nominal/previous steering), and other
Lattice proposals. All alternatives go through the same checker. If none pass,
it returns an explicitly labelled minimum-risk emergency control, with
`selected_safe=0` and `no_safe_candidate=1`, NEVER an asserted safe baseline.
Its risk ranking prioritizes predicted contact/offroad duration, kinematic
violations, proximity, and terminal speed. An unavoidable situation cannot
honestly be certified safe by changing a flag.

## Logs And Interpretation

- Lattice/SAC/MPPI: predicted_cost, value, safe, min_ttc, min_clearance,
  boundary_margin, soft_risk; risk_reduction_sac/mppi and correction magnitude.
- MPPI: safe_candidate_ratio, collision/offroad rejection rate,
  baseline_selected, planner_score_gain, no_safe_candidate, expanded_search,
  selected_safe, cost_validated, planning_ms. Rates distinguish whole-search
  aggregate rejection from final-batch collision/offroad rejection.
- SAC training: lagrangian/value, episode_cost, cost_limit,
  constraint_violation, loss_actor_safety, loss_cost_critic, predicted_cost,
  auxiliary losses and update counts. Pure evaluation losses are unavailable
  (null in summary), not falsely reported as newly computed zero losses.
- WM: cost_loss plus MAE/RMSE/correlation/ranking/sample count per horizon and
  cost_ready. Unvalidated predicted costs are unavailable, not zero risk.
- Episode: success, any collision/offroad, joint collision-success/offroad,
  actual episode cost, finite-TTC mean/minimum/coverage, predicted clearance,
  road/dynamic/static risk-field components, speed, distance and completion.
  `dynamic_collision_cost`/`static_collision_cost` in episode CSV are aliases
  for the respective risk-field potential sums, NOT separately attributable
  physical contact charges. `episode_cost` is the actual supervised constraint
  cost and must not be reconstructed by summing these diagnostic components.
- TTC NaN/infinity are not converted to zero seconds. Summary minimum TTC is
  the full-set minimum; mean TTC is finite-step weighted. Coverage is reported.
- Scores against an invalid baseline may be unavailable instead of artificial
  infinity gains; inspect geometric risk reduction and selected_safe as well.

## Checkpoint Compatibility

The original file formats and milestone names remain. Additional cost buffers
are tolerated when loading old WM weights; absent update/validation data means
cost is disabled. Existing reward/value interfaces still work. New metadata
captures encoder, residual mapping/schema/scales, cost definition and planner
configuration. Architecture is inferred from saved weights/metadata, not an
experiment folder name. Explicit evaluation settings are not silently replaced
by old MPPI settings.

Old 527-dimensional policies can be evaluated unchanged with the 527 schema.
The prior best `mpr_mpc_frozen_wm200k_rl500k_seed1_v2` milestone was actually
loaded successfully: flat encoder, linear residual, state_dim=527, cost updates
0, cost_ready=false. E5's 535-dimensional policy requires matching ordered
features; it cannot directly load a 527 actor. WM-only component loading does
not unnecessarily require matching SAC feature dimensions.

Optimizer resume rejects changed SAC constraint semantics and retains stricter
planner checks. Legacy inference compatibility is not a promise that resuming
an old optimizer with new critic semantics is valid. Train improved constraints
from scratch, or explicitly retain legacy semantics for a legacy resume.

## Changed Files

| File | Change |
| --- | --- |
| `__init__.py` | Prevent sibling-package mixing; support V2 module entry |
| `config.yaml` | Cost/validation/safety/mode/diagnostic switches; defaults |
| `agent.py` | Calibration, aligned costs, replay diagnostics, metadata/load guards |
| `trainer.py` | Fixed held-out map split, real-control calibration, whole-episode metrics |
| `planning/types.py` | Optional cost/risk fields and safety context |
| `planning/evaluator.py` | Same-rollout gated cost prediction and horizon mask |
| `planning/cost_validation.py` | Held-out executed-control multi-horizon calibration |
| `planning/safety.py` | Global footprint, road, dynamic/static collision, TTC/dynamics |
| `planning/local_mppi.py` | Hard-first scores, multimodal search, explicit emergency path |
| `planning/coordinator.py` | Preserve SAC intent, integrate safety, per-stage diagnostics |
| `planning/structured_proposal.py` | Defer collision-only rejection to active safety MPPI |
| `lattice/frenet_metadrive.py` | Optional obstacle-check deferral; retain geometry checks |
| `residual_rl/state_builder.py` | Configurable ordered risk pairs, compatibility masks |
| `residual_rl/replay.py` | Requested residual separate from executed control/risks |
| `residual_rl/sac_agent.py` | Aligned conservative cost critic and optional risk auxiliary |
| `vendor/tdmpc2_runtime/common/world_model.py` | Persist supervised-update/readiness/horizon buffers |
| `vendor/tdmpc2_runtime/tdmpc2.py` | Track genuine cost supervision and support legacy buffers |
| `tdmpc2/core.py` | Correct the learner extension documentation |
| `evaluation_diagnostics.py` | Shared event/trajectory recording, CSV/JSON/CI reports |
| `evaluate.py` | Architecture-aware load, explicit full stack, unique scenes, diagnostics |
| `evaluate_frozen_stack.py` | Composite weights, optional MPPI file, diagnostics/reports |
| `evaluate_model_suite.py` | Architecture inference, seed-range checks and new metrics |
| `experiment/E0.yaml` through `E5.yaml` | Independent cumulative ablation configurations |
| `schedule/frozen_20k_50k.yaml` | Explicit alternative to original actual default schedule |
| `scenarios/fixed100.json` | User-supplied fixed evaluation IDs, preserved order |
| `tests/test_v2_safety.py` | New safety/risk/replay/calibration/constraint tests |
| `tests/test_v2_cost_training.py` | Real CUDA cost-head optimizer/rollout/legacy test |
| `tests/test_v2_reporting.py` | Unique fixed scenarios, whole-episode events and strict JSON |
| `tests/test_residual_rl.py`, `tests/test_evaluation_ablation.py` | Update intentional new contracts/fixtures |
| `pytest.ini` | Ensure tests import this V2 rather than the old sibling |
| `tools/smoke_v2.py` | Actual MetaDrive controls, geometry and checkpoint roundtrips |
| `README.md`, `V2_IMPLEMENTATION.md` | V2 entry points and delivery details |

## Verification

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
/home/kzb/anaconda3/envs/tdmpc2/bin/python -m pytest -q
/home/kzb/anaconda3/envs/tdmpc2/bin/python tools/smoke_v2.py
```

Verified on the local CUDA/MetaDrive installation:

- 80 tests passing, including a real cost-head optimizer step (not only mocks).
- Static and moving/crossing obstacle footprints, clearance, road-lane union,
  predecessor-road rear coverage, terminal-action timing and unsafe-baseline
  rejection; braking escape and all-invalid explicit emergency behavior.
- Real steering sign positive; one-step position error about .0035m and
  maximum one-second gentle-turn rollout error about .251m in the smoke scene.
  This is calibration evidence for that scene, not a global dynamics bound.
- Actual full three-stage 24-step compressed training smoke: 4 WM cost updates,
  10 SAC updates, frozen WM, 4 MPPI calls and all three milestone saves.
  Its cost stayed correctly unvalidated; no risk auxiliary updates occurred.
- Single-checkpoint 2-episode evaluation and separate-WM/SAC 1-episode frozen
  evaluation on eight-step smoke scenes both completed and wrote results.
- Old best checkpoint successfully loaded without modifying or copying it.
- V2 module entry and Hydra E5/frozen schedule composition verified.

These tiny truncated episodes are API/training/inference checks, not meaningful
task-success measurements. Smoke artifacts are under `/tmp`, not copied into
the V2 model tree. GUI rendering code is retained but was not visually tested
in this headless session.

## Remaining Limits And Next Validation

1. No formal 50K/1M training or 100/500-scene performance run was launched.
   Success >=76% and crash <17% are research targets, not established results.
2. Constant-velocity traffic and a bicycle ego model do not reproduce all
   Bullet contact/tyre/slip/braking dynamics or future IDM reactions. Test
   fast curves, braking and lane changes across seeds; calibrate model margins
   and horizon-dependent error before making safety claims.
3. Lane surfaces and inflated footprint margins do not exactly reproduce all
   MetaDrive continuous-painted-line/sidewalk sensor termination semantics.
   The actual environment remains authoritative for crash/offroad reporting.
4. Hidden/unmodelled objects, wrong-way route semantics and interaction beyond
   H are not certified. Increasing H needs BOTH cost and geometry validation.
5. Q-ensemble spread is a value uncertainty proxy, not calibrated cost-head
   uncertainty. A single cost head can fail on counterfactual/OOD controls even
   after executed-window ranking passes. Collect diverse held-out interventions.
6. MPPI can reach an unavoidable state with no verified-safe action. Emergency
   is labelled explicitly; future work may need a separately validated shield.
7. The geometry/WM loops are not established real-time. A local smoke plan was
   roughly .4 seconds on RTX 3070, above the .1-second simulation control period.
   Profile warm-up and sustained latency before live use; H=20 costs more.
8. E5's risk auxiliary and extra input are experimental and need isolated
   training ablations. Do not interpret a masked/unready cost as zero danger.
9. The held-out split deliberately reduces available training maps (default
   50 minus 20). The recommended command expands the pool to 2000; maintain
   identical splits across ablations and separate final-test maps.
10. Original A/B stages retain their existing Lattice safeguards; the new MPPI
    safety shield is active only when MPPI is enabled. Training with MPPI off
    does not create an unconditional physical safety guarantee. A rejected
    residual's observed environment cost belongs to the guarded execution,
    not its unexecuted counterfactual path; invalid penalties and fallback
    diagnostics remain important, and failed-regeneration WM labels are masked.
