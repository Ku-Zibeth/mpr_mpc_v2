# MPR-MPC V3 Planner Delivery

Date: 2026-11-09 (Asia/Shanghai). Work performed in this existing repository on
branch `main`; no new project, commit, reset, checkout or push was performed.
Existing user changes and the running training job were preserved.

## Status and Important Limitations

The requested inference-time search, independent safety filter, fallback,
episode RNG, checkpoint override and diagnostic plumbing are implemented.
This is an experimental planner, not a demonstrated success-rate improvement
or a real-world safety guarantee.

- Original worktree tests: 230, retained. Added: 29 parameterized V3 cases.
  Final result: **259 passed, zero failures**, no tests deleted.
- Real CUDA/MetaDrive partial smoke: 30 environment steps across scenes 1714
  and 539 and B/C/D configurations. All 20 C/D decisions executed MPPI;
  17 changed the first executed action, without fallback in this short smoke.
- Complete paired A/B/C/D run on scene 539 finished and wrote all reports.
  B succeeded; C and D timed out; D also collided. This failure is retained,
  not hidden or reclassified.
- Fixed-state full-plan benchmark: 163.20 ms mean, versus 154.39 ms before
  the change. Mean overhead is about 5.7%. **The 100 ms reference target and
  the desired no-latency-regression outcome have NOT been achieved.**
- The interfaces and output pipeline are ready for a formal fixed100
  diagnostic ablation, but planning effectiveness and real-time acceptance
  have not passed. The formal 400-episode experiment was not started here.

## Files Changed in This Task

| File | Change |
| --- | --- |
| `planning/local_mppi.py` | Retain single Gaussian, elite/variance updates, trust region and exact candidate cache; add pre-WM filtering, finite-safe baseline gate, checked fallback, persistent generator and diagnostics. |
| `planning/mppi_safety.py` (new) | Independent measured-state context, body-footprint/swept road containment, CV swept collision, kinematic checks, verified braking mapping and cached CPU geometry. |
| `planning/kinematic_filter.py` | Batch the reference together with candidates in one bicycle rollout, instead of launching a second identical H-step rollout. |
| `planning/coordinator.py` | Prepare candidate safety independently; preserve legacy SAC features; skip unused SAC feature building in C; pass raw Lattice fallback; expose selected physical trajectory and full-plan timing. |
| `planning/types.py` | Add a separate `mppi_safety_context`, without changing residual-state layout. |
| `planning/native_config.py` | Allow the new safety configuration, while discarding retired outer/nested mode approval settings. |
| `agent.py` | Seed evaluation episodes without touching training SAC RNG; keep architecture/mapping/risk-order checks; authorize only MPPI runtime differences through the explicit restore flag. |
| `config.yaml` | V3 trial MPPI defaults and independent safety settings; `restore_checkpoint_runtime=true`, `restore_mppi_runtime=false`; empty legacy `risk_geometry` mapping for structured config restoration. Training schedule and WM/SAC hyperparameters unchanged. |
| `evaluate_model_suite.py` | Independent MPPI config restoration flag, paired episode resets, per-step JSONL and optional W&B evidence. |
| `evaluate_ablation.py` | Record the persistent episode RNG rule in the manifest. Uses the existing A/B/C/D entry point. |
| `evaluate.py` | Add the C stack switch and per-step planner W&B metrics. |
| `evaluate_frozen_stack.py` | Add per-step planner W&B metrics; avoid conflicting episode-based W&B step numbers. |
| `evaluation_diagnostics.py` | Complete fixed MPPI episode aliases, unavailable values for A/B, legacy metric compatibility, optional step logger. |
| `evaluation_statistics.py` | Include new numeric MPPI fields in clustered summaries, confidence intervals and paired comparisons. |
| `tools/benchmark_mppi.py` | Record safety times, validity, first-action changes, unique/WM counts and fallback details. |
| `tools/planning_profiler.py` | Independently profile candidate safety, kinematic rollout and swept collision; retain the distinction between CPU wall time and CUDA stream elapsed time. |
| `tools/smoke_v2.py` | Keep the trust-region assertion for normal Gaussian search; checked safety overrides are explicitly outside that normal-search region. |
| `tools/smoke_mppi_v3.py` (new) | Read-only real-checkpoint CUDA/MetaDrive inference smoke, action-delta/stack checks and checkpoint SHA256 verification. No gradient updates. |
| `tools/run_v3_ablation.sh` (new) | Launch the existing paired ablation entry on fixed100, save step evidence, directly log to a file without a blocking tee pipeline, and show progress in the current terminal. |
| `tests/test_mppi_v3.py` (new) | 29 V3 cases, including parameterized invalid baseline/checkpoint tests and formal-command validation. |
| `tests/test_mpr_mpc.py` | Explicit legacy algorithm-only safety opt-out in fixtures; deterministic reproducibility checked across episode resets. |
| `tests/test_native_mppi_cost.py` | Update default-parameter assertions for the requested V3 defaults; keep old Cost/cache/invalid-value tests. |
| `tests/test_evaluation_ablation.py` | Check seeded resets while retaining all stack/event tests. |
| `tests/test_residual_rl.py` | Add C-switch coverage to the existing evaluation-stack test; retain strict mismatch tests. |
| `tests/test_frozen_rl_launcher.py` | Isolate the positive launch test from the user's real active training job. The separate real duplicate-process guard test remains. |

`planning/safety.py` was reviewed and left unchanged: its legacy SAC feature
calculations, normalization inputs and mask semantics are preserved. The new
module uses a separate geometry helper instance and does not construct SAC
states. `planning/evaluator.py` was reviewed and its batched reward/Cost/Q
semantics were retained. `experiment/E5.yaml` was left unchanged and inherits
the new MPPI settings from the base config.

The earlier training/logger stability fixes, GIF tool and user's other changes
were not reverted. No WM/SAC network, training loss, learned weight, Lattice
candidate generator, environment success criterion, reward or real Cost was
changed. MPPI remains off during WM and SAC data collection. Default stage
boundaries remain 20k/50k; the user's explicit 500k/1m training preset is also
unchanged.

## Algorithm and Data Flow

### Search

The trial defaults remain 64 samples, 8 elites and 4 iterations. Initial/min/max
standard deviations are `[0.04,0.12]`, `[0.01,0.03]`, `[0.08,0.25]`; action
deviation bounds are `[0.10,0.25]`. Deviation/smoothness coefficients are
1.0/0.3; throttle weight is 0.25; improvement thresholds are 0.05 absolute and
0.001 relative. These are experimental settings, not tuned optima.

Each iteration samples a single Gaussian, checks the kinematic corridor, then
the independent safety layer. Only valid candidate indices are passed to the
batched WM evaluator. Scores map back to their exact action-byte keys. Both
safety results and WM consequences are cached only within the current plan;
traffic/observations are never cached across planning steps. Invalid candidates
cannot enter elites, and an empty valid set never reaches `topk`.

The best evaluated safe candidate is retained across iterations. Weighted mean
and variance updates use finite, valid elites only. Means themselves are never
assumed safe: they are checked if sampled as a candidate in the next iteration.

### Baseline and Fallback

There is no unconditional `valid[0]=True`. A normal baseline gate is available
only for a physically valid baseline with a finite score. NaN/-inf baselines
are not used in ordinary gain or threshold arithmetic. A safe finite alternative
can be chosen without an improvement gate when the original baseline is invalid.

Fallback order is: checked SAC baseline, best checked Gaussian candidate,
checked original Lattice sequence, then braking with the previous steering
command. Lattice/braking safety overrides need not stay inside the unsafe SAC
trust region. They are identified separately from ordinary Gaussian corrections.
The braking sequence is checked again; failure sets
`safety_fallback_unverified=1`. That output is not represented as certified safe.

Braking polarity was checked against the installed MetaDrive
`BaseVehicle._apply_throttle_brake`: with `enable_reverse=false`, negative
`action[1]` applies zero engine force and positive wheel brake force. The unit
test invokes that real method with a wheel-system mock. Reverse-enabled or
unknown braking context is rejected rather than guessed.

### Independent Geometry

Candidate checking does not consume `residual_state`. Current pose, measured
velocity, heading, real vehicle dimensions/steering/axle geometry, lane polygons
and nearby actor states form a separate context. A shared immutable live
snapshot may be reused, but caches/settings and SAC feature computation are
separate. Missing context/road geometry is an explicit error.

The body, not only its center, must remain on the inset road. Continuous road
envelopes include translation, rotation and arc-error bounds between substeps.
An enclosing whole-path box can cheaply certify interior cases; ambiguous
paths still receive interval boxes and conservative swept hulls. Prepared GEOS
road geometry is cached while retaining its source geometry reference.

Dynamic actors use measured constant velocity and fixed heading over the short
horizon, not invented future observations. Continuous swept OBB checks reuse
the existing collision primitive. Ego motion reuses the existing configurable
bicycle rollout. Default dt is 0.1 s with two safety substeps, H=10: exactly
H transitions, not H+1; the last action is only the terminal-Q input.

Candidate checks run as batches on CPU because GEOS is CPU-based. Fast corridor
checking uses the same `BatchedKinematicCorridor` equations on the already
copied candidate batch; it does not repeatedly launch small GPU kernels.

Default safety trial settings include 0.05 m model/road margins, 7 rad/s
steering-command slew and 6 m/s^2 lateral acceleration. The 7 rad/s limit
admits approximately one normalized steering unit per 0.1 s for the live
MetaDrive steering range; it is a configurable trial command-slew limit, not
a measured actuator specification. Acceleration/deceleration estimates remain
configurable 4/6 m/s^2; they are not exact Bullet dynamics certificates.

TTC and clearance hard thresholds default to zero. Explicit positive thresholds
reuse the existing defined short-horizon geometry diagnostics; they are not
silently introduced as arbitrary extra constraints. No TTC, clearance, boundary
or uncertainty soft score is added on top of the hard geometry filter.

### Cost and Diagnostics

The ranking score remains

```text
discounted_reward + gamma^H * terminal_Q
 - cost_coef * discounted_H_step_cost
 - deviation_coef * normalized_deviation
 - smoothness_coef * smoothness
```

WM Cost is a ranking term, not safety certification. Unvalidated finite
predictions remain marked unvalidated. Mixed invalid Cost predictions are
excluded from usable ranking rather than causing an invalid baseline gate;
an entirely invalid evaluated Cost batch remains an explicit model error,
preserving the existing invalid-model tests rather than substituting zero Cost.

First-step deltas are computed from the final selected sequence versus the SAC
refined sequence, including marked fallback actions. Blue debug trajectories
use the selected safety-model rollout rather than an unverified interpolated
target path when independent safety is enabled.

CSV/JSON exports contain first-action deltas, score/gate/std/trust-region
diagnostics, WM/unique counts, baseline/selected predicted Cost, Cost penalty
and validation status, predicted reward/Q, all safety rejection/fallback fields,
and sampling/safety/WM/Cost/elite/full-plan timing. A/B MPPI-only metrics are
unavailable, not fabricated zeros. Fallback deltas must not be mistaken for
Gaussian-search improvements.

`mppi_predicted_cost` is a per-plan H-step forecast. `total_predicted_cost` sums
only first-step predictions along executed actions; it does not repeatedly add
overlapping H-step forecasts. Unavailable fallback forecasts are not counted as
zero. Timing for Cost is nested in WM rollout; those times are not additive.

### RNG and Checkpoint Compatibility

One private generator is seeded at episode reset with
`(test_seed + scenario) % 2**32`; it advances at each plan. Repeated first-step
resets preserve that episode seed. C and D use the same rule and sample budget.
Training SAC's global RNG is not reseeded by these evaluation changes.

`restore_checkpoint_runtime=true` still restores saved encoder, residual
mapping, normalization/risk settings and stage configuration.
`restore_mppi_runtime=false` keeps current inference-time MPPI settings.
Setting it true restores saved MPPI parameters; checkpoints predating the new
safety block explicitly retain their historical disabled candidate-safety
setting. This restores parameters, not the old RNG bug or bitwise historical
trajectories.

Only MPPI runtime mismatch is authorized by that flag. Latent/model/Q sizes,
residual state/action sizes, mapping and risk-feature order remain checked;
strict network weight loading also remains in force.

## Verification and Evidence

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
/home/kzb/anaconda3/envs/tdmpc2/bin/python -m pytest tests env/tests -q
```

Final result: **259 passed** (230 existing + 29 new V3 cases), no deletions.
Five existing test files were adapted/extended for explicit legacy fixtures,
episode reset semantics, new defaults, C-switch coverage and active-job
isolation. No valid safety test was removed to obtain this result.

Real inference smoke:

```bash
/home/kzb/anaconda3/envs/tdmpc2/bin/python tools/smoke_mppi_v3.py \
  --checkpoint logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm200k_rl300k_20261103_195237/models/milestone_500k_full_mpr_mpc.pt \
  --output reports/v3/a_new_smoke.json
```

Existing evidence is `reports/v3/smoke_final.json`: 30 actual physics steps,
20 MPPI calls, 17 nonzero first-action corrections, 535-D SAC input and unchanged
checkpoint checksum. This partial smoke is not an episode success-rate test.

The final full-episode evidence is in `reports/v3/ablation_verified/`, using
one scene (539), seed 1, normal horizon/termination and all four configurations.
The earlier `ablation_smoke/` run independently completed with the same outcomes:

| Group | Steps | Success | Offroad | Crash | Route Completion |
| --- | ---: | --- | --- | --- | ---: |
| A | 228 | no | yes | no | 68.94% |
| B | 400 | yes | no | no | 98.77% |
| C | 1000 | no | no | no | 83.11% |
| D | 1000 | no | no | yes | 71.02% |

Both C/D execute MPPI at every step, including steps ending in a safety
fallback. A/B execute zero MPPI steps. The conservative fallback behavior is
currently a real limitation: the first complete run used emergency fallback
on about 63.6% of C and 75% of D steps. D first reported collision at step 298.
These observations do not establish a population success rate, and must not
be presented as evidence that V3 improves the original 75% success result.

The recorded failures point to narrow trust-region recovery around an unsafe
reference, model/road-margin conservatism and the limits of constant-velocity
interaction prediction. Required stopping fallback can become a progress
deadlock; it is not silently replaced with an unchecked maneuver. Further
calibration/experiments are needed before any effectiveness claim.

## Performance

`reports/v3/native_before.json` and `reports/v3/v3_verified.json` use the same
checkpoint, GPU, two scenes `[1714,539]`, 20-step common prefixes, two warmups
per scene, H=10, 64 samples, 8 elites, 4 iterations and 100 measured full-plan
calls. States and checkpoint SHA256 were programmatically compared and match.
The new trial search/safety settings intentionally differ from the old settings.
The user's training task was concurrently active; this is a shared-load
comparison, not an isolated real-time certification.

| Full-plan metric | Native Before | V3 |
| --- | ---: | ---: |
| mean | 154.39 ms | 163.20 ms |
| p50 | 151.88 ms | 162.24 ms |
| p95 | 166.61 ms | 169.41 ms |
| p99 | 179.99 ms | 190.51 ms |

Average complete filtering-phase time was 24.79 ms. That includes fast
corridor work, independent safety, cache handling and required transfers.
Cost prediction remains nested in WM rollout time. Static benchmark states
had all Gaussian candidates valid, so prefiltering did not reduce WM work
there; the driving test does exercise rejections and reduced WM batches.

Exact caches, prepared road geometry, conservative box broad phases and
batched CPU corridor work reduced the initial unoptimized safety version's
roughly 460 ms mean substantially. Nevertheless mean planning still regresses
about 5.7% versus the original native benchmark, and 100 ms has not been met.
Remaining main costs are preserved SAC/coarse feature preparation, four WM
batch rollouts, independent body/traffic checking and Python dispatch.
Skipping required safety or changing SAC feature semantics was not used to
make the timing look better.

Checkpoint SHA256 (before and after read-only evaluation):
`4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a`.
Runtime `.py/.yaml` snapshot SHA256 is recorded in `v3_verified.json`; launcher
shell scripts, tests and reports are outside that benchmark fingerprint scope.

## Formal Fixed100 Run

Run the existing ablation framework through the launcher:

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
bash tools/run_v3_ablation.sh
```

This loads the trained 200k WM + 300k SAC full checkpoint, does no training,
uses current V3 MPPI parameters and independent safety, and runs A/B/C/D on
exactly `scenarios/fixed100.json`, test seed 1: **400 complete episodes**.
The script launches evaluation with direct-file logging and automatically
shows progress in the current terminal. Ctrl+C stops the viewer, not evaluation.
It does not overwrite prior output directories and does not set
`CUDA_VISIBLE_DEVICES`.

For optional per-step W&B logging, use this command INSTEAD of the preceding
launch, not in addition:

```bash
ENABLE_WANDB=true bash tools/run_v3_ablation.sh
```

This uses the existing account/project and proxy port 7897. W&B console
capture is disabled, no model artifacts are uploaded, and cloud failure does
not discard the local step evidence. For three repeats, set
`REPEAT_SEEDS='[1,2,3]'` before the launcher; that is 1200 episodes on the same
100 unique maps, not 300 distinct maps.

Inspect the exact Hydra invocation without starting evaluation:

```bash
bash tools/run_v3_ablation.sh --dry-run
```

The equivalent foreground invocation is:

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
checkpoint="$PWD/logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm200k_rl300k_20261103_195237/models/milestone_500k_full_mpr_mpc.pt"
out="$PWD/logs/benchmark/mpr_mpc_v3_fixed100_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_CONSOLE=off \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -u evaluate_ablation.py \
  +experiment=E5 checkpoint="$checkpoint" \
  restore_checkpoint_runtime=true restore_mppi_runtime=false \
  seed=1 eval_episodes=100 enable_wandb=false save_video=false \
  mpr_mpc.mppi.safety.enabled=true \
  +benchmark.cost_ablation=false +benchmark.episodes=100 \
  '+benchmark.repeat_seeds=[1]' \
  +benchmark.scenarios_file="$PWD/scenarios/fixed100.json" \
  +benchmark.scenario_start=300 +benchmark.scenario_pool_count=2000 \
  +benchmark.scenario_seed=1 +benchmark.bootstrap_samples=2000 \
  +benchmark.bootstrap_seed=1 +benchmark.mppi_cost_coef=1.0 \
  +benchmark.save_step_diagnostics=true +benchmark.enable_wandb=false \
  +benchmark.output_dir="$out"
```

Outputs include `episodes.csv`, group/seed CSVs, `summary.csv/json`,
`seed_summary.csv`, `paired_comparisons.csv`, `detailed_summary.json`, effective
configuration, scenario/manifest files and `steps/*.jsonl` with actual executed
actions and per-step planner diagnostics. The launcher additionally writes
`run.log`, `command.sh` and `evaluation.pid`.

## Conclusion

MPPI genuinely executes and can change the first executed action. Unsafe
candidates are filtered before WM candidate inference; unsafe SAC cannot be
forced back by the gain gate. WM Cost affects ranking but does not certify
safety. C has an independent context even with no SAC residual-state building.
The implementation is runnable for diagnostic fixed100 ablations, but the
single-scene failures, frequent stopping fallback and measured latency mean
that effectiveness and real-time acceptance remain open requirements.
