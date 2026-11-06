# Native MPPI + World Model Cost

## Scope

Changes are confined to `mpr_mpc_v2`. The sibling `mpr_mpc`, trained model files,
environment success conditions, environment horizon, Lattice implementation,
Residual SAC networks, action mappings and SAC training hyperparameters are untouched.
The default schedule remains WM/Cost training at 0--20K, frozen-WM SAC training
at 20--50K, and MPPI for evaluation actions only. The existing 200K + 300K run
is loaded without retraining or overwriting its checkpoint.

## Changed Files

| Files | Change |
| --- | --- |
| `planning/local_mppi.py` | Single Gaussian, native elite updates, exact per-plan candidate cache, WM cost score and diagnostic outputs |
| `planning/evaluator.py` | One batched latent rollout per unique candidate; reward, Q and step-cost predictions from the same states/actions |
| `planning/coordinator.py` | Remove candidate geometry/emergency calls and repeated selected/baseline rollouts; preserve SAC input features; full-plan timing |
| `planning/timing.py` | CPU wall and CUDA-stream phase timing with no per-phase synchronization |
| `planning/native_config.py` | Restore only native checkpoint planner parameters, not retired search/safety settings |
| `planning/cost_validation.py` | Basic held-out horizon errors/correlation and optional same-origin candidate ranking, no mode approvals or anchor hashes |
| `planning/types.py` | Discounted reward and rollout timings in trajectory results |
| `vendor/tdmpc2_runtime/common/world_model.py` | Replace mode-approval buffers with simple validation metadata; all learned head weights are unchanged |
| `vendor/tdmpc2_runtime/tdmpc2.py` | Load historical cost weights; ignore retired approval buffers and revoke old approval schemas |
| `agent.py`, `trainer.py` | Remove obsolete mode-dataset setup; keep learner loss, held-out replay split and training stages |
| `config.yaml`, `experiment/E0.yaml`, `E2.yaml`, `E3.yaml`, `E4.yaml`, `E5.yaml` | Native MPPI + cost config; historical E3/E4 names are compatibility aliases, not new search algorithms |
| `evaluate_model_suite.py`, `evaluate_ablation.py` | Native/WM-cost paired ablation and checkpoint migration; in-episode progress |
| `evaluation_diagnostics.py`, `evaluation_statistics.py` | Terminal/timeout, speed, actual/predicted cost and complete planning-time metrics |
| `evaluate.py`, `evaluate_frozen_stack.py` | Handle textual end reasons in W&B summaries |
| `tools/benchmark_mppi.py`, `tools/planning_profiler.py`, `tools/smoke_v2.py` | Native timing benchmarks, explicit source snapshots, real cost-supervision smoke |
| `planning/safety.py` | Preserve existing road-polygon fix and SAC feature compatibility; remove obsolete emergency score output |
| `tests/test_native_mppi_cost.py`, existing planner/cost/round tests | Replace tests of removed algorithms with native search, cost ranking, validation, input and checkpoint regression tests |

Deleted modules:

- `planning/multimodal.py`
- `planning/emergency.py`
- `planning/calibration_anchor.py`
- `tools/calibrate_cost_modes.py`

No candidate-level `TrajectorySafetyEvaluator`, expanded search, emergency
controller, per-mode distribution update, uncertainty score, TTC/clearance/boundary
score or mode-specific cost approval remains in `LocalMPPI`.

### Explicit SAC Compatibility Exception

The trained E5 policy consumes 535 inputs, including values/masks for coarse-path
TTC, clearance and boundary margin. Deleting their calculation would silently
change policy inputs. Therefore `safety.py` and its swept-geometry primitive remain
ONLY as an on-demand coarse SAC state-feature extractor. There is one coarse
feature calculation, not one calculation per candidate/iteration. Stage A skips
it entirely. Policies without these features do not instantiate it.

This is deliberate retention of a dependency still used by SAC, not an MPPI
candidate filter or score. Removing even this legacy calculation requires a
separate SAC-input migration/retraining experiment, outside the request to keep
SAC inputs unchanged. Its cost is included in the reported full planning time.
The old E5 cost-input mask also remains disabled when `coarse_cost_feature=false`,
independently of MPPI's new use of cost predictions.

## Native Source and Parameters

Source: sibling `mpr_mpc` Git commit
`f09b64fa1ee432acc5eaa210b624ed5b24f1cbd4`, the native residual-SAC MPPI version.
Its planner blob is `76c71e67ecd3334fb77ef57d3047238ba8559525`; its config blob is
`26e96068159f60d9231b95aee535eaa65f789a35`. The earlier pre-SAC `6e728b6` config
had 512/64/6; the later native SAC config explicitly used 64/8/4. This delivery
uses the latter, rather than inventing a 48-sample/two-iteration workload.

| Parameter | Historical value | Current value |
| --- | --- | --- |
| samples / elites / iterations | 64 / 8 / 4 | 64 / 8 / 4 |
| initial std | [0.025, 0.10] | unchanged |
| min std | [0.005, 0.02] | unchanged |
| max std | [0.05, 0.20] | unchanged |
| max delta | [0.05, 0.20] | unchanged |
| temperature | 0.5 | unchanged |
| deviation / smoothness coefficients | 5.0 / 1.0 | unchanged |
| throttle deviation weight | 0.25 | unchanged |
| minimum absolute / relative improvement | 1.0 / 0.01 | unchanged |
| initialization / blend alpha | structured_baseline / 0.8 | unchanged |
| corridor ratio / margin | 0.5 / 0.15 m | unchanged |
| WM cost enabled / coefficient | new score extension | true / 1.0 |

The full-ensemble mean terminal Q evaluator is retained. There is no Q-standard-
deviation penalty. The historical baseline improvement gate, previous-mean/blend
warm-start options, scalar/vector std constraints and simple kinematic corridor
remain. A retained Lattice fallback is not described as independently certified safe.

## Score and Planning Flow

```text
V = sum(k=0..H-1, gamma^k * predicted_reward[k]) + gamma^H * terminal_Q
C = sum(k=0..H-1, cost_gamma^k * predicted_step_cost[k])
J = V - cost_coef*C - deviation_coef*D - smoothness_coef*S
```

Native episodic reward/terminal-Q masking is preserved. Predicted early termination
does not hide future predicted cost. The Cost Head is a softplus nonnegative
ONE-STEP predictor, trained against raw environment cost with MSE. No extra
normalization, terminal cost-Q, repeated accumulation or second cost subtraction
is introduced. `cost.gamma` controls the one cost accumulation in the evaluator.
The terminal action `u_H` is used only for Q, never as another transition/cost.

Flow:

1. Unchanged Lattice generation/filtering selects the coarse path.
2. Encode once and compute the coarse features used by the existing SAC policy.
3. Unchanged SAC outputs delta-d/delta-v; unchanged mapping and Lattice regeneration
   produce the baseline action sequence.
4. Sample one Gaussian around that baseline; always include the baseline and mean.
5. Evaluate unique action sequences in batches with the same latent dynamics,
   reward and cost heads. Exact duplicate baselines/means reuse their prediction
   only within the current plan, never across observations.
6. Rank with J, take Top-K, exponential-weight the elites, update mean and variance,
   project into native trust/action bounds and repeat four times.
7. Return the selected sequence's first action. Reuse its already computed value
   and cost in coordinator diagnostics; do not roll it out again for logging.

`cost_coef=0` removes the cost score exactly. With `cost_enabled=true`, predictions
are still computed for diagnostic comparison. `cost_enabled=false` skips the MPPI
candidate Cost Head forward as well. Cost Head training and SAC safety constraints
are independent of these evaluation score switches.

Non-finite/negative cost predictions raise an explicit diagnostic error. Finite
unvalidated predictions DO affect the cost-enabled score, with a warning and
`mppi/cost_validated=0`; they are never silently replaced by zero or called a guarantee.

## Cost Training and Validation Evidence

Both top-level `cost.enabled` and `metadrive.cost.enabled` remain true by default.
The original learner still adds its cost-supervised MSE to the loss, uses real
environment labels from replay, updates the same Cost Head parameters, and freezes
the whole model when SAC begins. No label quality, cost scale, network shape,
SAC cost critic or Lagrangian update was changed.

The real CUDA/MetaDrive smoke passed with:

- cost supervised loss: 0.7306962
- pre-clip Cost Head gradient norm: 26.50664
- real supervised samples: 40
- training-time MPPI: false, including a step beyond the deployment threshold
- residual state: 535 inputs
- three WM/SAC/full milestone save/load round trips
- finite predicted cost included in final MPPI score

Basic validation reports MAE, RMSE, correlation and per-horizon errors. Validation
metadata uses schema 3, records the number of cost updates at validation, and is
revoked on weight updates, stale schemas or a changed cost definition. It is a
diagnostic; it does not gate MPPI cost scoring or certify safety.

Same-origin candidate ranking is supported by `CostModelValidator.set_candidates`:
records must identify a common `origin_id`, the same observation, executed action
sequences and real cost labels (`executed=true`). Different starting observations
are not silently paired. No modes, six-way approvals, anchor hashes or per-plan
calibration are used. Executed-policy windows alone cannot establish this ranking;
when such candidate replay data is missing, the ranking metric is explicitly NaN/null.

## Measurements

Checkpoint used for BOTH score variants:

```text
logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm200k_rl300k_20261103_195237/models/milestone_500k_full_mpr_mpc.pt
SHA256: 4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a
```

### Complete Planning Latency

Measured from `coordinator.plan` entry through CPU action construction, including
Lattice, coarse SAC features, SAC inference and all MPPI work. The benchmark also
synchronizes CUDA around the outer call. Both variants use the SAME five live
states (1714, 539, 1525, 1084, 694), 20-step prefix, five warm-ups per scene,
seed 1, H=10, N=64, elites=8, iterations=4 and 100 measured calls each.

See `reports/native_cost/final_benchmark_no_cost.json`,
`final_benchmark_wm_cost.json`, matching raw CSVs and `benchmark_comparison.csv`
for final values and the exact runtime-source fingerprint. The source snapshot
is explicit because the user worktree is intentionally not committed or reset.
Cost is computed in both variants for matched diagnostics; only its score weight
changes from 0 to 1. This isolates scoring, not the cost of removing the head forward.

| Synchronized full-plan latency | Cost weight 0 | Cost weight 1 |
| --- | --- | --- |
| mean, ms | 144.38 | 146.80 |
| p50, ms | 145.15 | 147.41 |
| p95, ms | 149.25 | 155.16 |
| p99, ms | 153.89 | 173.02 |

Both final runs used runtime source snapshot SHA256
`9ebf355f291f63bb8b6ff1ea56e702c09b2a133c0864bac78645afd506839b85`.
The measured average is approximately 145 ms, above the 100 ms control-cycle
reference. The implementation must NOT be described as meeting the real-time
target. The coarse legacy SAC geometry and Python-driven model rollout still
cost time. Stream-elapsed phase timings include idle CPU gaps and are not pure
kernel times. `cost_prediction_ms` is nested inside `world_model_rollout_ms`, so
these phase values must not be summed as disjoint costs.

### Four Complete Driving Episodes, Two Shared Scenes

Pilot only: scenes [1714, 539], test seed 1, paired episode RNG seed = scenario+1,
same checkpoint and MPPI noise seed. Every episode ran to the unchanged environment
terminal condition. No 1000-step limit or success criterion was modified.

| Metric | Native MPPI, cost weight 0 | Native MPPI + WM Cost, weight 1 |
| --- | --- | --- |
| success | 2/2 | 2/2 |
| crash / out-of-road | 0 / 0 | 0 / 0 |
| timeout | 0/2 | 0/2 |
| mean route completion | 0.98756 | 0.98537 |
| mean speed, m/s | 7.9974 | 8.0079 |
| mean actual episode cost | 2.45602 | 1.41344 |
| mean predicted episode cost | 3.21821 | 3.32631 |
| mean complete planning time, ms | 157.99 | 158.08 |
| lengths, in scenario order | 496, 388 | 498, 385 |

Actual episode cost is the sum of real environment labels. Predicted episode cost
is the sum of ONLY the first-step predictions along executed controls, not the
sum of overlapping H-step forecasts. Detailed outputs, confidence intervals,
per-scenario metrics and paired comparisons are under
`reports/native_cost/paired_driving_smoke/`.

Post-evaluation diagnostics (no model weight updates) gave one-step MAE/RMSE
0.004943/0.012419 for the no-cost-score variant and 0.006192/0.017702 for the
cost-score variant. Corresponding correlations were 0.5773 and 0.3314. H=10 MAEs
were 0.05444 and 0.05366. These are executed-policy diagnostics, NOT evidence
that all counterfactual candidate costs are calibrated.

Two scenes are insufficient to claim a reliable reduction in risk, an optimal
cost coefficient or a general success-rate improvement. No full 100-scene run
was started automatically.

## 1000-Step Timeout Investigation

- The environment still uses `horizon=1000`, decision repeat 5 and physics dt .02.
- `terminated` and `truncated` remain distinct; training bootstrap semantics are unchanged.
- All four pilot episodes ended with `terminated=1`, `truncated=0`,
  `end_reason=arrive_dest`; SUCCESS triggers normally.
- Pilot mean requested delta-v was positive, not persistent negative braking.
  Mean SAC target speeds were about 11.7--12.3 m/s, actual speeds about 7.8--8.1 m/s.
- Mean applied cost penalty in the cost-enabled pilot was about 0.067--0.074 per
  planning score. There was no observed cost-induced standstill on these two maps.
- Earlier scenario 1714 in the complex planner had timed out at 1000 steps with
  about 11% completion and .47 m/s. The new paired pilot is not an identical
  historical RNG protocol, so this is diagnostic context, not a controlled
  attribution of the old timeout to any one module.
- Cost scale remains imperfect: predicted episode totals exceed actual totals,
  and candidate risk ordering has not been validated with real same-origin pairs.
  Inspect the new metrics before tuning weights; do not arbitrarily increase
  `cost_coef` to force low-speed behavior.

## Tests and Limits

The full suite plus environment tests passed (199 tests at the verification run).
Tests cover actual Cost Head gradient updates and checkpoint migration, exact
discount/terminal semantics, non-finite cost errors, cost-coefficient zero, lower-
cost candidate preference with equal value, no repeated candidate rollout,
trust region/baseline/elite updates, native parameter provenance, existing SAC
state layout/masks and separate terminated/truncated diagnostics. Tests specifically
requiring the removed modes, emergency controllers and mode approvals were replaced,
not silently skipped. Legacy SAC geometric-feature primitive tests remain.

Remaining limitations: the 100 ms target is unmet; only two complete paired pilot
scenes have been run; independent real candidate-ranking data is still missing;
the old risk-input SAC policy still requires one coarse geometry feature pass.
Historical E3/E4 profile names are now aliases, not meaningful new planner ablations.
Old six-mode approvals are not reusable, and old full-training optimizer resumes
remain subject to strict planner/config compatibility checks.

## Full Fixed-100 Cost Ablation Command

This runs 100 identical unique scenes per variant, 200 episodes total. It does
not train, overwrite checkpoints or upload W&B runs. For three test seeds, change
the repeat list to [1,2,3], giving 600 episodes but still 100 unique maps per variant.

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
set -o pipefail
checkpoint="$PWD/logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm200k_rl300k_20261103_195237/models/milestone_500k_full_mpr_mpc.pt"
out="$PWD/logs/benchmark/native_wm_cost_fixed100_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -u evaluate_ablation.py \
  +experiment=E5 checkpoint="$checkpoint" restore_checkpoint_runtime=true \
  seed=1 eval_episodes=100 enable_wandb=false save_video=false \
  metadrive.simulator.start_seed=300 metadrive.simulator.num_scenarios=2000 \
  metadrive.simulator.use_render=false \
  +benchmark.cost_ablation=true +benchmark.cost_coef=1.0 \
  +benchmark.episodes=100 '+benchmark.repeat_seeds=[1]' \
  +benchmark.scenarios_file="$PWD/scenarios/fixed100.json" \
  +benchmark.scenario_start=300 +benchmark.scenario_pool_count=2000 \
  +benchmark.scenario_seed=1 +benchmark.bootstrap_samples=2000 \
  +benchmark.bootstrap_seed=1 +benchmark.output_dir="$out" \
  2>&1 | tee "$out/run.log"
```

Read `summary.csv`, `episodes.csv`, `paired_comparisons.csv`,
`detailed_summary.json`, `manifest.json` and each per-variant effective config.
The existing four-way Lattice/SAC/MPPI entry remains available by leaving
`benchmark.cost_ablation` unset/false.
