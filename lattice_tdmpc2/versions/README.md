# Lattice TD-MPC2 Version Records

This directory stores immutable experiment records and executable V2/V3/V4
algorithm modules. Shared network/planner structure remains in
`lattice_tdmpc2/`, `sac/`, `lattice/`, and `env/`.

## Records

| File | Experiment | Purpose |
| --- | --- | --- |
| `v1_record.yaml` | 2026-09-08 V1 | Historical 16-candidate/WM-auxiliary run |
| `v2_record.yaml` | 2026-09-10 V2 | Historical online-WM PID-SACL run |
| `v3_record.yaml` | 2026-09-11 V3 | Current frozen-WM projected-dual SACL baseline |
| `v4_record.yaml` | V4, untrained | Lane-width residual + frozen H10 WM cost-to-go SACL |
| `checkpoint_sha256.txt` | V1/V2/V3 | Detect accidental checkpoint replacement |

Executable source is split by version:

| Version | SAC and multiplier | Replay/constraint | Environment cost | Launch preset |
| --- | --- | --- | --- | --- |
| V2 | `v2/sac_model_v2.py` | `v2/training_v2.py` | `v2/cost_v2.py` | `v2/config_v2.yaml` |
| V3 | `v3/sac_model_v3.py` | `v3/training_v3.py` | `v3/cost_v3.py` | `v3/config_v3.yaml` |
| V4 | `v4/sac_model_v4.py` | `v4/training_v4.py` | reuses V3 real-cost logging | `v4/config_v4.yaml` |

The version files contain executable training behavior, not pseudocode. The
shared trainer dispatches them through `algorithm_version`; the tests in
`../tests/test_versions.py` lock the intended behavioral differences.

Each YAML records the checkpoint pair, W&B source, action mapping, world-model
mode, actor/cost losses, constraint definition, and final evaluation summary.
The `*_record.yaml` files are documentation records. The version-directory
`config_v*.yaml` files are Hydra launch presets wired to executable code.

## Rollback Rules

1. To inspect a trained policy, load the exact SAC and TD-MPC2 paths recorded in
   that version's YAML. Never mix V2 SAC with the V3 frozen world model.
2. Launch V2 with `--config-name versions/v2/config_v2`; this dispatches to the
   restored PID/trust/unsafe-cost implementation under `versions/v2/`.
3. Launch V3 with `--config-name versions/v3/config_v3`; this dispatches to the
   projected-dual/Huber/deduplicated-cost implementation under `versions/v3/`.
4. Launch V4 with `--config-name versions/v4/config_v4`. V4 must start a new SAC
   policy because its normalized lateral action has a different physical meaning.
5. V1 uses the pre-`symmetric_coarse_speed_v2` action mapping and is rejected by
   the current evaluator. Its world-model checkpoint remains independently usable.
6. Verify large model files with `sha256sum -c` before moving or renaming them.

No version checkpoint contains the SAC replay buffer or MetaDrive RNG state.
Loading one for optimization is therefore a warm start, never a bit-for-bit
continuation of the original run.

## Source Ownership

- `sac/model.py`: reward critics, cost critics, actor loss and multiplier update.
- `lattice_tdmpc2/trainer.py`: replay cost, online/frozen WM mode and checkpoints.
- `lattice_tdmpc2/residual_action.py`: normalized-to-physical residual mapping.
- `lattice_tdmpc2/planner.py`: one-residual refinement and fallback behavior.
- `lattice/frenet_metadrive.py`: trajectory tracking, including zero target speed.
- `env/env.py`: risk-field and deduplicated event cost.
- `lattice_tdmpc2/evaluator.py`: diagnostic TD-MPC2 trajectory scoring.

W&B run files and checkpoint configs remain the primary experiment records.
V2 source was not committed or uploaded to W&B, but its exact V2-to-V3 patches
were recovered from the local Codex session and reconstructed under `v2/`.
