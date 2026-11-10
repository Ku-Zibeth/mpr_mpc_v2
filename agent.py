"""Training facade combining TD-MPC2, MPR-MPC planning, and Residual SAC."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .planning.config import mapping, section
from .planning.cost_validation import CostModelValidator, cost_definition
from .residual_rl import ResidualReplayBuffer, ResidualSACAgent
from .residual_rl.types import ResidualPolicyOutput


class MPRMPCAgent:
    """Expose the original trainer API while routing actions through MPR-MPC."""

    def __init__(self, cfg, tdmpc_agent, planner, env):
        self.cfg = cfg
        self.tdmpc_agent = tdmpc_agent
        self.planner = planner
        self.env = env
        self.model = tdmpc_agent.model
        self.device = tdmpc_agent.device

        self.residual_cfg = mapping(section(cfg, "residual_rl"))
        self.residual_rl_enabled = bool(self.residual_cfg.get("enabled", True))
        self.planner.allow_residual = bool(self.planner.allow_residual and self.residual_rl_enabled)
        self.freeze_tdmpc2_on_start = bool(
            self.residual_cfg.get("freeze_tdmpc2_on_start", True)
        )
        self.seed_steps = max(0, int(self.residual_cfg.get("seed_steps", 5000)))
        self.warmup_action_scale = min(
            1.0, max(0.0, float(self.residual_cfg.get("warmup_action_scale", 0.5)))
        )
        self.residual_learning_starts = max(
            0, int(self.residual_cfg.get("learning_starts", 5000))
        )
        self.residual_update_per_step = max(
            0.0, float(self.residual_cfg.get("update_per_step", 0.2))
        )
        self.residual_batch_size = max(1, int(self.residual_cfg.get("batch_size", 256)))
        self.residual_gamma = float(self.residual_cfg.get("gamma", 0.99))
        self.invalid_residual_penalty = max(
            0.0, float(self.residual_cfg.get("invalid_residual_penalty", 0.2))
        )
        self.zero_like_action_threshold = max(
            0.0, float(self.residual_cfg.get("zero_like_action_threshold", 0.05))
        )
        stats_window = max(1, int(self.residual_cfg.get("residual_stats_window", 1000)))
        self._residual_stats = deque(maxlen=stats_window)

        self.residual_sac = ResidualSACAgent(
            planner.residual_state_dim,
            self.residual_cfg,
            device=self.device,
        )
        self.residual_replay = ResidualReplayBuffer(
            int(self.residual_cfg.get("buffer_size", 200000)),
            planner.residual_state_dim,
            action_dim=2,
        )
        self.planner.set_residual_action_provider(self._select_residual_action)
        self._print_residual_startup_diagnostics()

        debug_cfg = mapping(section(cfg, "mpr_mpc").get("debug"))
        self.debug_enabled = bool(debug_cfg.get("enabled", False))
        self.debug_every = max(1, int(debug_cfg.get("every", 100)))
        self._planner_calls = 0
        self.global_env_step = 0
        self.best_eval_key = None
        self.best_eval_selection = None
        self.last_plan = None
        self.last_residual_transition_metrics: dict[str, float] = {}
        self.residual_rl_step = 0
        self.residual_update_budget = 0.0
        self._pending_residual: dict | None = None
        self._residual_episode_cost = 0.0
        self._residual_episode_steps = 0
        self._residual_episode_complete = False
        self._tdmpc_frozen = False
        self.last_loaded_checkpoint_metadata: dict = {}
        self.cost_validator = CostModelValidator(mapping(section(cfg, "cost")), label_definition=cost_definition(cfg))
        self.cost_validation_metrics = {}

    def _print_residual_startup_diagnostics(self) -> None:
        print(
            "\nSAC:\n"
            f"actor_lr={float(self.residual_cfg.get('actor_lr', 5e-4))} "
            f"critic_lr={float(self.residual_cfg.get('critic_lr', 1e-3))} "
            f"alpha={float(self.residual_cfg.get('alpha', 0.005))} "
            f"auto_alpha={bool(self.residual_cfg.get('auto_alpha', True))} "
            f"tau={float(self.residual_cfg.get('tau', 0.05))} "
            f"gamma={float(self.residual_cfg.get('gamma', 0.99))}\n\n"
            "Lagrangian:\n"
            f"enabled={self.residual_sac.use_lagrangian} "
            f"cost_limit={self.residual_sac.cost_limit:g} "
            f"lr={self.residual_sac.lagrangian.learning_rate:g}\n\n"
            f"Stage-B TD-MPC2 freeze={self.freeze_tdmpc2_on_start}; MPPI is evaluation-only"
        )

    def set_global_step(self, step: int) -> None:
        self.global_env_step = max(0, int(step))

    def _maybe_freeze_tdmpc2(self, step: int) -> None:
        if self._tdmpc_frozen or not self.freeze_tdmpc2_on_start:
            return
        if not self.planner.stage_for_step(step).use_residual:
            return
        self.validate_cost_model()
        self.tdmpc_agent.model.eval()
        for parameter in self.tdmpc_agent.model.parameters():
            parameter.requires_grad_(False)
        self._tdmpc_frozen = True

    @torch.no_grad()
    def _select_residual_action(self, state: torch.Tensor, *, eval_mode: bool) -> ResidualPolicyOutput:
        if eval_mode:
            return self.residual_sac.select_action(state, deterministic=True)
        if self.residual_rl_step < self.seed_steps:
            action = (
                torch.rand((1, 2), device=self.device, dtype=torch.float32) * 2.0 - 1.0
            ) * self.warmup_action_scale
            zeros = torch.zeros_like(action)
            return ResidualPolicyOutput(action=action, mean=zeros, log_std=zeros, warmup=True)
        return self.residual_sac.select_action(state, deterministic=False)

    def _complete_pending_with_next_state(self, next_state: torch.Tensor) -> None:
        pending = self._pending_residual
        if not pending or "reward" not in pending:
            return
        self.residual_replay.add(
            pending["state"],
            pending["action"],
            pending["reward"],
            pending["cost"],
            next_state.detach().cpu().numpy(),
            pending["done"],
            diagnostics=pending.get("diagnostics"),
        )
        self._pending_residual = None

    def observe_transition(self, reward: float, cost: float, done: bool, info: dict | None = None) -> None:
        del info
        pending = self._pending_residual
        if not pending or "reward" in pending:
            return
        env_reward = float(reward)
        env_cost = float(cost)
        invalid_penalty = self.invalid_residual_penalty if not pending["valid"] else 0.0
        residual_reward = env_reward - invalid_penalty
        pending.update(
            reward=residual_reward,
            cost=env_cost,
            done=bool(done),
        )
        weight = (self.residual_sac.cost_gamma ** self._residual_episode_steps
                  if self.residual_sac.constraint_alignment else 1.0)
        self._residual_episode_cost += weight * env_cost
        self._residual_episode_steps += 1
        self.last_residual_transition_metrics = {
            "env/reward": env_reward,
            "residual_rl/reward": residual_reward,
            "residual_rl/invalid_penalty": invalid_penalty,
            "residual_rl/env_cost": env_cost,
        }
        if done:
            terminal_state = pending["state"]
            self.residual_replay.add(
                terminal_state,
                pending["action"],
                residual_reward,
                env_cost,
                terminal_state,
                True,
                diagnostics=pending.get("diagnostics"),
            )
            self._pending_residual = None
            if self._residual_episode_steps > 0 and getattr(self, "_residual_episode_complete", True):
                self.last_residual_transition_metrics.update(
                    self.residual_sac.update_lagrangian(
                        self._residual_episode_cost,
                        self._residual_episode_steps,
                    )
                )
            self._residual_episode_cost = 0.0
            self._residual_episode_steps = 0

    def act(self, obs, t0=False, eval_mode=False, task=None, global_step=None):
        if not self.planner.enabled:
            return self.tdmpc_agent.act(obs, t0=t0, eval_mode=eval_mode, task=task)
        if task is not None:
            raise ValueError("The current MetaDrive MPR-MPC integration is single-task only.")
        step = self.global_env_step if global_step is None else int(global_step)
        self.set_global_step(step)
        self._maybe_freeze_tdmpc2(step)
        if t0 and eval_mode:
            scenario = int(self.env.unwrapped_metadrive.current_seed)
            self.planner.reset(seed=(int(self.cfg.seed) + scenario) % 2**32)
        result = self.planner.plan(
            obs,
            self.env.unwrapped_metadrive.agent,
            global_step=step,
            t0=t0,
            eval_mode=eval_mode,
        )
        self.last_plan = result
        if eval_mode and result.stage.use_mppi:
            import hashlib
            sequence = result.debug.get("selected_mppi_actions")
            certificate = result.debug.get("safety_certificate") or {}
            matches = (sequence is not None and torch.equal(result.action.cpu(), sequence[0].cpu())
                and hashlib.sha256(sequence.contiguous().cpu().numpy().tobytes()).hexdigest() == certificate.get("sequence_sha256"))
            failure = result.safety_failure or not certificate.get("valid", False) or not matches
            if failure:
                result.safety_failure = True
                result.metrics.update({"mppi/safety_failure": 1., "mppi/execution_authorized": 0.,
                    "mppi/selected_safety_valid": 0., "mppi/selected_exact_verified": 0., "mppi/emergency_verified": 0.})
                stop = getattr(self.env, "request_safety_failure", None)
                if stop is None:
                    raise RuntimeError("Planner safety_failure: environment cannot terminate without executing an unverified action.")
                stop("no_verified_action" if matches else "action_certificate_mismatch")
        context = self.planner.last_context
        if not eval_mode:
            if t0:
                self._residual_episode_complete = bool(result.stage.use_residual)
            if context is not None and context.residual_state is not None:
                self._complete_pending_with_next_state(context.residual_state)
            self._planner_calls += 1
            if self.debug_enabled and self._planner_calls % self.debug_every == 0:
                coarse = result.coarse_parameters
                refined = result.refined_parameters
                residual = result.residual.detach().cpu().numpy()
                print(
                    "[mpr] "
                    f"candidates={int(result.metrics['mpr/candidate_count'])} "
                    f"valid={int(result.metrics['mpr/feasible_count'])} "
                    f"selected={int(result.metrics['mpr/selected_index'])} "
                    f"coarse={coarse.tolist()} residual={residual.tolist()} "
                    f"refined={refined.tolist()} "
                    f"WM={result.metrics['mpr/baseline_wm_value']:.3f}->"
                    f"{result.metrics['mpr/selected_wm_value']:.3f} "
                    f"score={result.metrics['mpr/baseline_score']:.3f}->"
                    f"{result.metrics['mpr/selected_score']:.3f} "
                    f"time_ms={result.metrics['mpr/planning_ms']:.2f}"
                )
            if (
                self.residual_rl_enabled
                and result.stage.use_residual
                and context is not None
                and context.residual_state is not None
            ):
                self._pending_residual = {
                    "state": context.residual_state.detach().cpu().numpy(),
                    "action": result.requested_normalized_residual.detach().cpu().numpy(),
                    "valid": bool(result.residual_valid),
                    "diagnostics": {
                        **{key: result.metrics.get(f"mpr/{key}", float("nan")) for key in
                           ("lattice_predicted_cost", "sac_predicted_cost", "mppi_predicted_cost",
                            "mppi_correction_magnitude", "sac_risk_valid")},
                        "executed_action": result.action.detach().cpu().numpy(),
                    },
                }
                self._record_residual_stats(result)
                self.residual_rl_step += 1
        return result.action

    def _record_residual_stats(self, result) -> None:
        action = result.requested_normalized_residual.detach().cpu().numpy().reshape(2)
        delta = result.residual.detach().cpu().numpy().reshape(2)
        fallback = bool(result.metrics.get("mpr/used_lattice_fallback", 0.0))
        self._residual_stats.append(
            {
                "abs_action": np.abs(action).astype(np.float32),
                "abs_delta": np.abs(delta).astype(np.float32),
                "zero_like": float(
                    abs(action[0]) < self.zero_like_action_threshold
                    and abs(action[1]) < self.zero_like_action_threshold
                ),
                "fallback": float(fallback),
            }
        )

    def _residual_stats_metrics(self) -> dict[str, torch.Tensor]:
        if not self._residual_stats:
            return {}
        abs_action = np.stack([item["abs_action"] for item in self._residual_stats])
        abs_delta = np.stack([item["abs_delta"] for item in self._residual_stats])
        fallback = np.asarray([item["fallback"] for item in self._residual_stats], dtype=np.float32)
        normal = 1.0 - fallback

        def masked_mean(values, mask):
            if float(mask.sum()) <= 0.0:
                return 0.0
            return float(values[mask > 0.5].mean())

        return {
            "residual_stats/mean_abs_delta_d": torch.tensor(float(abs_delta[:, 0].mean()), device=self.device),
            "residual_stats/mean_abs_delta_v": torch.tensor(float(abs_delta[:, 1].mean()), device=self.device),
            "residual_stats/max_abs_delta_d": torch.tensor(float(abs_delta[:, 0].max()), device=self.device),
            "residual_stats/max_abs_delta_v": torch.tensor(float(abs_delta[:, 1].max()), device=self.device),
            "residual_stats/mean_abs_action_d": torch.tensor(float(abs_action[:, 0].mean()), device=self.device),
            "residual_stats/mean_abs_action_v": torch.tensor(float(abs_action[:, 1].mean()), device=self.device),
            "residual_stats/zero_like_ratio": torch.tensor(
                float(np.mean([item["zero_like"] for item in self._residual_stats])),
                device=self.device,
            ),
            "residual_stats/fallback_mean_abs_delta": torch.tensor(
                masked_mean(abs_delta, fallback), device=self.device
            ),
            "residual_stats/normal_mean_abs_delta": torch.tensor(
                masked_mean(abs_delta, normal), device=self.device
            ),
        }

    def update_tdmpc(self, buffer, *, enabled: bool) -> dict[str, torch.Tensor]:
        info = {}
        stage = self.planner.stage_for_step(self.global_env_step)
        if not enabled:
            info["wm/replay_ready"] = torch.tensor(0.0, device=self.device)
            return info
        if self._tdmpc_frozen and stage.use_residual:
            info["wm/frozen"] = torch.tensor(1.0, device=self.device)
            return info
        info.update(dict(self.tdmpc_agent.update(buffer)))
        interval = max(1, int(section(self.cfg, "cost").get("validation_interval", 1000)))
        if hasattr(self.model, "_cost_updates") and int(self.model._cost_updates) % interval == 0:
            info.update(self.validate_cost_model())
        for key in ("consistency_loss", "reward_loss", "value_loss", "termination_loss", "cost_loss",
                    "cost_grad_norm", "cost_updates", "cost_supervised_samples", "cost_label_mean"):
            if key in info:
                info[f"wm/{key}"] = info[key]
        info["wm/replay_ready"] = torch.tensor(1.0, device=self.device)
        return info

    def validate_cost_model(self, episodes=None):
        if not bool(section(self.cfg, "cost").get("enabled", False)):
            return {}
        if episodes is not None:
            self.cost_validator.set_episodes(episodes)
        self.cost_validation_metrics = self.cost_validator.validate(self.model, self.device, horizon=self.planner.horizon)
        return self.cost_validation_metrics

    def update_residual(self, *, enabled: bool = True) -> dict[str, torch.Tensor]:
        stage = self.planner.stage_for_step(self.global_env_step)
        if not (enabled and self.planner.enabled and self.residual_rl_enabled and stage.use_residual):
            return {}
        info = {
            "residual_rl/replay_size": torch.tensor(float(len(self.residual_replay)), device=self.device),
            "residual_rl/local_step": torch.tensor(float(self.residual_rl_step), device=self.device),
            "residual_rl/update_budget": torch.tensor(float(self.residual_update_budget), device=self.device),
            "residual_rl/update_steps": torch.tensor(float(self.residual_sac.update_steps), device=self.device),
            "lagrangian/value": torch.tensor(
                float(self.residual_sac.lagrangian_multiplier), device=self.device
            ),
            "lagrangian/cost_limit": torch.tensor(self.residual_sac.cost_limit, device=self.device),
        }
        if (
            self.residual_rl_step >= self.residual_learning_starts
            and len(self.residual_replay) >= self.residual_batch_size
        ):
            self.residual_update_budget += self.residual_update_per_step
            updates = 0
            while self.residual_update_budget >= 1.0:
                batch = self.residual_replay.sample(
                    self.residual_batch_size,
                    device=self.device,
                    gamma=self.residual_gamma,
                )
                info.update(self.residual_sac.update(batch))
                self.residual_update_budget -= 1.0
                updates += 1
            info["residual_rl/updates"] = torch.tensor(float(updates), device=self.device)
            info["residual_rl/update_budget"] = torch.tensor(
                float(self.residual_update_budget), device=self.device
            )
            info["residual_rl/update_steps"] = torch.tensor(
                float(self.residual_sac.update_steps), device=self.device
            )
        info.update(self._residual_stats_metrics())
        info.update(
            {
                key: torch.tensor(value, device=self.device)
                for key, value in self.last_residual_transition_metrics.items()
            }
        )
        return info

    def update(
        self,
        buffer,
        *,
        tdmpc_update_enabled: bool = True,
        residual_update_enabled: bool = True,
    ):
        info = {}
        info.update(self.update_tdmpc(buffer, enabled=tdmpc_update_enabled))
        info.update(self.update_residual(enabled=residual_update_enabled))
        if self.last_plan is not None:
            info.update(
                {
                    key: torch.tensor(value, device=self.device)
                    for key, value in self.last_plan.metrics.items()
                }
            )
        env_metrics = {
            "env/metadrive_reward": "metadrive_reward",
            "env/tdmpc2_reward": "tdmpc2_reward",
            "env/cost": "cost",
            "env/risk_field_cost": "risk_field_cost",
            "env/risk_field_normalized_cost": "risk_field_normalized_cost",
            "env/risk_field_reward_penalty": "risk_field_reward_penalty",
            "env/risk_field_boundary_cost": "risk_field_boundary_cost",
            "env/risk_field_lane_cost": "risk_field_lane_cost",
            "env/risk_field_offroad_cost": "risk_field_offroad_cost",
            "env/risk_field_vehicle_cost": "risk_field_vehicle_cost",
            "env/risk_field_object_cost": "risk_field_object_cost",
            "env/safety_risk_cost": "safety_risk_cost",
            "env/safety_event_cost": "safety_event_cost",
            "env/dynamic_collision_cost": "safety_dynamic_collision_cost",
            "env/static_collision_cost": "safety_static_collision_cost",
            "env/road_event_cost": "safety_road_event_cost",
            "env/other_event_cost": "safety_other_event_cost",
            "env/unallocated_event_cost": "safety_unallocated_event_cost",
            "env/route_completion": "route_completion",
        }
        last_env_info = self.env.last_info
        info.update(
            {
                metric: torch.tensor(float(last_env_info.get(source, 0.0)), device=self.device)
                for metric, source in env_metrics.items()
            }
        )
        return info

    def save(self, filepath) -> None:
        state = {
            "format": "mpr_mpc_v2_residual_sac",
            "algorithm_version": str(getattr(self.cfg, "algorithm_version", "unknown")),
            "experiment_profile": getattr(self.cfg, "experiment_profile", None),
            "metadata": self._checkpoint_metadata(),
            "model": self.tdmpc_agent.model.state_dict(),
            "tdmpc_optim": self.tdmpc_agent.optim.state_dict(),
            "tdmpc_pi_optim": self.tdmpc_agent.pi_optim.state_dict(),
            "tdmpc_scale": self.tdmpc_agent.scale.state_dict(),
            "global_env_step": self.global_env_step,
            "best_eval_key": self.best_eval_key,
            "best_eval_selection": getattr(self, "best_eval_selection", None),
            "mpr_mpc": {
                "residual_sac": self.residual_sac.state_dict(),
                "residual_rl_step": self.residual_rl_step,
                "residual_update_budget": self.residual_update_budget,
                "planner_calls": self._planner_calls,
                "tdmpc_frozen": self._tdmpc_frozen,
                "residual_invalid_count": self.planner.residual_invalid_count,
                "residual_attempt_count": self.planner.residual_attempt_count,
                "mppi_call_count": self.planner.mppi_call_count,
                "baseline_selected_count": self.planner.baseline_selected_count,
            },
        }
        torch.save(state, Path(filepath))

    @staticmethod
    def _plain(value):
        if value is None:
            return None
        try:
            return OmegaConf.to_container(value, resolve=True)
        except Exception:
            pass
        if isinstance(value, dict):
            return {key: MPRMPCAgent._plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [MPRMPCAgent._plain(item) for item in value]
        return value

    def _checkpoint_metadata(self) -> dict:
        sim = self._plain(section(self.cfg, "metadrive")).get("simulator", {})
        start, count = int(sim.get("start_seed", 0)), int(sim.get("num_scenarios", 0))
        fixed = list(getattr(self.cfg, "evaluation_scenarios", []) or [])
        reserved = fixed or list(range(start + max(0, count-int(getattr(self.cfg, "eval_episodes", 20))), start+count))
        return {
            "algorithm_version": str(getattr(self.cfg, "algorithm_version", "unknown")),
            "experiment_profile": getattr(self.cfg, "experiment_profile", None),
            "latent_dim": int(getattr(self.cfg, "latent_dim")),
            "model_size": int(getattr(self.cfg, "model_size")),
            "num_q": int(getattr(self.cfg, "num_q")),
            "horizon": int(getattr(self.cfg, "horizon")),
            "residual_state_dim": int(self.planner.residual_state_dim),
            "residual_action_dim": 2,
            "use_lagrangian": bool(self.residual_sac.use_lagrangian),
            "stage_config": self._plain(self.planner.mpr_cfg.get("stages", {})),
            "mppi_config": self._plain(self.planner.mppi_cfg),
            "risk_features": list(getattr(getattr(self.planner, "residual_state_builder", None), "risk_features", [])),
            "coarse_cost_feature": bool(self.residual_cfg.get("coarse_cost_feature", False)),
            "encoder_config": self._plain(section(self.cfg, "state_encoder")),
            "residual_mapping": getattr(self.planner, "residual_mapping", "linear"),
            "residual_config": self._plain(self.residual_cfg),
            "cost_definition": self._cost_definition(),
            "training_scene_range": self._plain(section(self.cfg, "metadrive").get("simulator", {})),
            "validation_scene_ids": reserved,
        }

    def _check_checkpoint_compatibility(
        self,
        state: dict,
        *,
        strict_planner_config: bool,
        allow_legacy: bool = False,
        check_residual: bool = True,
    ) -> None:
        metadata = state.get("metadata") if isinstance(state, dict) else None
        if not isinstance(metadata, dict):
            if allow_legacy:
                print("[checkpoint] Compatibility metadata missing; loading legacy checkpoint.")
                return
            metadata = {}
        current = self._checkpoint_metadata()
        checks = [
            ("algorithm_version", state.get("algorithm_version", metadata.get("algorithm_version"))),
            ("latent_dim", metadata.get("latent_dim")),
            ("model_size", metadata.get("model_size")),
            ("num_q", metadata.get("num_q")),
        ]
        if check_residual:
            checks.extend([("residual_state_dim", metadata.get("residual_state_dim")),
                           ("residual_action_dim", metadata.get("residual_action_dim"))])
        if strict_planner_config:
            planner_state = state.get("planner") or state.get("mpr_mpc") or {}
            checks.extend(
                [
                    ("horizon", metadata.get("horizon")),
                    ("use_lagrangian", metadata.get("use_lagrangian")),
                    (
                        "stage_config",
                        planner_state.get("stage_config", metadata.get("stage_config")),
                    ),
                ]
            )
            if bool(getattr(self.cfg, "restore_mppi_runtime", True)):
                checks.append(("mppi_config", planner_state.get("mppi_config", metadata.get("mppi_config"))))
        mismatches = []
        if check_residual and "risk_features" in metadata and metadata["risk_features"] != current["risk_features"]:
            raise ValueError("Checkpoint risk feature order differs from this run.")
        if check_residual and bool(metadata.get("coarse_cost_feature", False)) != current["coarse_cost_feature"]:
            raise ValueError("Checkpoint coarse cost-feature semantics differ; use its matching training profile.")
        if check_residual and "residual_mapping" in metadata and metadata["residual_mapping"] != current["residual_mapping"]:
            raise ValueError("Checkpoint residual action mapping differs from this run.")
        for field, checkpoint_value in checks:
            if checkpoint_value is None:
                if allow_legacy:
                    continue
                mismatches.append(
                    f"{field}: checkpoint value missing, current value={current.get(field)!r}"
                )
                continue
            current_value = current.get(field)
            checkpoint_value = self._plain(checkpoint_value)
            current_value = self._plain(current_value)
            if checkpoint_value != current_value:
                mismatches.append(
                    f"{field}: checkpoint value={checkpoint_value!r}, current value={current_value!r}"
                )
        if mismatches:
            raise ValueError("Checkpoint/config mismatch: " + "; ".join(mismatches))

    def _world_model_state(self) -> dict:
        return {
            "model": self.tdmpc_agent.model.state_dict(),
            "tdmpc_optim": self.tdmpc_agent.optim.state_dict(),
            "tdmpc_pi_optim": self.tdmpc_agent.pi_optim.state_dict(),
            "tdmpc_scale": self.tdmpc_agent.scale.state_dict(),
        }

    def _cost_definition(self):
        md = self._plain(section(self.cfg, "metadrive"))
        sim = md.get("simulator", {})
        return {"environment_cost": md.get("cost", {}),
                "risk_field": md.get("risk_field", {}),
                "simulator_cost": {key: value for key, value in sim.items()
                                   if "cost" in key or key.startswith("risk_field_")},
                "prediction_gamma": section(self.cfg, "cost").get("gamma", 0.99)}

    def _check_cost_definition(self, state):
        saved = state.get("metadata", {}).get("cost_definition")
        model = getattr(self, "model", self.tdmpc_agent.model)
        if saved != self._cost_definition() and hasattr(model, "_cost_validated"):
            model.invalidate_cost_validation()
            print("[cost] Cost definition changed: validation revoked. MPPI predictions remain explicitly UNVALIDATED.")

    def _planner_state(self) -> dict:
        return {
            "planner_calls": self._planner_calls,
            "tdmpc_frozen": self._tdmpc_frozen,
            "residual_invalid_count": self.planner.residual_invalid_count,
            "residual_attempt_count": self.planner.residual_attempt_count,
            "mppi_call_count": self.planner.mppi_call_count,
            "baseline_selected_count": self.planner.baseline_selected_count,
            "stage_config": dict(self.planner.mpr_cfg.get("stages", {})),
            "mppi_config": dict(self.planner.mppi_cfg),
            "residual_config": dict(self.residual_cfg),
        }

    def _residual_state(self) -> dict:
        return {
            "residual_sac": self.residual_sac.state_dict(),
            "residual_rl_step": self.residual_rl_step,
            "residual_update_budget": self.residual_update_budget,
            "residual_update_steps": self.residual_sac.update_steps,
        }

    def milestone_state(self, milestone_type: str, *, global_env_step: int | None = None) -> dict:
        step = self.global_env_step if global_env_step is None else int(global_env_step)
        include_residual = milestone_type in {"world_model_residual_rl", "full_mpr_mpc"}
        include_planner = milestone_type == "full_mpr_mpc"
        include_mppi = milestone_type == "full_mpr_mpc"
        milestone_stage = {
            "world_model": "A",
            "world_model_residual_rl": "B",
            "full_mpr_mpc": "C",
        }.get(str(milestone_type), self.planner.stage_for_step(step).name)
        state = {
            "format": "mpr_mpc_milestone_v1",
            "algorithm_version": str(getattr(self.cfg, "algorithm_version", "unknown")),
            "milestone_type": str(milestone_type),
            "global_env_step": step,
            "stage": milestone_stage,
            "components": {
                "world_model": True,
                "residual_rl": include_residual,
                "planner": include_planner,
                "mppi": include_mppi,
            },
            "metadata": {
                "residual_rl_trained": include_residual,
                "mppi_used_during_stage_b": False,
                "checkpoint_note": "Replay buffers are not serialized.",
                **self._checkpoint_metadata(),
            },
            **self._world_model_state(),
        }
        if include_residual:
            state["mpr_mpc"] = {**self._residual_state(), **self._planner_state()}
        elif milestone_type == "world_model":
            state["mpr_mpc"] = {
                "residual_rl_trained": False,
                "planner_calls": self._planner_calls,
            }
        if include_planner:
            state["planner"] = self._planner_state()
        return state

    def save_milestone(
        self,
        filepath,
        *,
        milestone_type: str,
        global_env_step: int | None = None,
    ) -> None:
        torch.save(
            self.milestone_state(milestone_type, global_env_step=global_env_step),
            Path(filepath),
        )

    def load_milestone(
        self,
        filepath,
        *,
        components=None,
        load_optimizer: bool = False,
    ) -> dict:
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        if not isinstance(state, dict) or state.get("format") != "mpr_mpc_milestone_v1":
            raise ValueError("Expected an mpr_mpc_milestone_v1 checkpoint.")
        strict = state.get("milestone_type") == "full_mpr_mpc"
        self._check_checkpoint_compatibility(
            state,
            strict_planner_config=strict and load_optimizer,
            allow_legacy=not load_optimizer,
            check_residual=components is None or "residual_rl" in components,
        )
        available = state.get("components", {})
        requested = (
            [name for name, enabled in available.items() if enabled]
            if components is None
            else list(components)
        )
        missing = [name for name in requested if not bool(available.get(name, False))]
        if missing:
            raise ValueError(
                f"Milestone {filepath} does not contain requested component(s): {missing}."
            )
        if "world_model" in requested:
            self.tdmpc_agent.load({"model": state["model"]})
            self._check_cost_definition(state)
            if load_optimizer:
                self.tdmpc_agent.optim.load_state_dict(state["tdmpc_optim"])
                self.tdmpc_agent.pi_optim.load_state_dict(state["tdmpc_pi_optim"])
                self.tdmpc_agent.scale.load_state_dict(state["tdmpc_scale"])
        mpr_state = state.get("mpr_mpc") or {}
        if "residual_rl" in requested:
            self.residual_sac.load_state_dict(
                mpr_state["residual_sac"],
                load_optimizer=load_optimizer,
            )
            self.residual_rl_step = int(mpr_state.get("residual_rl_step", 0))
            self.residual_update_budget = float(mpr_state.get("residual_update_budget", 0.0))
        if "planner" in requested:
            planner_state = state.get("planner") or mpr_state
            self._load_planner_counters(planner_state)
        self.global_env_step = int(state.get("global_env_step", 0))
        self.last_loaded_checkpoint_metadata = {
            "format": state.get("format"),
            "milestone_type": state.get("milestone_type"),
            "components": dict(state.get("components", {})),
            "metadata": dict(state.get("metadata", {})),
        }
        return state

    def _load_planner_counters(self, mpr_state: dict) -> None:
        self._planner_calls = int(mpr_state.get("planner_calls", 0))
        self._tdmpc_frozen = bool(mpr_state.get("tdmpc_frozen", False))
        if self._tdmpc_frozen:
            self.tdmpc_agent.model.eval()
            for parameter in self.tdmpc_agent.model.parameters():
                parameter.requires_grad_(False)
        self.planner.residual_invalid_count = int(mpr_state.get("residual_invalid_count", 0))
        self.planner.residual_attempt_count = int(mpr_state.get("residual_attempt_count", 0))
        self.planner.mppi_call_count = int(mpr_state.get("mppi_call_count", 0))
        self.planner.baseline_selected_count = int(mpr_state.get("baseline_selected_count", 0))

    def load(self, filepath, *, load_optimizer: bool = False) -> dict:
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and state.get("format") == "mpr_mpc_milestone_v1":
            components = [
                name for name, enabled in state.get("components", {}).items() if enabled
            ]
            loaded = self.load_milestone(
                filepath,
                components=components,
                load_optimizer=load_optimizer,
            )
            return loaded
        if not isinstance(state, dict) or state.get("format") != "mpr_mpc_v2_residual_sac":
            raise ValueError(
                "Refusing to resume an incompatible MPR-MPC checkpoint. Start from scratch with "
                "resume_checkpoint=null or provide an mpr_mpc_v2_residual_sac checkpoint."
            )
        self._check_checkpoint_compatibility(
            state,
            strict_planner_config=False,
            allow_legacy=True,
        )
        self.tdmpc_agent.load(state)
        self._check_cost_definition(state)
        mpr_state = state.get("mpr_mpc") if isinstance(state, dict) else None
        if not mpr_state:
            raise ValueError("MPR-MPC checkpoint is missing planner state.")
        self.residual_sac.load_state_dict(
            mpr_state["residual_sac"], load_optimizer=load_optimizer
        )
        if load_optimizer:
            self.tdmpc_agent.optim.load_state_dict(state["tdmpc_optim"])
            self.tdmpc_agent.pi_optim.load_state_dict(state["tdmpc_pi_optim"])
            self.tdmpc_agent.scale.load_state_dict(state["tdmpc_scale"])
        self.residual_rl_step = int(mpr_state.get("residual_rl_step", 0))
        self.residual_update_budget = float(mpr_state.get("residual_update_budget", 0.0))
        self._planner_calls = int(mpr_state.get("planner_calls", 0))
        self.global_env_step = int(state.get("global_env_step", 0))
        self.best_eval_selection = state.get("best_eval_selection")
        # Old reward-first tuples cannot be compared with the safety-first schema.
        self.best_eval_key = None
        if state.get("best_eval_key") is not None and self.best_eval_selection is None:
            print("[checkpoint] Legacy best key reset; safety-first reevaluation required.")
        self._load_planner_counters(mpr_state)
        self.last_loaded_checkpoint_metadata = {
            "format": state.get("format"),
            "milestone_type": None,
            "components": {},
            "metadata": dict(state.get("metadata", {})),
        }
        return state

    def eval(self):
        self.tdmpc_agent.eval()
        self.residual_sac.eval()
        return self


__all__ = ["MPRMPCAgent"]
