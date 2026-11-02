"""Lattice-driven online trainer with staged MPR-MPC activation."""

from __future__ import annotations

from pathlib import Path
from time import time

import numpy as np
import torch

from mpr_mpc._bootstrap import bootstrap
from mpr_mpc.planning.config import mapping, section
from mpr_mpc.training_schedule import tdmpc_update_gate
from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics


bootstrap()
from trainer.online_trainer import OnlineTrainer  # noqa: E402


class MPROnlineTrainer(OnlineTrainer):
    """Collect from Lattice at step zero and perform one smooth update per step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = int(self.agent.global_env_step)
        self._resuming_without_replay = self._step > 0
        if self._resuming_without_replay and not bool(
            self.cfg.get("resume_replay_warmup", True)
        ):
            raise ValueError(
                "Replay buffers are not serialized; resume_replay_warmup must remain true."
            )
        self._replay_steps = 0
        self._best_eval_key = self.agent.best_eval_key
        self._checkpoint_freq = int(self.cfg.get("checkpoint_freq", 50000))
        self._start_time = time()
        stage_training = mapping(section(self.cfg, "tdmpc_stage_training"))
        self._tdmpc_update_ratios = {
            "A": float(stage_training.get("stage_a_update_ratio", 1.0)),
            "B": float(stage_training.get("stage_b_update_ratio", 1.0)),
            "C": float(stage_training.get("stage_c_update_ratio", 0.25)),
        }
        self._tdmpc_update_budget = 0.0
        self._milestone_cfg = mapping(section(self.cfg, "milestone_checkpoints"))
        self._reported_missed_milestones: set[str] = set()
        self._report_missed_historical_milestones()
        simulator = self.cfg.metadrive["simulator"]
        pool = np.arange(int(simulator["start_seed"]), int(simulator["start_seed"]) + int(simulator["num_scenarios"]))
        fixed = list(self.cfg.get("evaluation_scenarios", []))
        count = int(self.cfg.eval_episodes)
        self._eval_scenarios = fixed if fixed else pool[-count:].tolist()
        if len(self._eval_scenarios) != count or len(set(self._eval_scenarios)) != count or not set(self._eval_scenarios) <= set(pool):
            raise ValueError("Training evaluation requires unique in-pool scenario IDs matching eval_episodes.")
        self._training_scenarios = np.setdiff1d(pool, self._eval_scenarios)
        if not len(self._training_scenarios):
            raise ValueError("num_scenarios must exceed eval_episodes to reserve held-out validation scenes.")
        self._scenario_rng = np.random.default_rng(int(self.cfg.seed))
        print(f"[validation] Held-out scenarios (excluded from train replay): {self._eval_scenarios}")

    def _save_checkpoint(self, identifier: str) -> None:
        self.agent.set_global_step(self._step)
        self.logger.save_agent(self.agent, identifier)

    @staticmethod
    def _step_label(step: int) -> str:
        step = int(step)
        if step % 1_000_000 == 0:
            return f"{step // 1_000_000}m"
        if step % 1000 == 0:
            return f"{step // 1000}k"
        return str(step)

    def _milestone_specs(self) -> list[dict]:
        if not bool(self._milestone_cfg.get("enabled", True)):
            return []
        specs = []
        world_step = int(self._milestone_cfg.get("world_model_step", 20000))
        residual_step = int(self._milestone_cfg.get("world_model_residual_step", 50000))
        full_step = int(self._milestone_cfg.get("full_model_step", int(self.cfg.steps)))
        if bool(self._milestone_cfg.get("save_world_model", True)):
            specs.append(
                {
                    "key": "world_model_milestone_saved",
                    "step": world_step,
                    "type": "world_model",
                    "filename": f"milestone_{self._step_label(world_step)}_world_model.pt",
                    "message": "Stage-A World Model",
                }
            )
        if bool(self._milestone_cfg.get("save_world_model_residual", True)):
            specs.append(
                {
                    "key": "world_model_residual_milestone_saved",
                    "step": residual_step,
                    "type": "world_model_residual_rl",
                    "filename": (
                        f"milestone_{self._step_label(residual_step)}_"
                        "world_model_residual_rl.pt"
                    ),
                    "message": "Stage-B World Model + Residual SAC",
                }
            )
        if bool(self._milestone_cfg.get("save_full_model", True)):
            specs.append(
                {
                    "key": "full_model_milestone_saved",
                    "step": full_step,
                    "type": "full_mpr_mpc",
                    "filename": f"milestone_{self._step_label(full_step)}_full_mpr_mpc.pt",
                    "message": "Stage-C Full MPR-MPC",
                }
            )
        return specs

    def _report_missed_historical_milestones(self) -> None:
        if self._step <= 0:
            return
        for spec in self._milestone_specs():
            if self._step > int(spec["step"]):
                path = Path(self.logger.model_dir) / str(spec["filename"])
                if not path.exists() and spec["key"] not in self._reported_missed_milestones:
                    print(
                        "[checkpoint] Historical milestone missed; "
                        f"not saving {path.name} from resumed step {self._step:,}."
                    )
                    self._reported_missed_milestones.add(spec["key"])

    def _maybe_save_milestones(self, previous_step: int, current_step: int) -> dict[str, float]:
        metrics = {}
        for spec in self._milestone_specs():
            milestone_step = int(spec["step"])
            if not (int(previous_step) < milestone_step <= int(current_step)):
                continue
            path = Path(self.logger.model_dir) / str(spec["filename"])
            metric = f"checkpoint/{spec['key']}"
            metrics["checkpoint/milestone_step"] = float(milestone_step)
            if path.exists():
                metrics[metric] = 0.0
                continue
            self.agent.set_global_step(milestone_step)
            self.agent.save_milestone(
                path,
                milestone_type=str(spec["type"]),
                global_env_step=milestone_step,
            )
            self.agent.set_global_step(current_step)
            print(
                f"[checkpoint] Saved {spec['message']} milestone at {milestone_step:,}: {path}"
            )
            metrics[metric] = 1.0
        return metrics

    def _tdmpc_update_gate(self, stage_name: str, replay_ready: bool) -> tuple[bool, float, int]:
        enabled, self._tdmpc_update_budget, ratio, updates = tdmpc_update_gate(
            stage_name=stage_name,
            replay_ready=replay_ready,
            budget=self._tdmpc_update_budget,
            ratios=self._tdmpc_update_ratios,
        )
        return enabled, ratio, updates

    def _handle_eval(self) -> dict[str, float]:
        from .checkpoint_selection import select_checkpoint
        metrics = self.eval()
        metrics.update(self.common_metrics())
        self.logger.log(metrics, "eval")
        improved, selection, reason = select_checkpoint(metrics,
            getattr(self.agent, "best_eval_selection", None), getattr(self.cfg, "checkpoint_selection", {}) or {})
        print(f"[checkpoint] best selection: {reason}")
        if improved:
            self.agent.best_eval_selection = selection
            self._save_checkpoint("best")
        self._save_checkpoint("latest")
        return metrics

    @torch.no_grad()
    def eval(self):
        rewards, costs, successes, lengths = [], [], [], []
        collisions, offroads, route_completions = [], [], []
        base_rewards, risk_costs, normalized_risks, risk_penalties = [], [], [], []
        validation_episodes, collision_success, collision_offroad, ttc_values = [], [], [], []
        plan_metrics = {}
        episode_diagnostics = []
        for episode in range(int(self.cfg.eval_episodes)):
            observation = self.env.reset(seed=int(self._eval_scenarios[episode]))
            diagnostics = EpisodeDiagnostics()
            episode_collision = episode_offroad = False
            done = False
            episode_reward = episode_cost = episode_base_reward = 0.0
            episode_risk_cost = episode_normalized_risk = episode_risk_penalty = 0.0
            step, info = 0, {}
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=episode == 0)
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(
                    observation,
                    t0=step == 0,
                    eval_mode=True,
                    global_step=self._step,
                )
                previous_observation = observation
                observation, reward, done, info = self.env.step(action)
                diagnostics.observe(previous_observation, action, info, self.agent.last_plan, self.env.unwrapped_metadrive.agent)
                episode_collision |= any(bool(info.get(k, False)) for k in
                    ("crash", "crash_vehicle", "crash_object", "crash_human", "crash_building", "crash_sidewalk"))
                episode_offroad |= bool(info.get("out_of_road", False))
                ttc = float(info.get("risk_field_min_ttc", float("nan")))
                if np.isfinite(ttc):
                    ttc_values.append(ttc)
                if self.agent.last_plan is not None:
                    for key, value in self.agent.last_plan.metrics.items():
                        if np.isfinite(value):
                            plan_metrics.setdefault(key, []).append(value)
                episode_reward += float(reward)
                episode_cost += float(info.get("cost", 0.0))
                episode_base_reward += float(info.get("metadrive_reward", 0.0))
                episode_risk_cost += float(info.get("risk_field_cost", 0.0))
                episode_normalized_risk += float(info.get("risk_field_normalized_cost", 0.0))
                episode_risk_penalty += float(info.get("risk_field_reward_penalty", 0.0))
                step += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            rewards.append(episode_reward)
            costs.append(episode_cost)
            successes.append(float(info.get("success", 0.0)))
            lengths.append(step)
            collisions.append(float(episode_collision))
            offroads.append(float(episode_offroad))
            collision_success.append(float(episode_collision and bool(info.get("success", False))))
            collision_offroad.append(float(episode_collision and episode_offroad))
            validation_episodes.append(diagnostics.trajectory())
            episode_diagnostics.append(diagnostics.summary(info, self.agent))
            route_completions.append(float(info.get("route_completion", 0.0)))
            base_rewards.append(episode_base_reward)
            risk_costs.append(episode_risk_cost)
            normalized_risks.append(episode_normalized_risk)
            risk_penalties.append(episode_risk_penalty)
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        mean = lambda values: float(np.nanmean(values))
        diagnostics_mean = {}
        for key in episode_diagnostics[0] if episode_diagnostics else []:
            finite = [row[key] for row in episode_diagnostics if np.isfinite(row[key])]
            diagnostics_mean[key] = mean(finite) if finite else float("nan")
        clearances = [row["min_obstacle_clearance"] for row in episode_diagnostics if np.isfinite(row["min_obstacle_clearance"])]
        diagnostics_mean["min_obstacle_clearance"] = min(clearances) if clearances else float("nan")
        return {
            **diagnostics_mean,
            **self.agent.validate_cost_model(validation_episodes),
            **{key: mean(values) for key, values in plan_metrics.items()},
            "collision_and_success_rate": mean(collision_success),
            "collision_and_offroad_rate": mean(collision_offroad),
            "mean_ttc": mean(ttc_values) if ttc_values else float("nan"),
            "min_ttc": min(ttc_values) if ttc_values else float("nan"),
            "episode_reward": mean(rewards),
            "episode_cost": mean(costs),
            "average_safety_cost": mean(costs),
            "success_rate": mean(successes),
            "out_of_road_rate": mean(offroads),
            "collision_rate": mean(collisions),
            "success": mean(successes),
            "offroad": mean(offroads),
            "collision": mean(collisions),
            "route_completion": mean(route_completions),
            "episode_length": mean(lengths),
            "episode_metadrive_reward": mean(base_rewards),
            "episode_risk_field_cost": mean(risk_costs),
            "episode_normalized_risk": mean(normalized_risks),
            "episode_risk_penalty": mean(risk_penalties),
        }

    def train(self):
        train_metrics, done, info = {}, True, {}
        eval_next = self._step == 0
        eval_freq = max(1, int(self.cfg.eval_freq))
        next_eval = ((self._step // eval_freq) + 1) * eval_freq
        next_checkpoint = ((self._step // self._checkpoint_freq) + 1) * self._checkpoint_freq
        minimum_replay_steps = int(self.cfg.batch_size) * (int(self.cfg.horizon) + 1)

        while self._step < int(self.cfg.steps):
            if done:
                if eval_next:
                    self._handle_eval()
                    eval_next = False
                if self._step > 0:
                    if info["terminated"] and not self.cfg.episodic:
                        raise ValueError("Termination detected but episodic=false.")
                    episode = torch.cat(self._tds)
                    self._ep_idx = self.buffer.add(episode)
                    self._replay_steps += max(0, len(episode) - 1)
                    replay_ready = self._replay_steps >= minimum_replay_steps
                    train_metrics.update(
                        episode_reward=torch.tensor([td["reward"] for td in self._tds[1:]]).sum(),
                        episode_cost=torch.tensor([td["cost"] for td in self._tds[1:]]).sum(),
                        episode_success=info["success"],
                        episode_length=len(self._tds),
                        episode_terminated=info["terminated"],
                        Reason=info.get("end_reason", "unknown"),
                        replay_steps=float(self._replay_steps),
                        replay_ready=float(replay_ready),
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, "train")
                observation = self.env.reset(seed=int(self._scenario_rng.choice(self._training_scenarios)))
                self._tds = [self.to_td(observation)]

            # No random seed policy: Stage A is Lattice-controlled from the first step.
            self.agent.set_global_step(self._step)
            action = self.agent.act(
                observation,
                t0=len(self._tds) == 1,
                global_step=self._step,
            )
            observation, reward, done, info = self.env.step(action)
            env_cost = float(info.get("cost", info.get("risk_field_cost", 0.0)))
            cost = torch.tensor(env_cost, dtype=torch.float32)
            if hasattr(self.agent, "observe_transition"):
                self.agent.observe_transition(float(reward), env_cost, bool(done), info)
            self._tds.append(self.to_td(observation, action, reward, info["terminated"], cost))

            # TD-MPC2 replay readiness and Residual SAC replay readiness are independent.
            replay_ready = self._replay_steps >= minimum_replay_steps
            stage = self.agent.planner.stage_for_step(self._step)
            tdmpc_update_enabled, tdmpc_ratio, tdmpc_updates = self._tdmpc_update_gate(
                stage.name,
                replay_ready,
            )
            train_metrics["replay_steps"] = float(self._replay_steps)
            train_metrics["replay_ready"] = float(replay_ready)
            update_metrics = self.agent.update(
                self.buffer,
                tdmpc_update_enabled=tdmpc_update_enabled,
                residual_update_enabled=True,
            )
            train_metrics.update(update_metrics)
            train_metrics.update(
                {
                    "train/stage": float(ord(stage.name) - ord("A")),
                    "train/tdmpc_update_ratio": float(tdmpc_ratio),
                    "train/tdmpc_updates": float(tdmpc_updates),
                    "train/residual_updates": float(
                        torch.as_tensor(
                            update_metrics.get("residual_rl/updates", 0.0),
                            device="cpu",
                        )
                    ),
                    "train/residual_active": float(stage.use_residual),
                    "train/mppi_active": float(self.agent.last_plan.stage.use_mppi),
                }
            )
            previous_step = self._step
            self._step += 1
            self.agent.set_global_step(self._step)
            train_metrics.update(self._maybe_save_milestones(previous_step, self._step))
            if self._step >= next_eval:
                eval_next = True
                while next_eval <= self._step:
                    next_eval += eval_freq
            if self._step >= next_checkpoint:
                self._save_checkpoint("latest")
                while next_checkpoint <= self._step:
                    next_checkpoint += self._checkpoint_freq

        self.agent.set_global_step(self._step)
        final_milestones = self._maybe_save_milestones(self._step - 1, self._step)
        if final_milestones:
            self.logger.log(final_milestones, "train")
        # The loop exits before the next episode boundary; do not leave best.pt
        # comparing only initialization when eval_freq equals the full 50K run.
        self._handle_eval()
        self.logger.finish(self.agent)


__all__ = ["MPROnlineTrainer"]
