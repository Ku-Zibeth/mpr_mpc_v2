import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion

try:
	from fm_tools.config import cfg_bool, cfg_str
	from tdmpc2_fm.guidance import evaluate_action_sequences
	from tdmpc2_fm.planner import FlowGuidedPlanner
except ImportError:  # pragma: no cover - baseline TD-MPC2 can run without fm_tools on PYTHONPATH
	cfg_bool = None
	cfg_str = None
	evaluate_action_sequences = None
	FlowGuidedPlanner = None


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		model_param_groups = [
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._cost.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		]
		if self.model.cost_q_enabled:
			model_param_groups.append({'params': self.model._cost_Qs.parameters()})
		self.optim = torch.optim.Adam(model_param_groups, lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		planner_type = cfg_str(self.cfg, "planner.type", "mppi") if cfg_str is not None else "mppi"
		flow_enabled = cfg_bool(self.cfg, "flow.enabled", False) if cfg_bool is not None else False
		if self.cfg.compile and not (flow_enabled and planner_type in {"flow", "flow_guided"}):
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		calibration_keys = {"_cost_updates", "_cost_validated", "_cost_validated_horizons",
			"_cost_validated_by_mode", "_cost_validated_by_horizon", "_cost_validation_samples_by_mode",
			"_cost_validation_scenes_by_mode", "_cost_mode_validated_at", "_cost_validation_schema", "_cost_validated_coarse"}
		for key in calibration_keys:
			if key not in state_dict:
				getattr(self.model, key).zero_()
		complete_mode_validation = all(key in state_dict for key in calibration_keys)
		missing_cost_weights = any(key.startswith("_cost.") and key not in state_dict for key in self.model.state_dict())
		cost_q_prefixes = (
			"_cost_Qs.", "_detach_cost_Qs_params.", "_target_cost_Qs_params."
		)
		# TensorDictParams cannot load a wholly missing nested state, even with
		# strict=False. Keep freshly initialized cost-Q parameters when warming
		# from checkpoints that predate this optional head.
		current_state_dict = self.model.state_dict()
		for key, value in current_state_dict.items():
			if key.startswith(cost_q_prefixes) and key not in state_dict:
				state_dict[key] = value
		missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
		if not complete_mode_validation or missing_cost_weights:
			self.model.invalidate_cost_validation()
		if missing_cost_weights:
			self.model._cost_updates.zero_()
		missing_keys = [
			key for key in missing_keys
			if not key.startswith("_cost.") and not key.startswith(cost_q_prefixes)
			and key not in calibration_keys
		]
		# Ignore weights from checkpoints produced before the unused score head was removed.
		unexpected_keys = [
			key for key in unexpected_keys
			if not key.startswith("_score.") and not key.startswith(cost_q_prefixes)
		]
		if missing_keys or unexpected_keys:
			pieces = []
			if missing_keys:
				pieces.append(f"Missing key(s) in state_dict: {missing_keys}")
			if unexpected_keys:
				pieces.append(f"Unexpected key(s) in state_dict: {unexpected_keys}")
			raise RuntimeError("Error(s) in loading state_dict for WorldModel: " + "; ".join(pieces))
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		num_samples = actions.shape[1]
		termination = torch.zeros(num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _estimate_cost(self, z, actions, task):
		"""Estimate discounted learned cost for [H, N, A] action sequences."""
		G, discount = 0, 1
		cost_cfg = getattr(self.cfg, "cost", {})
		if isinstance(cost_cfg, dict) and "gamma" in cost_cfg:
			discount_update = float(cost_cfg["gamma"])
		elif torch.is_tensor(self.discount):
			discount_update = float(self.discount.detach().mean())
		else:
			discount_update = float(self.discount)
		for t in range(actions.shape[0]):
			G = G + discount * self.model.cost(z, actions[t], task)
			z = self.model.next(z, actions[t], task)
			discount = discount * discount_update
		return G

	@torch.no_grad()
	def sample_mppi_candidates(self, obs=None, z=None, t0=False, task=None):
		"""
		Run TD-MPC2 MPPI and return the final candidate set plus unaveraged elites.

		Returned tensors use [N, H, A] for teacher/dataset code. The MPPI weighted
		mean is reported separately and is never used as the only Flow target.
		"""
		if z is None:
			if obs is None:
				raise ValueError("Either obs or z must be provided.")
			z = self.model.encode(obs, task)

		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		z_samples = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		for _ in range(self.cfg.iterations):
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			value = self._estimate_value(z_samples, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		return {
			"candidates": actions.permute(1, 0, 2).contiguous(),
			"values": value.squeeze(1).contiguous(),
			"elite_indices": elite_idxs.contiguous(),
			"elite_actions": elite_actions.permute(1, 0, 2).contiguous(),
			"elite_values": elite_value.squeeze(1).contiguous(),
			"weights": score.squeeze(1).contiguous(),
			"mean": mean.contiguous(),
			"std": std.contiguous(),
		}

	def _flow_plan(self, obs, t0=False, eval_mode=False, task=None):
		if FlowGuidedPlanner is None:
			raise ImportError("fm_tools is not importable; cannot use planner.type=flow/flow_guided")
		planner = getattr(self, "_flow_guided_planner", None)
		if planner is None:
			planner = FlowGuidedPlanner(self, cfg=self.cfg)
			self._flow_guided_planner = planner
		return planner.plan(obs, t0=t0, eval_mode=eval_mode, task=task)

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		planner_type = cfg_str(self.cfg, "planner.type", "mppi") if cfg_str is not None else "mppi"
		flow_enabled = cfg_bool(self.cfg, "flow.enabled", False) if cfg_bool is not None else False
		if flow_enabled and planner_type in {"flow", "flow_guided"}:
			return self._flow_plan(obs, t0=t0, eval_mode=eval_mode, task=task)
		info = self.sample_mppi_candidates(obs=obs, t0=t0, task=task)
		elite_actions = info["elite_actions"].permute(1, 0, 2).contiguous()
		score = info["weights"].unsqueeze(1)
		mean = info["mean"]
		std = info["std"]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = {
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		}
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	@torch.no_grad()
	def _cost_td_target(self, next_z, cost, terminated, task):
		"""Bootstrap discounted safety cost without changing the reward policy."""
		action, _ = self.model.pi(next_z, task)
		cost_cfg = self.cfg.cost if isinstance(getattr(self.cfg, "cost", None), dict) else {}
		discount = float(cost_cfg.get("gamma", self.discount))
		return cost + discount * (1-terminated) * self.model.cost_Q(
			next_z, action, task, return_type='avg', target=True
		)

	def _update(self, obs, action, reward, terminated, task=None, cost=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)
			cost_cfg = self.cfg.cost if isinstance(getattr(self.cfg, "cost", None), dict) else {}
			cost_q_enabled = self.model.cost_q_enabled and cost is not None
			cost_td_targets = (
				self._cost_td_target(next_z, cost, terminated, task)
				if cost_q_enabled else None
			)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		cost_preds = self.model.cost(_zs, action, task)
		cost_qs = self.model.cost_Q(_zs, action, task, return_type='all') if cost_q_enabled else None
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t
		cost_loss = reward_loss.new_zeros(())
		cost_enabled = bool(cost_cfg.get("enabled", False))
		if cost_enabled and cost is not None:
			valid_cost = torch.isfinite(cost)
			if valid_cost.any():
				cost_loss = F.mse_loss(cost_preds[valid_cost], cost[valid_cost])
		cost_q_loss = reward_loss.new_zeros(())
		if cost_q_enabled:
			for t, (cost_q_t, target_t) in enumerate(zip(cost_qs.unbind(1), cost_td_targets.unbind(0))):
				cost_q_loss = cost_q_loss + F.smooth_l1_loss(
					cost_q_t, target_t.unsqueeze(0).expand_as(cost_q_t)
				) * self.cfg.rho**t
			cost_q_loss = cost_q_loss / self.cfg.horizon

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			float(cost_cfg.get("loss_coef", 1.0)) * cost_loss +
			float(cost_cfg.get("q_loss_coef", 1.0)) * cost_q_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)
		# Update model
		total_loss.backward()
		cost_grads = [p.grad.detach().norm() for p in self.model._cost.parameters() if p.grad is not None]
		cost_grad_norm = torch.stack(cost_grads).norm() if cost_grads else reward_loss.new_zeros(())
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)
		# Any representation or cost-weight change invalidates held-out approvals.
		self.model.invalidate_cost_validation()
		cost_step_supervised = bool(cost_enabled and float(cost_cfg.get("loss_coef", 1.0)) > 0 and cost is not None
			and torch.isfinite(cost).any() and cost_grads and torch.isfinite(cost_grad_norm))
		if cost_step_supervised:
			self.model._cost_updates.add_(1)
			if int(self.model._cost_updates) % max(1, int(cost_cfg.get("validation_interval", 1000))) == 0:
				self.model._cost_validated.fill_(False)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = {
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"cost_loss": cost_loss,
			"cost_grad_norm": cost_grad_norm,
			"cost_updates": self.model._cost_updates.float(),
			"cost_supervised_samples": torch.isfinite(cost).sum().float() if cost_step_supervised else 0,
			"cost_step_supervised": float(cost_step_supervised),
			"cost_label_mean": cost[torch.isfinite(cost)].mean() if cost_enabled and cost is not None and torch.isfinite(cost).any() else float("nan"),
			"cost_q_loss": cost_q_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		}
		if cost_q_enabled:
			info.update({
				"cost_q": cost_qs.mean(),
				"cost_q_target": cost_td_targets.mean(),
			})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return {k: v.detach().mean() if isinstance(v, torch.Tensor) else torch.tensor(v) \
			for k, v in info.items()}

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task, cost = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		if cost is not None:
			kwargs["cost"] = cost
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)
