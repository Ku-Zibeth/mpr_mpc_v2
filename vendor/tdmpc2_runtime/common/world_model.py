from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._cost = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1)
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout).apply(init.weight_init) for _ in range(cfg.num_q)])
		cost_cfg = cfg.cost if isinstance(getattr(cfg, "cost", None), dict) else {}
		self.cost_q_enabled = bool(cost_cfg.get("q_enabled", False))
		self.num_cost_q = int(cost_cfg.get("num_q", 2))
		if self.cost_q_enabled:
			if self.num_cost_q < 2:
				raise ValueError("cost.num_q must be at least 2 when terminal cost-Q is enabled.")
			self._cost_Qs = layers.Ensemble([
				layers.mlp(
					cfg.latent_dim + cfg.action_dim + cfg.task_dim,
					2*[cfg.mlp_dim],
					1,
					dropout=cfg.dropout,
				).apply(init.weight_init)
				for _ in range(self.num_cost_q)
			])
		else:
			self._cost_Qs = None
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.register_buffer("_cost_updates", torch.tensor(0, dtype=torch.long))
		self.register_buffer("_cost_validated", torch.tensor(False))
		self.register_buffer("_cost_validated_horizons", torch.tensor(0, dtype=torch.long))
		self.register_buffer("_cost_validated_at", torch.tensor(0, dtype=torch.long))
		self.register_buffer("_cost_validation_schema", torch.tensor(0, dtype=torch.long))
		self.init()

	@property
	def cost_ready(self):
		return (int(self._cost_updates.item()) > 0
			and int(self._cost_validation_schema.item()) == 3
			and int(self._cost_validated_at.item()) == int(self._cost_updates.item())
			and bool(self._cost_validated))

	def invalidate_cost_validation(self):
		self._cost_validated.zero_()
		self._cost_validated_horizons.zero_()
		self._cost_validated_at.zero_()
		self._cost_validation_schema.zero_()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)
		if self.cost_q_enabled:
			self._detach_cost_Qs_params = TensorDictParams(self._cost_Qs.params.data, no_convert=True)
			self._target_cost_Qs_params = TensorDictParams(self._cost_Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)
		if self.cost_q_enabled:
			with self._detach_cost_Qs_params.data.to("meta").to_module(self._cost_Qs.module):
				self._detach_cost_Qs = deepcopy(self._cost_Qs)
				self._target_cost_Qs = deepcopy(self._cost_Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params
		if self.cost_q_enabled:
			delattr(self._detach_cost_Qs, "params")
			self._detach_cost_Qs.__dict__["params"] = self._detach_cost_Qs_params
			delattr(self._target_cost_Qs, "params")
			self._target_cost_Qs.__dict__["params"] = self._target_cost_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Cost', 'Termination', 'Policy prior', 'Q-functions', 'Cost Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._cost, self._termination, self._pi, self._Qs, self._cost_Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			if m is None:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		if self.cost_q_enabled:
			self._target_cost_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)
		if self.cost_q_enabled:
			self._target_cost_Qs_params.lerp_(self._detach_cost_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def cost(self, z, a, task):
		"""
		Predict instantaneous nonnegative safety cost.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return F.softplus(self._cost(z))

	def cost_Q(self, z, a, task, return_type='avg', target=False, detach=False):
		"""Predict nonnegative discounted cost-to-go for a state-action pair."""
		if not self.cost_q_enabled:
			raise RuntimeError("Terminal cost-Q is disabled; set cost.q_enabled=true.")
		if return_type not in {'min', 'avg', 'all'}:
			raise ValueError(f"Unknown cost-Q return_type: {return_type}")
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_cost_Qs
		elif detach:
			qnet = self._detach_cost_Qs
		else:
			qnet = self._cost_Qs
		out = F.softplus(qnet(z))
		if return_type == 'all':
			return out
		if return_type == 'min':
			return out.min(0).values
		return out.mean(0)
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
