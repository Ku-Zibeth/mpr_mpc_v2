"""Optional alternating online updates for the existing TD-MPC2 replay format."""

from __future__ import annotations

import torch
from tensordict.tensordict import TensorDict

from common.buffer import Buffer


class OnlineWorldModelReplayAdapter:
    def __init__(self, cfg, agent):
        self.cfg = cfg
        self.agent = agent
        self.buffer = Buffer(cfg)
        self.episode = []
        self.ready = False

    @staticmethod
    def _transition(obs, action=None, reward=None, terminated=None, cost=None):
        observation = obs.unsqueeze(0).cpu() if torch.is_tensor(obs) else torch.as_tensor(obs).unsqueeze(0)
        if action is None:
            action = torch.full((2,), float("nan"))
        action = torch.as_tensor(action, dtype=torch.float32).cpu()
        reward = torch.tensor(float("nan") if reward is None else float(reward), dtype=torch.float32)
        terminated = torch.tensor(float("nan") if terminated is None else float(terminated), dtype=torch.float32)
        cost = torch.tensor(float("nan") if cost is None else float(cost), dtype=torch.float32)
        return TensorDict(
            obs=observation,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            cost=cost.unsqueeze(0),
            terminated=terminated.unsqueeze(0),
            batch_size=(1,),
        )

    def start_episode(self, observation) -> None:
        self.episode = [self._transition(observation)]

    def add(self, next_observation, action, reward, terminated, cost, done: bool) -> None:
        if not self.episode:
            raise RuntimeError("start_episode() must be called before adding TD-MPC2 transitions.")
        self.episode.append(self._transition(next_observation, action, reward, terminated, cost))
        if done:
            if len(self.episode) > int(self.cfg.horizon):
                self.buffer.add(torch.cat(self.episode))
                self.ready = True
            self.episode = []

    def update(self) -> dict[str, float]:
        if not self.ready:
            return {}
        metrics = self.agent.update(self.buffer)
        return {f"online_wm/{key}": float(value.detach().cpu()) for key, value in metrics.items()}
