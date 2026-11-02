"""Independent bounded Gaussian updates for nominal, braking and signed turns."""

import torch


MODE_NAMES = ("nominal", "braking", "lateral", "joint")
# Actual component IDs: negative steering is right, positive steering is left
# in the validated MetaDrive XY/heading convention.
MODE_COMPONENT_NAMES = ("nominal", "braking", "right_lateral", "left_lateral", "joint_right", "joint_left")


class MultiModalProposals:
    def __init__(self, cfg, baseline, mean, std, *, num_samples, num_elites,
                 temperature, min_std, max_std, max_delta):
        self.baseline = baseline
        self.num_elites, self.temperature = num_elites, temperature
        self.min_std = min_std
        # Left/right distributions stay separate so their means cannot cancel.
        self.groups = (0, 1, 2, 2, 3, 3)
        counts = cfg.get("mode_counts")
        if counts is None:
            weights = torch.as_tensor(cfg.get("mode_weights", [1, 1, 1, 1]), dtype=torch.float64)
            counts = ((num_samples - 2) * weights / weights.sum()).floor().long().tolist()
            counts[0] += num_samples - 2 - sum(counts)
        else:
            counts = list(counts)
            if len(counts) != 4 or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != num_samples - 2:
                raise ValueError("mode_counts must contain four nonnegative integers summing to num_samples - 2.")
        self.counts = (counts[0], counts[1], counts[2]//2, counts[2]-counts[2]//2,
                       counts[3]//2, counts[3]-counts[3]//2)
        self.means = baseline.expand(6, -1, -1).clone()
        self.means[0] = mean
        self.lower = (baseline - max_delta).expand_as(self.means).clone()
        self.upper = (baseline + max_delta).expand_as(self.means).clone()
        brake_center = float(cfg.get("braking_center", -.6))
        brake_delta = float(cfg.get("braking_delta", .4))
        lateral_delta = float(cfg.get("lateral_delta", .25))
        lateral_shift = float(cfg.get("lateral_shift", .12))
        if not -1 <= brake_center <= 0 or min(brake_delta, lateral_delta, lateral_shift) < 0:
            raise ValueError("Invalid braking/lateral mode center or bounds.")
        for i, group in enumerate(self.groups):
            if group in (1, 3):
                self.means[i, :, 1] = brake_center
                self.lower[i, :, 1] = max(-1., brake_center - brake_delta)
                self.upper[i, :, 1] = min(0., brake_center + brake_delta)
            if group in (2, 3):
                sign = -1 if i in (2, 4) else 1
                self.means[i, :, 0] += sign * lateral_shift
                self.lower[i, :, 0] = baseline[:, 0] - (lateral_delta if sign < 0 else 0)
                self.upper[i, :, 0] = baseline[:, 0] + (lateral_delta if sign > 0 else 0)
        self.lower.clamp_(-1, 1)
        self.upper.clamp_(-1, 1)
        self.means = self._project(self.means)
        self.stds = std.expand_as(self.means).clone()
        configured = dict(cfg.get("mode_std", {}))
        if set(configured) - set(MODE_NAMES):
            raise ValueError("mode_std contains an unknown proposal mode.")
        for i, group in enumerate(self.groups):
            if group in (1, 3):
                self.stds[i, :, 1] = float(cfg.get("braking_std", .15))
            if MODE_NAMES[group] in configured:
                value = baseline.new_tensor(configured[MODE_NAMES[group]])
                if value.shape != (2,) or not torch.isfinite(value).all() or (value <= 0).any():
                    raise ValueError("Each mode_std must contain two finite positive values.")
                self.stds[i] = value
        self.max_std = torch.maximum(self.stds, max_std).clone()
        self.stds = self.stds.clamp_min(min_std)
        self.ids = None
        self.checked = [0] * 4
        self.accepted = [0] * 4

    def _project(self, means):
        return torch.maximum(torch.minimum(means, self.upper), self.lower)

    def sample(self, generator=None):
        actions = [self.baseline[None], self.means[0:1]]
        labels = [torch.zeros(2, dtype=torch.long, device=self.baseline.device)]
        for i, count in enumerate(self.counts):
            noise = torch.randn((count, *self.baseline.shape), device=self.baseline.device, generator=generator)
            candidate = self.means[i] + self.stds[i] * noise
            actions.append(torch.maximum(torch.minimum(candidate, self.upper[i]), self.lower[i]))
            labels.append(torch.full((count,), i, dtype=torch.long, device=self.baseline.device))
        self.ids = torch.cat(labels)
        return torch.cat(actions)

    def update(self, actions, scores, valid, ids):
        for i, group in enumerate(self.groups):
            belongs = ids == i
            eligible = belongs & valid & torch.isfinite(scores)
            self.checked[group] += int(belongs.sum())
            self.accepted[group] += int(eligible.sum())
            indices = torch.where(eligible)[0]
            if not len(indices):
                continue
            top = indices[torch.topk(scores[indices], min(self.num_elites, len(indices))).indices]
            weights = torch.softmax(self.temperature * (scores[top] - scores[top].max()), dim=0)
            mean = (weights[:, None, None] * actions[top]).sum(0)
            self.means[i] = torch.maximum(torch.minimum(mean, self.upper[i]), self.lower[i])
            variance = (weights[:, None, None] * (actions[top] - self.means[i]).square()).sum(0)
            self.stds[i] = torch.maximum(torch.minimum(variance.sqrt(), self.max_std[i]), self.min_std)

    def metrics(self):
        return {key: value for i, name in enumerate(MODE_NAMES) for key, value in (
            (f"mode_{name}_samples", float(self.checked[i])),
            (f"mode_{name}_safe_ratio", self.accepted[i] / self.checked[i] if self.checked[i] else float("nan")),
        )}
