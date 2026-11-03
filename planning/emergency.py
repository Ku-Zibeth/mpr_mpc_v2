"""Independently checked braking, or an explicitly unverified stop command."""

from dataclasses import dataclass

import torch


@dataclass
class EmergencyResult:
    sequence: torch.Tensor
    verified: bool
    safety: object | None
    selected_index: int


def emergency_braking(baseline, context, safety_evaluator, *, control_dt):
    # MetaDrive BaseVehicle._apply_throttle_brake: negative = friction brake
    # ONLY with enable_reverse=false, checked when creating the live context.
    previous = float(context.previous_action[0]) if context is not None else 0.
    max_steer = float(context.max_steering) if context is not None else .6981317
    rate = float(safety_evaluator.cfg.get("max_steering_rate", 2.5)) if safety_evaluator is not None else 2.5
    max_step = .999 * rate * control_dt / max_steer
    targets = baseline.new_tensor([0., previous, -.3, .3, -.15, .15])
    sequences = baseline.new_zeros((len(targets), *baseline.shape))
    steering = baseline.new_full((len(targets),), previous)
    for t in range(baseline.shape[0]):
        steering = steering + (targets-steering).clamp(-max_step, max_step)
        sequences[:, t, 0] = steering.clamp(-1, 1)
        sequences[:, t, 1] = -1.
    if safety_evaluator is None or context is None:
        return EmergencyResult(sequences[0], False, None, 0)
    safety = safety_evaluator.check(sequences, context)
    safe = safety.valid.nonzero().flatten()
    if len(safe):
        index = int(safe[safety.soft_risk[safe].argmin()])
        return EmergencyResult(sequences[index], True, safety, index)
    # Do not execute the least-bad ordinary candidate. This predetermined
    # full-braking/slew-to-neutral command is NOT certified collision-free.
    return EmergencyResult(sequences[0], False, safety, 0)
