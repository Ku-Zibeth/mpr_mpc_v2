"""Single SAC-initialised Gaussian MPPI; no safety/recovery execution policy."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import warnings
import torch
from .timing import PhaseTimings


MPPI_DIAGNOSTIC_KEYS = (
    'baseline_predicted_return','optimized_predicted_return','selected_predicted_return','predicted_return_gain',
    'predicted_reward','predicted_first_reward','terminal_q','baseline_score','selected_score','final_score','score_gain',
    'baseline_selected','invalid_prediction','iterations_completed','wm_evaluated_candidates',
    'sampling_diversity','sampling_diversity_steering','sampling_diversity_throttle',
    'action_correction_norm','action_correction_steering','action_correction_throttle',
    'first_action_delta_norm','first_steer_delta','first_throttle_delta','sac_to_selected_delta',
    'initial_std_steer','initial_std_throttle','final_std_steer','final_std_throttle',
    'max_delta_steer','max_delta_throttle','planning_mode','mppi_sampling_ms','elite_update_ms',
    'world_model_rollout_ms','final_value_evaluation_ms','planning_time_ms',
) + tuple(f'iteration_{i}_{name}' for i in range(4) for name in ('elite_value_mean','elite_value_max','sampling_diversity'))
MPPI_EPISODE_METRICS = {f'mppi_{k}':f'mppi/{k}' for k in (*MPPI_DIAGNOSTIC_KEYS,'total_planning_ms')}


def _action_vector(value, action_dim, name, device):
    result = torch.as_tensor(value,dtype=torch.float32,device=device)
    if result.ndim == 0:
        result = result.repeat(action_dim)
    if result.shape != (action_dim,) or not torch.isfinite(result).all() or (result<0).any():
        raise ValueError(f'{name} must be finite, nonnegative, scalar or [{action_dim}].')
    return result


@dataclass(frozen=True)
class LocalMPPIResult:
    action: torch.Tensor
    selected_sequence: torch.Tensor
    initial_mean: torch.Tensor
    final_mean: torch.Tensor
    initial_std: torch.Tensor
    final_std: torch.Tensor
    baseline_wm_value: torch.Tensor
    optimized_wm_value: torch.Tensor
    selected_wm_value: torch.Tensor
    wm_value_gain: torch.Tensor
    elite_values: torch.Tensor
    baseline_selected: bool
    max_action_delta: torch.Tensor  # measured correction, NOT a search constraint
    metrics: dict = field(default_factory=dict)
    iterations: list = field(default_factory=list)
    numerical_error: str | None = None
    baseline_consequence: dict = field(default_factory=dict)
    selected_consequence: dict = field(default_factory=dict)
    planning_mode: str = 'baseline'  # output provenance, never a search mode
    baseline_score = property(lambda self:self.baseline_wm_value)
    baseline_value = property(lambda self:self.baseline_wm_value)
    selected_score = property(lambda self:self.selected_wm_value)
    final_value = property(lambda self:self.selected_wm_value)
    planner_score_gain = property(lambda self:self.wm_value_gain)
    planner_gain = property(lambda self:self.wm_value_gain)


class LocalMPPI:
    def __init__(self,evaluator,cfg,*,horizon,action_dim,device):
        self.evaluator,self.device = evaluator,torch.device(device)
        self.horizon,self.action_dim = int(horizon),int(action_dim)
        self.num_samples = int(cfg.get('num_samples',64))  # Gaussian proposals; baseline is additional
        self.num_elites = int(cfg.get('num_elites',8))
        self.iterations = int(cfg.get('iterations',4))
        self.temperature = float(cfg.get('temperature',.5))  # divisor tau, official beta=1/tau
        self.eval_seed = int(cfg.get('eval_seed',0))
        self.initial_std_vector = _action_vector(cfg.get('initial_std',[.04,.12]),action_dim,'initial_std',device)
        self.min_std_vector = _action_vector(cfg.get('min_std',[.01,.03]),action_dim,'min_std',device)
        self.max_std_vector = _action_vector(cfg.get('max_std',[.08,.25]),action_dim,'max_std',device)
        self.noise_rho = _action_vector(cfg.get('noise_rho',[.8,.8]),action_dim,'noise_rho',device)
        if (self.horizon<=0 or self.action_dim<=0 or self.num_samples<=0 or self.iterations<=0
                or not 0<self.num_elites<=self.num_samples+1 or not math.isfinite(self.temperature) or self.temperature<=0
                or (self.noise_rho>=1).any() or (self.min_std_vector<=0).any()
                or (self.min_std_vector>self.initial_std_vector).any() or (self.initial_std_vector>self.max_std_vector).any()):
            raise ValueError('Invalid Gaussian, horizon, elite budget or divisor temperature.')
        self.generator = torch.Generator(device=self.device).manual_seed(self.eval_seed)

    def reset(self,seed=None):
        self.generator.manual_seed(self.eval_seed if seed is None else int(seed)%2**32)

    def initial_mean(self,baseline):
        return baseline.clone()  # deliberately no history, shift or mixed warm start

    def _candidate_batch(self,mean,std,baseline):
        innovations = torch.randn(self.num_samples,self.horizon+1,self.action_dim,
                                  device=self.device,generator=self.generator)
        noise = innovations.clone()
        scale = (1-self.noise_rho.square()).sqrt()
        for t in range(1,self.horizon+1):
            noise[:,t] = self.noise_rho*noise[:,t-1]+scale*innovations[:,t]
        sampled = (mean[None]+std[None]*noise).clamp(-1.,1.)
        return torch.cat((baseline[None],sampled),dim=0)

    def _elite_weights(self,values):
        indices = torch.topk(values,self.num_elites,dim=0).indices
        elite = values[indices]
        return indices,elite,torch.softmax((elite-elite.max())/self.temperature,dim=0)

    def _update_distribution(self,actions,values):
        indices,elite,weights = self._elite_weights(values)
        selected = actions[indices]
        mean = (weights[:,None,None]*selected).sum(0)
        variance = (weights[:,None,None]*(selected-mean[None]).square()).sum(0)
        std = variance.sqrt().clamp(self.min_std_vector,self.max_std_vector)
        return mean,std,elite

    @torch.no_grad()
    def plan(self,z_t,baseline,*,eval_mode,task=None):
        baseline = baseline.to(self.device)
        if (baseline.shape!=(self.horizon+1,self.action_dim) or not torch.isfinite(baseline).all()
                or (baseline.abs()>1).any()):
            raise ValueError('SAC baseline must be finite, in [-1,1], and [H+1,action_dim].')
        mean = self.initial_mean(baseline)
        std = self.initial_std_vector.expand_as(mean).clone()
        initial_mean,initial_std = mean.clone(),std.clone()
        phase,timing,history = PhaseTimings(self.device),{},[]
        evaluated = 0
        error = None
        nan = baseline.new_tensor(float('nan'))
        baseline_value = optimized_value = selected_value = nan
        base = selected = dict(value=nan,reward=nan,terminal_q=nan,first_reward=nan,cost=nan,validated=False)
        elite = baseline.new_empty(0)

        def evaluate(actions):
            nonlocal evaluated
            result = self.evaluator.evaluate_mppi_candidates(z_t,actions,task=task,compute_cost=False)
            evaluated += len(actions)
            for key,value in getattr(result,'timings',{}).items():
                timing[key] = timing.get(key,0.)+value
            values = result.total_value.reshape(-1)
            if values.shape!=(len(actions),):
                raise ValueError('WM must return one value per candidate.')
            return result,values

        def row(result,values,index):
            def item(name):
                value = getattr(result,name,None)
                return nan if value is None else value[index]
            rewards = getattr(result,'reward_sequence',None)
            return dict(value=values[index],reward=item('discounted_reward'),terminal_q=item('terminal_q'),
                        first_reward=nan if rewards is None else rewards[index,0],cost=nan,validated=False)

        for iteration in range(self.iterations):
            with phase.measure('mppi_sampling'):
                candidates = self._candidate_batch(mean,std,baseline)
            result,values = evaluate(candidates)
            baseline_value,base = values[0],row(result,values,0)
            if not bool(torch.isfinite(values).all()):
                error = 'non_finite_candidate_return'
                break
            with phase.measure('elite_update'):
                mean,std,elite = self._update_distribution(candidates,values)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
                error = 'non_finite_distribution'
                break
            diversity = candidates[1:].var(0,unbiased=False).mean(0).sqrt()
            history.append(dict(iteration=iteration,mean=mean.clone(),std=std.clone(),elite=elite.clone(),
                                diversity=diversity,candidate_count=len(candidates),elite_count=len(elite)))
        mean = mean.clamp(-1.,1.)
        sequence = baseline
        if error is None:
            with phase.measure('final_value_evaluation'):
                result,values = evaluate(torch.stack((baseline,mean)))
            baseline_value,optimized_value = values[0],values[1]
            base = row(result,values,0)
            if not bool(torch.isfinite(values).all()):
                error = 'non_finite_final_return'
            elif bool(values[1]>values[0]):
                sequence,selected_value,selected = mean,values[1],row(result,values,1)
        baseline_selected = torch.equal(sequence,baseline)
        if baseline_selected:
            selected_value,selected = baseline_value,base
        if error:
            warnings.warn(f'MPPI WM numerical anomaly ({error}); retaining finite SAC baseline.',RuntimeWarning,stacklevel=2)
        timing.update(phase.finish())
        delta = sequence[0]-baseline[0]
        maximum = (sequence-baseline).abs().amax(0)
        metrics = dict(timing,baseline_predicted_return=float(baseline_value),optimized_predicted_return=float(optimized_value),
            selected_predicted_return=float(selected_value),predicted_return_gain=float(optimized_value-baseline_value),
            baseline_score=float(baseline_value),selected_score=float(selected_value),final_score=float(selected_value),
            score_gain=float(selected_value-baseline_value),predicted_reward=float(selected['reward']),
            predicted_first_reward=float(selected['first_reward']),terminal_q=float(selected['terminal_q']),
            baseline_selected=float(baseline_selected),invalid_prediction=float(error is not None),iterations_completed=float(len(history)),
            wm_evaluated_candidates=float(evaluated),action_correction_norm=float(delta.norm()),
            action_correction_steering=float(delta[0]),action_correction_throttle=float(delta[-1]),
            first_action_delta_norm=float(delta.norm()),first_steer_delta=float(delta[0]),first_throttle_delta=float(delta[-1]),
            sac_to_selected_delta=float(delta.norm()),max_delta_steer=float(maximum[0]),max_delta_throttle=float(maximum[-1]),
            initial_std_steer=float(initial_std[:,0].mean()),initial_std_throttle=float(initial_std[:,-1].mean()),
            final_std_steer=float(std[:,0].mean()),final_std_throttle=float(std[:,-1].mean()),planning_mode=float(not baseline_selected))
        records = []
        for h in history:
            diversity = h['diversity'].cpu()
            i = h['iteration']
            metrics.update({f'iteration_{i}_elite_value_mean':float(h['elite'].mean()),
                            f'iteration_{i}_elite_value_max':float(h['elite'].max()),
                            f'iteration_{i}_sampling_diversity':float(diversity.square().mean().sqrt())})
            records.append(dict(iteration=i,mean=h['mean'].cpu().tolist(),std=h['std'].cpu().tolist(),
                elite_value_mean=float(h['elite'].mean()),elite_value_max=float(h['elite'].max()),
                sampling_diversity=diversity.tolist(),candidate_count=h['candidate_count'],elite_count=h['elite_count']))
        avg = torch.stack([h['diversity'] for h in history]).mean(0) if history else baseline.new_zeros(self.action_dim)
        metrics.update(sampling_diversity=float(avg.square().mean().sqrt()),sampling_diversity_steering=float(avg[0]),sampling_diversity_throttle=float(avg[-1]))
        return LocalMPPIResult(action=sequence[0],selected_sequence=sequence,initial_mean=initial_mean,final_mean=mean,
            initial_std=initial_std,final_std=std,baseline_wm_value=baseline_value,optimized_wm_value=optimized_value,
            selected_wm_value=selected_value,wm_value_gain=selected_value-baseline_value,elite_values=elite,
            baseline_selected=baseline_selected,max_action_delta=maximum,metrics=metrics,iterations=records,numerical_error=error,
            baseline_consequence=base,selected_consequence=selected,planning_mode='baseline' if baseline_selected else 'normal')


__all__ = ['LocalMPPI','LocalMPPIResult','_action_vector','MPPI_DIAGNOSTIC_KEYS','MPPI_EPISODE_METRICS']
