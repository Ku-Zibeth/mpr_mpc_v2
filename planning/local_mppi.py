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
    'weight_entropy','effective_sample_size','prefix_reward_gain','terminal_q_gain','discounted_terminal_gain',
    'selected_prefix_reward_gain','selected_discounted_terminal_gain','q_ensemble_std','fixed_q_pair',
    'q_pair_0','q_pair_1','active_action_steps','terminal_q_weight','weighted_terminal_q_gain',
    'uses_sac_value','effective_cost_weight','cost_penalty','cost_penalty_gain','audit_terminal_candidates',
    'mean_best_sequence_l2','mean_best_first_action_l2','mean_best_predicted_gap','elite_pairwise_l2',
    'selection_is_elite','selected_source_code','final_elite_value','final_elite_gain',
) + tuple(f'iteration_{i}_{name}' for i in range(4) for name in ('elite_value_mean','elite_value_max','sampling_diversity','weight_entropy','effective_sample_size'))
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
    evaluation: dict = field(default_factory=dict)
    candidate_audit: dict = field(default_factory=dict)
    selection_candidates: dict = field(default_factory=dict)
    candidate_trace: dict = field(default_factory=dict)
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
        self.temperature = float(cfg.get('temperature',.5))
        self.temperature_definition = str(cfg.get('temperature_definition','inverse'))
        self.terminal_action = str(cfg.get('terminal_action','candidate'))
        self.q_aggregation = str(cfg.get('q_aggregation','random_pair'))
        self.terminal_q_weight = float(cfg.get('terminal_q_weight',1.0))
        self.terminal_value = str(cfg.get('terminal_value','tdmpc2_q'))
        self.audit_candidates = bool(cfg.get('audit_candidates',False))
        self.record_cost = bool(cfg.get('record_cost',False))
        self.cost_weight = float(cfg.get('cost_weight',0.0))
        self.selection_mode = str(cfg.get('selection_mode','weighted_mean'))
        self.compare_final_elite = 'selection_mode' in cfg  # exact legacy behavior when absent
        self.record_candidates = bool(cfg.get('record_candidates',False))
        if self.selection_mode not in {'weighted_mean','best_final_elite'}:
            raise ValueError('Unknown final MPPI selection_mode.')
        if self.compare_final_elite and self.audit_candidates:
            raise ValueError('Use record_candidates with explicit selection_mode; legacy audit has a separate Q rescore.')
        if self.terminal_value not in {'tdmpc2_q','sac_value'} or not math.isfinite(self.cost_weight) or self.cost_weight < 0:
            raise ValueError('Invalid MPPI terminal value or continuous cost weight.')
        if self.terminal_value == 'sac_value' and self.terminal_q_weight != 1.:
            raise ValueError('V_SAC uses unit terminal weight.')
        if not math.isfinite(self.terminal_q_weight) or self.terminal_q_weight < 0:
            raise ValueError('terminal_q_weight must be finite and nonnegative.')
        if (self.temperature_definition not in {'inverse','divisor'} or self.terminal_action not in {'candidate','policy_prior'}
                or self.q_aggregation not in {'random_pair','fixed_pair','ensemble_mean'}):
            raise ValueError('Unknown MPPI temperature, terminal action or Q definition.')
        self.eval_seed = int(cfg.get('eval_seed',0))
        self.initial_std_vector = _action_vector(cfg.get('initial_std',[.04,.12]),action_dim,'initial_std',device)
        self.min_std_vector = _action_vector(cfg.get('min_std',[.01,.03]),action_dim,'min_std',device)
        self.max_std_vector = _action_vector(cfg.get('max_std',[.08,.25]),action_dim,'max_std',device)
        self.noise_rho = _action_vector(cfg.get('noise_rho',[.8,.8]),action_dim,'noise_rho',device)
        if (self.horizon<=0 or self.action_dim<=0 or self.num_samples<=0 or self.iterations<=0
                or not 0<self.num_elites<=self.num_samples+1 or not math.isfinite(self.temperature) or self.temperature<=0
                or (self.noise_rho>=1).any() or (self.min_std_vector<=0).any()
                or (self.min_std_vector>self.initial_std_vector).any() or (self.initial_std_vector>self.max_std_vector).any()):
            raise ValueError('Invalid Gaussian, horizon, elite budget or temperature.')
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
        if self.terminal_action == 'policy_prior' or self.terminal_value == 'sac_value':
            sampled[:,-1] = baseline[-1]  # Coordinator padding, never optimised
        return torch.cat((baseline[None],sampled),dim=0)

    def _weights(self,elite):
        beta = self.temperature if self.temperature_definition == 'inverse' else 1/self.temperature
        return torch.softmax(beta*(elite-elite.max()),dim=0)

    def _elite_weights(self,values):
        indices = torch.topk(values,self.num_elites,dim=0).indices
        elite = values[indices]
        return indices,elite,self._weights(elite)

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
        active = self.horizon if self.terminal_action == 'policy_prior' or self.terminal_value == 'sac_value' else self.horizon+1
        if active == self.horizon:
            std[-1] = 0.
        initial_mean,initial_std = mean.clone(),std.clone()
        q_pair = self.evaluator.new_mppi_q_pair() if self.q_aggregation == 'fixed_pair' and self.terminal_value == 'tdmpc2_q' else None
        options = {}
        if self.terminal_action != 'candidate' or self.q_aggregation != 'random_pair':
            options = dict(terminal_action=self.terminal_action,q_aggregation=self.q_aggregation,q_indices=q_pair)
        if self.terminal_q_weight != 1.0:
            options['terminal_q_weight'] = self.terminal_q_weight
        if self.terminal_value != 'tdmpc2_q':
            options['terminal_value'] = self.terminal_value
        if self.cost_weight:
            options['cost_weight'] = self.cost_weight
        cost_cfg = getattr(getattr(self.evaluator,'agent',None),'cfg',None)
        cost_available = bool(getattr(cost_cfg,'cost',{}).get('enabled',False)) if cost_cfg is not None else False
        compute_cost = cost_available and (self.record_cost or self.cost_weight > 0)
        definition = dict(temperature_definition=self.temperature_definition,temperature=self.temperature,
                          terminal_action=self.terminal_action,q_aggregation=self.q_aggregation,q_indices=q_pair,
                          terminal_q_weight=self.terminal_q_weight,terminal_value=self.terminal_value,cost_weight=self.cost_weight,
                          selection_mode=self.selection_mode,uniform_final_comparison=self.compare_final_elite)
        phase,timing,history = PhaseTimings(self.device),{},[]
        evaluated = 0
        error = None
        nan = baseline.new_tensor(float('nan'))
        baseline_value = optimized_value = selected_value = nan
        base = selected = dict(value=nan,reward=nan,terminal_q=nan,first_reward=nan,cost=nan,validated=False,q_std=nan,penalty=nan.new_zeros(()))
        optimized = base
        elite = baseline.new_empty(0)
        audit, cached, sampled, original_scores, original_elites = {}, [], [], [], []
        trace_actions, trace_values, trace_members = [], [], []
        selection_candidates = {}
        final_elite_value = nan
        selected_source = 'baseline'
        last_result = last_candidates = last_best = None

        def evaluate(actions, *, final=False):
            nonlocal evaluated
            kwargs = dict(options)
            if self.record_candidates or (final and self.compare_final_elite):
                kwargs['capture_q_indices'] = True
            result = self.evaluator.evaluate_mppi_candidates(z_t,actions,task=task,compute_cost=compute_cost,**kwargs)
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
            penalties = getattr(result,'cost_penalty',None)
            return dict(value=values[index],reward=item('discounted_reward'),terminal_q=item('terminal_q'),
                        first_reward=nan if rewards is None else rewards[index,0],cost=item('cumulative_cost'),
                        validated=bool(getattr(result,'cost_ranking_validated',False)),
                        penalty=nan.new_zeros(()) if penalties is None else penalties[index],
                        q_std=item('terminal_q_std'))

        for iteration in range(self.iterations):
            with phase.measure('mppi_sampling'):
                candidates = self._candidate_batch(mean,std,baseline)
            result,values = evaluate(candidates)
            baseline_value,base = values[0],row(result,values,0)
            if not bool(torch.isfinite(values).all()):
                error = 'non_finite_candidate_return'
                break
            last_result, last_candidates, last_best = result, candidates, int(values.argmax())
            if self.record_candidates:
                trace_actions.append(candidates.clone())
                trace_values.append(values.clone())
                trace_members.append(getattr(result,'q_member_indices',None))
            if self.audit_candidates:
                cached.append((result,slice(1,None)))
                sampled.append(candidates[1:].clone())
                original_scores.append(values[1:].clone())
                original_elites.append(candidates[self._elite_weights(values)[0]].clone())
            with phase.measure('elite_update'):
                mean,std,elite = self._update_distribution(candidates,values)
            if active == self.horizon:
                mean[-1],std[-1] = baseline[-1],0.
            if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
                error = 'non_finite_distribution'
                break
            diversity = candidates[1:,:active].var(0,unbiased=False).mean(0).sqrt()
            weights = self._weights(elite)
            entropy = -(weights*weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum()
            ess = weights.square().sum().reciprocal()
            history.append(dict(iteration=iteration,mean=mean.clone(),std=std.clone(),elite=elite.clone(),
                                diversity=diversity,candidate_count=len(candidates),elite_count=len(elite),
                                entropy=entropy,ess=ess,best_actions=candidates[values.argmax()].clone()))
        mean = mean.clamp(-1.,1.)
        sequence = baseline
        if error is None:
            with phase.measure('final_value_evaluation'):
                result,values = evaluate(torch.stack((baseline,mean)), final=True)
            if self.audit_candidates:
                with phase.measure('audit_terminal_rescore'):
                    # All old samples, baseline and mean share ONE reference
                    # terminal batch; reward/dynamics/cost rollouts are cached.
                    result = self.evaluator.rescore_cached_candidates(
                        [(result,slice(None)),*cached], task=task, terminal_value=self.terminal_value,
                        terminal_action=self.terminal_action,q_aggregation=self.q_aggregation,q_indices=q_pair,
                        terminal_q_weight=self.terminal_q_weight,cost_weight=self.cost_weight)
                    pool = torch.cat((torch.stack((baseline,mean)),*sampled),0)
                    common_values = result.total_value.reshape(-1)
                    if bool(torch.isfinite(common_values).all()):
                        best = 2+int(common_values[2:].argmax())
                        last = len(pool)-self.num_samples+int(common_values[-self.num_samples:].argmax())
                        pairwise = torch.pdist(torch.cat(original_elites)[:,:active].flatten(1))
                        audit = dict(actions=pool,values=common_values,indices=dict(baseline=0,mean=1,best_elite=best,final_elite=last),
                            iteration_actions=sampled,iteration_values=original_scores,original_elites=original_elites,
                            terminal_latents=result.terminal_latent,discounted_rewards=result.discounted_reward,
                            terminal_discounts=result.terminal_discount,costs=result.cumulative_cost,
                            elite_pairwise_l2=float(pairwise.mean()) if len(pairwise) else 0.)
                    else:
                        error = 'non_finite_audit_return'
                    values = common_values[:2]
            baseline_value,optimized_value = values[0],values[1]
            base,optimized = row(result,values,0),row(result,values,1)
            candidate_sequence, candidate_value, candidate_row = mean,values[1],optimized
            if self.compare_final_elite:
                members = getattr(result,'q_member_indices',None)
                reducer = 'fixed_pair' if self.q_aggregation == 'random_pair' and self.terminal_value == 'tdmpc2_q' else self.q_aggregation
                if reducer == 'fixed_pair' and members is None:
                    raise RuntimeError('Final selection requires the exact shared Q member indices.')
                with phase.measure('final_elite_evaluation'):
                    elite_result = self.evaluator.rescore_cached_candidates(
                        [(last_result,[last_best])],task=task,terminal_value=self.terminal_value,
                        terminal_action=self.terminal_action,q_aggregation=reducer,
                        q_indices=members if reducer == 'fixed_pair' else q_pair,
                        terminal_q_weight=self.terminal_q_weight,cost_weight=self.cost_weight)
                final_elite_value = elite_result.total_value.reshape(-1)[0]
                elite_sequence = last_candidates[last_best]
                selection_candidates = dict(mode=self.selection_mode,
                    actions=torch.stack((baseline,mean,elite_sequence)),
                    values=torch.stack((values[0],values[1],final_elite_value)),
                    q_member_indices=members,elite_original_index=last_best,
                    elite_original_value=float(last_result.total_value[last_best]),
                    source='last_iteration_top1; cached reward/dynamics; same final Q members')
                if self.selection_mode == 'best_final_elite':
                    candidate_sequence,candidate_value=elite_sequence,final_elite_value
                    candidate_row=row(elite_result,elite_result.total_value.reshape(-1),0)
            if error is not None or not bool(torch.isfinite(values).all()):
                error = error or 'non_finite_final_return'
            elif not bool(torch.isfinite(candidate_value)):
                error = 'non_finite_final_elite'
            elif bool(candidate_value>values[0]):
                sequence,selected_value,selected = candidate_sequence,candidate_value,candidate_row
                selected_source = 'elite' if self.selection_mode == 'best_final_elite' else 'mean'
        baseline_selected = torch.equal(sequence,baseline)
        if baseline_selected:
            selected_value,selected = baseline_value,base
            selected_source = 'baseline'
        if selection_candidates:
            selection_candidates['selected_source'] = selected_source
        if error:
            warnings.warn(f'MPPI WM numerical anomaly ({error}); retaining finite SAC baseline.',RuntimeWarning,stacklevel=2)
        timing.update(phase.finish())
        delta = sequence[0]-baseline[0]
        maximum = (sequence-baseline).abs().amax(0)
        metrics = dict(timing,baseline_predicted_return=float(baseline_value),optimized_predicted_return=float(optimized_value),
            selected_predicted_return=float(selected_value),predicted_return_gain=float(optimized_value-baseline_value),
            baseline_score=float(baseline_value),selected_score=float(selected_value),final_score=float(selected_value),
            score_gain=float(selected_value-baseline_value),predicted_reward=float(selected['reward']),
            prefix_reward_gain=float(optimized['reward']-base['reward']),
            terminal_q_gain=float(optimized['terminal_q']-base['terminal_q']),
            discounted_terminal_gain=float((optimized_value-optimized['reward']+optimized['penalty'])-(baseline_value-base['reward']+base['penalty'])),
            weighted_terminal_q_gain=float((optimized_value-optimized['reward']+optimized['penalty'])-(baseline_value-base['reward']+base['penalty'])),
            cost_penalty_gain=float(optimized['penalty']-base['penalty']),
            terminal_q_weight=self.terminal_q_weight,
            selected_prefix_reward_gain=float(selected['reward']-base['reward']),
            selected_discounted_terminal_gain=float((selected_value-selected['reward']+selected['penalty'])-(baseline_value-base['reward']+base['penalty'])),
            q_ensemble_std=float(selected['q_std']),fixed_q_pair=float(q_pair is not None),
            q_pair_0=float(q_pair[0]) if q_pair else float('nan'),q_pair_1=float(q_pair[1]) if q_pair else float('nan'),
            uses_sac_value=float(self.terminal_value == 'sac_value'),
            selection_is_elite=float(self.selection_mode == 'best_final_elite'),
            selected_source_code=float({'baseline':0,'mean':1,'elite':2}[selected_source]),
            final_elite_value=float(final_elite_value),final_elite_gain=float(final_elite_value-baseline_value),
            effective_cost_weight=float(getattr(result,'effective_cost_weight',0.)),
            cost_penalty=float(result.cost_penalty[0 if baseline_selected else 1]) if getattr(result,'cost_penalty',None) is not None else 0.,
            audit_terminal_candidates=float(len(audit.get('values',[]))),
            active_action_steps=float(active),
            predicted_first_reward=float(selected['first_reward']),terminal_q=float(selected['terminal_q']),
            baseline_selected=float(baseline_selected),invalid_prediction=float(error is not None),iterations_completed=float(len(history)),
            wm_evaluated_candidates=float(evaluated),action_correction_norm=float(delta.norm()),
            action_correction_steering=float(delta[0]),action_correction_throttle=float(delta[-1]),
            first_action_delta_norm=float(delta.norm()),first_steer_delta=float(delta[0]),first_throttle_delta=float(delta[-1]),
            sac_to_selected_delta=float(delta.norm()),max_delta_steer=float(maximum[0]),max_delta_throttle=float(maximum[-1]),
            initial_std_steer=float(initial_std[:active,0].mean()),initial_std_throttle=float(initial_std[:active,-1].mean()),
            final_std_steer=float(std[:active,0].mean()),final_std_throttle=float(std[:active,-1].mean()),planning_mode=float(not baseline_selected))
        records = []
        if audit:
            best = audit['indices']['best_elite']
            metrics.update(mean_best_sequence_l2=float((mean[:active]-audit['actions'][best,:active]).norm()),
                mean_best_first_action_l2=float((mean[0]-audit['actions'][best,0]).norm()),
                mean_best_predicted_gap=float(audit['values'][1]-audit['values'][best]),elite_pairwise_l2=audit['elite_pairwise_l2'])
        for h in history:
            diversity = h['diversity'].cpu()
            i = h['iteration']
            metrics.update({f'iteration_{i}_elite_value_mean':float(h['elite'].mean()),
                            f'iteration_{i}_elite_value_max':float(h['elite'].max()),
                            f'iteration_{i}_sampling_diversity':float(diversity.square().mean().sqrt()),
                            f'iteration_{i}_weight_entropy':float(h['entropy']),
                            f'iteration_{i}_effective_sample_size':float(h['ess'])})
            records.append(dict(iteration=i,mean=h['mean'].cpu().tolist(),std=h['std'].cpu().tolist(),
                elite_value_mean=float(h['elite'].mean()),elite_value_max=float(h['elite'].max()),
                sampling_diversity=diversity.tolist(),candidate_count=h['candidate_count'],elite_count=h['elite_count'],
                weight_entropy=float(h['entropy']),effective_sample_size=float(h['ess']),best_elite_actions=h['best_actions'].cpu().tolist()))
        avg = torch.stack([h['diversity'] for h in history]).mean(0) if history else baseline.new_zeros(self.action_dim)
        metrics.update(weight_entropy=float(torch.stack([h['entropy'] for h in history]).mean()) if history else float('nan'),
                       effective_sample_size=float(torch.stack([h['ess'] for h in history]).mean()) if history else float('nan'),
                       sampling_diversity=float(avg.square().mean().sqrt()),sampling_diversity_steering=float(avg[0]),sampling_diversity_throttle=float(avg[-1]))
        trace = dict(actions=trace_actions,values=trace_values,q_members=trace_members) if self.record_candidates else {}
        return LocalMPPIResult(action=sequence[0],selected_sequence=sequence,initial_mean=initial_mean,final_mean=mean,
            initial_std=initial_std,final_std=std,baseline_wm_value=baseline_value,optimized_wm_value=optimized_value,
            selected_wm_value=selected_value,wm_value_gain=selected_value-baseline_value,elite_values=elite,
            baseline_selected=baseline_selected,max_action_delta=maximum,metrics=metrics,iterations=records,numerical_error=error,evaluation=definition,
            baseline_consequence=base,selected_consequence=selected,candidate_audit=audit,
            selection_candidates=selection_candidates,candidate_trace=trace,
            planning_mode='baseline' if baseline_selected else 'normal')


__all__ = ['LocalMPPI','LocalMPPIResult','_action_vector','MPPI_DIAGNOSTIC_KEYS','MPPI_EPISODE_METRICS']
