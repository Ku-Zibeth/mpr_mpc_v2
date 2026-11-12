"""Summarise bounded replay evidence; never turn an abort into a passing test."""
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parent
raw=json.loads((ROOT/'fixed_states_checked.json').read_text())
cases=raw['cases'];rows=[r for c in cases for r in c.get('records',[])]
metrics={}
for name in ('mpr/total_planning_ms','mppi/preview_ms','mppi/safety_check_ms','mppi/world_model_rollout_ms','mppi/candidate_evaluation_ms','mppi/mppi_sampling_ms'):
    values=[r['metrics'][name] for r in rows if name in r['metrics'] and np.isfinite(r['metrics'][name])]
    metrics[name]={f'p{q}':float(np.percentile(values,q)) for q in (50,95)} if values else {}
    if values:metrics[name].update(mean=float(np.mean(values)),max=float(np.max(values)))
summary=dict(requested_scenarios=13,recorded_scenarios=len(cases),planning_attempts=len(rows),
    physical_steps=sum(c.get('physical_steps',0) for c in cases),
    unverified_action_executions=sum(c.get('unverified_action_executions',0) for c in cases),
    safety_failure_attempts=sum(r['safety_failure'] for r in rows),
    new_physical_collision_count=sum(r['executed'] and r['crash'] for r in rows),
    new_physical_offroad_count=sum(r['executed'] and r['offroad'] for r in rows),
    decision_states=dict(Counter(r['decision']['decision_state_name'] for r in rows)),
    certification_reasons=dict(Counter(r['decision']['baseline_certification_reason'] for r in rows)),
    timings=metrics,fixed_state_acceptance_passed=raw['fixed_state_acceptance_passed'],formal_100_run=False,
    scope='Zero newly executed steps is NOT successful recovery, a safe driving rate, or proof of accident prevention. Cold fixed states include resets/model loading effects and do not isolate a latency improvement against the old full100 experiment.',
    cases=[dict(scenario=c['scenario'],before_step=c.get('before_step'),provenance=c.get('state_provenance'),
        prior_replay_unavailable=bool(c.get('historical_replay_unavailable')),physical_steps=c.get('physical_steps',0),
        replay_max_speed_error=c.get('replay_max_speed_error'),recorded_route_gain=c.get('route_gain'),
        route_gain=c.get('route_gain') if c.get('physical_steps',0)>0 else 0.) for c in cases])
(ROOT/'fixed_state_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2))
