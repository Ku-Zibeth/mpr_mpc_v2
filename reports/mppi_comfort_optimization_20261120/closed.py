"""True 0.1s rolling closed-loop A/B/C, native termination; optional fixed dev smoke."""
import argparse,time
from runtime import *
from metrics import comfort_metrics,path_error
ap=argparse.ArgumentParser();ap.add_argument('--group',required=True,choices=['A','B','C']);ap.add_argument('--scene',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--timing',action='store_true');a=ap.parse_args()
def vehicle_state(v):return np.array([*np.asarray(v.position)[:2],float(np.linalg.norm(np.asarray(v.velocity)[:2])),float(v.heading_theta)],dtype=float)

@torch.no_grad()
def main():
    p=load(R/'protocol.json');out=R/('latency_episodes' if a.timing else 'development' if a.smoke else 'episodes')/f'{a.group}_{a.scene}_r{a.seed}'
    out.mkdir(parents=True,exist_ok=False);parsed,cfg,env,agent=make_runtime(a.scene,a.seed,a.group)
    from common.seed import set_seed
    from omegaconf import OmegaConf
    wm=tensor_hash(agent.model);actor=tensor_hash(agent.residual_sac.actor);local_time=[0.];original=agent.planner.local_mppi.plan
    def measured(*args,**kwargs):
        torch.cuda.synchronize();begin=time.perf_counter();result=original(*args,**kwargs);torch.cuda.synchronize();local_time[0]=1000*(time.perf_counter()-begin);return result
    agent.planner.local_mppi.plan=measured
    try:
        combined=a.scene+a.seed;set_seed(combined);agent.planner.reset(seed=combined);obs=env.reset(seed=a.scene)
        sim=env.unwrapped_metadrive;v=sim.vehicle
        assert int(sim.current_seed)==a.scene and sim.config['decision_repeat']*sim.config['physics_world_step_size']==.1
        initial=snapshot(env,obs);dump(out/'initial.json',initial);OmegaConf.save(parsed,out/'config.yaml')
        # A map identity comes from actual road geometry, not renamed scenario labels.
        graph=v.navigation.map.road_network.graph;geometry=[]
        for start,ends in sorted(graph.items()):
            for end,lanes in sorted(ends.items()):
                for lane in lanes:geometry.append(dict(start=start,end=end,index=lane.index,length=float(lane.length),points=[np.asarray(lane.position(float(t),0)).tolist() for t in np.linspace(0,lane.length,5)]))
        dump(out/'map_identity.json',dict(scene=a.scene,current_seed=int(sim.current_seed),road_geometry_sha256=digest(geometry)))
        states=[vehicle_state(v)];observations=[obs.cpu().numpy().copy()];actions=[];rewards=[];costs=[];doneflags=[];term=[];trunc=[];rows=[]
        done=crash=oor=False;lane_changes=lane_reversals=road_transitions=0;prior_lane=tuple(v.lane_index);last_lane_move=0
        max_steer=np.deg2rad(v.max_steering)
        while not done:
            previous=np.asarray(v.current_action,dtype=float).copy();local_time[0]=0.
            torch.cuda.synchronize();begin=time.perf_counter();selected=agent.act(obs,t0=not actions,eval_mode=True).cpu().numpy().copy()
            baseline=agent.last_plan.refined_actions[0].cpu().numpy().copy()
            if a.group=='B':
                alpha=p['lowpass']['alpha'];bound=p['lowpass']['steering_delta_bound']
                selected=baseline.copy();selected[0]=np.clip(alpha*baseline[0]+(1-alpha)*previous[0],baseline[0]-bound,baseline[0]+bound)
            if a.group=='C':
                assert selected[1]==baseline[1]
                assert agent.last_plan.metrics['mppi/wm_evaluated_candidates']==0
                assert agent.last_plan.metrics['mppi/iterations_completed']==4
                assert np.array_equal(agent.planner.comfort_evaluator._observed_action,previous)
            torch.cuda.synchronize();planner_ms=1000*(time.perf_counter()-begin)
            path=np.asarray(agent.last_plan.refined_path.xy).copy();old_state=states[-1];begin_env=time.perf_counter()
            obs,reward,done,info=env.step(selected);env_ms=1000*(time.perf_counter()-begin_env)
            assert np.array_equal(np.asarray(v.current_action,dtype=np.float32),selected.astype(np.float32))
            assert abs(float(reward)-float(info['tdmpc2_reward']))<1e-6
            current=vehicle_state(v);lateral,error=path_error(current[:2],path)
            current_lane=tuple(v.lane_index)
            if current_lane[:2]==prior_lane[:2] and current_lane[2]!=prior_lane[2]:
                lane_changes+=1;move=int(np.sign(current_lane[2]-prior_lane[2]));lane_reversals+=int(last_lane_move!=0 and move!=last_lane_move);last_lane_move=move
            elif current_lane[:2]!=prior_lane[:2]:road_transitions+=1;last_lane_move=0
            prior_lane=current_lane
            angle_method=getattr(v.system,'getSteeringValue',None)
            wheel=float(np.deg2rad(angle_method(0))) if callable(angle_method) else float(v.steering)*max_steer
            rows.append(dict(configuration=a.group,scene=a.scene,seed=a.seed,step=len(actions)+1,planner_ms=planner_ms,local_mppi_ms=local_time[0],env_ms=env_ms,control_cycle_ms=planner_ms+env_ms,
                steering=float(selected[0]),throttle=float(selected[1]),sac_steering=float(baseline[0]),sac_throttle=float(baseline[1]),front_wheel_angle_rad=wheel,wheel_angle_source='Bullet getSteeringValue' if callable(angle_method) else 'live steering mapping',
                x=current[0],y=current[1],speed=current[2],heading=current[3],reference_tracking_error=error,lateral_deviation=lateral,lane_index=json.dumps(current_lane),
                lattice_target_d=float(agent.last_plan.coarse_path.target_d),refined_target_d=float(agent.last_plan.refined_path.target_d),target_speed=float(agent.last_plan.refined_path.target_speed),
                reward=float(reward),cost=float(info.get('cost',0)),route_completion=float(info['route_completion']),
                physical_replacement=bool(np.linalg.norm(selected-baseline)>1e-5),first_action_deviation=float(np.linalg.norm(selected-baseline)),
                **{k:float(value) for k,value in agent.last_plan.metrics.items() if k.startswith('mppi/comfort_') or k.endswith('_ms') or k in ['mppi/baseline_selected','mppi/wm_evaluated_candidates']}))
            actions.append(selected);observations.append(obs.cpu().numpy().copy());states.append(current);rewards.append(float(info['tdmpc2_reward']));costs.append(float(info.get('cost',0)))
            doneflags.append(bool(done));term.append(bool(info['terminated']));trunc.append(bool(info['truncated']))
            crash|=any(bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk']);oor|=bool(info.get('out_of_road',False))
            if a.smoke and len(actions)>=a.smoke:break
            assert len(actions)<=1000
            if len(actions)%200==0:print('COMFORT_PROGRESS',a.group,a.scene,a.seed,len(actions),round(info['route_completion'],3),flush=True)
        assert wm==tensor_hash(agent.model) and actor==tensor_hash(agent.residual_sac.actor)
        states=np.array(states);actions=np.array(actions);scales=p['comfort_index']['scales']
        metrics,signal=comfort_metrics(actions,states,.1,max_steer,scales=scales,braking_threshold=p['hard_braking_threshold_mps2'])
        metrics.update(reference_tracking_rms=float(np.sqrt(np.mean([x['reference_tracking_error']**2 for x in rows]))),
            lateral_deviation_rms=float(np.sqrt(np.mean([x['lateral_deviation']**2 for x in rows]))),
            lateral_oscillation_count=int(np.count_nonzero(np.diff(np.sign(np.diff([x['lateral_deviation'] for x in rows]))))),
            lane_change_count=lane_changes,lane_change_reversal=lane_reversals,road_segment_transitions=road_transitions)
        np.savez_compressed(out/'trace.npz',states=states,actions=actions,observations=observations,rewards=rewards,costs=costs,done=doneflags,terminated=term,truncated=trunc)
        np.savez_compressed(out/'comfort_signals.npz',**{f'{label}_{key}':value for label,data in signal.items() for key,value in data.items()})
        table(out/'steps.csv',rows)
        dump(out/'episode.json',dict(configuration=a.group,scene=a.scene,seed=a.seed,native_done=bool(done),smoke_limit=a.smoke,success=float(info['success']),crash=float(crash),out_of_road=float(oor),route_completion=float(info['route_completion']),timeout=float(info['truncated']),
            episode_return=sum(rewards),discounted_return=discounted(rewards),episode_length=len(actions),safety_cost=sum(costs),end_reason=info.get('end_reason',''),
            replacement_rate=float(np.mean([x['physical_replacement'] for x in rows])),mean_action_deviation=float(np.mean([x['first_action_deviation'] for x in rows])),weights_unchanged=True,initial_hashes=initial['hashes'],**metrics))
        print('COMFORT_DONE',a.group,a.scene,a.seed,len(actions),'native',done,'success',info['success'],'CI',metrics['comfort_index'],flush=True)
    finally:env.close()
if __name__=='__main__':main()
