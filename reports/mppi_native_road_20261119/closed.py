"""Actual full receding-horizon driving; record native events and geometry separately."""
import argparse,time
from runtime import *
from mpr_mpc.planning.safety import _road_polygon_parts
from shapely.ops import unary_union
from shapely.geometry import Polygon
sys.path.insert(0,str(OLD))
from metrics import comfort_metrics,path_error
ap=argparse.ArgumentParser();ap.add_argument('--scene',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--method',required=True)
ap.add_argument('--split',choices=['development','holdout','latency'],required=True);ap.add_argument('--limit',type=int,default=0)
ap.add_argument('--output-root',type=Path,help='New destination for reproductions; existing episode directories are never overwritten.')
a=ap.parse_args()
def vstate(v):return np.r_[np.asarray(v.position)[:2],np.linalg.norm(np.asarray(v.velocity)[:2]),v.heading_theta].astype(float)
def footprint(state,v):
    x,y,_,h=state;f=np.array([np.cos(h),np.sin(h)]);n=np.array([-np.sin(h),np.cos(h)])
    return Polygon([np.array([x,y])+sx*v.LENGTH/2*f+sy*v.WIDTH/2*n for sx,sy in [(1,1),(1,-1),(-1,-1),(-1,1)]])

@torch.no_grad()
def main():
    p=load(R/'protocol.json');destination=R if a.output_root is None else a.output_root.resolve()
    out=destination/a.split/f'{a.method}_{a.scene}_r{a.seed}';out.mkdir(parents=True,exist_ok=False)
    parsed,cfg,env,agent,traffic_seed=make_runtime(a.scene,a.seed,a.method)
    from common.seed import set_seed
    from omegaconf import OmegaConf
    tracked=a.method.startswith('T') or a.method=='C';wm=tensor_hash(agent.model);actor=tensor_hash(agent.residual_sac.actor)
    local=[0.];original=agent.planner.local_mppi.plan
    def timed(*args,**kwargs):
        begin=time.perf_counter();result=original(*args,**kwargs);local[0]=1000*(time.perf_counter()-begin);return result
    agent.planner.local_mppi.plan=timed
    try:
        set_seed(a.scene+a.seed);agent.planner.reset(seed=a.scene+a.seed);obs=env.reset(seed=a.scene);sim=env.unwrapped_metadrive;v=sim.vehicle
        assert int(sim.current_seed)==a.scene and sim.config['decision_repeat']*sim.config['physics_world_step_size']==.1
        initial=snapshot(env,obs);dump(out/'initial.json',initial);OmegaConf.save(parsed,out/'config.yaml')
        graph=v.navigation.map.road_network.graph;geometry=[];parts=[];invalid=0
        for start,ends in sorted(graph.items()):
            for end,lanes in sorted(ends.items()):
                for lane in lanes:
                    geometry.append(dict(start=start,end=end,index=lane.index,length=float(lane.length),points=[np.asarray(lane.position(float(t),0)).tolist() for t in np.linspace(0,lane.length,5)]))
                    invalid+=int(not lane.shapely_polygon.is_valid);parts.extend(_road_polygon_parts(lane.shapely_polygon))
        road=unary_union(parts);assert road.is_valid
        dump(out/'map_identity.json',dict(scene=a.scene,traffic_seed=traffic_seed,road_geometry_sha256=digest(geometry),invalid_lane_polygons_repaired=invalid,
            actual_map_seed=int(sim.current_seed),traffic_initial_hash=initial['hashes']['traffic'],spawn_body_on_road=bool(road.covers(footprint(vstate(v),v))),spawn_on_lane=bool(v.on_lane)))
        from mpr_mpc.planning.tracking_dynamics import TrackingVehicle,rollout
        from dataclasses import asdict
        vehicle_model=TrackingVehicle.from_vehicle(v,float(sim.config['physics_world_step_size']))
        resolved_method=load(R/'selection.json')['selected'] if a.method=='C' else a.method
        motion_file='motion_model_corrected.json';motion=load(R/motion_file)
        dump(out/'motion_model_provenance.json',dict(file=motion_file,sha256=file_sha(R/motion_file)))
        dump(out/'vehicle_parameters.json',asdict(vehicle_model))
        states=[vstate(v)];observations=[obs.cpu().numpy().copy()];actions=[];rewards=[];costs=[];dones=[];terms=[];truncs=[];rows=[];done=crash=oor=False
        while not done:
            previous=np.asarray(v.current_action).copy();local[0]=0.;torch.cuda.synchronize();start=time.perf_counter();action=agent.act(obs,t0=not actions,eval_mode=True).cpu().numpy().copy()
            plan=agent.last_plan;base=plan.refined_actions[0].cpu().numpy().copy()
            if not actions:
                c=agent.planner.last_context
                dump(out/'first_policy_fingerprint.json',dict(latent=digest(c.latent),state535=digest(c.residual_state),coarse_actions=digest(c.coarse_actions),
                    refined_actions=digest(plan.refined_actions),normalized_residual=digest(plan.requested_normalized_residual),refined_parameters=digest(plan.refined_parameters)))
            if a.method=='B':
                lp=p['lowpass'];action=base.copy();action[0]=np.clip(lp['alpha']*base[0]+(1-lp['alpha'])*previous[0],base[0]-lp['steering_delta_bound'],base[0]+lp['steering_delta_bound'])
            if tracked:
                assert plan.metrics['mppi/wm_evaluated_candidates']==0
                active=plan.metrics['mppi/native_search_active']
                assert plan.metrics['mppi/iterations_completed']==(4 if active else 0)
                if not agent.planner.tracking_evaluator.cfg.get('optimize_throttle',False):assert action[1]==base[1]
                else:assert abs(float(action[1]-base[1]))<=.250001
                ctx=agent.planner.tracking_evaluator.context(v,plan.refined_path,agent.planner.last_context.safety_context,float(sim.config['physics_world_step_size']),agent.planner.structured.controller)
                expected=ctx.feedback(ctx.state[None])[0]
                assert np.allclose(expected,base,rtol=0,atol=2e-7),'Batched tracker changed original low-level policy'
            torch.cuda.synchronize();planning=1000*(time.perf_counter()-start);path=np.asarray(plan.refined_path.xy).copy()
            prediction=rollout(action[None,None],states[-1],vehicle_model,dt=.1,model=motion['model'],coefficients=motion['coefficients'])[0,1]
            start=time.perf_counter();obs,reward,done,info=env.step(action);env_ms=1000*(time.perf_counter()-start)
            assert np.array_equal(np.asarray(v.current_action,dtype=np.float32),action.astype(np.float32))
            assert abs(float(reward)-float(info['tdmpc2_reward']))<1e-6
            current=vstate(v);lat,track=path_error(current[:2],path)
            native_crash=any(bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk'])
            native_oor=bool(info.get('out_of_road',False));crash|=native_crash;oor|=native_oor
            rows.append(dict(configuration=a.method,scene=a.scene,seed=a.seed,traffic_seed=traffic_seed,step=len(actions)+1,
                steering=float(action[0]),throttle=float(action[1]),sac_steering=float(base[0]),sac_throttle=float(base[1]),
                x=current[0],y=current[1],speed=current[2],heading=current[3],planner_ms=planning,local_mppi_ms=local[0],env_ms=env_ms,control_cycle_ms=planning+env_ms,
                reward=float(info['tdmpc2_reward']),cost=float(info.get('cost',0)),route_completion=float(info['route_completion']),crash=float(native_crash),out_of_road=float(native_oor),
                reference_tracking_error=track,lateral_deviation=lat,on_lane=float(v.on_lane),on_white_continuous_line=float(v.on_white_continuous_line),on_yellow_continuous_line=float(v.on_yellow_continuous_line),crash_sidewalk=float(v.crash_sidewalk),
                prediction_position_error=float(np.linalg.norm(prediction[:2]-current[:2])),prediction_speed_error=float(abs(prediction[2]-current[2])),
                prediction_heading_error=float(abs(np.arctan2(np.sin(prediction[3]-current[3]),np.cos(prediction[3]-current[3])))),
                first_action_deviation=float(np.linalg.norm(action-base)),physical_replacement=bool(np.linalg.norm(action-base)>1e-5),
                **{k:float(val) for k,val in plan.metrics.items() if k.startswith('mppi/') or k.endswith('_ms')}))
            actions.append(action);states.append(current);observations.append(obs.cpu().numpy().copy());rewards.append(float(info['tdmpc2_reward']));costs.append(float(info.get('cost',0)))
            dones.append(bool(done));terms.append(bool(info['terminated']));truncs.append(bool(info['truncated']))
            if a.limit and len(actions)>=a.limit:break
            assert len(actions)<=1000
            if len(actions)%200==0:print('TRACK_PROGRESS',a.split,a.method,a.scene,a.seed,len(actions),round(info['route_completion'],3),flush=True)
        assert wm==tensor_hash(agent.model) and actor==tensor_hash(agent.residual_sac.actor)
        states=np.array(states);actions=np.array(actions)
        metrics,signals=comfort_metrics(actions,states,.1,np.deg2rad(v.max_steering),scales=p['comfort_index']['scales'],braking_threshold=p['hard_braking_threshold_mps2'])
        np.savez_compressed(out/'trace.npz',states=states,observations=observations,actions=actions,rewards=rewards,costs=costs,done=dones,terminated=terms,truncated=truncs)
        table(out/'steps.csv',rows)
        dump(out/'episode.json',dict(configuration=a.method,scene=a.scene,seed=a.seed,traffic_seed=traffic_seed,split=a.split,success=float(info['success']),crash=float(crash),out_of_road=float(oor),route_completion=float(info['route_completion']),timeout=float(info['truncated']),
            episode_return=sum(rewards),discounted_return=discounted(rewards),safety_cost=sum(costs),episode_length=len(actions),native_done=bool(done),end_reason=info.get('end_reason',''),
            weights_unchanged=True,replacement_rate=float(np.mean([x['physical_replacement'] for x in rows])),mean_action_deviation=float(np.mean([x['first_action_deviation'] for x in rows])),reference_tracking_rms=float(np.sqrt(np.mean([x['reference_tracking_error']**2 for x in rows]))),
            final_body_inside_road=bool(road.covers(footprint(states[-1],v))),final_on_lane=bool(v.on_lane),final_white_line=bool(v.on_white_continuous_line),final_sidewalk=bool(v.crash_sidewalk),initial_hashes=initial['hashes'],**metrics))
        print('TRACK_DONE',a.split,a.method,a.scene,a.seed,len(actions),'success',info['success'],'RTG',round(discounted(rewards),3),flush=True)
    finally:env.close()
if __name__=='__main__':main()
