"""Every physical tick replans Lattice/SAC/MPPI and executes exactly its first action."""
import argparse,sys,time
from pathlib import Path
_data_root=Path(__file__).resolve().parents[2]
_group=sys.argv[sys.argv.index('--group')+1] if '--group' in sys.argv else 'B'
# Freeze B/C to clean main; D alone loads the opt-in corrected worktree.
_code_root=_data_root if _group=='D' else _data_root.parent/'mpr_mpc_v2'
sys.path.insert(0,str(_code_root));sys.path.insert(1,str(Path(__file__).parent))
import numpy as np
import torch
from omegaconf import OmegaConf
from study_io import R,ROOT,load,dump,table,make_runtime
from instrumentation import SamplerAudit
from tools.first_action_study import snapshot,tensor_hash
from planning.time_alignment import align_frenet_path


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['B','C','D'],required=True)
    ap.add_argument('--scene',type=int,required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    p=load(R/'protocol.json');assert args.scene in p['scenarios'] and args.seed in p['repeat_seeds']
    name=f'{args.group}_{args.scene}_r{args.seed}';out=R/'episodes'/name;out.mkdir(parents=True,exist_ok=False)
    parsed,cfg,env,agent=make_runtime(args.scene,args.seed,args.group,record=True)
    from common.seed import set_seed
    initial_model=tensor_hash(agent.model);initial_sac=tensor_hash(agent.residual_sac.actor)
    try:
        combined=args.scene+args.seed;set_seed(combined);agent.planner.reset(seed=combined)
        obs=env.reset(seed=args.scene);assert env.unwrapped_metadrive.current_seed==args.scene
        initial=snapshot(env,obs);dump(f'episodes/{name}/initial.json',initial)
        OmegaConf.save(parsed,out/'config.yaml')
        sim=env.unwrapped_metadrive
        assert agent.planner.control_dt==sim.config['decision_repeat']*sim.config['physics_world_step_size']
        planner=agent.planner;sampler=SamplerAudit(planner.local_mppi) if args.group!='B' else None
        local_result=[None];phases={};active=[False];counts=dict(prepare=0,mppi=0)
        original_plan=planner.local_mppi.plan;original_prepare=planner.prepare
        def capture_plan(*a,**kw):
            counts['mppi']+=1;result=original_plan(*a,**kw);local_result[0]=result;return result
        def prepare(*a,**kw):counts['prepare']+=1;return original_prepare(*a,**kw)
        planner.local_mppi.plan=capture_plan;planner.prepare=prepare
        def measured(obj,method,key):
            old=getattr(obj,method)
            def fn(*a,**kw):
                begin=time.perf_counter();result=old(*a,**kw)
                if active[0]:phases[key]=phases.get(key,0.)+1000*(time.perf_counter()-begin)
                return result
            setattr(obj,method,fn)
        measured(planner.structured,'select_coarse','lattice_selection_wall_ms')
        measured(planner.action_adapter,'path_to_actions','trajectory_adapter_wall_ms')
        observations=[obs.cpu().numpy().copy()];actions=[];rewards=[];costs=[];dones=[];terminated=[];truncated=[]
        physical=[];selected_sequences=[];virtual_positions=[];virtual_speeds=[];virtual_headings=[]
        act_ms=[];diag_ms=[];step_rows=[];coverage=[];alignment=[];done=crash=oor=False
        while not done:
            step=len(actions)+1;vehicle=sim.agent
            before=dict(position=np.asarray(vehicle.position[:2]).copy(),speed=float(vehicle.speed),heading=float(vehicle.heading_theta))
            if sampler:sampler.reset()
            phases.clear();active[0]=True;torch.cuda.synchronize();begin=time.perf_counter()
            action=agent.act(obs,t0=step==1,eval_mode=True)
            torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-begin);active[0]=False
            result=agent.last_plan;ctx=planner.last_context
            assert result.stage.use_residual and result.stage.use_mppi==(args.group!='B')
            assert counts['prepare']==step and counts['mppi']==(step if args.group!='B' else 0)
            sequence=result.debug['selected_mppi_actions'];assert torch.equal(action.cpu(),sequence[0].cpu())
            diagstart=time.perf_counter();mp=local_result[0]
            predicted_first=None;predicted_prefix=None;predicted_q=None;predicted_total=None
            if sampler:
                trace=mp.candidate_trace
                assert len(trace['actions'])==4 and not mp.numerical_error
                values=[v.cpu().numpy() for v in trace['values']]
                metadata=dict(configuration=args.group,scene=args.scene,repeat_seed=args.seed,step=step)
                coverage.extend(sampler.rows(metadata,values,trace['q_members']))
                predicted_first=float(mp.selected_consequence['first_reward']);predicted_prefix=float(mp.selected_consequence['reward'])
                predicted_q=float(mp.selected_consequence['terminal_q']);predicted_total=float(mp.selected_wm_value)
            aligned=align_frenet_path(result.refined_path,required_steps=11,control_dt=planner.control_dt,
                **({'cartesian_speed':True} if args.group=='D' else {}))
            selection=ctx.selection
            modes={(round(float(path.target_d),6),round(float(path.target_speed),6),round(float(path.horizon),6)) for path in selection.feasible}
            row=dict(configuration=args.group,scene=args.scene,repeat_seed=args.seed,step=step,act_wall_ms=elapsed,
                predicted_first_reward=predicted_first,predicted_prefix_reward=predicted_prefix,predicted_terminal_q=predicted_q,
                predicted_total_return=predicted_total,baseline_selected=float(result.metrics['mpr/baseline_selected']),
                first_action_delta=float(np.linalg.norm(action.cpu().numpy()-result.refined_actions[0].cpu().numpy())),
                lattice_candidates=len(selection.candidates),lattice_feasible=len(selection.feasible),feasible_parameter_modes=len(modes),
                coarse_target_d=float(result.coarse_path.target_d),refined_target_d=float(result.refined_path.target_d),
                coarse_target_speed=float(result.coarse_path.target_speed),refined_target_speed=float(result.refined_path.target_speed),
                path_end_time=float(result.refined_path.t[-1]),clipped_future_times=int((aligned.times>result.refined_path.t[-1]).sum()),
                **phases,**{k:float(v) for k,v in result.metrics.items() if 'ms' in k})
            if sampler:
                pg=float(mp.selected_consequence['reward']-mp.baseline_consequence['reward'])
                row.update(selected_predicted_gain=float(mp.wm_value_gain),selected_prefix_gain=pg,
                           selected_discounted_terminal_gain=float(mp.wm_value_gain)-pg)
            if step in p['detailed_snapshot_steps']:
                snap=dict(observation=obs.cpu().numpy(),position=before['position'],speed=before['speed'],heading=before['heading'],
                    coarse_xy=result.coarse_path.xy,coarse_t=result.coarse_path.t,coarse_speed=result.coarse_path.s_d,
                    refined_xy=result.refined_path.xy,refined_t=result.refined_path.t,refined_speed=result.refined_path.s_d,
                    coarse_actions=result.coarse_actions.cpu().numpy(),refined_actions=result.refined_actions.cpu().numpy(),
                    selected_actions=sequence.cpu().numpy(),aligned_xy=aligned.positions,aligned_heading=aligned.headings,aligned_speed=aligned.speeds)
                for i,path in enumerate(selection.candidates):
                    snap[f'lattice_{i}_xy']=path.xy;snap[f'lattice_{i}_parameters']=np.array([path.target_d,path.target_speed,path.horizon])
                if sampler:
                    snap['gaussians']=torch.stack(mp.candidate_trace['actions']).cpu().numpy()
                    snap['gaussian_values']=torch.stack(mp.candidate_trace['values']).cpu().numpy()
                    snap['weighted_mean']=mp.final_mean.cpu().numpy()
                np.savez_compressed(out/f'snapshot_{step}.npz',**snap)
                dump(f'episodes/{name}/snapshot_{step}_origin.json',snapshot(env,obs))
            diagnostic=1000*(time.perf_counter()-diagstart)
            obs,reward,done,info=env.step(action)
            assert abs(float(reward)-float(info['tdmpc2_reward']))<1e-4
            after=sim.agent
            actual_control=np.array([float(after.steering),float(after.throttle_brake)])
            actual_position=np.asarray(after.position[:2]).copy();actual_speed=float(after.speed);actual_heading=float(after.heading_theta)
            tracker_next,_=planner.structured.controller._track_path(after,result.refined_path)
            alignment.append(dict(configuration=args.group,scene=args.scene,repeat_seed=args.seed,step=step,
                expected_control_dt=planner.control_dt,actual_control_dt=sim.config['decision_repeat']*sim.config['physics_world_step_size'],
                first_action_submission_equal=True,normalized_control_execution_error=float(np.max(np.abs(action.cpu().numpy()-actual_control))),
                virtual_next_position_error=float(np.linalg.norm(aligned.positions[1]-actual_position)),
                virtual_next_speed_error=float(aligned.speeds[1]-actual_speed),
                virtual_next_heading_error=float((aligned.headings[1]-actual_heading+np.pi)%(2*np.pi)-np.pi),
                old_path_next_control_delta=float(np.linalg.norm(np.asarray(tracker_next)-result.refined_actions[1].cpu().numpy())),
                clipped_future_times=row['clipped_future_times'],speed_units='m/s actual and virtual',
                virtual_speed_mode='Cartesian XY/time derivative' if args.group=='D' else 'legacy Frenet s_d',
                note='Ideal path virtual state vs real controlled state; an error alone does not prove an index bug.'))
            row.update(actual_first_reward=float(info['tdmpc2_reward']),diagnostic_post_act_ms=diagnostic,
                actual_crash=bool(info.get('crash',False)),actual_oor=bool(info.get('out_of_road',False)),route_completion=float(info['route_completion']))
            step_rows.append(row);act_ms.append(elapsed);diag_ms.append(diagnostic)
            observations.append(obs.cpu().numpy().copy());actions.append(action.cpu().numpy().copy());rewards.append(float(info['tdmpc2_reward']))
            costs.append(float(info['cost']));dones.append(bool(done));terminated.append(bool(info['terminated']));truncated.append(bool(info['truncated']))
            physical.append([*actual_position,actual_speed,actual_heading]);selected_sequences.append(sequence.cpu().numpy().copy())
            virtual_positions.append(aligned.positions);virtual_speeds.append(aligned.speeds);virtual_headings.append(aligned.headings)
            crash|=any(bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk'])
            oor|=bool(info.get('out_of_road',False))
            assert step<=1000
            if step%100==0:print(name,'step',step,'route',round(info['route_completion'],3),'act_ms',round(elapsed,1),flush=True)
        rtg=np.zeros(len(rewards)+1)
        for i in range(len(rewards)-1,-1,-1):rtg[i]=rewards[i]+p['gamma']*rtg[i+1]
        for i,row in enumerate(step_rows):row['actual_replanning_policy_rtg']=float(rtg[i])
        np.savez_compressed(out/'trace.npz',observations=observations,actions=actions,rewards=rewards,costs=costs,done=dones,
            terminated=terminated,truncated=truncated,rtg=rtg,physical_after=physical,selected_sequences=selected_sequences,
            virtual_positions=virtual_positions,virtual_speeds=virtual_speeds,virtual_headings=virtual_headings,act_wall_ms=act_ms,diagnostic_ms=diag_ms)
        table(f'episodes/{name}/steps.csv',step_rows);table(f'episodes/{name}/coverage.csv',coverage);table(f'episodes/{name}/alignment.csv',alignment)
        assert tensor_hash(agent.model)==initial_model and tensor_hash(agent.residual_sac.actor)==initial_sac
        row=dict(configuration=args.group,scene=args.scene,repeat_seed=args.seed,success=float(info['success']),crash=float(crash),out_of_road=float(oor),
            route_completion=float(info['route_completion']),timeout=float(info.get('end_reason')=='timeout' or bool(info['truncated'])),
            episode_return=sum(rewards),discounted_return=float(rtg[0]),episode_length=len(rewards),safety_cost=sum(costs),
            planning_p50_ms=float(np.quantile(act_ms,.5)),planning_p95_ms=float(np.quantile(act_ms,.95)),planning_max_ms=max(act_ms),
            baseline_selected_rate=float(np.mean([r['baseline_selected'] for r in step_rows])),first_action_delta=float(np.mean([r['first_action_delta'] for r in step_rows])),
            end_reason=info.get('end_reason',''),native_done=True,weights_unchanged=True,counts=counts,
            initial_hashes=initial['hashes'],latency_scope='two-worker shared GPU synchronized full agent.act; observational recording included; post-act diagnostics separate')
        row['runtime_code_root']=str(_code_root)
        dump(f'episodes/{name}/episode.json',row);print('DONE',name,row['discounted_return'],row['success'],row['crash'],row['out_of_road'],flush=True)
    finally:env.close()


if __name__=='__main__':main()
