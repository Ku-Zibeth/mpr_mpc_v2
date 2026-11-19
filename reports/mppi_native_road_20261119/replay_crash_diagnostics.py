"""Read-only native replay of existing C trajectories, not new model selection."""
from runtime import *
import pandas as pd
@torch.no_grad()
def main():
    from common.seed import set_seed
    rows=[];directory=R/'crash_replays';directory.mkdir(exist_ok=False)
    for row in pd.read_csv(R/'introduced_crash_cases.csv').itertuples(index=False):
        scene,seed=int(row.scene),int(row.seed);source=R/'holdout'/f'C_{scene}_r{seed}';trace=np.load(source/'trace.npz');steps=pd.read_csv(source/'steps.csv')
        _,_,env,agent,_=make_runtime(scene,seed,'C')
        try:
            set_seed(scene+seed);agent.planner.reset(seed=scene+seed);obs=env.reset(seed=scene);first=False
            for i,action in enumerate(trace['actions']):
                assert np.array_equal(obs.cpu().numpy(),trace['observations'][i]),(scene,seed,i,'fixed-replay obs')
                obs,_,done,info=env.step(action);assert np.array_equal(obs.cpu().numpy(),trace['observations'][i+1])
                assert abs(float(info['tdmpc2_reward'])-float(trace['rewards'][i]))<1e-8
                collision=any(bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk'])
                if collision and not first:
                    first=True;v=env.unwrapped_metadrive.vehicle;old=steps.iloc[i]
                    flags={k:bool(info.get(k,False)) for k in ['crash','crash_vehicle','crash_object','crash_human','crash_building','crash_sidewalk']}
                    rows.append(dict(scene=scene,seed=seed,step=i+1,**flags,contact_types=sorted(v.contact_results),search_active=float(old['mppi/native_search_active']),selected_overlap_cost=float(old['mppi/tracking_obstacle_cost']),selected_line_risk=float(old['mppi/native_selected_risk']),first_action_deviation=float(old.first_action_deviation),speed=float(old.speed)))
                    dump(directory/f'{scene}_r{seed}_first_collision.json',dict(info=clean(info),environment=snapshot(env,obs),original_step=clean(old.to_dict())))
                assert bool(done)==bool(trace['done'][i])
            assert first
            print('FIXED_CRASH_REPLAY_EXACT',scene,seed,flush=True)
        finally:env.close()
    table(R/'introduced_crash_diagnostics.csv',rows);dump(R/'crash_replay_verification.json',dict(trajectories=3,all_observations_rewards_termination_exact=True,new_independent_maps=0,selection_or_tuning=False))
if __name__=='__main__':
    from mpr_mpc._bootstrap import bootstrap
    bootstrap();main()
