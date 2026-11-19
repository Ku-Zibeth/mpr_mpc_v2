"""Capture current-only contexts from exact old development-prefix replays.

Not used to choose or tune the frozen independent driving method.
"""
from runtime import *
import pickle,copy
from dataclasses import replace
from types import SimpleNamespace
@torch.no_grad()
def main():
    from common.seed import set_seed
    targets={50,100,150,200};out=R/'performance_states';out.mkdir(exist_ok=False)
    for scene in load(R/'protocol.json')['development_scenes'][:2]:
        _,_,env,agent,_=make_runtime(scene,1,'Tconditional');agent.planner.allow_mppi=False
        try:
            set_seed(scene+1);agent.planner.reset(seed=scene+1);obs=env.reset(seed=scene);sim=env.unwrapped_metadrive;trace=np.load(R/'development'/f'A_{scene}_r1/trace.npz')
            for i in range(200):
                assert np.array_equal(obs.cpu().numpy(),trace['observations'][i])
                action=agent.act(obs,t0=i==0,eval_mode=True).cpu().numpy();assert np.array_equal(action,trace['actions'][i])
                if i+1 in targets:
                    plan=agent.last_plan;e=agent.planner.tracking_evaluator
                    context=e.context(sim.vehicle,plan.refined_path,agent.planner.last_context.safety_context,float(sim.config['physics_world_step_size']),agent.planner.structured.controller)
                    feedback=copy.copy(context.feedback);feedback.cfg={k:v for k,v in feedback.cfg.items() if k.startswith('frenet_')}
                    obstacles=tuple(SimpleNamespace(position=np.asarray(x.position).copy(),velocity=np.asarray(x.velocity).copy(),heading=x.heading,length=x.length,width=x.width) for x in context.obstacles)
                    context=replace(context,feedback=feedback,obstacles=obstacles)
                    with (out/f'{scene}_{i+1}.pkl').open('xb') as f:pickle.dump(dict(context=context,baseline=plan.refined_actions.cpu(),configuration=e.cfg,scene=scene,control=i+1),f)
                obs,_,done,_=env.step(action);assert not done
            print('CAPTURE_EXACT_CURRENT_CONTEXTS',scene,sorted(targets),flush=True)
        finally:env.close()
if __name__=='__main__':
    from mpr_mpc._bootstrap import bootstrap
    bootstrap();main()
