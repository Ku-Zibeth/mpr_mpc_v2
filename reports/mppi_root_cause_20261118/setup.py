"""Freeze NEW development maps and exact B/C protocol before any outcomes."""
import importlib.metadata as md,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from study_io import *


def main():
    h=load(R/'history_review.json');assert not h['unreadable_files']
    previous=load(MAIN/'reports/mppi_selection_20261117/protocol.json')
    scenes=h['proposed_new_development_maps'];assert not set(scenes)&set(h['historical_scene_ids'])
    assert min(scenes)>=130000 and len(scenes)==8
    # Additional broad scene-ID scan of ALL historical report protocols/results.
    seen=set();files=[]
    def visit(x,key=''):
        if isinstance(x,dict):
            for k,v in x.items():visit(v,k)
        elif isinstance(x,list):
            if any(s in key.lower() for s in ['scen','maps','origins']):
                for v in x:
                    if isinstance(v,int) and not isinstance(v,bool):seen.add(v)
                    elif isinstance(v,list) and v and isinstance(v[0],int):seen.add(v[0])
            for v in x:
                if isinstance(v,dict):visit(v)
        elif isinstance(x,int) and key.lower() in ['scene','scenario','scene_id','scenario_id']:seen.add(x)
    for base in [MAIN,PRIOR]:
        for f in (base/'reports').rglob('*.json'):
            visit(load(f));files.append(str(f))
    assert not set(scenes)&seen,'New scenes overlap historical metadata'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert commit=='e26d669574e10d5184135f5fa629582371c45636'
    ck=Path(previous['checkpoint']);assert sha(ck)==previous['checkpoint_sha256']
    protocol=dict(base_commit=commit,branch='experiment/mppi-root-cause-20261118',
        checkpoint=str(ck),checkpoint_sha256=previous['checkpoint_sha256'],source_environment_config=previous['source_environment_config'],
        scenarios=scenes,repeat_seeds=[1,2],groups=['B','C'],episodes_per_group=16,
        gamma=.995,horizon=10,episode_horizon=1000,control_dt=.1,
        B='Original deterministic Lattice+SAC, freshly recomputed each step; MPPI off',
        C='Current default LocalMPPI:64 Gaussian+Baseline,8elite,4iterations,rho .8/.8,std .04/.12,beta .5,candidate/random_pair,cost0; no selection_mode inserted',
        native_loop='Every control step: prepare Lattice, original535/SAC, optional MPPI, execute selected[0], native env.step, new observation. No fixed-H replay.',
        D='No D predefined: only add if a reproducible deterministic bug/one-factor correction is established and separately frozen before D outcomes. No forced refactor.',
        diagnostics='record_candidates observational only. Exact pre-clamp sampler reconstructed using cloned private RNG AFTER act; original/global RNG unchanged. Post-act diagnostics timed separately.',
        detailed_snapshot_steps=[1,50,150],coverage='every native MPPI step and every iteration',
        probe='After all closed-loop jobs: one isolated worker, first2newmaps repeat1, Baseline prefix29 to step30,5warmup+50calls per map per group, record_candidates false; no probe controls executed',
        parallel_workers=2,closed_loop_latency_scope='Complete synchronized agent.act under two-worker shared load, NOT isolated real-time certification; separate isolated full-planner probe follows.',
        bootstrap=dict(samples=2000,seed=20261118,unit='whole map; both repeat seeds paired within map'),
        acceptance='At least one extra success, no material crash/OOR/route deterioration, paired discounted gain>0 with sceneCI, consistency beyond single map/seed, isolatedP95 target100ms. If no positive signal stop, no100episode expansion.',
        near_overlap_l2=.01,unknown_actual_alternative_paths='No true return labels for all Gaussian/Lattice alternatives; never substitute WM/oracle scores for true performance.',
        training=0,scoring_heads=0,formal100=False,default_config_changes=False,commit=False,push=False)
    dump('protocol.json',protocol)
    protected={str(f.relative_to(ROOT)):sha(f) for d in ['planning','lattice','residual_rl','vendor/tdmpc2_runtime'] for f in (ROOT/d).rglob('*.py')}
    for name in ['agent.py','config.yaml','evaluate_model_suite.py']:protected[name]=sha(ROOT/name)
    dump('experiment_manifest.json',dict(base_commit=commit,main_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=MAIN,text=True).strip(),
        main_status_at_start=subprocess.check_output(['git','status','--short'],cwd=MAIN,text=True),
        worktree=str(ROOT),branch=protocol['branch'],remote_main_verified=commit,checkpoint_sha256=sha(ck),
        source_config_sha256=sha(Path(protocol['source_environment_config'])),default_config_sha256=sha(ROOT/'config.yaml'),
        historical_review_sha256=sha(R/'history_review.json'),protocol_sha256=sha(R/'protocol.json'),
        historical_scene_ids=sorted(seen),historical_json_scanned=len(files),scenarios=scenes,seeds=[1,2],
        protected_sources_before=protected,packages={n:md.version(n) for n in ['torch','numpy','metadrive-simulator']},
        python=sys.version,real_future_observations_only_in_diagnostics=True))
    print('NEW8maps frozen:',scenes,'B/C x2 seeds =32 native episodes. No historical validation reused.',flush=True)


if __name__=='__main__':main()
