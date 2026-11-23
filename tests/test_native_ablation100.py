"""Four-way launcher compatibility and faithful live failure exports."""
from types import SimpleNamespace
from unittest.mock import Mock
import json,csv
import torch
from mpr_mpc.planning.coordinator import MPRMPCPlanner
from tools.run_native_ablation100 import export_failures


def test_lattice_mppi_gets_geometry_without_building_sac_state():
    p=object.__new__(MPRMPCPlanner);p.horizon=10;p.device='cpu';p.control_dt=.1
    path=SimpleNamespace(target_d=0.,target_speed=8.,horizon=2.)
    p.structured=SimpleNamespace(select_coarse=lambda v:SimpleNamespace(coarse_path=path),planner=object())
    p.action_adapter=SimpleNamespace(path_to_actions=lambda *a,**kw:torch.zeros(11,2))
    p.residual_bounds=lambda path:torch.zeros(2);p._vehicle_geometry=lambda v:(1.852,2.46894,.698)
    geometry=object();p.safety=SimpleNamespace(context=Mock(return_value=geometry),check=Mock(side_effect=AssertionError('No SAC feature check needed')))
    p.evaluator=SimpleNamespace(encode=lambda observation:torch.zeros(1,512))
    result=p.prepare(torch.zeros(259),SimpleNamespace(),need_world_model_features=True,need_residual_state=False,need_tracking_geometry=True)
    assert result.safety_context is geometry and result.residual_state is None and result.coarse_consequence is None
    p.safety.context.assert_called_once();p.safety.check.assert_not_called()
    p.safety.context.reset_mock()
    result=p.prepare(torch.zeros(259),SimpleNamespace(),need_world_model_features=True,need_residual_state=False)
    assert result.safety_context is None;p.safety.context.assert_not_called()


def test_failure_exports_distinguish_failure_successful_collision_and_pending(tmp_path):
    rows=[dict(configuration='A_lattice',episode=1,scenario=1000001,final_success=0.,episode_crash=0.,episode_out_of_road=1.,end_reason='out_of_road'),
          dict(configuration='B_lattice_sac',episode=1,scenario=1000001,final_success=1.,episode_crash=1.,episode_out_of_road=0.,end_reason='arrive_dest')]
    export_failures(tmp_path,rows);f=tmp_path/'failures'
    assert json.loads((f/'A_lattice_seeds.json').read_text())==[1000001]
    assert json.loads((f/'B_lattice_sac_seeds.json').read_text())==[]
    assert json.loads((f/'B_lattice_sac_unsafe_seeds.json').read_text())==[1000001]
    matrix=list(csv.DictReader((f/'failure_matrix.csv').open()))[0]
    assert matrix['completed_groups']=='2' and matrix['all_failed']=='0' and matrix['C_lattice_mppi_failed']==''


def test_native_reset_discards_previous_barrier_velocity_and_keeps_pair_guard(tmp_path,monkeypatch):
    import tools.run_native_ablation100 as runner
    config={'force_destroy':False};state={'barrier_velocity':7.,'observation':0}
    mgr=SimpleNamespace(random_traffic=False,seed=Mock(),reset=Mock())
    engine=SimpleNamespace(global_config=config,traffic_manager=mgr)
    native=SimpleNamespace(config=config,engine=engine)
    def reset(*,seed):
        # Mimic MetaDrive: a reused static object retains velocity unless destroyed.
        if config['force_destroy']:state['barrier_velocity']=0.
        mgr.reset();return state['observation']
    env=SimpleNamespace(unwrapped_metadrive=native,reset=reset)
    def snap(env,obs):
        return {'hashes':{'observation':str(obs),'vehicle':'same','traffic':str(state['barrier_velocity']),'rng':'same'}}
    monkeypatch.setattr(runner,'snapshot',snap)
    refs={};observed={};cfg=SimpleNamespace(seed=123)
    env=runner.make_paired_env(cfg,tmp_path,'B_lattice_sac',refs,observed,factory=lambda cfg:env)
    env.reset(seed=1043250);state['barrier_velocity']=-5.065686225891113
    env.reset(seed=1043250)
    assert state['barrier_velocity']==0.
    assert mgr.seed.call_args.args==((1043250*1009+123)%2**32,)
    state['observation']=1
    import pytest
    with pytest.raises(RuntimeError,match='components='):
        env.reset(seed=1043250)
    evidence=json.loads((tmp_path/'pairing_mismatches/B_lattice_sac_1043250.json').read_text())
    assert evidence['different_components']==['observation']


def test_episode_journal_resume_schema_and_duplicate_guard(tmp_path):
    import pytest
    from tools.run_native_ablation100 import EpisodeJournal
    j=EpisodeJournal(tmp_path,[10,20],123)
    row=dict(configuration='A_lattice',scenario=10,episode=1,test_seed=123,
             final_success=1.,episode_crash=0.,episode_out_of_road=0.,end_reason='arrive_dest')
    j.save(row,{'traffic':'same'});j.export()
    # A crash before regenerating CSV must still retain the committed episode.
    j.save(dict(row,configuration='B_lattice_sac',new_inference_metric=3.),{'traffic':'same'})
    resumed=EpisodeJournal(tmp_path,[10,20],123);resumed.export()
    assert len(resumed.records)==2 and resumed.pending('A_lattice')==[20]
    values=list(csv.DictReader((tmp_path/'episodes.csv').open()))
    assert len(values)==2 and values[0]['new_inference_metric']=='' and values[1]['new_inference_metric']=='3.0'
    with pytest.raises(ValueError,match='already committed'):resumed.save(row,{})


def test_import_reuses_matching_hashes_and_preserves_polluted_old_data(tmp_path):
    from tools.run_native_ablation100 import EpisodeJournal,import_compatible,atomic_csv,atomic_json
    source=tmp_path/'old';source.mkdir();out=tmp_path/'fixed';out.mkdir()
    manifest=dict(scenarios=[10,20],seed=123,checkpoint_sha256='same_checkpoint')
    atomic_json(source/'launch_manifest.json',manifest);atomic_json(out/'launch_manifest.json',manifest)
    rows=[dict(configuration='A_lattice',scenario=scene,episode=i,test_seed=123,final_success=1.,
               episode_crash=0.,episode_out_of_road=0.,end_reason='arrive_dest') for i,scene in enumerate([10,20],1)]
    atomic_csv(source/'episodes.csv',rows);old_bytes=(source/'episodes.csv').read_bytes()
    refs={10:{'traffic':'zero'},20:{'traffic':'zero'}}
    (source/'initial_state_hashes.jsonl').write_text('\n'.join(json.dumps(dict(configuration='A_lattice',scenario=scene,
        hashes={'traffic':'zero' if scene==10 else 'stale_velocity'})) for scene in [10,20]))
    j=EpisodeJournal(out,[10,20],123);import_compatible(source,j,refs,[10,20],123)
    assert j.pending('A_lattice')==[20] and len(j.records)==1
    # Repeating the import on resume cannot duplicate a completed episode.
    import_compatible(source,j,refs,[10,20],123)
    assert len(j.records)==1 and (source/'episodes.csv').read_bytes()==old_bytes
    audit=json.loads((out/'resume_import_audit.json').read_text())
    assert audit['retained']==1 and audit['rerun_required']==1


def test_resume_keeps_full_scenario_range_with_runtime_dict_config():
    from omegaconf import OmegaConf
    from tools.run_native_ablation100 import preserve_scenario_range
    effective=OmegaConf.create(dict(metadrive=dict(simulator=dict(start_seed=20,num_scenarios=1))))
    parsed=SimpleNamespace(metadrive={'simulator':{'start_seed':20,'num_scenarios':1}})
    effective,parsed=preserve_scenario_range(effective,parsed,[10,20,30])
    assert parsed.metadrive['simulator']=={'start_seed':10,'num_scenarios':21}
    assert effective.metadrive.simulator.start_seed==10 and effective.metadrive.simulator.num_scenarios==21
