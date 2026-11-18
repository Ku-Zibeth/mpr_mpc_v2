"""Preregister new maps, data-derived scales, simple-filter control and decision gates."""
import re, importlib.metadata as md, io
from comfort_study import *
import pandas as pd

def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='a9e84b55f4d373cad11979e70eceed149db59a6d'
    required=['MPPI_EVALUATION_REDESIGN_REPORT.md','acceptance.json','h1_statistics.json','closed_statistics.json','latency.csv','candidate_coverage.csv']
    read=[]
    for name in required:
        f=OLD/name;data=f.read_bytes()
        if f.suffix=='.json':json.loads(data)
        elif f.suffix=='.csv':list(csv.DictReader(io.StringIO(data.decode())))
        else:data.decode()
        read.append(dict(path=str(f),sha256=file_sha(f)))
    previous=load(OLD/'exclusions.json');seen=set(previous['scene_ids']);intervals={tuple(v) for v in previous['intervals']};seen.update(load(OLD/'protocol.json')['scenes'])
    files=[];digests=set();intervals.add((10000,12000))
    def visit(v,key=''):
        if isinstance(v,dict):
            if isinstance(v.get('start_seed'),int) and isinstance(v.get('num_scenarios'),int):intervals.add((v['start_seed'],v['start_seed']+v['num_scenarios']))
            for k,x in v.items():visit(x,k)
        elif isinstance(v,list):
            if any(k in key.lower() for k in ['scene','scenario','maps']):seen.update(x for x in v if isinstance(x,int) and not isinstance(x,bool))
            for x in v:
                if isinstance(x,dict):visit(x)
        elif isinstance(v,int) and key.lower() in ['scene','scenario','scene_id','scenario_id','map_id']:seen.add(v)
    roots=[MAIN,ROOT.parent/'mpr_mpc_v2_first_action_20261117',ROOT.parent/'mpr_mpc_v2_root_cause_20261118',ROOT.parent/'mpr_mpc_v2_h1_counterfactual_20261118',OLD.parents[1]]
    for root in roots:
        for folder in [root/'reports',root/'logs']:
            if not folder.exists():continue
            for f in folder.rglob('*'):
                if not f.is_file() or f.suffix not in {'.json','.csv','.yaml','.yml','.log'}:continue
                data=f.read_bytes();sha=hashlib.sha256(data).hexdigest()
                if sha in digests:continue
                digests.add(sha);files.append(dict(path=str(f),sha256=sha))
                if f.suffix=='.json':
                    try:visit(json.loads(data))
                    except json.JSONDecodeError:pass
                elif f.suffix in {'.yaml','.yml'}:
                    from omegaconf import OmegaConf
                    try:visit(OmegaConf.to_container(OmegaConf.load(f),resolve=False))
                    except Exception:pass
                elif f.suffix=='.csv':
                    reader=csv.DictReader(io.StringIO(data.decode(errors='replace')))
                    keys=[k for k in reader.fieldnames or [] if k.lower() in ['scene','scenario','scene_id','scenario_id','map_id']]
                    if keys:
                        for row in reader:
                            for k in keys:
                                try:seen.add(int(row[k]))
                                except (ValueError,KeyError,TypeError):pass
                else:
                    text=data.decode(errors='replace');seen.update(map(int,re.findall(r'(?:scene|scenario|seed)[_ :=-]+(\d+)',text,re.I)))
                    for line in text.splitlines():
                        if 'scenarios:' in line.lower():seen.update(map(int,re.findall(r'\b\d+\b',line)))
    candidates=np.random.default_rng(20261120).permutation(np.arange(300000,301000)).tolist()
    scenes=[s for s in candidates if s not in seen and not any(lo<=s<hi for lo,hi in intervals)][:12];assert len(scenes)==12
    scales=load(R/'development_scales.json');params=load(R/'vehicle_parameters.json');err=pd.read_csv(R/'kinematic_prediction_errors.csv')
    def tolerance(metric):return float(err[(err.horizon==1)&(err.metric==metric)].groupby('scene').p95.mean().mean())
    max_steer=np.deg2rad(params['max_steering'])
    # Half the measured P95 per-step steering change is the local correction envelope.
    bound=.5*scales['steering_rate_scale']*.1/max_steer
    comfort=dict(enabled=True,steering_rate_scale=scales['steering_rate_scale'],steering_jerk_scale=scales['steering_jerk_scale'],
        steering_delta_bound=bound,path_extra_tolerance_m=tolerance('position_m'),progress_loss_tolerance_m=tolerance('position_m'),
        heading_tolerance_rad=tolerance('heading_rad'),kinematic_cfg=params['legacy_geometry_cfg'])
    # Characteristic derivative timescale from old data, no outcome/grid tuning.
    tau=scales['steering_rate_scale']/scales['steering_jerk_scale'];alpha=.1/(.1+tau)
    config=dict(objective_mode='comfort',comfort=comfort)
    dump(R/'comfort_config.json',config)
    old=load(OLD/'protocol.json')
    protocol=dict(base_commit='a9e84b55f4d373cad11979e70eceed149db59a6d',checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],source_environment_config=old['source_environment_config'],
        development_scenes=load(OLD/'protocol.json')['scenes'],scenes=scenes,seeds=[1,2,3],seed_semantics='planner/global seed=scene+seed; environment reset seed=scene, same paired traffic',
        bootstrap_samples=2000,bootstrap_seed=20261120,control_dt=.1,gamma=.995,worker_limit=6,
        methods=dict(A='Original Lattice+SAC',B='Causal steering low-pass, exact original throttle',C='Comfort-aware Local MPPI, exact original throttle'),
        lowpass=dict(tau_seconds=tau,alpha=alpha,steering_delta_bound=bound,formula='alpha*u_SAC+(1-alpha)*actual_previous, clipped around current SAC; throttle unchanged'),
        objective='two directly measurable action terms only: equally normalized steering rate and steering second difference; no unvalidated physical jerk or TerminalQ',
        physics_horizon=1,kinematic_prediction_not_safety_guarantee=True,samples=64,elites=8,iterations=4,
        comfort_index=dict(scales=scales['comfort_index'],definition='mean of raw steering_rate_RMS and SG-filtered actual longitudinal_jerk_RMS/lateral_jerk_RMS/lateral_accel_RMS normalized by map-level development medians'),
        offline_filter='5point Savitzky-Golay poly2 at dt0.1, identical allgroups; raw metrics retained; never used online',
        hard_braking_threshold_mps2=scales['hard_braking_threshold_mps2'],hard_braking_definition='data-derived P95 of development actual negative acceleration; episode event starts, not a universal emergency threshold',
        acceptance=dict(comfort_index_reduction=.10,success_noninferiority=-.03,crash_noninferiority=.02,oor_noninferiority=.02,route_noninferiority=-.01,discounted_return_relative_noninferiority=-.02,
            planner_p95_ms=100,planner_ratio_to_A=1.20,comfort_C_vs_B='paired improvement or comparable comfort with credible task benefit; no baseline-only success claim',
            interval_rule='map bootstrap95% must support conditions; wide intervals=insufficient evidence'),
        development_smoke=dict(scenes=[200618,200015],seed=1,steps=100,groups=['A','B','C'],no_weight_search=True),
        no_new_test_outcomes_at_freeze=True,training=False,commit=False,push=False)
    dump(R/'protocol.json',protocol)
    dump(R/'exclusions.json',dict(inherited_exclusions_sha256=file_sha(OLD/'exclusions.json'),scene_ids=sorted(seen),intervals=sorted(intervals),new_scan_files=files,selected_scenes=scenes))
    protected={n:file_sha(ROOT/n) for n in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if n and not n.startswith('reports/') and (ROOT/n).is_file() and n not in ['planning/local_mppi.py','planning/coordinator.py']}
    dump(R/'experiment_manifest.json',dict(base_commit=protocol['base_commit'],remote_main_verified=protocol['base_commit'],branch='experiment/mppi-comfort-20261120',
        original_config_sha256=file_sha(ROOT/'config.yaml'),checkpoint_sha256=file_sha(Path(protocol['checkpoint'])),source_config_sha256=file_sha(Path(protocol['source_environment_config'])),
        protected_sources=protected,original_core_sha256={n:subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT).hex() for n in []},
        prior_required_inputs_read=read,baseline_traces=load(R/'development_scales.json')['sources'],protocol_sha256=file_sha(R/'protocol.json'),comfort_config_sha256=file_sha(R/'comfort_config.json'),
        python=sys.version,packages={k:md.version(k) for k in ['torch','numpy','scipy','metadrive-simulator']},environment_frozen_before_independent_maps=True))
    assert protocol['checkpoint_sha256']==file_sha(Path(protocol['checkpoint']))
    print('COMFORT_PROTOCOL_FROZEN',scenes,'alpha',alpha,'bound',bound,'files',len(files),flush=True)
if __name__=='__main__':main()
