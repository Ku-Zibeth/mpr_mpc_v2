"""Four frozen-policy ablations, paired native traffic and live failure-seed exports.

No training or default changes. Uses the independently tested Tfast configuration.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import evaluate_model_suite as suite
import evaluate_ablation as ablation
from evaluation_pairing import snapshot

VARIANTS=tuple(ablation.VARIANTS)
DEFAULT_SCENARIOS=ROOT/'scenarios/selected100.json'


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def export_failures(out,rows):
    directory=out/'failures';directory.mkdir(exist_ok=True)
    fields=['configuration','episode','scenario','episode_seed','traffic_seed','end_reason','native_success','final_success','strict_success','strict_failure_reason','episode_crash','episode_out_of_road','final_route_completion','actual_discounted_return','sum_cost','length']
    failed_seeds={};unsafe_seeds={}
    for label,_ in VARIANTS:
        group=[r for r in rows if r['configuration']==label]
        failed=[r for r in group if float(r['final_success'])<.5]
        failed_seeds[label]=[int(r['scenario']) for r in failed]
        unsafe_seeds[label]=[int(r['scenario']) for r in group if float(r.get('episode_crash',0))>0 or float(r.get('episode_out_of_road',0))>0]
        with (directory/f'{label}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(failed)
        write_json(directory/f'{label}_seeds.json',failed_seeds[label])
        write_json(directory/f'{label}_unsafe_seeds.json',unsafe_seeds[label])
    write_json(directory/'all_failed_seeds.json',failed_seeds)
    write_json(directory/'all_unsafe_seeds.json',unsafe_seeds)
    matrix=[]
    for scene in dict.fromkeys(int(r['scenario']) for r in rows):
        row={'scenario':scene};present=[]
        for label,_ in VARIANTS:
            item=next((r for r in rows if r['configuration']==label and int(r['scenario'])==scene),None)
            row[f'{label}_failed']=None if item is None else int(float(item['final_success'])<.5)
            row[f'{label}_end_reason']=None if item is None else item['end_reason']
            if item is not None:present.append(row[f'{label}_failed'])
        row.update(completed_groups=len(present),any_failed=int(any(present)),all_failed=int(len(present)==4 and all(present)))
        matrix.append(row)
    if matrix:
        with (directory/'failure_matrix.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)


def finite_json(value):
    """JSON journals use null for missing diagnostics, never nonstandard NaN."""
    import math
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def atomic_json(path, value):
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(finite_json(value),stream,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_csv(path, rows):
    """Regenerate views atomically; the per-episode journals are authoritative."""
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with temporary.open('w',newline='') as stream:
        if fields:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader();writer.writerows(finite_json(rows))
    temporary.replace(path)


def read_rows(path):
    if not Path(path).is_file():return []
    with Path(path).open() as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for key,value in row.items():
            if value=='':row[key]=None
            elif isinstance(value,str):
                try:row[key]=float(value)
                except ValueError:pass
        for key in ['scenario','episode','test_seed','episode_seed','traffic_seed','length']:
            if row.get(key) is not None:row[key]=int(row[key])
    return finite_json(rows)


def row_key(row):return str(row['configuration']),int(row['scenario'])


def environment_protocol(raw):
    sim=raw.metadrive.simulator
    mode=str(sim.get('out_of_road_mode','legacy'))
    result=dict(traffic_density=float(sim.traffic_density),accident_prob=float(sim.accident_prob),
        out_of_road_mode=mode,out_of_road_margin_widths=float(sim.get('out_of_road_margin_widths',0.)) if mode=='body_margin' else 0.)
    if mode=='body_margin' and sim.get('out_of_road_margin_m') is not None:
        result['out_of_road_margin_m']=float(sim.out_of_road_margin_m)
    return result


def apply_environment_options(raw,traffic_density=None,margin_widths=None,margin_m=None):
    if margin_widths is not None and margin_m is not None:raise ValueError('Choose either metres or vehicle-width margin, not both')
    if traffic_density is not None:
        if not np.isfinite(traffic_density) or not 0<=traffic_density<=1:raise ValueError('Traffic density must be in [0,1]')
        raw.metadrive.simulator.traffic_density=float(traffic_density)
    if margin_widths is not None:
        if not np.isfinite(margin_widths) or margin_widths<0:raise ValueError('Road margin widths must be finite/nonnegative')
        raw.metadrive.simulator.out_of_road_mode='body_margin'
        raw.metadrive.simulator.out_of_road_margin_widths=float(margin_widths)
        raw.metadrive.simulator.out_of_road_margin_m=None
    if margin_m is not None:
        if not np.isfinite(margin_m) or margin_m<0:raise ValueError('Road margin metres must be finite/nonnegative')
        raw.metadrive.simulator.out_of_road_mode='body_margin'
        raw.metadrive.simulator.out_of_road_margin_widths=0.
        raw.metadrive.simulator.out_of_road_margin_m=float(margin_m)


def preserve_scenario_range(effective,parsed,scenarios):
    start=min(scenarios);count=max(scenarios)-start+1
    effective.metadrive.simulator.start_seed=start
    effective.metadrive.simulator.num_scenarios=count
    # The runtime dataclass stores metadrive as a plain nested dict.
    parsed.metadrive['simulator']['start_seed']=start
    parsed.metadrive['simulator']['num_scenarios']=count
    return effective,parsed


class EpisodeJournal:
    """Commit completed episodes before exporting any CSV or summary."""
    def __init__(self,out,scenarios,seed):
        self.out=Path(out);self.scenarios=list(scenarios);self.seed=int(seed);self.records={}
        for path in sorted((self.out/'completed').glob('*/*.json')):
            record=json.loads(path.read_text());row=record['row'];key=row_key(row)
            if key in self.records or key[0] not in dict(VARIANTS) or key[1] not in self.scenarios:
                raise ValueError(f'Invalid/duplicate episode journal: {path}')
            if int(row['test_seed'])!=self.seed:raise ValueError(f'Seed changed: {path}')
            self.records[key]=record

    def save(self,row,hashes,source='new_run'):
        row=finite_json(dict(row));key=row_key(row)
        if key in self.records:raise ValueError(f'Episode already committed: {key}')
        record=dict(row=row,initial_hashes=hashes,source=source)
        atomic_json(self.out/'completed'/key[0]/f'{key[1]}.json',record)
        self.records[key]=record

    def rows(self):
        return [self.records[(label,scene)]['row'] for label,_ in VARIANTS
                for scene in self.scenarios if (label,scene) in self.records]

    def pending(self,label):
        return [scene for scene in self.scenarios if (label,scene) not in self.records]

    def export(self,summary=False):
        rows=self.rows();atomic_csv(self.out/'episodes.csv',rows);export_failures(self.out,rows)
        for label,_ in VARIANTS:
            group=[row for row in rows if row['configuration']==label]
            atomic_csv(self.out/f'episodes_{label}_seed{self.seed}.csv',group)
        if summary and rows:
            from evaluation_statistics import summarize_runs,paired_comparisons
            summaries=[]
            for label,_ in VARIANTS:
                group=[row for row in rows if row['configuration']==label]
                if group:
                    result=summarize_runs(label,group,samples=2000,seed=20261124)
                    result['completed_episodes']=len(group);result['expected_episodes']=len(self.scenarios)
                    result['complete']=len(group)==len(self.scenarios);summaries.append(result)
            atomic_csv(self.out/'summary.csv',summaries);atomic_json(self.out/'summary.json',summaries)
            atomic_csv(self.out/'paired_comparisons.csv',paired_comparisons(rows,samples=2000,seed=20261124))


def make_paired_env(cfg, out, label, references, observed, factory=None):
    """Native destruction on reset prevents stale road-object Bullet velocities.

    BaseStaticObject.reset() relocates pooled barriers without clearing their
    linear/angular velocity. force_destroy is MetaDrive's native bookkeeping
    switch, not a change to roads, traffic, collision, rewards or termination.
    """
    env=(factory or suite.make_env)(cfg);native_env=env.unwrapped_metadrive
    native_env.config['force_destroy']=True
    original_reset=env.reset;holder={'initialized':False,'scene':None}
    def reset(*,seed=None,options=None):
        scene=int(seed);holder['scene']=scene
        if not holder['initialized']:
            original_reset(seed=scene)
            engine=native_env.engine
            if not engine.global_config['force_destroy']:
                raise RuntimeError('MetaDrive ignored force_destroy; cannot guarantee paired reset.')
            mgr=engine.traffic_manager;native_reset=mgr.reset
            if mgr.random_traffic:raise RuntimeError('Expected native random_traffic=false protocol.')
            def traffic_reset(*args,**kwargs):
                mgr.seed((holder['scene']*1009+int(cfg.seed))%2**32)
                return native_reset(*args,**kwargs)
            mgr.reset=traffic_reset;holder['initialized']=True
        obs=original_reset(seed=scene)
        current=snapshot(env,obs);hashes=current['hashes'];observed[scene]=hashes
        # Persist the evidence BEFORE the guard, including a mismatching state.
        with (out/'initial_state_hashes.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(configuration=label,scenario=scene,hashes=hashes))+'\n')
        reference=references.get(scene)
        if reference is not None and reference!=hashes:
            changed=[k for k in reference if reference[k]!=hashes.get(k)]
            path=out/'pairing_mismatches'/f'{label}_{scene}.json'
            atomic_json(path,dict(expected=reference,actual=current,different_components=changed))
            raise RuntimeError(f'Unpaired initial state, scene={scene}, method={label}, components={changed}; evidence={path}')
        if reference is None:
            references[scene]=hashes
            atomic_json(out/'initial_states'/f'{scene}.json',current)
        return obs
    env.reset=reset
    return env


def import_compatible(source,journal,references,scenarios,seed):
    """Import only completed rows whose FULL recorded initial hashes still match."""
    source=Path(source);manifest=json.loads((source/'launch_manifest.json').read_text())
    if (source/'launch_config.yaml').exists() and (journal.out/'launch_config.yaml').exists():
        old_protocol=environment_protocol(OmegaConf.load(source/'launch_config.yaml'))
        new_protocol=environment_protocol(OmegaConf.load(journal.out/'launch_config.yaml'))
        if old_protocol!=new_protocol:raise ValueError('Environment/departure rules changed; old episode results cannot be imported.')
    if manifest['scenarios']!=scenarios or int(manifest['seed'])!=seed:
        raise ValueError('Resume source scenarios/seed differ; refusing to mix experiments.')
    if manifest['checkpoint_sha256']!=json.loads((journal.out/'launch_manifest.json').read_text())['checkpoint_sha256']:
        raise ValueError('Resume source checkpoint differs.')
    hashes={}
    for line in (source/'initial_state_hashes.jsonl').read_text().splitlines():
        record=json.loads(line);key=(record['configuration'],int(record['scenario']))
        if key in hashes and hashes[key]!=record['hashes']:raise ValueError(f'Conflicting old hashes: {key}')
        hashes[key]=record['hashes']
    audit=[];seen=set()
    for row in read_rows(source/'episodes.csv'):
        key=row_key(row)
        if key in seen:raise ValueError(f'Duplicate source episode: {key}')
        seen.add(key)
        if key[0] not in dict(VARIANTS) or key[1] not in scenarios or int(row['test_seed'])!=seed:
            raise ValueError(f'Invalid source episode: {key}')
        expected=references[key[1]];old=hashes.get(key)
        changed=list(expected) if old is None else [k for k in expected if old.get(k)!=expected[k]]
        compatible=not changed
        audit.append(dict(configuration=key[0],scenario=key[1],reused=int(compatible),different_components=','.join(changed)))
        if compatible and key not in journal.records:
            journal.save(row,expected,source=str(source))
    atomic_csv(journal.out/'resume_import_audit.csv',audit)
    atomic_json(journal.out/'resume_import_audit.json',dict(source=str(source),source_rows=len(audit),
        retained=sum(x['reused'] for x in audit),rerun_required=sum(not x['reused'] for x in audit),
        rule='All observation, vehicle, traffic and RNG hashes must match; original files remain untouched.',rows=audit))
    print(f"RESUME AUDIT: {len(audit)} saved episodes; {sum(x['reused'] for x in audit)} retained, "
          f"{sum(not x['reused'] for x in audit)} require rerun due to changed initial state.",flush=True)


def main():
    import fcntl
    ap=argparse.ArgumentParser(description='Evaluate the retained WM200k+RL300k native-road release; collision means failure.')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'logs/selected_evaluation')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--smoke-steps',type=int,default=0,help='Technical truncated run on the first3 fixed maps, not formal evaluation.')
    args=ap.parse_args()
    if args.smoke_steps<0:raise ValueError('smoke-steps must be nonnegative')
    out=args.output_dir.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.runner.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError(f'Another evaluation is already running in {out}')
    release=json.loads((ROOT/'models/manifest.json').read_text())
    checkpoint=(ROOT/release['checkpoint']).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(f'{checkpoint}; retrieve the selected weight with git lfs pull')
    digest=hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    if digest.hexdigest()!=release['checkpoint_sha256']:raise ValueError('Selected checkpoint checksum mismatch')
    profile=ROOT/release['profile'];profile_sha=hashlib.sha256(profile.read_bytes()).hexdigest()
    payload=json.loads((ROOT/release['scenarios']).read_text());all_scenarios=payload['scenarios'] if isinstance(payload,dict) else payload
    if len(all_scenarios)!=100 or len(set(all_scenarios))!=100:raise ValueError('Selected release requires100 unique fixed scenarios')
    scenarios=all_scenarios[:3] if args.smoke_steps else all_scenarios
    seed=int(release['test_seed']);version=release['version']
    raw=OmegaConf.load(profile)
    raw.checkpoint=str(checkpoint);raw.seed=seed;raw.eval_episodes=len(scenarios);raw.work_dir=str(out);raw.log_root=str(ROOT/'logs')
    raw.enable_wandb=raw.save_video=raw.save_agent=False
    raw.benchmark.output_dir=str(out);raw.benchmark.scenarios_file=str(ROOT/release['scenarios'])
    raw.benchmark.collision_as_failure=True;raw.benchmark.paired_episode_seeds=True;raw.benchmark.mppi_eval_seed=seed
    marker=out/'manifest.json'
    expected=dict(runner_protocol='selected_strict_v1',checkpoint_sha256=digest.hexdigest(),profile_sha256=profile_sha,
        scenarios=scenarios,seed=seed,smoke_steps=args.smoke_steps,success_definition='native arrival AND no collision at any episode step')
    if marker.exists():
        if not args.resume:raise FileExistsError('Use --resume or a fresh output directory')
        previous=json.loads(marker.read_text())
        for key,value in expected.items():
            if previous.get(key)!=value:raise ValueError(f'Resume setting changed: {key}')
    else:
        if (out/'episodes.csv').exists():raise FileExistsError('Refusing to overwrite previous results')
        atomic_json(marker,dict(expected,checkpoint=str(checkpoint),variants=dict(VARIANTS),environment_protocol=environment_protocol(raw)))
    OmegaConf.save(raw,out/'config.yaml');atomic_json(out/'scenarios.json',dict(scenarios=scenarios))
    print(f'Checkpoint: {checkpoint}\nOutput: {out}\nEnvironment: {environment_protocol(raw)}\nSuccess: arrive AND no collision',flush=True)
    if args.prepare_only:return
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    from common.seed import set_seed
    references={int(path.stem):json.loads(path.read_text())['hashes'] for path in (out/'initial_states').glob('*.json')}
    observed={};base_factory=suite.make_env;original_cfg=suite._make_cfg
    def fixed_cfg(*a,**kw):return preserve_scenario_range(*original_cfg(*a,**kw),all_scenarios)
    suite._make_cfg=fixed_cfg
    _,cfg=fixed_cfg(raw,version,checkpoint,scenarios);set_seed(seed)
    env=make_paired_env(cfg,out,'reset_preflight',references,observed,factory=base_factory)
    try:
        for index,scene in enumerate(scenarios,1):
            set_seed((seed+scene)%2**32);env.reset(seed=scene)
            if index==1 or index%10==0 or index==len(scenarios):print(f'[reset preflight {index}/{len(scenarios)}] scene={scene} OK',flush=True)
    finally:env.close()
    journal=EpisodeJournal(out,scenarios,seed)
    for (label,scene),record in journal.records.items():
        if record['initial_hashes']!=references[scene]:raise RuntimeError(f'Resume initial state changed: {label}/{scene}')
    active={'label':None}
    def factory(cfg):
        env=make_paired_env(cfg,out,active['label'],references,observed,factory=base_factory)
        if args.smoke_steps:
            native_step=env.step;native_reset=env.reset;counter=[0]
            def reset(**kwargs):counter[0]=0;return native_reset(**kwargs)
            def step(action):
                obs,reward,done,info=native_step(action);counter[0]+=1
                if counter[0]>=args.smoke_steps:done=True;info=dict(info,end_reason='technical_smoke_cutoff')
                return obs,reward,done,info
            env.reset=reset;env.step=step
        return env
    suite.make_env=factory
    try:
        for label,stack in VARIANTS:
            pending=journal.pending(label);print(f'\n=== {label}: completed={len(scenarios)-len(pending)}/{len(scenarios)}, remaining={len(pending)} ===',flush=True)
            if not pending:continue
            active['label']=label
            variant=OmegaConf.create(OmegaConf.to_container(raw,resolve=True));variant.evaluation_stack=stack
            OmegaConf.save(variant,out/f'config_{label}.yaml')
            def save_episode(row):
                from evaluation_success import apply_collision_failure
                apply_collision_failure(row)
                row.update(configuration=label,test_seed=seed,episode=all_scenarios.index(int(row['scenario']))+1,
                    traffic_seed=(int(row['scenario'])*1009+seed)%2**32)
                journal.save(row,observed[int(row['scenario'])]);journal.export()
                print(f"[saved {len(journal.records)}/{4*len(scenarios)}] {label} scene={row['scenario']} native_success={row['native_success']:.0f} success={row['final_success']:.0f}",flush=True)
            suite._evaluate_checkpoint(variant,version,checkpoint,pending,on_episode=save_episode)
            journal.export(summary=True)
    finally:
        suite.make_env=base_factory;suite._make_cfg=original_cfg;journal.export(summary=True)
    print('Summary:',out/'summary.csv','\nStrict failure seeds:',out/'failures/all_failed_seeds.json',flush=True)

if __name__=='__main__':main()
