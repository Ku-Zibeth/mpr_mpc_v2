"""Freeze new maps/seeds, margin and decision gates BEFORE any new-map execution."""
import re,subprocess,sys,importlib.metadata as md
from study import *

def main():
    assert load(R/'frozen_margin.json')['no_new_map_outcomes_seen']
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert base=='75b8a212a62f50ca76e29e37085fe1cd639ac718'
    seen=set();intervals=set();files=[];digests=set()
    def visit(value,key=''):
        if isinstance(value,dict):
            if isinstance(value.get('start_seed'),int) and isinstance(value.get('num_scenarios'),int):intervals.add((value['start_seed'],value['start_seed']+value['num_scenarios']))
            for k,v in value.items():visit(v,k)
        elif isinstance(value,list):
            if any(k in key.lower() for k in ['scene','scenario','maps','origin']):
                for x in value:
                    if isinstance(x,int) and not isinstance(x,bool):seen.add(x)
                    elif isinstance(x,list) and x and isinstance(x[0],int):seen.add(x[0])
            for x in value:
                if isinstance(x,dict):visit(x)
        elif isinstance(value,int) and key.lower() in ['scene','scenario','scene_id','scenario_id','map_id']:seen.add(value)
    roots=[MAIN,ROOT.parent/'mpr_mpc_v2_first_action_20261117',ROOT.parent/'mpr_mpc_v2_root_cause_20261118',ROOT.parent/'mpr_mpc_v2_h1_counterfactual_20261118']
    for root in roots:
        bases=[root/'reports']+([root/'logs'] if root==MAIN else [])
        for folder in bases:
            for file in folder.rglob('*'):
                if file.suffix not in ['.json','.csv','.yaml','.yml','.log']:continue
                data=file.read_bytes();digest=hashlib.sha256(data).hexdigest()
                if digest in digests:continue
                digests.add(digest);files.append(dict(path=str(file),sha256=digest))
                if file.suffix=='.json':
                    try:visit(json.loads(data))
                    except json.JSONDecodeError:pass
                elif file.suffix in ['.yaml','.yml']:
                    from omegaconf import OmegaConf
                    try:visit(OmegaConf.to_container(OmegaConf.load(file),resolve=False))
                    except Exception:pass
                elif file.suffix=='.csv':
                    reader=csv.DictReader(io.StringIO(data.decode(errors='replace')))
                    keys=[k for k in (reader.fieldnames or []) if k.lower() in ['scene','scenario','scene_id','scenario_id','map_id']]
                    if keys:
                        for row in reader:
                            for key in keys:
                                if row.get(key,'').isdigit():seen.add(int(row[key]))
                else:
                    text=data.decode(errors='replace')
                    seen.update(int(x) for x in re.findall(r'(?:scene|scenario|seed)[_ :=-]+(\d+)',text,re.I))
                    for line in text.splitlines():
                        if 'scenarios:' in line.lower():seen.update(int(x) for x in re.findall(r'\b\d+\b',line))
    # Explicit original WM/SAC training window, irrespective of metadata duplication.
    intervals.add((10000,12000))
    options=np.random.default_rng(20261119).permutation(np.arange(200000,201000)).tolist()
    scenes=[s for s in options if s not in seen and not any(lo<=s<hi for lo,hi in intervals)][:12];assert len(scenes)==12
    old=load(HIST/'protocol.json');margin=load(R/'frozen_margin.json')
    protocol=dict(base_commit=base,checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],source_environment_config=old['source_environment_config'],
        scenes=scenes,seeds=[1,2,3],origin_step=50,gamma=.995,control_dt=.1,worker_limit=6,bootstrap_samples=2000,bootstrap_seed=20261119,
        candidate_pool='Baseline, final WeightedMean, every actual Gaussian from all4iterations =258 sequences; preserve source and aliases; uniform unique nonBaseline FIRST controls L2>1e-5',
        methods=dict(A='Original Lattice+SAC',B='Exact native Baseline/Mean final selection',C='Uniform random actual nonBaseline first control',
            D='Reward-only argmax over the entire SAME captured pool',E='Independent native RQ final2 replication, MUST equal B',
            F='Native proposed Mean protected by development-only frozen finite margin',G='Supplement: same-pool RQ argmax, paired with D to isolate terminal-Q from pool eligibility'),
        margin=margin['margin'],margin_development_supported=margin['development_positive'],
        random_seed_rule='Independent default_rng(20261119 + scene*31 + seed), never global environment RNG',
        dedup_l2=1e-5,pair_tie_return=1.,h1='Exactly one action, then original B feedback to native done',
        closed='Whole episode, replan and execute first action every period; A originalSAC,B originalNative,C evidence-selected opt-in change',
        unreachable='If original episode ends before fixed50, retain full episode, mark origin unreachable, no map/origin replacement',
        decision_rules=dict(reward_case='D must beat Native B and matched-pool G in paired95%CI, no crash/OOR mean increase,positive gain on >=8/12maps; reward gain vsA CI lower>0',
            margin_case='F-vsA paired gain CI lower>0, no crash/OOR mean increase,nonzero physical replacement; frozen margin never retuned',
            unsupported='Otherwise implement opt-in baseline guard with replacement_supported=false; deny unsupported replacement, explicitly protection not performance success',
            no_gaussian_change='Positive finite old oracle already proves some coverage; no evidence supporting new samples/std/rho/iterations',
            screening_note='New maps independent of all older studies. H1 informs a predeclared mechanism branch; same-map subsequent closed-loop is SCREENING, not untouched final validation'),
        latency_probe='After all jobs: isolated first2maps seed1 pre50,5warmup+50full calls/method, no probe controls executed; closed shared-load timings separately',
        acceptance='Success>A,crash/OOR not higher,positive map-paired discounted gain and route not lower,beyond one map/seed,H1gain better thanrandom,physicalreplacement>0;targetfullcycleP95<=100ms',
        formal100=False,training=0,commit=False,push=False)
    dump(R/'exclusions.json',dict(scene_ids=sorted(seen),intervals=sorted(intervals),unique_metadata_files=files,selection_seed=20261119))
    dump(R/'protocol.json',protocol)
    protected={n:file_sha(ROOT/n) for n in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if n and not n.startswith('reports/') and (ROOT/n).is_file()}
    dump(R/'experiment_manifest.json',dict(base_commit=base,remote_verified_sha=base,branch='experiment/mppi-evaluation-redesign-20261119',
        checkpoint_sha256=file_sha(Path(old['checkpoint'])),source_config_sha256=file_sha(Path(old['source_environment_config'])),
        original_config_sha256=file_sha(ROOT/'config.yaml'),protected_sources=protected,protocol_sha256=file_sha(R/'protocol.json'),
        historical_inputs=load(R/'historical_inputs_read.json'),python=sys.version,packages={k:md.version(k) for k in ['torch','numpy','metadrive-simulator','omegaconf']},
        no_new_map_outcomes_before_freeze=True))
    assert protocol['checkpoint_sha256']==file_sha(Path(protocol['checkpoint']))
    print('NEW_MAPS_FROZEN',scenes,'seeds',[1,2,3],'metadata_files',len(files),flush=True)
if __name__=='__main__':main()
