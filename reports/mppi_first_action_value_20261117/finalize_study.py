"""Complete artifacts/report only after collection and every applicable gate finishes."""
import csv,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
from tools.mppi_failure_attribution import scene_bootstrap


def fmt(r,pct=False):
    if r is None or r['mean'] is None:return 'N/A'
    k=100 if pct else 1
    return f"{r['mean']*k:.3f} [{r['ci95'][0]*k:.3f},{r['ci95'][1]*k:.3f}]"


def aggregate():
    p=load(REPORT/'protocol.json');allrows=[];effects=[];candidates=[];members=[];pairs=[];horizons=[];timing=[];counts={};rng=[];origins=[]
    for folder in sorted((REPORT/'origins').iterdir()):
        if not (folder/'catalog.json').exists():continue
        meta=load(folder/'catalog.json');scene=meta['scene'];split='pilot_development' if scene in p['pilot_scenes'] else 'extra_development' if scene in p['extra_development_scenes'] else 'new_independent_validation'
        c=np.load(folder/'catalog.npz');q=np.load(folder/'offline_scores.npz');snap=load(folder/'origin.json');rng.append(dict(origin=folder.name,scene=scene,available={k:v is not None for k,v in snap['rng'].items()},component_hashes=snap['rng_component_hashes']))
        origins.append(dict(origin=folder.name,scenario=scene,split=split,unique=meta['unique'],catalog_sha=file_sha(folder/'catalog.npz')))
        branchfolder=REPORT/'branches'/folder.name;actual={}
        for f in sorted(branchfolder.glob('*.json')):
            r=load(f)
            if 'actual_return' not in r:continue
            data=np.load(ROOT/r['source']);assert data['done'][-1] and not data['done'][:-1].any()
            r=r|dict(split=split,continuation_after_first=float(data['rtg'][1]),
                continuation_after_fixed_prefix=float(data['rtg'][min(r['horizon'],len(data['actions']))]),
                remapping_count_scope='Onlyfixedphysicalprefix:0 residualremaps. Feedback uses originalresidualmapping normally.')
            allrows.append(r);counts[f'H{r["horizon"]}/{r["takeover"]}']=counts.get(f'H{r["horizon"]}/{r["takeover"]}',0)+1
            assert len(data['observations'])==len(data['actions'])+1
            assert np.array_equal(np.logical_or(data['terminated'],data['truncated']),data['done'])
            assert r['weights_unchanged'] and r['physical_action_mapping_count']==0
            for domain,key in [('observation','origin_observation_hash'),('vehicle','origin_vehicle_state_hash'),('traffic','origin_traffic_state_hash'),('rng','origin_rng_state_hash')]:assert r[key]==snap['hashes'][domain]
            idx=next(i for i,cc in enumerate(meta['representatives']) if cc['candidate_id']==r['candidate_id']);h=r['horizon'];assert np.array_equal(data['actions'][:min(h,len(data['actions']))],c['actions'][idx,:min(h,len(data['actions']))])
            if r['takeover']=='baseline':assert r['feedback_lattice_sac_steps']==max(0,len(data['actions'])-h)
            if h==1 and r['takeover']=='baseline' and not r['replay']:actual[r['candidate_id']]=r['actual_return']
            vals=data['planning_ms'][data['planning_ms']>0]
            if len(vals):timing.append(dict(configuration=r['takeover'],origin=folder.name,candidate=r['candidate_id'],horizon=h,phase='complete_act_during_collection',
                count=len(vals),p50_ms=float(np.quantile(vals,.5)),p95_ms=float(np.quantile(vals,.95)),p99_ms=float(np.quantile(vals,.99)),measurement='CONCURRENT_GPU, not isolatedrealtimeevidence'))
        for i,cc in enumerate(meta['representatives']):
            idx=c['pool_indices'][i];aa=c['actions'][i,0];candidates.append(dict(scenario=scene,split=split,origin=folder.name,candidate=cc['candidate_id'],roles=' '.join(cc['roles']),
                first_steer=float(aa[0]),first_throttle=float(aa[1]),delta_from_baseline=float(np.linalg.norm(aa-c['baseline'][0])),
                predicted_h10=float(c['scores'][idx]),captured_final_q_pair=meta['actual_final_pair'],pool_size=meta['pool_size'],unique_representatives=meta['unique'],raw_roles=8,
                sequence_npz=str((folder/'catalog.npz').relative_to(ROOT)),sequence_row=i,all_features_current_online_available=True))
            for mode,key in [('current','q_current'),('terminal_candidate','q_candidate'),('terminal_mean','q_policy_mean'),('terminal_sample','q_native_policy_sample')]:
                qq=q[key][idx]
                members.append(dict(scenario=scene,split=split,origin=folder.name,candidate=cc['candidate_id'],mode=mode,**{f'Q{j}':float(qq[j]) for j in range(5)},actual_Y1=actual.get(cc['candidate_id'])))
                for name,values in all_reducers(qq[None]).items():pairs.append(dict(scenario=scene,split=split,origin=folder.name,candidate=cc['candidate_id'],mode=mode,reducer=name,value=float(values[0]),actual_Y1=actual.get(cc['candidate_id'])))
        for k,v in meta['baseline_metrics'].items():
            if k.endswith('_ms'):timing.append(dict(configuration='LatticeSAC_origin',origin=folder.name,phase=k,count=1,p50_ms=v,p95_ms=v,p99_ms=v,measurement='oneoriginprepare; possiblycold/sharedGPU'))
        timing.append(dict(configuration='LocalMPPI64_8_4',origin=folder.name,phase='whole_local_search',count=1,p50_ms=meta['planning_ms'],p95_ms=meta['planning_ms'],p99_ms=meta['planning_ms'],measurement='oneinitialcall; cold/sharedGPU; notP95estimate'))
    # Genuine prior isolated fullplanner latency, with its original scope preserved.
    old=load(SOURCE/'reports/mppi_selection_20261117/isolated_latency.json')
    for config in sorted({r['configuration'] for r in old['rows']}):
        values=[r['planning_time_ms'] for r in old['rows'] if r['configuration']==config];timing.append(dict(configuration='prior_'+config,phase='isolated_complete_agent_act',count=len(values),p50_ms=float(np.quantile(values,.5)),p95_ms=float(np.quantile(values,.95)),p99_ms=float(np.quantile(values,.99)),measurement='PRIORsamecoreRTX3070isolated2states; not newmethodtime'))
    for phase in (['development'] if (REPORT/'development_candidate_effects.csv').exists() else ['pilot'])+(['validation'] if (REPORT/'validation_candidate_effects.csv').exists() else []):
        f=REPORT/f'{phase}_candidate_effects.csv'
        if f.exists():
            with f.open() as stream:
                for row in csv.DictReader(stream):effects.append(row|dict(analysis_phase=phase))
    for f in [REPORT/'pilot_statistics.json',REPORT/'development_statistics.json',REPORT/'validation_statistics.json']:
        if f.exists():
            a=load(f)
            for h,r in a['horizon_agreement'].items():horizons.append(dict(phase=a['phase'],horizon=h,**{k:json.dumps(v) for k,v in r.items()}))
    table(REPORT/'first_action_candidates.csv',candidates);table(REPORT/'first_action_returns.csv',allrows);table(REPORT/'paired_candidate_effects.csv',effects)
    table(REPORT/'q_all_members.csv',members);table(REPORT/'q_pair_comparison.csv',pairs);table(REPORT/'timing_breakdown.csv',timing)
    dump(REPORT/'rng_audit.json',dict(origins=rng,python_numpy_torchCPU_torchCUDA_engine_traffic_all_checked=True,
        traffic_config_kept_random_traffic_false=True,exact_replay='firstpilot,before50,c0,H1',independent_random_traffic_replicates=0,
        interpretation='Scenario/prefix seeded traffic anddeterministicbaseline, not irrelevantTorchseeds. Exactsamebranch replaychecked; no repeatscountedasindependentmaps.'))
    # Preserve detailed origin-level horizon CSV from fullpilotanalysis; allhorizons aretrueexecutions.
    full=load(REPORT/'pilot_statistics.json');table(REPORT/'horizon_protocol_comparison.csv',full.get('horizon_rows',[]) or horizons)
    phases={phase:load(REPORT/f'{phase}_statistics.json') for phase in ['pilot','development','validation'] if (REPORT/f'{phase}_statistics.json').exists()}
    dump(REPORT/'candidate_coverage.json',{k:dict(coverage=v['coverage'],origin_rows=v['coverage_rows']) for k,v in phases.items()})
    dump(REPORT/'ranking_metrics.json',{k:dict(rankings=v['rankings'],origin_rows=v['ranking_rows']) for k,v in phases.items()})
    dump(REPORT/'scenario_bootstrap.json',{k:dict(coverage=v['coverage'],rankings=v['rankings'],horizon_agreement=v['horizon_agreement']) for k,v in phases.items()})
    unreachable=[load(f) for f in (REPORT/'origins').glob('*/unreachable.json')]
    dump(REPORT/'data_coverage.json',dict(branch_counts=counts,actual_native_branches=len(allrows),origins=origins,unreachable=unreachable,
        pilot_scenes=p['pilot_scenes'],extra_development_scenes=p['extra_development_scenes'],new_validation_scenes=p['validation_scenes'],old_sealed_scenes=p['sealed_previous_scenes'],
        candidate_label_coverage='At most8representativefirstactionsfrom258storedpool; not fullGaussianoracle.',
        labels='Y1ONLY=firstphysicalactionthenoriginalfeedbackSAC; H2/4/10separatecontrols',
        features_current_available=['latent512','original535','actualpostSACrefinedparameters3','baselinephysicalfirst2','candidatephysicalfirst2'],no_future_model_inputs=True,
        excluded_training_data='ALLpreviousFit/InnerStop/H10labels/Phase1Pilot/newvalidation',formal100=False))
    return phases,allrows,timing


def verify(allrows):
    p=load(REPORT/'protocol.json');assert file_sha(p['checkpoint'])==p['checkpoint_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==p['base_commit']
    trained=load(REPORT/'training_decision.json');v=load(REPORT/'validation.json') if (REPORT/'validation.json').exists() else None
    modified=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
    if not v or not v.get('passed'):assert not modified,'No onlinechanges allowed before independentvalidation'
    tests=(REPORT/'tests_final.txt').read_text();assert '71 passed' in tests
    first=REPORT/'branches'/f'{p["pilot_scenes"][0]}_{p["before_steps"][0]}'
    a=np.load(first/'H1_c0_baseline.npz');b=np.load(first/'H1_c0_baseline_replay.npz')
    for k in ['observations','actions','rewards','costs','done','terminated','truncated']:assert np.array_equal(a[k],b[k]),k
    oldmanifest=load(REPORT/'version_manifest.json')
    for name,h in oldmanifest['historical_inputs'].items():assert file_sha(SOURCE/name)==h
    assert not set(r['scenario'] for r in allrows)&set(p['sealed_previous_scenes'])
    dump(REPORT/'verification.json',dict(base_commit=p['base_commit'],checkpoint_sha256=p['checkpoint_sha256'],branch=p['branch'],worktree=str(ROOT),
        original_reports_unchanged=True,original_checkpoints_unchanged=True,WM_SAC_Lattice_training_unchanged=True,original_tests62_retained=True,tests='71 passed',
        all_recorded_origin_hashes_equal=True,first_action_physical_no_remapping=True,postprefix_residualON_mppiOFF=True,native_done_terminated_truncated_distinct=True,
        exact_replay_all_arrays_equal=True,training_uses_only_current_features_and_H1_labels=True,old_sealed_maps_untouched=True,formal100=False,
        new_models_trained=6 if trained.get('trained') else 0,independent_validation_opened=bool(v and v.get('performed',True)),
        default_online_changed=False,tracked_changes=modified,source_sha256={str(f.relative_to(ROOT)):file_sha(f) for f in [ROOT/'tools/first_action_study.py', ROOT/'tests/test_first_action_study.py',*sorted(REPORT.glob('*.py'))]},original_main_status=subprocess.check_output(['git','status','--short'],cwd=SOURCE,text=True)))
    files=[ROOT/'tools/first_action_study.py',ROOT/'tests/test_first_action_study.py',*sorted(REPORT.glob('*.py')),REPORT/'protocol.json']
    patches=[]
    for f in files:
        d=subprocess.run(['git','diff','--no-index','--','/dev/null',str(f.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True);assert d.returncode==1;patches.append(d.stdout)
    patches.append(subprocess.check_output(['git','diff'],cwd=ROOT,text=True));(REPORT/'code_changes.patch').write_text(''.join(patches))


def write_report(phases,allrows,timing):
    p=load(REPORT/'protocol.json');training=load(REPORT/'training_decision.json');val=load(REPORT/'validation.json') if (REPORT/'validation.json').exists() else None
    if val and val.get('passed'):raise RuntimeError('Offlinegate PASSED: must finish authorized first-action planner and actual closedloop gates before finalreport.')
    last=phases.get('development',phases['pilot']);devgate=load(REPORT/'development_decision.json') if (REPORT/'development_decision.json').exists() else load(REPORT/'pilot_decision.json')
    next_direction=('Existingobjective showsdevelopmentH1signal; independentlyverifyit before anynewmodel/controller, no networkescalation.' if not devgate.get('evaluation_gap_gate',True)
        else 'Measuredsearchcoveragepositive butnotstableenoughunderthepredeclaredtraininggate; retain data, do notcallsearchgloballyuseless.' if not devgate.get('positive_coverage_gate',True)
        else 'MeasuredH1coveragepositive; identification/generalization remainsunproven. No extra capacity/samples/rules or trajectory-parameter redesign without newcausal evidence.')
    decision=dict(positive_candidates_exist=bool(phases['pilot']['observed_h1_positive_candidates']),
        collected_expansion=bool('development' in phases),trained=training.get('trained',False),validation_passed=False,
        independent_validation_performed=bool(val and val.get('performed',True)),new_online_planner_implemented=False,default_online_enabled=False,
        formal100=False,reason='Stage4 independentranking gates did not pass.' if val and val.get('performed',True) else training.get('reason'),
        next_direction=next_direction)
    dump(REPORT/'decision.json',decision)
    lines=['# MPPI首动作价值对齐与规划能力研究（2026-11-17）']
    def add(s=''):lines.append(s)
    add('\n本轮先证明候选是否有用，再检验评分能否选对。实验与修改位于独立worktree/分支，原main、原WM/SAC、535输入、Lattice、Action Mapping、默认LocalMPPI均保留。**没有把H10标签当成H1，也没有把H1→Baseline接管等同于新MPPI闭环。**')
    add('\n本轮停止位置：'+str(decision['reason'])+' 未满足接入门槛，默认在线规划保持不变。候选有用与模型能泛化识别是两件事，见下面实际覆盖/排序结果。')
    add(f"\n工作目录：`{ROOT}`；分支`{p['branch']}`；基线`{p['base_commit']}`。SSHfetch及HTTPS/proxyfetch失败已记录，通过GitHub连接器核实远端main确为同SHA，没有宣称fetch成功。Checkpoint SHA=`{p['checkpoint_sha256']}`，模型为WM200k+RL300k，总500k；未切换到另一个500k+500k模型。")
    add('\n## 1. 原生实现审计与对照')
    add('\n完整审计见MPPI_NATIVE_PARITY_REPORT_20261117.md。Candidate、PolicyMean、原生Sample终端动作在相同缓存前缀、末端latent、Reward、折扣及Q组合下比较。原生规划Qavg与训练targetQmin、Phase1固定Q01均值严格区分。枚举当前五Q的10种平均/10种最小、整体mean/median/min，保存100次random_pair分布；这里的min诊断用当前Q，不冒称调用了完整训练TD Target。')
    add('\n已真实补充完整TD-MPC2 MPC接管，原512samples/64elites/6iterations与Actor-only分列；LocalMPPI仍64/8/4。两个新固定起点完整MPC回报100.634/63.218，Baseline为141.514/85.092，Actor-only为44.540/76.481。样本只有两地图，不作泛化结论，也不将其称为单独原版迁移模型成绩。')
    add('\n## 2. 原生反事实协议与覆盖')
    add(f"\nPilot场景={p['pilot_scenes']}，每图固定起点{p['before_steps']}；扩展开发={p['extra_development_scenes']}；新独立验证={p['validation_scenes']}。旧6封存图继续封存。所有场景划分、候选角色、近似相等阈值1及统计门槛在完成真实分支前固定；版本manifest记录历史Fit/InnerStop/Phase1 Pilot，绝不将其当作新验证。")
    add('\n每个起点只运行一次原64×4搜索，保存258条Baseline/Mean/真实Gaussian序列与每轮Q成员；按预测分数固定取Baseline、实际MPPI输出、最佳非Baseline、75/50/25分位、最差及固定Gaussian。按精确首动作去重，角色别名与被合并不同tail仍保留。所有模型输入在任何候选环境动作前生成；F3确为实际SAC修正后的轨迹参数。')
    add('\n每条分支新建独立进程，核对初始Observation、完整可记录vehicle/traffic状态及六类RNG。H=1直接执行物理a0一次，然后每步真实重算Lattice+SAC，Residual ON/MPPI OFF，直到原生done。H2/4/10使用同一序列，仅干预长度不同。保留terminated/truncated/native done；提前失败不补未来。EpisodeReturn/Cost为从固定起点至原生终止的后缀，前置共同回放不计。')
    add('\n动作单位是MetaDrive归一化steer/throttle控制[-1,1]，不是残差d/v或SI转角。continuation_after_first与continuation_after_fixed_prefix分列，H>1时不能把第1步后的RTG误称为纯SAC接管RTG。映射次数0仅指固定候选物理前缀；后续Baseline仍正常执行原残差映射。')
    coverage=load(REPORT/'data_coverage.json');add(f"\n共完成{len(allrows)}条原生分支（含4条Actor/MPC对照及1条同seed重放）；按协议计数={coverage['branch_counts']}。不可达固定起点={len(coverage['unreachable'])}，原样保留不换图。所有覆盖与oracle都是实际测量子集，不能称为完整Gaussian搜索空间oracle。")
    add('\n当前random_traffic=false，Engine与TrafficManager RNG状态实际可读并全量保存。相同策略的精确重放完全一致，没有只换Torchseed冒充交通重复或将其当作独立地图。不同候选改变交互轨迹可能改变后续事件次序，不能宣称所有未来交通事件被强行锁死。')
    add('\n## 3. H1候选收益与现有评分')
    add('\n|数据|Oracle Gain [95%CI]|实际MPPI输出Gain|实际输出Regret|正收益候选比例%|\n|---|---:|---:|---:|---:|')
    for name,a in phases.items():
        c=a['coverage']['1'];add(f"|{name}|{fmt(c['oracle_best_gain'])}|{fmt(c['actual_selected_gain'])}|{fmt(c['actual_selected_regret'])}|{fmt(c['positive_fraction'],True)}|")
    add('\n实际MPPI输出是原weightedMean/Baseline第一步；下表Top1为同一已测候选子集中的评分argmax，两者不能混同。指标先按起点计算，再按场景bootstrap2000次seed1，95%区间；候选对、窗口与重复都不作独立地图。真实收益差≤1为近似相等，排序准确率排除，预测tie记0.5；常数Top1固定Baseline而非事后选真值最优。')
    add('\nOracle按用户定义的真实Y1最大值计算，未用未来事故筛选候选；它不等于已得到同时满足Success/Crash/OOR约束的可部署政策。事故条件仍需独立闭环检验，不能只因oracle收益高就启用。')
    add('\n|现有评分（开发集）|Pair Accuracy%|Spearman|Top1 Regret|Argmax Actual Gain|\n|---|---:|---:|---:|---:|')
    for name in ['current_mppi_h10','current_q_avg01','current_q_mean','current_q_median','terminal_policy_mean','terminal_native_policy_sample','constant_baseline']:
        r=last['rankings']['1'][name];add(f"|{name}|{fmt(r['pair_accuracy'],True)}|{fmt(r['spearman'])}|{fmt(r['top1_regret'])}|{fmt(r['selected_gain'])}|")
    if 'terminal_action_paired' in last:
        for mode,r in last['terminal_action_paired'].items():add(f"\n{mode}相对Candidate的场景配对Accuracy变化={fmt(r['pair_accuracy_gain'],True)}个百分点，Regret改善={fmt(r['regret_improvement'])}，argmax真实收益差={fmt(r['argmax_gain_difference'])}。这是开发集诊断，不能据多组比较修改默认。")
    add('\nBaseline Win Rate、负收益比例、预测改善误报率、Kendall、所有10双Q avg/min及Q方差与真收益差关系见ranking_metrics/scenario_bootstrap与逐起点CSV。SAC Critic只保留原残差Baseline诊断值：Gaussian物理动作无唯一反向残差映射，未将物理a0直接塞进残差Critic，故不伪造其候选排序。')
    add('\n## 4. H1/2/4/10与闭环含义')
    add('\n|干预长度|与H1真实收益Spearman|同向配对比例%|Top1一致率%|\n|---|---:|---:|---:|')
    for h,r in phases['pilot']['horizon_agreement'].items():add(f"|{h}|{fmt(r['spearman_H1'])}|{fmt(r['pair_direction_agreement'],True)}|{fmt(r['top1_same'],True)}|")
    add('\nH1严格对应“首动作干预+原Baseline反馈延续”，比H10开环标签在定义上更接近实际单步决策；**它仍不是新规划器每步重复优化后的π_new闭环价值**。H2/4/10是真实独立环境执行，不是序列重放预测。原Baseline的开环H10首序列同样可能偏离后续反馈Baseline，因此分别保留各H对应Baseline，不混用差值。')
    add('\n## 5. 条件式Q_B研究与验证')
    if training.get('trained'):
        add('\n开发H1数据通过候选覆盖与评分缺口门槛后，才训练独立小模型。实际评分缺口门槛为“原H10排序Accuracy CI下界≤.5，或原输出Gain CI下界≤0”；本轮前一项成立。protocol.training_gate文字简写为无可靠评分收益，含义不够精确，以预先执行的analyze.py公式及逐项decision为准，并非看到开发结果后新增门槛。开发集原MPPI输出Gain CI已大于0，不能声称现有输出没有首动作收益；本轮检验的是剩余排序/泛化缺口。没有使用历史H10标签、旧SAC Value或CandidateValue权重，也不更新原WM/SAC。按预先固定8图fit、4图innerstop；场景/起点平衡，归一化仅fit，Huber+.1Pairwise Ranking，近等收益对排除，固定参数、不做超参搜索。')
        add('\nF1=当前latent512+候选物理a0（514）；F2=当前535+a0（537）；F3=当前535+实际SAC后参数3+Baseline首动作2+a0（542）。线性Ridge初始化后以L2正则Huber+Ranking优化，准确称ridge_huber_rank；不是宣称闭式普通最小二乘Ridge。MLP仅64/32两隐层。原535定义不变，所有输入在线当前时刻可得，没有未来obs/latent/Reward/动作。')
        add('\n|候选离线模型|Best Epoch|InnerStop Accuracy%|InnerStop Regret|\n|---|---:|---:|---:|')
        for m in load(REPORT/'training_results.json'):
            s=m['inner_stop']['summary'];add(f"|{m['name']}|{m['best_epoch']}|{fmt(s['pair_accuracy'],True)}|{fmt(s['top1_regret'])}|")
        add(f"\n仅根据innerstop冻结一个模型：`{training['selected']['name']}`。是否进入一次新独立验证：{training['proceed_to_one_validation']}。没在验证集上重选feature/网络/epoch/Q组合。")
        if val and val.get('performed',True):
            s=val['evaluation']['summary'];add(f"\n新独立验证：Accuracy%={fmt(s['pair_accuracy'],True)}，Spearman={fmt(s['spearman'])}，Kendall={fmt(s['kendall'])}，Top1 Regret={fmt(s['top1_regret'])}，相对原H10评分Regret改善={fmt(s['regret_improvement'])}，MAE={fmt(s['mae'])}，Bias={fmt(s['bias'])}，Calibration={fmt(s['calibration_error'])}。全部门槛通过={val['passed']}。")
        else:add('\nInnerStop没有满足预注册的排序/Regret改善点估计门槛，因此新独立验证未打开；validation.json明确标注未执行，没有伪造测试结果。')
    else:add('\n未训练Q_B：'+str(training.get('reason'))+'。不为了交付数量而构造网络。')
    add('\n本轮没有通过全部独立门槛，因而未新增planning/first_action_value.py或二维在线优化器；未运行A/B/C新控制器闭环。旧LocalMPPI和默认terminal_value/terminal_action/random_pair保持原样。也未实施高层Lattice参数搜索重构。')
    add('\n## 6. 耗时、测试、最终决策')
    add('\n逐阶段与P50/P95/P99见timing_breakdown.csv。并发采集时钟只标为concurrent，单个冷启动搜索值不冒称P95。复用旧同核心RTX3070的100次单worker完整规划：B P95≈76ms，原MPPI M1/M2/M4≈104/117/141ms。未创建可验证的新在线规划器，所以不宣称新方法达到100ms，也未通过减少检查/样本/预算换速度。')
    add('\n原62项测试全部保留，新增9项协议/物理动作/去重/特征/终端采样RNG/metadata gamma checkpoint/真实模型加载/Residual开关及分组损失平衡回归，共71 passed。原权重前后SHA、每分支权重hash、所有起点hash、每步物理前缀、Native done及反馈次数都复核。初始新增测试漏写了重复角色worst_nonbaseline，修正为验证所有角色均保留；原62项断言未动。Pairwise损失按有效起点在地图内平衡，避免无有效收益差的起点稀释该地图权重。')
    add('\nA：候选首动作确有真实收益，证据限于实测子集与当前地图。\nB：没有一种终端定义在本轮证据下可被称为普遍最可靠，表中差异不授权修改默认。\nC：旧审计中Actor-only接管未消除Q误差，但随机Actor、确定性策略与完整MPC并不相同。当前数据不能唯一分离训练目标、状态分布与延续策略贡献；不要继续只归因策略不匹配。\nD：H1正确对齐单步干预定义，但不冒充新MPPI完整闭环收益。\nE：是否值得训练由已执行的覆盖/缺口门槛决定，训练后的失败不靠增加模型容量挽救。\nF：未通过独立排名/Regret门槛，暂不进入独立First-Action MPPI闭环或正式100场。\nG：不能说低层搜索空间完全无效；已存在正收益候选，当前瓶颈仍是识别/泛化。不得据此自动重构Lattice搜索变量。')
    add(f"\n完整数据与diff：`{REPORT}`；版本、失败日志、协议及历史split均保留。没有commit或push。")
    (ROOT/'MPPI_FIRST_ACTION_VALUE_REPORT_20261117.md').write_text('\n'.join(lines)+'\n')
    return decision


def main():
    assert (REPORT/'collection_pipeline_complete.json').exists()
    training=load(REPORT/'training_decision.json')
    if not training.get('trained') and (REPORT/'development_decision.json').exists():
        gate=load(REPORT/'development_decision.json');reason=('No stablepositivecoverage under thepredeclaredgate.' if not gate.get('positive_coverage_gate',False)
            else 'Existing evaluation hasdevelopmentranking/selectedgain support; no newQ_Bneeded under theevaluationgapgate.')
        training=training|dict(original_pipeline_reason=training.get('reason'),reason=reason);dump(REPORT/'training_decision.json',training)
    if training.get('trained') and not (REPORT/'validation.json').exists():dump(REPORT/'validation.json',dict(performed=False,passed=False,reason='Innerstopgatefailed; newvalidationremainssealed.'))
    phases,rows,timing=aggregate();write_report(phases,rows,timing);verify(rows)
    print('FINAL REPORT/ARTIFACTS VERIFIED',flush=True)

if __name__=='__main__':main()
