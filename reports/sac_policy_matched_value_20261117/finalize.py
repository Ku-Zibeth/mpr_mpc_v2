"""Paired inference, repeat-noise audit and a non-deployment Phase1 report."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
sys.path.insert(0,str(R));from analyze_values import load,dump,table,rank
import numpy as np
from tools.mppi_failure_attribution import scene_bootstrap
from tools.policy_value_diagnosis import discounted_return


def read_csv(path):
    rows=[]
    strings={'origin','origin_id','candidate_id','candidate','split','source','observation_source','policy','end_reason','reward_definition','td_first_action','note'}
    for r in csv.DictReader(Path(path).open()):
        for k,v in r.items():
            if k in strings:continue
            if v=='':r[k]=None
            elif v in ['True','False']:r[k]=v=='True'
            else:r[k]=float(v)
        rows.append(r)
    return rows


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(p):
    assert sha(p['checkpoint'])==p['checkpoint_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==p['base_commit']
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()
    assert '62 passed' in (R/'tests.txt').read_text()
    assert not list(R.glob('*.pt')) and not list(R.glob('*.pth'))
    expected_context=load(ROOT/'reports/sac_value_20261117/dataset_manifest.json')['context'];paired=[];native=[]
    for folder in sorted((R/'fresh').iterdir()):
        meta=load(folder/'terminal_metadata.json');assert meta['policy_context_sha256']==expected_context
        for seed in p['repeat_seeds']:
            a=load(folder/f'lattice_sac_r{seed}_start.json');b=load(folder/f'td_actor_r{seed}_start.json');assert a==b
            paired.append(dict(case=folder.name,repeat=seed,initial_observation_state_traffic_recorded_RNG_equal=True))
        reference_hashes=None
        for f in folder.glob('*_verification.json'):
            v=load(f);assert v['weights_unchanged'] and v['terminal_state_observation_traffic_RNG_paired']
            if reference_hashes is None:reference_hashes=v['original_hashes']
            else:assert reference_hashes==v['original_hashes']
        for f in folder.glob('*.npz'):
            if f.name=='terminal.npz':continue
            a=np.load(f);assert len(a['observations'])==len(a['actions'])+1 and a['done'][-1] and not a['done'][:-1].any()
            native.append(dict(path=str(f.relative_to(ROOT)),sha256=sha(f),steps=len(a['actions'])))
    assert len(native)==33 and len(paired)==16
    first=R/'fresh'/f"{p['fresh_scenarios'][0]}_baseline";a=np.load(first/'lattice_sac_r1.npz');b=np.load(first/'lattice_sac_r1_replay.npz')
    repeat={k:bool(np.array_equal(a[k],b[k])) for k in ['observations','actions','rewards','costs','done','physical']};assert all(repeat.values()),repeat
    dump('rollout_pair_verification.json',dict(rows=paired,exact_same_seed_replay=repeat,native_artifacts=native))
    dump('verification.json',dict(checkpoint_sha256=p['checkpoint_sha256'],checkpoint_unchanged=True,head=p['base_commit'],existing_tracked_files_unchanged=True,
        native_suffixes=33,fresh_scenes=4,fresh_terminal_states=8,independent_paired_initial_conditions=True,exact_replay_equal=True,
        original_sac_and_wm_weights_unchanged=True,inputs535_unchanged=True,reward_gamma_done_unchanged=True,tests='62 passed',
        trained_networks=0,new_value_heads=0,online_score_changes=False,default_q_or_sampling_changes=False,baseline_path_changes=False,
        sealed_six_map_set_untouched=True,formal100=False,
        confounds_explicit=['SAC .99 soft target vsWM .995 rawlabels','stochastic training vsdeterministic SACeval','invalidresidual penalty','partial535representation','nativeActor first action differs between takeovers','only two repeats; puretrafficnoise not isolated'],
        failed_attempts='Config Path serialization and non-tensor TensorDict hash errors occurred before env.reset. Same-process reset later failed strict pairing after two SACsuffixes; retained outside scientific pairedset. Corrected collector to freshprocessperbranch, no policy change.'))
    patches=[]
    for path in [ROOT/'tools/policy_value_diagnosis.py',ROOT/'tests/test_policy_value_diagnosis.py',*sorted(R.glob('*.py'))]:
        r=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True);assert r.returncode==1;patches.append(r.stdout)
    (R/'code_changes.patch').write_text(''.join(patches))


def paired_analysis(p):
    rows=read_csv(R/'paired_rollout_results.csv');rows=[r for r in rows if not r['replay']];keys=['discounted_return','discounted_return_sac_gamma','episode_return','success','crash','out_of_road','route_completion','safety_cost']
    differences=[];variance=[]
    for scene in p['fresh_scenarios']:
        for candidate in p['candidates']:
            rr=[r for r in rows if r['scenario']==scene and r['candidate']==candidate]
            for seed in p['repeat_seeds']:
                a=next(r for r in rr if r['policy']=='lattice_sac' and r['repeat']==seed);b=next(r for r in rr if r['policy']=='td_actor' and r['repeat']==seed)
                differences.append(dict(scenario=scene,candidate=candidate,repeat=seed,**{k:b[k]-a[k] for k in keys}))
            for policy in p['takeover_policies']:
                a=[r for r in rr if r['policy']==policy];v=[r['discounted_return'] for r in a]
                variance.append(dict(scenario=scene,candidate=candidate,policy=policy,mean_return=float(np.mean(v)),sample_variance=float(np.var(v,ddof=1)),
                    repeat_difference=float(v[1]-v[0]),repeats=2,noise_kind='stochastic TDactor vsdeterministic SAC; internaltrafficRNG held fixed; NOT isolatedtrafficresampling'))
    for r in rows:r['scenario']=int(r['scenario'])
    summary={policy:scene_bootstrap([r for r in rows if r['policy']==policy],keys) for policy in p['takeover_policies']}
    table('paired_takeover_differences.csv',differences);table('repeat_rollout_variance.csv',variance)
    noise=[]
    for scene in p['fresh_scenarios']:
        for policy in p['takeover_policies']:
            gains=[]
            for seed in p['repeat_seeds']:
                g=next(r for r in rows if r['scenario']==scene and r['candidate']=='gaussian1' and r['policy']==policy and r['repeat']==seed)
                b=next(r for r in rows if r['scenario']==scene and r['candidate']=='baseline' and r['policy']==policy and r['repeat']==seed);gains.append(g['discounted_return']-b['discounted_return'])
            noise.append(dict(scenario=scene,policy=policy,mean_candidate_gain=float(np.mean(gains)),paired_gain_variance=float(np.var(gains,ddof=1)),
                same_direction_both_seeds=bool(gains[0]*gains[1]>0),gain_seed1=gains[0],gain_seed2=gains[1]))
    table('candidate_effect_vs_rollout_noise.csv',noise)
    return dict(by_policy=summary,td_actor_minus_lattice_sac=scene_bootstrap(differences,keys),variance=variance,
        candidate_effect_noise=noise,native_suffixes_excluding_replay=32,fresh_maps=4,
        scope='Each32suffix uses native done, fourmaps, two candidate prefixes andtwo suffixseeds. Not32 independentmaps. TDactor usesstochastic directactor and differentfirstphysical action, NOT TD-MPC2 MPC or firstaction-controlledtail intervention.')


def nested_noise_bootstrap(p):
    data=read_csv(R/'value_comparison.csv');fresh=[r for r in data if r['split']=='fresh_independent_pilot'];raw=read_csv(R/'paired_rollout_results.csv');raw=[r for r in raw if r['policy']=='lattice_sac' and not r['replay']]
    cache={}
    for scene in p['fresh_scenarios']:
        rr=sorted([r for r in fresh if r['scenario_id']==scene],key=lambda r:r['candidate_id']!='baseline');mat=[]
        for r in rr:mat.append([next(q['discounted_return'] for q in raw if q['scenario']==scene and q['candidate']==r['candidate_id'] and q['repeat']==seed) for seed in p['repeat_seeds']])
        cache[scene]=(np.asarray(mat),np.array([r['td_q_matched'] for r in rr]),np.array([r['q_min'] for r in rr]))
    rng=np.random.default_rng(1);accuracy=[];regret=[];empty=0
    for _ in range(2000):
        aa=[];gg=[]
        for scene in rng.choice(p['fresh_scenarios'],len(cache),replace=True):
            matrix,predA,predB=cache[int(scene)];repeat=rng.integers(0,2,size=2);y=matrix[:,repeat].mean(1)
            ia=int(np.argmax(predA));ib=int(np.argmax(predB));gg.append(float(y[ib]-y[ia]))
            if abs(y[1]-y[0])>1:
                def correct(v):
                    x=(v[1]-v[0])*(y[1]-y[0]);return 1. if x>0 else 0. if x<0 else .5
                aa.append(correct(predB)-correct(predA))
        if aa:accuracy.append(float(np.mean(aa)))
        else:empty+=1
        regret.append(float(np.mean(gg)))
    return dict(pair_accuracy_gain_ci95=np.quantile(accuracy,[.025,.975]).tolist() if accuracy else [None,None],
        regret_improvement_ci95=np.quantile(regret,[.025,.975]).tolist(),draws_without_effective_pair=empty,
        definition='Nested scene bootstrap then paired repeat-index resampling inside eachscene, candidates retain common repeat draws. Only2repeats; percentileCI exploratory, not unbiased stochastic-noise decomposition.')


def invalid_penalty_sensitivity():
    """Native labels unchanged; isolate known residual-training reward penalty offline."""
    rows=[r for r in read_csv(R/'value_comparison.csv') if r['split']=='fresh_independent_pilot'];stats=[]
    for r in rows:
        folder=R/'fresh'/f"{int(r['scenario_id'])}_{r['candidate_id']}";targets=[];counts=[]
        for seed in [1,2]:
            d=np.load(folder/f'lattice_sac_r{seed}.npz');counts.append(int(d['invalid_residual'].sum()))
            targets.append(discounted_return(d['rewards']-.2*d['invalid_residual'],.99,d['done']))
        r['diagnostic_penalized_gamma99']=float(np.mean(targets))
        stats.append(dict(scenario=int(r['scenario_id']),candidate=r['candidate_id'],penalized_return_gamma99=r['diagnostic_penalized_gamma99'],
            raw_return_gamma99=r['actual_continuation_sac_gamma'],discounted_invalid_penalty=r['actual_continuation_sac_gamma']-r['diagnostic_penalized_gamma99'],
            invalid_steps=counts[0],critic_prediction=r['q_min'],absolute_error=abs(r['q_min']-r['diagnostic_penalized_gamma99'])))
    table('invalid_penalty_sensitivity.csv',stats)
    result=rank(rows,'diagnostic_penalized_gamma99')
    result['scope']='OfflineactualSACdeterministictrajectory, rewardminus.2invalid penalty,gamma99. NOT originalenvironmentReturn, still lacks softfutureentropy/stochasticSAC expectation. Does not overwriteprimarylabels or modifyanyreward.'
    dump('training_reward_sensitivity.json',result)
    return result


def fmt(s,percent=False):
    if s['mean'] is None:return 'N/A'
    scale=100 if percent else 1
    return f"{s['mean']*scale:.2f} [{s['ci95'][0]*scale:.2f},{s['ci95'][1]*scale:.2f}]"


def report(p,ranking,paired,nested,coverage):
    audit=load(R/'critic_architecture_audit.json');a=ranking['primary'];fresh=a['by_split']['fresh_independent_pilot'];diff=a['paired']['fresh_independent_pilot'];secondary=ranking['secondary_gamma99']
    advantage=bool(diff['pair_accuracy_gain']['ci95'][0] is not None and diff['pair_accuracy_gain']['ci95'][0]>0 and diff['regret_improvement']['ci95'][0]>0 and nested['regret_improvement_ci95'][0]>0)
    decision=dict(existing_sac_critic_supported_on_pilot=advantage,online_deployment=False,formal100=False,new_value_head_training=False,
        reason='Phase1 only. Four freshmaps,2prefix candidates,2repeats; existing critics have irreducible objective mismatches. No validated predicted-terminal535 online interface. No newonlinepolicy or score.',
        ranking_gate=dict(pair_accuracy_gain=diff['pair_accuracy_gain'],regret_improvement=diff['regret_improvement'],nested_noise=nested))
    dump('decision.json',decision)
    lines=['# 策略匹配型终端价值验证：Phase1（2026-11-17）']
    def add(x=''):lines.append(x)
    add('\n结论：**现有Residual SAC Critic没有表现出比TD-MPC2 Q更好的真实候选排序，不能直接替换MPPI终端价值。** 它是残差参数动作下的soft Q，通过实际Lattice/轨迹再生/跟踪路径学习组合系统收益，不能等同于任意物理动作Q。其gamma=.99、熵项、残差无效惩罚与当前MPPI gamma=.995原始环境回报不一致。本轮没有修改任何原有网络、权重、输入、映射或在线评分。')
    add(f"\n本轮新增{coverage['fresh_terminal_states']}个真实末端状态，来自4个预先固定新场景；每状态两种接管策略各2个随机种子，32条有效原生后续轨迹，加1条同seed精确重放。另复用197条历史完整535末端记录。新场景ID={p['fresh_scenarios']}；不属于训练10000–11999、既有开发场景或封存6场景。它们是本轮独立的小样本验证，不因结果补图、选参或训练；不能称为充分泛化证明。")
    add('\n## 1. Residual SAC完整审计')
    add('\n- Actor：535→256→256→4，ReLU；输出2均值及2logStd，tanh后为归一化[d残差,v残差]。训练随机rsample，部署评测用tanh(mean)。\n- Reward Q1/Q2：独立537→256→256→1，输入concat(535状态,2残差)。另有独立Cost Q，不混成Reward Q。\n- Target Q1/Q2：分别保存、无梯度，Polyak tau=.05；next action来自**当前**随机Actor，无target actor。Target=`r_res +.99*(1-done)*(min(targetQ1,targetQ2)-alpha*logpi_next)`。\n- 状态顺序：latent512、路径12、coarse WMreturn1、可行性2、风险值/有效位8。含target_d/target_speed与路径摘要，**不含显式Lattice控制动作/完整序列/全部内部跟踪状态**。\n- Replay action存requested normalized residual；executed steer/throttle只在诊断字段。状态转移由这条残差经实际组合路径产生，这是合法的残差MDP语义，并非单凭数值不等就判定Bug。')
    alpha=audit['reward_critics']['alpha_checkpoint'];add(f"\nCheckpoint已有59000次SAC更新；alpha={alpha:.6f}。原始训练Replay未序列化，训练源码SHA未附：本审计能验证当前实现与Checkpoint配置，不能证明训练时每个样本的具体映射、无效比例或历史代码完全相同。")
    add('\n六项审计回答：1）Critic面向Lattice诱导环境中的当前随机Residual SAC soft策略。2）含Lattice路径摘要，不含显式完整基准控制。3）在535充分表征状态的前提下可近似组合策略收益，但不能断言完整Markov或评价任意MPPI动作。4）训练action是残差，真实执行action是steer/throttle，两者数值不同；Critic动作必须使用残差定义。5）不与当前WM原始回报/折扣完全兼容。6）Target使用当前随机SAC策略，但不等于deterministic评测的raw Return。')
    add('\n## 2. 统一离线数据和评价定义')
    add('\nvalue_comparison.csv逐状态记录场景/seed/起点/候选/H10、Actor残差、Q1/Q2/targetQ、原TD Q及真实RTG。完整向量在historical_terminal_dataset.npz和fresh_terminal_dataset.npz，按split及array_index连接，含predicted latent、actual observation/latent、original535、12Lattice特征、Actor残差与Qensemble；fresh另存完整coarse控制与原始Lattice特征。旧缓存缺少完整rawLattice状态，明确不补造。')
    add('\n主比较A：真实末端z与**实际Lattice+SAC首个物理执行动作**上的原TD Q，固定成员01均值；B：同一真实535及确定性SAC残差均值上的min(Q1,Q2)，其残差对应同一个物理动作。因而没有把未执行候选a_H或残差直接当物理动作。预测latent下TDQ、Actor mean Q及未执行a_H只作独立敏感性诊断，不混进主结果。C是SAC实际延续RTG oracle；D=0。')
    add('\n本轮MAE是绝对末端RTG误差；前轮8.46是候选相对Baseline的延续收益差误差，两者定义不同，不能直接横比。真实535和由真实末端规划得到的动作都只用于离线验证，没有成为MPPI在线未来输入。')
    add('\n所有标签使用真实info.tdmpc2_reward、原生done、H10后的后续轨迹。主标签统一gamma=.995；另对同一真实奖励重算gamma=.99作为敏感性对照，**不重标或缩放任何Critic参数**。现成两个Critic固有训练gamma不同，无法在不改权重的条件下同时严格兼容；这是被审计出的混杂，而非宣称已消除。TDQ只由原stochastic TD Actor bootstrap，并无SAC式soft Q entropy项。')
    add('\n## 3. 四组价值结果（新独立4场景）')
    add('\n所有排序先同起点候选内计算，再场景平均/bootstrap2000次seed1；两个起点候选、重复seed不当作地图。真实收益差≤1排除准确率计数，预测tie计0.5；Constant Top1固定原Baseline顺序，不用真实收益打破tie。Spearman/Kendall对常数预测未定义，记N/A。Calibration用预先固定值域bins，未拟合校准器。C的完美指标只是标签自比较，不能部署。')
    add('\n|模型|Pair Accuracy% [95%CI]|Spearman|Kendall|Top1 Regret|MAE|Calibration Error|\n|---|---:|---:|---:|---:|---:|---:|')
    for name in ['A_tdmpc2_q','B_sac_critic','C_actual_oracle','D_zero']:
        r=fresh[name];add(f"|{name}|{fmt(r['pair_accuracy'],True)}|{fmt(r['spearman'])}|{fmt(r['kendall'])}|{fmt(r['top1_regret'])}|{fmt(r['mae'])}|{fmt(r['calibration_error'])}|")
    add(f"\nB相对A的配对准确率改善={fmt(diff['pair_accuracy_gain'],True)}个百分点，Top1 Regret改善={fmt(diff['regret_improvement'])}（正值才改善）。同时考虑场景和两次重复标签噪声，Regret改善nested95%CI={nested['regret_improvement_ci95']}；Pair Accuracy改善nested95%CI={nested['pair_accuracy_gain_ci95']}。")
    add(f"\n两项排序门槛是否同时有统计支持：**{advantage}**。不能因为MAE下降就判定通过。独立4地图配对CI很粗，尤其有效收益差候选对不足时不应夸大确定性；数值与有效数量详见ranking_metrics.json。")
    add('\n主排序只有3个场景的候选收益差>1，SAC为0/3、TDQ为1/3；第4场景属于近似相等。经验bootstrap对0/3给出0–0的退化区间，**不表示总体准确率确定为0**。为展示小样本不确定性，SAC的三场景Bernoulli精确95%区间约[0%,70.8%]；这不改变“没有优势证据”的结论。')
    add('\n历史数据与同奖励gamma=.99敏感性（按split严格分开；历史不能冒称独立验证）：')
    for label,case in [('gamma995',a),('gamma99',secondary)]:
        for split in case['by_split']:
            for name in ['A_tdmpc2_q','B_sac_critic']:
                r=case['by_split'][split][name];add(f"\n- {split}/{label}/{name}：准确率 {fmt(r['pair_accuracy'],True)}%，Regret {fmt(r['top1_regret'])}，MAE {fmt(r['mae'])}。")
    sensitivity=load(R/'training_reward_sensitivity.json')['by_split']['fresh_independent_pilot']['B_sac_critic']
    add(f"\n进一步仅在新真实轨迹上，离线减去原训练定义中的0.2×无效残差标志、使用gamma=.99：SAC Critic准确率 {fmt(sensitivity['pair_accuracy'],True)}%，Regret {fmt(sensitivity['top1_regret'])}，MAE {fmt(sensitivity['mae'])}。这没有恢复排序优势。该辅助标签明确不替换环境rawReturn，仍缺随机SACsoft目标的熵/期望，不能称为已彻底消除训练目标混杂。")
    add('\n## 4. 策略接管与随机波动')
    add('\n每条分支均用独立进程重建同一起点、固定H10动作前缀，并核对末端观测、ego/交通可记录状态及RNG。SAC继续原确定性评测路径；TD Actor直接执行原model.pi(z)[0]随机动作，不运行TD-MPC2 MPC，也不宣称是确定性Actor mean。两者首动作可能不同，因此策略接管对照包含首动作及后续策略的共同影响，不能唯一分离“相同首动作下的tail不匹配”。')
    add('\n这里的TD Actor和TD Q均来自同一个WM200k+RL300k Checkpoint，绝非另一个单独训练/迁移的原版TD-MPC2完整MPC模型。该Checkpoint配置是Lattice采集StageA、随后冻结WM/Q；内置Actor的表现不能代表原版TD-MPC2规划器表现，也不能据此宣称所有TD-MPC2 Actor都失效。')
    add('\n|接管策略|Discounted Return|Episode Return|Success%|Collision%|OOR%|Route Completion%|Safety Cost|\n|---|---:|---:|---:|---:|---:|---:|---:|')
    for name,r in paired['by_policy'].items():add(f"|{name}|{fmt(r['discounted_return'])}|{fmt(r['episode_return'])}|{fmt(r['success'],True)}|{fmt(r['crash'],True)}|{fmt(r['out_of_road'],True)}|{fmt(r['route_completion'],True)}|{fmt(r['safety_cost'])}|")
    native=read_csv(R/'td_actor_value_consistency.csv');own_pred=np.mean([r['td_q_at_own_first_action'] for r in native]);own_actual=np.mean([r['actual_td_actor_continuation'] for r in native]);own_mae=np.mean([r['absolute_error'] for r in native])
    add(f"\nTD Actor−SAC的配对折扣回报差={fmt(paired['td_actor_minus_lattice_sac']['discounted_return'])}。更关键的是，TDQ在其自身真实Actor首动作上平均预测{own_pred:.2f}，真实Actor RTG仅{own_actual:.2f}，MAE={own_mae:.2f}。因此，**不能把TDQ失准全部解释成“换成SAC延续策略了”**：Q对原Actor接管也严重失准，仍有价值校准/泛化/数据覆盖问题。首动作不同的限制仍然存在，不能据此定量分配唯一因果贡献。")
    add('\n两个suffix seeds在**相同**末端状态上施加，交通内部RNG保留场景前缀状态。审计进一步确认：SAC状态中的coarse WMreturn使用完整Q ensemble mean，**不是MPPI的random_pair**。故SAC确定性评测没有这项随机性，8个末端的两次SAC RTG样本方差实际均为0；TD Actor存在采样波动。预注册protocol中的“coarse-Q随机性”设想已在protocol_implementation_notes.json更正，不改历史protocol或结果。重复并非独立交通重采样，不能把TDActor采样波动归为Critic或纯环境误差。repeat_rollout_variance.csv及candidate_effect_vs_rollout_noise.csv分别记录重复方差和候选效果。每末端只有2次，不足以识别纯环境噪声分布。另一次相同seed重放动作/观测/奖励/Cost/物理状态完全一致。')
    add('\n## 5. 最终A—F与下一阶段')
    add('\n- A：现有SAC Critic可近似残差诱导组合系统收益，但不是已验证的完整Lattice+SAC确定性raw V，也不是仅latent可调用的末端价值。\n- B：在本轮新场景中**未优于TDQ**，SAC Critic Top1 Regret约5.12，TDQ约2.10；已有历史数据亦无优势。\n- C：**没有正向统计支持**；配对Regret改善CI包含0且点估计为负。SAC gamma=.99下MAE更低，但排序和Regret仍未改善。\n- D：能确认训练目标和策略定义不一致，但仅靠“策略匹配”不足以解释：TDQ对自身Actor也失准。优先审计已有Q训练目标、泛化及状态—动作覆盖；535状态遗漏、gamma/entropy及Replay分布各自贡献尚不能唯一分离。\n- E：当前数据支持否定“直接使用现成SAC Critic”的方案，**不足以支持马上再训练新的策略匹配价值模型**；需先明确目标并定位真实状态下Q为什么失准。\n- F：**暂不值得推进MPPI在线动作选择或正式100场**；保留Lattice+SAC Baseline，不改在线架构。')
    add('\n## 6. 交付与限制')
    add('\n新增离线tools/policy_value_diagnosis.py、3项针对性测试及本目录数据采集/分析脚本。原59项相关测试保留，总计**62 passed**。原WM/SAC/535/Action Mapping/Lattice/采样预算/默认Q聚合/在线选择/环境定义全部不改，Checkpoint前后SHA及每条分支权重hash复核通过。没有新价值网络训练、没有100场、没有commit/push。')
    add('\n采集工具的初始问题及日志保留：Path配置序列化、TensorDict非tensor元数据hash曾在reset前退出；之后同进程reset在2条SAC后续轨迹完成后被严格起点比较拒绝。两个未配对pilot保留在failed_same_process_reset，未混入33条有效分支；改为freshprocess，未改任何行为策略/场景/参数。最终有效轨迹和起点配对检查见rollout_pair_verification.json；真实执行总数另有这2条pilot，不隐瞒失败尝试。')
    add('\n数据文件：critic_architecture_audit.json、value_comparison.csv、continuation_returns.csv、ranking_metrics.json、paired_rollout_results.csv、scenario_bootstrap.json、data_coverage.json、verification.json。结构化数组与CSV通过split/array_index连接；所有旧报告、日志和权重保留。')
    (ROOT/'SAC_POLICY_MATCHED_VALUE_REPORT_20261117.md').write_text('\n'.join(lines)+'\n')
    return decision


def main():
    p=load(R/'protocol.json');ranking=load(R/'ranking_metrics.json');paired=paired_analysis(p);nested=nested_noise_bootstrap(p)
    invalid_penalty_sensitivity()
    audit=load(R/'critic_architecture_audit.json');audit['state535']['coarse_wm_q_aggregation']='ensemble_mean in evaluate_trajectory_consequence, not MPPIrandom_pair';dump('critic_architecture_audit.json',audit)
    dump('protocol_implementation_notes.json',dict(
        original_protocol_preserved=True,
        correction='Protocol variance sentence hypothesized random coarseQ features. Source audit shows coarseWM fullensemble mean; random_pair belongs toMPPI only. MPPIOFF + deterministicSAC has zero observed repeatvariance atall8endpoints.',
        independent_simulator_process_per_suffix=True,
        failed_reset_handling='Same-process reset did not reproduce origin. Two completed unpairedSACpilot trajectories retained, excludedfromscientific33suffix dataset; allpairedbranches reruninfreshprocess. No policy/config/scenario/weights change.',
        return_mae_definition='Thisphase absolute continuationRTG MAE; prior8.46 was candidate-minus-baseline continuation difference MAE. Not directly comparable.',
        terminal_action='TDQ(REALz, actualSACfirstphysicalaction) vsSACQ(REAL535, residualmean mapping to thatsameaction); objectives inherently differ. NativeTDtakeover also evaluatesQ atownactualfirstaction separately.'))
    # Exact Bernoulli intervals only for fresh one-pair-per-map outcomes.
    from scipy.stats import beta
    fresh_origin=ranking['primary']['origin_rows'];small={}
    for name in ['A_tdmpc2_q','B_sac_critic']:
        rr=[r for r in fresh_origin if r['split']=='fresh_independent_pilot' and r['model']==name and r['eligible_pairs']>0]
        successes=sum(r['pair_accuracy']==1 for r in rr);n=len(rr)
        small[name]=dict(effective_scenes=n,correct=successes,clopper_pearson95=[float(beta.ppf(.025,successes,n-successes+1)) if successes else 0.,float(beta.ppf(.975,successes+1,n-successes)) if successes<n else 1.],
            note='Only fresh onecandidatepair/map, supplementary small-n uncertainty. Degenerate0-0 empiricalbootstrap is notpopulationcertainty.')
    dump('small_sample_uncertainty.json',small)
    coverage=load(R/'historical_coverage.json')|dict(historical_terminal_states=197,fresh_terminal_states=8,fresh_scenarios=p['fresh_scenarios'],fresh_maps=4,
        paired_native_suffixes=32,exact_replay_suffixes=1,unpaired_pilot_suffixes_preserved=2,
        new_training_samples=0,selection_or_tuning_on_fresh=False,
        missing=['historicalfull rawLattice/controllerstate','serializedoriginaltrainingReplay','originaltrainingcodeSHA','independenttraffic-only repeatednoise','predictedfull535 onlineendpoint features'],
        label_source='RealnativeLattice+SAC RTG; no imagined labels',arrays={'historical':'historical_terminal_dataset.npz','fresh':'fresh_terminal_dataset.npz'})
    dump('data_coverage.json',coverage);dump('scenario_bootstrap.json',dict(value_comparison=ranking['primary']['paired'],
        secondary_gamma99=ranking['secondary_gamma99']['paired'],policy_takeover=paired,nested_repeat_noise=nested,small_sample_exact_intervals=small,
        td_native_value_consistency=scene_bootstrap(read_csv(R/'td_actor_value_consistency.csv'),['td_q_at_own_first_action','actual_td_actor_continuation','absolute_error'])))
    report(p,ranking,paired,nested,coverage);verify(p);print('Phase1 report and all checks complete',flush=True)

if __name__=='__main__':main()
