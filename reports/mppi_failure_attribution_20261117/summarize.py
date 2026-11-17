"""Scene-balanced reporting from existing real outcomes; no online score changes."""
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from scipy.stats import spearmanr
from tools.mppi_failure_attribution import ranking_metrics,scene_bootstrap

def load(p):return json.loads(Path(p).read_text())
def dump(name,v):(R/name).write_text(json.dumps(v,indent=2,allow_nan=False))
def read(name):
    out=[]
    for r in csv.DictReader((R/name).open()):
        for k,v in r.items():
            if v=='':r[k]=None
            elif k in {'origin','cohort','protocol','candidate','source','configuration','score','physical_prediction_source'}:continue
            elif v in ['True','False']:r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except (ValueError,TypeError):pass
        out.append(r)
    return out
def table(name,rows):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def stat(rows,keys):return scene_bootstrap(rows,keys)


def main():
    assert (R/'analysis_complete.json').exists()
    returns=read('candidate_returns.csv');coverage=[];components=[];realq=[];qrelation=[];meancomparisons=[]
    for cohort in ['fit_innerstop','fixed_rounds']:
        for origin in sorted({r['origin'] for r in returns if r['cohort']==cohort}):
            protocols=sorted({r['protocol'] for r in returns if r['cohort']==cohort and r['origin']==origin})
            for protocol in protocols:
                rr=[r for r in returns if r['cohort']==cohort and r['origin']==origin and r['protocol']==protocol]
                base=next(r for r in rr if r['candidate'] in ['B','baseline'])
                y=np.array([r['actual_return'] for r in rr]);b=rr.index(base);p=np.array([r['J00'] if r.get('J00') is not None else r['predicted_prefix'] for r in rr])
                # Coverage includes native early failures, even if oracle H-state is absent.
                if cohort=='fixed_rounds':pool=261;denominator='1 baseline +256 sampled identities +4 updated Means; measured Top1s are not allElites'
                else:
                    audit=ROOT/'reports/candidate_ranking_20261117/data'/origin/'audit.npz'
                    pool=258 if audit.exists() else 3;denominator='legacy3 fixed interventions' if pool==3 else '1Baseline+1FinalMean+256 sampled identities'
                row=dict(cohort=cohort,scenario=int(base['scenario']),origin=origin,protocol=protocol,
                    measured_candidates=len(rr),saved_pool_size=pool,label_coverage=len(rr)/pool,
                    coverage_family='legacy_fixed3_no_full_gaussian_pool' if cohort=='fit_innerstop' and pool==3 else 'saved_gaussian_search_pool',
                    pool_definition=denominator,positive_candidates=int(np.sum(y>y[b]+1)),negative_candidates=int(np.sum(y<y[b]-1)),
                    equal_candidates=int(np.sum(np.abs(y-y[b])<=1))-1,oracle_best_gain=float(y.max()-y[b]),
                    measured_subset_has_benefit=float(y.max()-y[b]>1),scope='Measured-subset lower bound; not full search-pool oracle')
                coverage.append(row)
                means=[r for r in rr if r['candidate'].startswith('mean')]
                for mean in means:
                    if mean.get('J00') is None:continue
                    gain=mean['actual_return']-base['actual_return'];pg=mean['J00']-base['J00']
                    meancomparisons.append(dict(scenario=int(base['scenario']),origin=origin,cohort=cohort,protocol=protocol,candidate=mean['candidate'],
                        actual_gain=gain,predicted_gain=pg,eligible=float(abs(gain)>1),baseline_mean_accuracy=float(pg*gain>0) if abs(gain)>1 else None))
                if cohort!='fit_innerstop':continue
                complete=[r for r in rr if r['full_horizon']]
                if base not in complete:continue
                qy=np.array([r['actual_continuation'] for r in complete]);qr=np.array([r['real_latent_terminal_q'] for r in complete])
                realq.append(dict(scenario=int(base['scenario']),origin=origin,**ranking_metrics(qr,qy)))
                nonbaseline=[r for r in complete if r is not base]
                qstd=[r['q_gain_std_across_pairs'] for r in nonbaseline]
                actual_abs=[abs(r['actual_return']-base['actual_return']) for r in nonbaseline]
                qrelation.append(dict(scenario=int(base['scenario']),origin=origin,
                    nonbaseline_candidates=len(nonbaseline),
                    q_gain_std_vs_abs_return_spearman=float(spearmanr(qstd,actual_abs).statistic) if len(nonbaseline)>=3 and np.ptp(qstd)>0 and np.ptp(actual_abs)>0 else None))
                for r in complete:
                    if r is base:continue
                    predprefix=r['predicted_prefix']-base['predicted_prefix'];actualprefix=r['actual_prefix']-base['actual_prefix']
                    predterm=.995**10*(r['predicted_terminal_q']-base['predicted_terminal_q'])
                    actualterm=.995**10*(r['actual_continuation']-base['actual_continuation'])
                    e1=predprefix-actualprefix;e2=predterm-actualterm
                    residual=abs((e1+e2)-((r['J00']-base['J00'])-(r['actual_return']-base['actual_return'])))
                    # Cached Q discounts were float32; row scalars are decimal float64.
                    assert residual<2e-5, residual
                    components.append(dict(scenario=int(base['scenario']),origin=origin,candidate=r['candidate'],
                        predicted_prefix_gain=predprefix,actual_prefix_gain=actualprefix,predicted_terminal_gain=predterm,actual_terminal_gain=actualterm,
                        prefix_gain_error=e1,terminal_gain_error=e2,absolute_prefix_error=abs(e1),absolute_terminal_error=abs(e2),
                        prefix_squared_error=e1**2,terminal_squared_error=e2**2,cross_term=2*e1*e2,total_squared_error=(e1+e2)**2,
                        predicted_gain=predprefix+predterm,actual_gain=actualprefix+actualterm,float32_decomposition_residual=residual))
    table('candidate_coverage.csv',coverage);table('reward_continuation_error_decomposition.csv',components)
    table('real_latent_q_vs_sac_continuation.csv',realq);table('q_variation_vs_returns.csv',qrelation);table('baseline_mean_accuracy.csv',meancomparisons)
    # Real selected counterfactual output at eachK, distinguished from native online trajectories.
    selections=[];d=ROOT/'reports/mppi_root_cause_20261117';p=load(d/'protocol.json')
    for scene,before in p['development_origins']:
        origin=f'{scene}_{before}';c=load(d/'catalog'/origin/'catalog.json')
        for protocol in ['H10_SAC','first_action_SAC']:
            rr=[r for r in returns if r['cohort']=='fixed_rounds' and r['origin']==origin and r['protocol']==protocol]
            y={r['candidate']:r['actual_return'] for r in rr}
            for k in [1,2,4]:
                name=c['iterations'][str(k)]['selected'];gain=y[name]-y['baseline']
                selections.append(dict(scenario=scene,origin=origin,protocol=protocol,k=k,selected=name,actual_gain=gain,
                    measured_subset_selection_regret=max(y.values())-y[name],best_measured_gain=max(y.values())-y['baseline'],
                    scope='Frozen starting-state prefix intervention thenSAC; NOT receding MPPI closed-loop'))
    table('actual_selected_counterfactual.csv',selections)
    # Aggregate dynamics/reward errors at eachH with independent MAPS, not timesteps.
    errors=read('rollout_prediction_error.csv');errorstats={}
    for h in [1,3,5,10]:
        rr=[r for r in errors if r['step']==h]
        for r in rr:
            r['bicycle_xy_error_m']=np.sqrt(r['bicycle_xy_squared_error'])
            r['absolute_reward_error']=abs(r['reward_error']) if r['reward_error'] is not None else None
        errorstats[str(h)]=stat(rr,['latent_l2','latent_cosine','bicycle_xy_error_m','bicycle_yaw_error','bicycle_speed_error','absolute_reward_error'])
    # Aggregate per-branch physical XY RMSE across full prefix, with explicit different model source.
    phys=[]
    for source in sorted({(r['origin'],r['candidate']) for r in errors}):
        rr=[r for r in errors if (r['origin'],r['candidate'])==source and r['step']>0]
        phys.append(dict(scenario=int(rr[0]['scenario']),origin=source[0],candidate=source[1],
            bicycle_xy_rmse=float(np.sqrt(np.mean([r['bicycle_xy_squared_error'] for r in rr]))),
            reward_rmse=float(np.sqrt(np.mean([r['reward_error']**2 for r in rr if r['reward_error'] is not None])))))
    table('branch_rollout_errors.csv',phys)
    counter=load(R/'counterfactual_statistics.json');q=load(R/'q_stability_statistics.json');search=load(R/'search_statistics.json');timing=load(R/'timing_statistics.json')
    paired=dict(counterfactual=counter,component_errors=stat(components,['absolute_prefix_error','absolute_terminal_error','prefix_squared_error','terminal_squared_error','cross_term','total_squared_error']),
        real_latent_q_continuation=stat(realq,['pair_accuracy','spearman','kendall','top1_regret']),
        q_variation_vs_returns=stat(qrelation,['q_gain_std_vs_abs_return_spearman']),
        coverage={f'{cohort}/{protocol}':stat([r for r in coverage if r['cohort']==cohort and r['protocol']==protocol],['oracle_best_gain','measured_subset_has_benefit'])
                  for cohort,protocol in sorted({(r['cohort'],r['protocol']) for r in coverage})},
        label_coverage_by_family={f'{cohort}/{protocol}/{family}':stat([r for r in coverage if r['cohort']==cohort and r['protocol']==protocol and r['coverage_family']==family],['label_coverage'])
                  for cohort,protocol,family in sorted({(r['cohort'],r['protocol'],r['coverage_family']) for r in coverage})},
        baseline_mean={f'{cohort}/{protocol}':stat([r for r in meancomparisons if r['cohort']==cohort and r['protocol']==protocol],['baseline_mean_accuracy']) for cohort,protocol in sorted({(r['cohort'],r['protocol']) for r in meancomparisons})},
        rollout_error_by_horizon=errorstats,branch_errors=stat(phys,['bicycle_xy_rmse','reward_rmse']),q_stability=q['summary'],
        prior_A_vs_B=load(d/'iteration_results.json')['agreement'],prior_relative_dynamics=load(d/'relative_dynamics.json')['by_horizon'],
        native_closed_loop_stage2=load(ROOT/'reports/mppi_selection_20261117/stage2_analysis.json')['comparisons'],
        note='Only scene-cluster CIs; origins/repeats/pairs never independent maps. All historical labels retrospective development/seen evidence, no new independent validation. No selected bestQ combination.')
    # Score separation versus weight concentration, from all saved search rounds.
    separation=[]
    for f in sorted((ROOT/'reports/mppi_selection_20261117/episodes/development').glob('*/episode.json')):
        ep=load(f)
        if not ep['options']['mppi']:continue
        with np.load(f.parent/'trace.npz') as a:
            v=a['iteration_values'];top=np.sort(v,axis=2)[:,:,-8:];w=np.exp(.5*(top-top[:,:,-1:]));w/=w.sum(2,keepdims=True)
            for k in range(v.shape[1]):
                separation.append(dict(scenario=ep['scenario'],configuration=ep['configuration'],seed=ep['test_seed'],k=k+1,
                    candidate_value_std=float(v[:,k].std(1).mean()),candidate_value_range=float(np.ptp(v[:,k],axis=1).mean()),
                    elite_value_gap=float(np.ptp(top[:,k],axis=1).mean()),maximum_elite_weight=float(w[:,k].max(1).mean())))
    table('distribution_score_separation.csv',separation)
    paired['score_separation']={g:stat([r for r in separation if r['configuration']==g],['candidate_value_std','candidate_value_range','elite_value_gap','maximum_elite_weight']) for g in sorted({r['configuration'] for r in separation})}
    dump('paired_statistics.json',paired)
    hypothesis=dict(primary_direction='D: continuation-value ranking/target mismatch; G is an execution-protocol confound that must be held explicit',
        observed='H10 prefix versusfirst-action intervention ranks poorly agree; real-prefix/real-latent oracles do not restore reliable Q ranking; actual closed-loop outcomes negative.',
        permitted_next='Only predeclared same-origin continuation target auditing, keeping H10 andfirst-action labels separate. Existing real-prefix/real-latent Q versus actual SAC RTG is the oracle test. Do not train/deploy a new scorer or tune Q/temperature/std.',
        predicted_metric='A target consistent withfirst-action execution should improve within-origin truefirst-action selection regret, not merely H10 predicted score.',
        falsification='If any proposed existing-score target fails paired held-out development-origin regret/Return improvement, stop it. If evidence comes only from old seen labels, no independent or formal100 claim.',
        next_code_change='None in online planner. Keep default selection andrandom_pair. Add only these offline paired-target diagnostics until the hypothesis has independent support.',
        independent_config=True,online_enabled=False,independent_validation_opened=False,formal100=False)
    dump('next_hypothesis.json',hypothesis)
    plot(counter,coverage,search)
    report(paired,timing,search)
    verify()


def plot(counter,coverage,search):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f,axs=plt.subplots(1,3,figsize=(14,4))
    a=counter['ranking']['fit_innerstop'];names=['J00','J10','J01','J11'];means=[a[n]['pair_accuracy']['mean'] for n in names]
    lo=[a[n]['pair_accuracy']['ci95'][0] for n in names];hi=[a[n]['pair_accuracy']['ci95'][1] for n in names]
    axs[0].bar(names,means,color='#457b9d');axs[0].errorbar(names,means,yerr=[np.array(means)-lo,np.array(hi)-means],fmt='none',color='black',capsize=3);axs[0].axhline(.5,color='red',ls='--');axs[0].set_ylim(0,1);axs[0].set_title('Paired oracle ranking (28 scenes)')
    rr=[r for r in coverage if r['cohort']=='fit_innerstop'];axs[1].scatter(np.arange(len(rr)),[r['oracle_best_gain'] for r in rr],s=12);axs[1].axhline(0,color='black');axs[1].set_title('Best measured Return gain / origin');axs[1].set_xlabel('Existing origin (not all65 candidates)')
    for g in ['M1','M2','M4']:
        vals=[search['replanning'][g][str(h)]['first_plan_vs_executed_rmse']['mean'] for h in [1,3,5,9]]
        axs[2].plot([1,3,5,9],vals,marker='o',label=g)
    axs[2].set_title('Saved sequence vs actual future action');axs[2].set_xlabel('Offset in control ticks');axs[2].set_ylabel('Action RMSE');axs[2].legend();f.tight_layout();f.savefig(R/'attribution.png',dpi=160);plt.close(f)


def fmt(s,scale=1):
    if s['mean'] is None:return 'N/A'
    return f"{s['mean']*scale:.3f} [{s['ci95'][0]*scale:.3f}, {s['ci95'][1]*scale:.3f}]"


def report(p,timing,search):
    m=load(R/'experiment_manifest.json');rank=p['counterfactual']['ranking']['fit_innerstop'];c=p['component_errors'];q=p['q_stability'];lines=[]
    def add(t=''):lines.append(t)
    add('# MPPI失效归因：2026-11-17')
    add('\n结论：**最大可定位误差项是D类：终端延续价值对真实SAC后续收益差的评价失真；G（开环评价与首动作/反馈执行目标不一致）是其次必须处理的协议偏差。** 这不等于已经唯一证明“策略不匹配”是根因，Q泛化、MC噪声和状态缺失尚未完全分离。C存在明显误差但oracle替换未显著修复排序；E会扰动搜索却无收益证据；B不是当前可证实的主因。A不能解释全部退化：已执行子集中存在真实获益候选。F的Mean与Elite均已真实闭环失败。这个排序是证据优先级，不是可识别的因果贡献百分比。')
    add('\n## 1. 版本与真实性核验')
    add(f"\n当前HEAD：`{m['current_head']}`（本地main）。56回合记录基础SHA：`{m['historical_56_base_commit']}`，**SHA不同**，但运行源码指纹均为 `{m['historical_source_fingerprint']}`；旧补丁在当前5个规划文件上反向检查成功，当前提交包含在线Elite实现。不是把candidate_audit记录冒充执行。当前main的`best_final_elite`会实际执行最终轮原预测Top1与Baseline公平比较后的首动作。原有trace中E/PE实际source=2，其他方案不使用Elite执行。")
    add(f"\nCheckpoint：`{m['checkpoint']}`；SHA256 `{m['checkpoint_sha256']}`。这是WM200k+RL300k、总500k的既有模型，本轮没有换模型。56个配置逐项记录路径、配置hash、选项、共同源码指纹及checkpoint hash，见experiment_manifest.json。仅有旧基础SHA而非一个原子实验commit；不能给每回合伪造独立commit。")
    add('\n未执行fetch/reset/checkout/commit/push。远端main核验单独见remote_version.json；本地origin/main缓存不能冒充远端最新。')
    remote=load(R/'remote_version.json')
    add(f"\n实测`git ls-remote origin refs/heads/main`返回 **`{remote['remote_main_observed_by_git_ls_remote']}`**，与本地HEAD不同。远端这版local_mppi只有candidate_audit的BestElite记录，**没有在线selection_mode/best_final_elite实现**；在线Elite存在于本地45073e1。56回合应归属本地运行源码，不能归属当前远端main。未擅自同步或覆盖任何分支。")
    add('\n## 2. 数据覆盖及统计')
    add('\n本轮新增277条**历史分支记录的分析行**（197个精确重放分支+4起点×10候选×2干预），不代表新增277次环境执行。同一历史分支可能对应不同分析cohort，不能跨cohort相加当独立样本。197条分支来自28个Fit/InnerStop场景；H10完整oracle有55个起点，27场景存在超过1回报差的有效候选对。另完整分析56回合中48个MPPI回合的全部逐轮动作/Q记录。')
    add('\n真实收益差绝对值≤1固定为近似相等，排序准确率排除这类真实收益对；预测分数相等计0.5。Spearman/Kendall保留原连续收益及原生ties。场景内先平均，再按场景bootstrap2000次、seed1、95%CI；多个起点、种子、时间窗及候选对不当成独立地图。所有旧验证标签均已见，不称为新独立验证。')
    for key,value in p['coverage'].items():add(f"\n- {key}：测得最佳候选收益增量 {fmt(value['oracle_best_gain'])}；起点存在>1正收益比例 {fmt(value['measured_subset_has_benefit'],100)}%。")
    for key,value in p['label_coverage_by_family'].items():add(f"\n- {key}标签覆盖 {fmt(value['label_coverage'],100)}%。旧3分支的100%只表示3个固定干预均有标签，不代表整个Gaussian池均有标签。")
    add('\n**A：存在真实获益候选，但不能证明完整候选池充分覆盖。** candidate_coverage.csv按起点给出正/负/近似相等数量、分母及覆盖；小量已执行候选的oracle是下界。全部Gaussian/Top8的大多数真实收益缺失，不能输出完整池真实SelectionRegret。candidate_ranking.csv中argmax仅为已测子集离线诊断，不冒称原在线动作。actual_selected_counterfactual.csv才是原K=1/2/4选择的实测前缀后果；56回合真实闭环结果另外引用。')
    add('\n## 3. Reward / Dynamics / 延续价值分解')
    add('\n所有J使用相同candidate终端a_H、Q01、gamma=.995、H10与cost=0；真实latent只做离线oracle。按题设J00/J10/J01/J11使用未额外加入预测终止mask的gamma^H公式。H前原生终止保留原始失败及收益，但没有z_H时排除完整oracle，绝不补造未来。完整候选前缀后实际由SAC接管，而a_H未实际执行，故J11仍不是实际策略价值。')
    add('\n| 评分 | 配对准确率% [95%CI] | Spearman | Kendall | Top1 Regret |\n|---|---:|---:|---:|---:|')
    for name in ['J00','J10','J01','J11']:
        r=rank[name];add(f"|{name}|{fmt(r['pair_accuracy'],100)}|{fmt(r['spearman'])}|{fmt(r['kendall'])}|{fmt(r['top1_regret'])}|")
    for name,value in p['counterfactual']['oracle_paired'].items():add(f"\n- {name}相对J00：配对准确率变化 {fmt(value['pair_accuracy_change'],100)}个百分点；Regret变化 {fmt(value['regret_change'])}（负值才改善）。")
    add(f"\n相对Baseline的前缀收益误差MAE：{fmt(c['absolute_prefix_error'])}；折扣后终端延续收益差误差MAE：{fmt(c['absolute_terminal_error'])}。分解逐候选保留e_prefix、e_terminal、平方项及交叉项，避免把相关误差平方直接相加当独立贡献。Q在真实末端latent上对实际SAC RTG的配对准确率为 {fmt(p['real_latent_q_continuation']['pair_accuracy'],100)}%。")
    add('\n**B/D/E回答：** 已测子集原Q识别能力不足，不能稳定提取oracle收益；真实前缀奖励没有恢复总收益排序，真实末端latent也没有显著恢复。真实状态加真实奖励仍保留较大后续价值误差，支持优先检查Q估计/实际延续策略目标，而不支持直接重训Dynamics或继续造ValueHead。不能仅据这些数据断言所有残差都是“策略不匹配”：MC回报噪声、状态/交通上下文和Q泛化同样未被完全隔离。')
    add('\n原Reward自身对真实prefix的有效大差值对准确率约93%，但只有12场景存在>1有效差值，全部场景Spearman约0.17。它既不是“Reward已经精准”的证明，也不能据此宣称全体候选Reward排序可靠。详见表中prefix_reward行。')
    add('\n逐步latent误差及cosine、Reward误差、物理RMSE见rollout_prediction_error.csv / branch_rollout_errors.csv。相对Dynamics方向cosine从H1约0.75降至H10约0.47，H10误差/真实候选差异约1.07，说明C是真问题；但不能由latent距离本身推断排序因果。WM没有物理decoder，XY/航向/速度误差明确标为既有bicycle模型，绝不称为WM位置误差。56回合缺少物理轨迹，不能补造其XY诊断。')
    add('\n## 4. Q Ensemble与搜索分布')
    add(f"\n枚举全部10双Q组合及五成员均值，在4个原始起点×4轮×65同池候选重评。排名Spearman {fmt(q['ranking_spearman'])}；Top8 Jaccard {fmt(q['top8_jaccard'])}；条件单次Mean首动作差 {fmt(q['first_action_delta'])}；序列RMSE {fmt(q['mean_sequence_rmse'])}。实际每轮Q成员见actual_q_member_indices.csv及原目录catalog；random_pair确实每轮重抽，且最终评分另抽一对。原生TD-MPC2同样如此，不是Bug证据。")
    eff=p['counterfactual']['q_fixed_vs_ensemble'];add(f"\n同已有实际标签，ensemble相对预先固定Q01的配对准确率变化 {fmt(eff['pair_accuracy_change'],100)}个百分点，Regret变化 {fmt(eff['regret_change'])}。这是缓存同池比较，不是新闭环；没有独立起点显著改善证据，不推荐固定Q或ensemble在线替换，也不从10对中反向挑最优。")
    add('\nsearch_distribution_diagnostics.csv记录各轮ValueGap、Entropy、ESS、Baseline入Elite、Mean/Std变化、动作改变量、分维标准差、Elite距离及边界饱和率。**精确裁剪率缺少pre-clamp创新量，不伪造**。跨轮样本是新抽样，索引Top8重合只作index描述，不能冒充同轨迹Jaccard。离线复现使用原逐向量CUDA topk：CPU批量topk在分数相等时会换候选，曾导致诊断assert失败；工具已修复，未改变在线planner。各回合最终Mean已与保存值匹配。')
    for g in ['M1','M2','M4']:
        r=search['by_configuration_iteration'][g][str({'M1':1,'M2':2,'M4':4}[g])]
        add(f"\n- {g}末轮ESS {fmt(r['effective_sample_size'])}，Entropy {fmt(r['weight_entropy'])}，Elite Gap {fmt(r['elite_value_gap'])}，Baseline入Elite {fmt(r['baseline_in_elite'],100)}%。")
    add('\nM4末轮ESS约7.91/8，说明Elite权重接近均匀；这不证明全部65候选都同分。distribution_score_separation.csv另外分开给出65候选得分范围/标准差、Top8 Gap及最大权重，避免把Elite内近等权与候选完全无差异混同。它是原beta下的实际行为，不据此擅自调temperature。')
    add('\n## 5. 首动作与真实闭环一致性')
    for g in ['M1','M2','M4','E','PE']:
        a=search['action_changes'][g];h=search['replanning'][g]['9']
        add(f"\n- {g}实际首动作ΔL2 {fmt(a['correction_l2_mean'])}，方向盘|Δ| {fmt(a['steer_abs_change'])}，油门|Δ| {fmt(a['throttle_abs_change'])}；原计划第9步与真实执行动作RMSE {fmt(h['first_plan_vs_executed_rmse'])}。")
    agreement=p['prior_A_vs_B'];add(f"\n相同候选在H10前缀→SAC与首动作→SAC两协议下，Spearman={agreement['spearman']:.3f}，95%CI[{agreement['spearman_ci95_low']:.3f},{agreement['spearman_ci95_high']:.3f}]；Top1一致率25%。仅4个已见开发起点，不能直接推广。但这足以否定把H10标签当作首动作控制质量等价标签；首动作→SAC也不等于每步重跑MPPI。replanning_consistency.csv覆盖全部48个真实MPPI回合，后续执行确实偏离原序列。closed_loop_prediction_error.csv的多步latent差同时含动力学误差与重规划动作改变，不能只归因Dynamics。")
    add('\n56回合真实闭环已表明：M1/M2/M4相对B折扣收益约−3.03/−3.02/−6.50，所有六个MPPI定义在两个种子上均负；没有成功率提升，M4新增碰撞，P/E单项加重越界。F中Mean/Elite已做过实际对照而非只审计；均没有正收益。动作修正量非零且随迭代增大，但其有效控制价值尚未被证明。')
    add('\n## 6. 完整规划耗时与缺失阶段')
    add('\n|配置|P50 ms|P95 ms|P99 ms|\n|---|---:|---:|---:|')
    for r in timing['rows']:
        if r['phase']=='total_planning':add(f"|{r['configuration']}|{r['p50_ms']:.2f}|{r['p95_ms']:.2f}|{r['p99_ms']:.2f}|")
    add('\n上表复用同代码RTX3070、单GPU worker、相同两固定状态的原同步完整agent.act测量，每方案100次；不是并发回合的污染耗时。所有MPPI P95>100ms，M1也未达标。本轮另对缓存z/actions测了WM/Q、采样、Elite更新及局部planner，记录在timing_breakdown.csv；分层同步有额外开销且嵌套阶段不可相加。Lattice/SAC/适配与最终选择独占时间未在旧回合保存，且缓存缺完整几何上下文，明确N/A。未另开MetaDrive伪补原始阶段耗时，也不将历史其他版本profile混用。')
    add('\n单独追加了非侵入PhaseTimings：同缓存origin的M4，每计划累积Reward、Dynamics、Termination、Q、采样及Elite耗时，30次有效测量；并验证插桩前后selected_sequence、全局CPU/CUDA RNG及私有采样RNG完全相同。见cached_phase_profile.json。以下是局部planner每计划的CUDA stream累计阶段，不包含Lattice/SAC，也不应与上表混加：')
    for r in timing['rows']:
        if r['configuration']=='M4_cached_phase_profile' and r['phase'] in ['wm_reward_ms','wm_dynamics_ms','wm_termination_ms','q_evaluation_ms','mppi_sampling_ms','elite_update_ms']:
            add(f"\n- {r['phase']}：P50={r['p50_ms']:.2f}ms，P95={r['p95_ms']:.2f}ms，P99={r['p99_ms']:.2f}ms。")
    add('\n## 7. 根因证据排序与下一步门槛')
    add('\n|优先级|因素|证据及限制|\n|---|---|---|\n|1|D 延续价值/评价目标不足|终端收益差误差远大于prefix；真实Reward+真实latent仍未恢复可靠排序。无法唯一分离策略差异、Q泛化与MC噪声|\n|2|G 目标/反馈不一致|同池A/B排序弱、真实执行偏离H10计划；直接协议对照，但只有4起点|\n|3|C 相对Dynamics误差|随H扩大且方向恶化；oracle替换无显著收益，暂未证实是主导|\n|4|E Q抽样波动|可观测排名/Mean变化，但ensemble无改善，不改默认|\n|5|F 分布更新/提取|未发现算法实现错误；Mean/Elite真实闭环都失败，非单一提取方式可解释|\n|6|B Reward|有预测误差，替换真实prefix未修复，证据不支持优先改Reward|\n|7|A 搜索覆盖不足|有正收益候选，不能解释全部；完整池真实标签缺失，仍不能排除在部分状态覆盖不足|')
    add('\n只保留一个后续方向：**D，核对实际Lattice+SAC延续目标与Q排序的一致性**，并严格分开G中的H10/首动作协议；对应最小代码改动限于本轮离线oracle及独立next_hypothesis.json，不改在线评分/采样/动作选择。预期指标是同起点实际延续收益排序、首动作Regret改善，不能以预测分数上涨验收；若配对开发样本真实收益/Regret无改善就否定方案。当前没有独立改善证据，不进入封存验证，不启动正式100场。无法从本诊断直接导出已验证的在线修复，也不训练第六个评分头。')
    add('\n交付问题A—H逐项回答：')
    add('\n|问题|回答|\n|---|---|\n|A 搜索空间有真实好候选吗|有：已测子集约32.4%候选为正收益，约一半起点至少有一个>1收益候选；完整Gaussian池标签不足|\n|B 原Q识别得出来吗|不可靠：J00配对准确率57.4%，CI跨50%；已测子集最优平均+4.63，Q所选平均约−0.28|\n|C Q成员影响排名吗|影响：Top8 Jaccard约0.67；但ensemble没有改善真实排序，不能据此改默认|\n|D 真实latent能恢复吗|未证实：58.1%，配对变化+0.62pp且CI跨0，Regret未改善|\n|E 真实Reward能恢复吗|未证实：56.4%，配对变化−1.05pp且CI跨0，Regret未改善|\n|F 有SAC后续策略不匹配迹象吗|有：真实latent的Q对SAC RTG约55.8%，终端收益差误差显著更大；但与Q泛化/MC噪声不能唯一分离|\n|G 首动作改变多少、有用吗|ΔL2约0.070/0.113/0.172（K1/2/4），控制改变真实存在；闭环折扣收益均下降，未证明有效收益|\n|H 最大可定位解释项|终端延续价值差误差MAE8.46，对比prefix0.39；优先D，G为协议层第二瓶颈。不是已识别的唯一因果比例|')
    add('\n## 8. 修改、测试及可复核性')
    add('\n新增tools/mppi_failure_attribution.py（离线统计）、tests/test_mppi_failure_attribution.py（3项关键回归）及本目录run.py/summarize.py/profile_cached.py。未改任何已有规划/网络/训练/配置文件，原56项相关测试保留，与新增3项共59项通过，详见tests.txt。checkpoint、535输入、动作映射、Cost Validator、cost_weight=0、Gaussian/AR、64/8/4、std/rho/temperature与默认在线选择均未修改。')
    add('\n主要可复核数据：experiment_manifest.json、candidate_returns.csv、candidate_ranking.csv、candidate_coverage.csv、q_ensemble_rank_stability.csv、rollout_prediction_error.csv、search_distribution_diagnostics.csv、timing_breakdown.csv、paired_statistics.json、verification.json。全部输出到新目录；旧实验结果原样保留。两个工具开发失败日志保留：旧Checkpoint验证buffer加载需复用原加载器；CPU/GPU topk ties需按原CUDA操作复现。两者均未执行环境、改权重或影响历史结果。')
    add('\n![归因诊断图](reports/mppi_failure_attribution_20261117/attribution.png)')
    (ROOT/'MPPI_FAILURE_ATTRIBUTION_REPORT_20261117.md').write_text('\n'.join(lines)+'\n')


def verify():
    import hashlib
    m=load(R/'experiment_manifest.json')
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    digest=hashlib.sha256()
    for name in sorted(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')):
        p=Path(name)
        if not name or p.parts[0] in {'logs','reports','tests'} or p.suffix not in {'.py','.yaml','.yml'}:continue
        if (ROOT/p).is_file():digest.update(name.encode());digest.update((ROOT/p).read_bytes())
    assert digest.hexdigest()==m['historical_source_fingerprint']
    m['tracked_runtime_fingerprint_now']=digest.hexdigest();m['tracked_runtime_matches_56_episodes']=True
    m['source_fingerprint_note']='Full now fingerprint includes NEW untracked offline tools; original tracked runtime byte-for-byte fingerprint still matches56. No historical source was modified.'
    m['remote_version']=load(R/'remote_version.json');dump('experiment_manifest.json',m)
    for name,h in m['core_sha256'].items():assert sha(ROOT/name)==h
    for e in m['episodes']:assert sha(ROOT/e['config_path'])==e['config_sha256']
    old_artifacts=load(ROOT/'reports/mppi_selection_20261117/artifacts_manifest.json')
    for e in old_artifacts:assert sha(ROOT/e['path'])==e['sha256']
    tests=(R/'tests.txt').read_text();assert '59 passed' in tests
    assert not list(R.glob('*.pt')) and not list(R.glob('*.pth'))
    diff=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True);assert not diff.strip()
    dump('verification.json',dict(head_unchanged=True,current_head=m['current_head'],checkpoint_sha256=m['checkpoint_sha256'],
        checkpoint_rehashed_after_run=load(R/'analysis_complete.json')['checkpoint_unchanged'],
        existing_tracked_files_unchanged=True,online_default_unchanged=True,original56_tests_retained=True,tests='59 passed',
        training_runs=0,environment_runs=0,new_value_heads=0,independent_validation_opened=False,formal100=False,
        same_Q_oracle=True,same_gamma_and_terminal_action=True,actual_candidate_returns_never_imputed=True,
        missing_data_explicit=True,attribution_not_a_unique_causal_decomposition=True,
        exact_historical_patch_matches_current=True,commit_ids_differ=True))
    patch=[]
    files=['tools/mppi_failure_attribution.py','tests/test_mppi_failure_attribution.py',str((R/'run.py').relative_to(ROOT)),str((R/'summarize.py').relative_to(ROOT)),str((R/'profile_cached.py').relative_to(ROOT))]
    for f in files:
        d=subprocess.run(['git','diff','--no-index','--','/dev/null',f],cwd=ROOT,capture_output=True,text=True);assert d.returncode==1;patch.append(d.stdout)
    (R/'code_changes.patch').write_text(''.join(patch))
    print('report and verification complete',flush=True)

if __name__=='__main__':main()
