"""Finalize the H1 study from completed native branches, including negative findings."""
import subprocess
import pandas as pd
from study import *

def write_patch():
    patch=''
    for file in sorted(R.iterdir()):
        if file.suffix in ['.py','.md'] or file.name in ['protocol.json','experiment_manifest.json']:
            result=subprocess.run(['git','diff','--no-index','--','/dev/null',str(file.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True)
            assert result.returncode==1;patch+=result.stdout
    (R/'code_changes.patch').write_text(patch)
    subprocess.run(['git','apply','--reverse','--check',str(R/'code_changes.patch')],cwd=ROOT,check=True)

def main():
    p=load(R/'protocol.json');m=load(R/'experiment_manifest.json');stats=load(R/'paired_statistics.json');audit=load(R/'data_audit.json')
    o=pd.read_csv(R/'oracle_gain_analysis.csv');e=pd.read_csv(R/'counterfactual_episodes.csv');a=pd.read_csv(R/'candidate_actions.csv')
    checks=pd.read_csv(R/'state_reconstruction_checks.csv');metrics=stats['metrics'];c=metrics['C_selected_gain'];d=metrics['D_selected_gain'];oracle=metrics['oracle_gain']
    if c['mean']<=0 and oracle['mean']>0:
        decision='当前有限候选池存在正收益，但C选择的平均H1收益为负，优先归为B：有限池内的评价/选择失效。'
        if c['ci95'][1]>=0:decision+='但C的地图区间跨0，尚不能认定稳定负效应或唯一因果根因。'
        investment='停止扩大当前LocalMPPI在线投入，继续使用Lattice+SAC；不因此训练新头或修改默认评分。'
        next_test='若继续研究，唯一下一实验是在预注册新地图上，将当前MPPI选择与同候选池预先固定的均匀随机选择做H1→SAC配对对照。它只验证现有评分是否提供超出无评分选择的真实价值，不训练网络，也不先改搜索。'
    elif c['mean']>0 and c['ci95'][0]>0:
        decision='本组数据中C首动作平均H1收益为正且地图区间下界为正；结合既有持续MPPI无改善结果，C：后续滚动交互值得专门检验，但本轮尚不能因果归因。'
        investment='可以继续一次严格的延续策略对照，不能据本轮开发地图直接上线MPPI或宣称系统优于B。'
        next_test='唯一下一实验：从同一起点执行完全相同的C首动作，t+1以后分别由原B接管和原C持续滚动重规划，到原生终止；只改变后续策略，直接测量收益保留或抵消。'
    elif oracle['mean']<=0:
        decision='当前有限候选池没有观察到平均正Oracle Gain，A：候选不足是优先方向；不能推广为整个连续动作空间无有效动作。'
        investment='暂停当前局部Gaussian候选与价值头方向的在线扩展。'
        next_test='唯一下一实验：预注册从现有Lattice不同可行驾驶意图提取少量首动作，保持WM/SAC/接管策略不变，执行同起点H1反事实；先验证候选收益，暂不改在线搜索。'
    else:
        decision='C选中动作平均H1收益为正，但地图聚类区间包含0；当前不足以区分稳定首动作收益与样本波动，也不足以认定C的滚动抵消。'
        investment='暂不扩大在线MPPI或训练新头，保留Lattice+SAC。'
        next_test='唯一下一实验：在全新、预注册的地图和起点上只复核A0/A1的H1配对收益及安全性，不再筛选模型或参数；先确认首动作正收益能否独立复现。'
    def ci(key):
        x=metrics[key];return f"{x['mean']:+.4f}（地图聚类95%CI [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]）"
    role=pd.read_csv(R/'role_results.csv').groupby('configuration').mean(numeric_only=True)
    role_table='| 首动作来源（此后全部B） | Success | Crash | OOR | Route completion | Safety Cost | Discounted Return |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for mode,label in [('B','A0 SAC'),('C','A1 原C选中'),('D','A2 原D选中')]:
        v=role.loc[mode];role_table+=f'| {label} | {v.success:.2%} | {v.crash:.2%} | {v.out_of_road:.2%} | {v.route_completion:.2%} | {v.safety_cost:.3f} | {v.discounted_return:.3f} |\n'
    safety='\n'.join(f"- {mode}−A0：Crash {ci(mode+'_crash_delta')}；OOR {ci(mode+'_out_of_road_delta')}；Success {ci(mode+'_success_delta')}。" for mode in ['C','D'])
    sensitivity=[]
    for key in ['oracle_gain','C_selected_gain','D_selected_gain']:
        values=metrics[key]['leave_one_map_out'].values();maps=metrics[key]['per_map'].values()
        sensitivity.append(f"- {key}：正均值地图{sum(v>0 for v in maps)}/8；Leave-One-Map-Out均值范围[{min(values):+.4f}, {max(values):+.4f}]。")
    sensitivity='\n'.join(sensitivity)
    predicted=[]
    for scene in p['scenes']:
        for step in p['steps']:
            meta=load(origin_dir(scene,step)/'catalog.json');z=np.load(origin_dir(scene,step)/'catalog.npz')
            predicted.append(float(meta['C']['score']-z['C_final_values'][0]))
    source_ok=all(file_sha(Path(path))==value for path,value in m['sources'].items())
    protected_ok=all(file_sha(ROOT/name)==value for name,value in m['protected_sources'].items())
    runtime_ok=all(file_sha(R/name)==value for name,value in load(R/'runtime_source_manifest.json').items())
    assert source_ok and protected_ok and runtime_ok
    assert file_sha(Path(p['checkpoint']))==p['checkpoint_sha256']
    assert '41 passed' in (R/'tests_v2.log').read_text() and '9 passed' in (R/'tests_statistics_v2.log').read_text() and audit['all_passed']
    main_status=subprocess.check_output(['git','status','--porcelain'],cwd=MAIN,text=True)
    source_B_commit=load(SOURCE/'protocol.json')['base_commit']
    dump(R/'verification.json',dict(base_commit=p['base_commit'],github_main_at_start=p['remote_main_commit'],source_B_base_commit=source_B_commit,
        old_data_unchanged=source_ok,protected_online_sources_unchanged=protected_ok,runtime_scripts_unchanged_during_execution=runtime_ok,
        checkpoint_sha256=p['checkpoint_sha256'],checkpoint_unchanged=True,default_config_unchanged=True,
        tests_passed=42,new_tests=9,data_audit_all_passed=True,maps=8,origins=16,branches=stats['branches'],
        full_baseline_tails_exact=int(checks.baseline_tail_exact.sum()),all_branch_origins_and_policy_history_exact=True,
        no_continuation_mppi=True,one_control_period_intervention_only=True,formal_h1_ranking='NOT ESTIMABLE',
        main_worktree_status_end=main_status,training=0,online_changes=0,commit=False,push=False))
    text=f'''# MPPI H1 首动作真实反事实实验

**结论：{decision}**

{investment} 本轮是8张已使用开发地图、16个预定状态的探索性反事实，不是新的独立验证，也不是正式100场。不能把{stats['branches']}条执行分支当成{stats['branches']}张独立地图。

## 开头直接回答8个问题

1. **候选中有真正优于SAC的动作吗？** 有限池Oracle Gain地图均值为{ci('oracle_gain')}，严格正收益起点{stats['strict_positive_origins']}/16；较稳健的收益差>1起点{stats['gain_over_one_origins']}/16。Oracle只对本次6–8个左右的有限候选集合成立，不是连续动作空间全局最优，也不是可执行的预测模型。
2. **正收益动作状态占比？** 严格>0为{stats['strict_positive_origins']/16:.2%}，>1为{stats['gain_over_one_origins']/16:.2%}。两种阈值同时报告，未根据结果选阈值。逐地图与每状态分布见oracle_gain_analysis.csv。
3. **MPPI最终首动作实际表现？** 原C Selected Gain {ci('C_selected_gain')}，D {ci('D_selected_gain')}；C严格正收益{stats['C_positive_origins']}/16，D {stats['D_positive_origins']}/16。C Top1 Regret {ci('C_top1_regret')}，D {ci('D_top1_regret')}。
4. **能识别真实好动作吗？** 在Oracle Gain>1的{stats['gain_over_one_origins']}个起点，C选中收益>1的有{stats['C_positive_when_oracle_over_one']}个，D有{stats['D_positive_when_oracle_over_one']}个。以上是实际选择后果，不是假称模型H1排序准确率。正式H1 Spearman/Kendall/Pairwise Accuracy全部标为NOT ESTIMABLE：没有独立验证合格、匹配当前Lattice+SAC接管策略的V_SAC。
5. **明显WM预测偏差？** 实际C原生H10预测平均Selected Gain={np.mean(predicted):.4f}，真实H1 Selected Gain={c['mean']:.4f}；其目标不同，不能相减当作同目标校准误差。WM即时Reward MAE={stats['immediate_reward_mae']:.4f}，这才是同一步可直接配对的误差。H10分数与H1回报的相关性仅作描述，不证明H1价值排序能力，更不能单独归咎Dynamics。
6. **主问题属于哪一环节？** {decision} 候选覆盖、原TD终端价值/继续策略不匹配及滚动交互的唯一因果贡献仍未全部分离。该实验直接识别的是一个动作改变后的同一B延续策略后果；不是持续MPPI策略的效果。
7. **是否继续投入LocalMPPI？** {investment} 无论均值方向如何，本轮均不能证明在独立地图上系统成功率提高。
8. **唯一下一实验？** {next_test}

## 1. 配对结果

{role_table}

次要对照D−C的同起点折扣收益差为{ci('D_minus_C_gain')}；不将其与原持续MPPI的D−C混同。

这些是从指定状态开始的分支结果，非从地图初始时刻独立运行的16张地图；Crash/OOR按干预后任一真实事件统计，允许Success与Crash同为1，未修改原环境。每张地图两个起点先平均，再以8图聚类bootstrap2000次，seed20261118。安全配对差：

{safety}

随机非baseline候选的均匀平均收益差为{ci('random_candidate_mean_gain')}。Oracle与随机基线均仅用于离线分析，不读取真实未来来执行在线动作。所有Leave-One-Map-Out值保留于paired_statistics.json；8地图区间可靠性有限，不以分支数夸大置信度。

地图敏感性：

{sensitivity}

## 2. 状态恢复和流程放行

按固定地图与seed=scene+1，重放前49/149条原B动作。每步同时调用原策略重建Lattice缓存，并逐位核验计划动作、真实观测以及reward。目标周期原B快照保存于“目标决策计算完、env.step前”，故再计算一次A0后匹配原observation/vehicle/traffic/RNG全部哈希。并非错误重放50/150步。

所有16个A0分支不仅前10周期，而且整个余下轨迹的动作、观测、物理状态、reward、terminated/truncated/done均与原B逐步一致，总计{int(checks.baseline_tail_exact.sum())}个原生控制周期。所有其他分支也匹配同一起点原哈希、新policy指纹和扩展环境指纹。详见STATE_RECONSTRUCTION_AUDIT.md。

源B数据协议基线为`{source_B_commit}`，本轮运行代码为`{p['base_commit']}`，两者SHA不同。本轮未默认启用Cartesian修正，B始终关闭MPPI；整段A0严格复现是新旧版本Baseline行为一致的实测证据，而非假设SHA相同。

单状态Phase1通过后，先在130719步50跑A1及3个Gaussian至原生done，检查仅1步干预、后续全SAC。通过Phase2后才准备其余起点；各起点A0完整复现后才放行剩余候选。所有gate和失败前诊断记录均保留。

最初扩展诊断触发了Gymnasium带副作用的np_random getter，创建原本未初始化的unused RNG，导致两个字段跨进程不一致，A0在执行前被阻止。修复的只是新增诊断工具：读取既有_np_random状态，保持None而不新建。引擎、manager、交通policy和全局RNG仍严格比较，不放宽容差。初始失败目录与state_diff.log完整保留，正式分支未使用该失败起点。

## 3. 动作注入与内部历史

原MPR评测每步编码真实观测，没有递归latent隐状态；原535 SAC是MLP。原TD `_prev_mean`仅在另一条nativeTD规划入口使用，所有分支执行前后不变。last_plan/last_result只作诊断，不作为上一实际动作读入。执行器不伪造训练residual transition，不调用observe或update。

首步实际MetaDrive steering/throttle逐位等于候选；下一周期vehicle.current_action及新SafetyContext.previous_action逐位等于实际执行动作，新的latent逐位等于冻结Encoder对真实后继obs的编码。后续每步都是原SAC，MPPI入口被执行器设为报错守卫，任何误用即失败；实际续航MPPI调用数全部0。因此不会只覆盖env.step而让一个有效历史输入继续记错动作。

## 4. 候选来源和预测目标

16起点中11个有8个独立首动作、5个去重为7个，总计{stats['branches']}条分支。A0、实际C selected、实际D selected、C最后一轮Gaussian预测分位候选全部在真实结果前固定。保持64samples+baseline、8elites、4iterations、原std/rho/temperature/Q/gamma及Cost=0。没有手工候选、额外筛选或新模型。

实际捕获每轮65×11×2，最后一个Gaussian batch是iteration3；其后2×11×2的baseline/weighted-mean复评不是第五轮Gaussian。A3–A7只取最后一轮row1:65，稳定按该批原分数取分位。A1/A2是agent.act实际输出，不以best sampled action或加权均值名义替换。原trace、weighted mean、实际selected、原始Q成员和别名均保留。

候选生成器和执行器分进程；生成C/D使用相同pre-decision全局RNG及原私有采样seed，结束后恢复原B决策后RNG，核验环境和交通未变。执行器从不继承候选准备器的mutable planner cache。C/D动作是在相同B状态重新运行原C/D得到，不复制来自不同C/D真实轨迹状态的动作；本轮只有一个固定规划seed，不能外推所有采样随机性。

原H10分数附其原Q成员与批次；8角色另用同一个C最终双Q组合H10复评，仅作描述，不影响原选择。模型评分定义是H10候选前缀+TD-Q，而真实标签为H1后原B接管，故没有可信H1正式排名指标。candidate_ranking.csv逐项标明NOT ESTIMABLE，图中的H10/H1散点也明确目标不匹配。

## 5. 图表

![每状态实际回报差](origin_candidate_returns.png)

![实际C/D与SAC配对收益](paired_selected_vs_sac.png)

![有限池Oracle收益](oracle_gain_distribution.png)

![实际选择Regret](top1_regret_distribution.png)

![首动作空间](candidate_first_action_space.png)

![仅描述性的异目标预测对照，非正式H1排序](descriptive_h10_score_vs_h1_gain.png)

## 6. 工程交付与限制

原线上代码、默认配置、WM/SAC权重及checkpoint完全未改，原数据重新hash一致。42项不同的相关测试通过，其中新增9项（初期41项，后补充地图聚类统计回归）；不把通过测试当作算法有效。本轮没有训练、微调、新网络、正式100场、commit或push。所有脚本/原始轨迹/结果在独立worktree本目录，主目录未改。

原始分支观测、动作、reward、cost、物理状态及done在branches/*/*/trace.npz，合并逐步日志在counterfactual_steps.csv；episode/role/候选/源数据hash及脚本指纹完整保存。源码Diff见code_changes.patch，运行说明见RUNNING.md。

仍有限制：只有8张已使用开发地图；每起点一个固定交通及规划seed；Oracle是有限集合后验最大值；未得到可信的匹配H1模型；不能把单步后B接管的结果当成持续MPPI重规划收益。没有通过独立场景的价值模型不进入在线规划。
'''
    with (R/'FINAL_REPORT.md').open('x') as f:f.write(text)
    rows='| '+' | '.join(checks.columns)+' |\n| '+' | '.join(['---']*len(checks.columns))+' |\n'
    rows+='\n'.join('| '+' | '.join(str(value) for value in record)+' |' for record in checks.itertuples(index=False,name=None))
    audit_text=f'''# 状态恢复审计

全部16起点和{stats['branches']}个分支通过严格恢复。源数据目录：`{SOURCE}`。

{rows}

## 核验内容

- replay49/149而非50/150；每个过去周期调用冻结原B并执行保存动作，核验全部观测与reward。
- 原快照在该周期agent.act后/env.step前；新边界与原snapshot_*_origin.json的observation、vehicle、traffic、Python/NumPy/Torch/引擎/traffic-manager RNG哈希逐位相等。
- 新指纹对照当前latent、535状态、SAC残差、coarse/refined序列、原TD dormant prev_mean、私有RNG及Lattice历史；所有分支逐位相同。
- 扩展指纹还包括全部manager RNG、对象物理和RNG、交通policy的数值/PID历史。未初始化的Gym RNG不经getter创建。
- 每个A0整段尾轨迹精确重现原B，累计{int(checks.baseline_tail_exact.sum())}周期；每个分支实际收到的第一动作与提交值一致。
- 后续每步上一实际动作由真实vehicle.current_action进入SafetyContext；第一后继latent重编码核验，无递归隐藏latent需要手工覆盖。下一planner缓存全部由新真实状态计算。
- 候选准备过程不执行env.step，不改变物理/交通/RNG；执行器为独立进程，目标状态后MPPI调用守卫保证0。

## 诊断工具问题的处理

初次失败只有Gym未使用的懒初始化RNG两字段不同；inspect已确认其getter会使用系统熵创建RNG。保留diagnostic_attempts、state_diff.log及原始失败日志。新增工具修正为只读现有_np_random，未放宽任一关键比较。正式分支均在修正后执行。

## 证据范围

确定性前缀重放实际重建Bullet/车辆与策略状态，而非手工设坐标。原历史文件没有保存旧policy指纹：新policy指纹是本轮所有分支间比较；旧latent由相同冻结Encoder+相同观测确定，并由完整A0尾轨迹复现补充验证。哈希是记录到的状态域一致性证据，不声称对未观测物理引擎内存逐字节序列化。
'''
    with (R/'STATE_RECONSTRUCTION_AUDIT.md').open('x') as f:f.write(audit_text)
    # Valid Git-style patch of newly added experiment sources/docs, no old models or results binaries.
    write_patch()
    dump(R/'artifact_manifest.json',{str(f.relative_to(R)):dict(sha256=file_sha(f),bytes=f.stat().st_size) for f in sorted(R.rglob('*')) if f.is_file() and '__pycache__' not in str(f)})
    print('FINAL_REPORT_COMPLETE',decision,flush=True)
if __name__=='__main__':main()
