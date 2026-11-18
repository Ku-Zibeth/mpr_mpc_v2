"""Freeze provenance and write the evidence-led report; never modify original experiments."""
import difflib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from study_io import R,ROOT,MAIN,load,sha,dump

def write_code_patch():
    patch=subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)
    for file in sorted(R.iterdir()):
        if file.suffix in ['.py','.md'] or file.name in ['protocol.json','D_protocol.json','experiment_manifest.json']:
            name=str(file.relative_to(ROOT))
            result=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=ROOT,text=True,capture_output=True)
            assert result.returncode==1,result.stderr
            patch+=result.stdout
    (R/'code_changes.patch').write_text(patch)
    subprocess.run(['git','apply','--reverse','--check',str(R/'code_changes.patch')],cwd=ROOT,check=True)

def main():
    manifest=load(R/'experiment_manifest.json');protocol=load(R/'protocol.json')
    changes={name:dict(before=value,after=sha(ROOT/name)) for name,value in manifest['protected_sources_before'].items() if sha(ROOT/name)!=value}
    assert set(changes)=={'planning/coordinator.py','planning/time_alignment.py','planning/trajectory_adapter.py'}
    assert sha(Path(protocol['checkpoint']))==protocol['checkpoint_sha256']
    assert sha(ROOT/'config.yaml')==manifest['default_config_sha256']
    assert sha(Path(protocol['source_environment_config']))==manifest['source_config_sha256']
    main_status=subprocess.check_output(['git','status','--short'],cwd=MAIN,text=True)
    assert not main_status,main_status
    review=load(R/'history_review.json');historical=review['reports_fully_read']+review['associated_files_fully_read']
    invalid=[x['path'] for x in historical if sha(Path(x['path']))!=x['sha256']];assert not invalid,invalid
    for scene in protocol['scenarios'][:2]:
        states=[load(R/f'probe_{g}_{scene}.json')['physical_state']['hashes'] for g in ['B','C','D']]
        assert states[0]==states[1]==states[2]
    def failures(file):return sorted(line for line in file.read_text().splitlines() if line.startswith('FAILED '))
    assert failures(R/'tests_audit.log')==failures(R/'tests_native_cost_clean_main.log')
    assert '34 passed' in (R/'tests_alignment.log').read_text()
    assert '12 passed' in (R/'tests_audit_relevant.log').read_text()
    dump('verification.json',dict(base_commit=manifest['base_commit'],main_status_end=main_status,
        checkpoint_sha256=sha(Path(protocol['checkpoint'])),checkpoint_unchanged=True,default_config_unchanged=True,
        protected_source_changes=changes,historical_files_rehashed=len(historical),historical_data_unchanged=True,
        core_and_targeted_tests_passed=46,new_tests=7,
        additional_legacy_cost_tests=dict(passed=15,failed=20,same_failures_on_clean_main=True,failures=failures(R/'tests_audit.log')),
        native_episodes=48,independent_maps=8,repeat_seeds=[1,2],isolated_probe_calls=300,
        probe_initial_observation_vehicle_traffic_rng_paired=True,default_output_path_unchanged=True,
        train_calls=0,new_value_heads=0,formal100=False,commit=False,push=False))
    dump('experiment_manifest_initial.json',manifest)
    manifest.update(D_protocol=load(R/'D_protocol.json'),source_versions=dict(B=str(MAIN),C=str(MAIN),D=str(ROOT)),
        D_build_modified_files=changes,verification='verification.json',native_episode_configs={str(p.relative_to(R)):sha(p) for p in sorted((R/'episodes').glob('*/config.yaml'))},
        main_status_end=main_status,decision='Do not deploy or expand: no additional success; D-B discounted gain not positive; crash not improved; P95>100ms',
        counters=dict(native_episodes=48,maps=8,paired_seeds=2,isolated_probe_calls=300,training=0,formal100=0))
    (R/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    e=pd.read_csv(R/'summary.csv');absolute=e[e.comparison=='absolute mean'].set_index('configuration')
    stats=load(R/'paired_statistics.json')['comparisons'];chain=load(R/'control_chain_statistics.json')['groups']
    hist=load(R/'historical_prediction_statistics.json');lat=pd.read_csv(R/'latency.csv')
    isolated=lat[(lat.scope=='isolated_frozen_state')&(lat.stage=='total_wall_ms')].set_index('configuration')
    cov=pd.read_csv(R/'candidate_coverage_summary.csv');c=cov[cov.configuration=='C'].set_index('iteration')
    def number(g,k):return f'{absolute.loc[g,k]:.3f}'
    def diff(key):
        x=stats[key]['discounted_return'];return f"{x['mean']:+.3f}，地图聚类95%CI [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"
    result_table='| 配置 | Success | Crash(any) | OOR(any) | Route completion | Timeout | Episode Return | Discounted Return | 步数 | 独立规划P95 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for g,name in [('B','B Lattice+SAC'),('C','C 原MPPI'),('D','D Cartesian速度修正')]:
        v=absolute.loc[g];result_table+=f"| {name} | {v.success:.2%} | {v.crash:.2%} | {v.out_of_road:.2%} | {v.route_completion:.2%} | {v.timeout:.2%} | {v.episode_return:.2f} | {v.discounted_return:.2f} | {v.episode_length:.1f} | {isolated.loc[g,'p95_ms']:.2f}ms |\n"
    text=f'''# MPR-MPC 闭环性能根因审计与最小修复（2026-11-18）

**结论：修复了可复现的虚拟速度物理量不一致，但没有证明 MPPI 优于 Lattice+SAC。停止扩大实验；保持原默认配置，不将 D 设为默认，不训练新评分头。**

最强证据指向“候选价值评价与真正部署的滚动控制目标不一致”，特别是终端TD-Q，而不是大量重复候选或首动作执行错位。这个判断是证据优先级，**尚不能识别唯一因果根因或各因素贡献比例**。确定性转换问题是真实但次要的问题：修正后成功率仍未提升，不能称为闭环退化的主因。

## 1. 真实闭环结果与验收

{result_table}

预先冻结8张新开发地图：{protocol['scenarios']}；每图2个规划种子[1,2]，每组16原生回合，共48。未运行正式100场、未触碰封存验证。B重复种子在8图上轨迹均完全相同；C/D均因MPPI随机采样而不同。因此独立单位始终只有8图，不是48图。

折扣回报配对差：

- C−B：{diff('C-B')}。
- D−B：{diff('D-B')}。
- D−C：{diff('D-C')}。

D相对C的路线完成率增加3.69个百分点，Episode Return增加21.32；Success/Crash/OOR总体未变。相对B，D路线完成率增加2.15个百分点，但Success增加0回合，Crash增加1/16=6.25个百分点，折扣回报仍为负差。D-C回报区间也包含0，不能宣称可靠改善。逐地图方向并不一致，见 paired_statistics.json；不能以原始Episode Return增加替代折扣回报和事故验收。

**验收不通过**：没有额外成功回合；未证明事故指标不恶化及折扣收益正向改善；完整规划P95仍超100ms。保留B为工作基准，停止该轮开发采样和参数搜索，不进入新的独立验证或正式100场。

每周期实际重新生成Lattice、原SAC、可选MPPI并执行首动作直到环境原生done；不是H步序列重放。逐场景/种子、失败原因、成功但碰撞场景及安全Cost均在 paired_closed_loop_episodes.csv。Crash/OOR按全回合任一真实事件记录；环境允许碰撞后继续驾驶，Success和Crash可同时为1，未改环境终止条件。

## 2. A：候选生成证据

C组6068个规划步、每步4轮，64个Gaussian加baseline。完整候选序列去重率每轮均100%；首动作去重率约99.98%以上。平均首动作距SAC从第0轮{c.loc[0,'first_action_distance_mean']:.4f}增加到第3轮{c.loc[3,'first_action_distance_mean']:.4f}；最终实际首动作平均改变量{number('C','first_action_delta')}（归一化二维L2）。不能描述为根本没有动作变化。

原始采样被夹取的分量比例从{c.loc[0,'clipping_ratio']:.2%}降到{c.loc[3,'clipping_ratio']:.2%}；第0轮油门分量夹取{c.loc[0,'throttle_clipping']:.2%}。这些是克隆私有RNG精确重建夹取前样本得出的比例，不是用“恰等于边界”代替夹取计数。近SAC首动作比例（固定L2≤0.01）从{c.loc[0,'sac_near_first_overlap']:.2%}降到{c.loc[3,'sac_near_first_overlap']:.2%}。

输入std steering从.0400→{c.loc[3,'input_std_steer']:.4f}，throttle从.1200→{c.loc[3,'input_std_throttle']:.4f}；Elite全序列距离从{c.loc[0,'elite_diversity']:.4f}降到{c.loc[3,'elite_diversity']:.4f}。这是明显收缩，但仅凭这些数字不能叫“错误的过早塌缩”。std仍非零，首动作距离在增大，并无大量完全重复动作。

Lattice每步平均约{chain['C']['lattice_feasible_mean']:.2f}个不同(d,v,H)可行参数组合，进入SAC/MPPI前只选1条。固定快照原始横向参数跨度平均约7.62m，包含不同速度、时间与横向意图；完整快照保留在 episodes/*/snapshot_*.npz。**但没有这些被舍弃路径的真实收益标签，不能断言 nominal 经常不是实际最好路径**；唯一性也不等于有效行为多样性，仍缺少64候选全部真实轨迹可达性/收益标签。因此没有实施结构化参数搜索或多模式扩张。

历史已执行候选子集存在Oracle正收益，但模型不稳定识别。H1旧数据的地图均衡Oracle Gain为{hist['horizons']['1']['statistics']['oracle_gain']['mean']:.3f}，原H10评分选中增益{hist['horizons']['1']['statistics']['model_gain']['mean']:.3f}且区间跨0；H10子集Oracle Gain为{hist['horizons']['10']['statistics']['oracle_gain']['mean']:.3f}，模型选中{hist['horizons']['10']['statistics']['model_gain']['mean']:.3f}。这是有限代表候选子集的后验Oracle，不是256候选的完整真实覆盖，也不是新的独立验证。

## 3. B：评价与执行目标

实现确为先reward(z_t,a_t)，two_hot_inv一次，再next，原gamma=.995和预测termination掩码。未发现一步Reward索引或折扣重复错误。random_pair同批共享、跨轮变化；现有实验已显示改变Q成员会扰动排名，但不足以证明改成固定Q能提高真实收益。本轮不更改Q聚合。

新C组选择的预测正增益步比例{chain['C']['predicted_positive_gain_ratio']:.2%}，平均预测增益{chain['C']['predicted_gain_mean']:.3f}；绝对terminal增益占prefix与terminal绝对分量和的{chain['C']['terminal_gain_absolute_share']:.2%}。与此同时整回合C−B折扣收益为负。该现象支持评价目标失配/模型利用问题，但**不是同一步同状态反事实的因果收益证明**。

新C单步Reward MAE {chain['C']['first_reward_mae']:.3f}。预测H10总分与后续真实滚动策略RTG的MAE为{chain['C']['predicted_total_vs_rolling_actual_mae']:.2f}，其中包含策略/执行协议差异，不能当作纯WM预测MAE。原历史终端延续价值误差约8.46、prefix约0.39；真实reward+真实末端latent oracle未可靠恢复排序，故不能仅归咎Dynamics或Reward。

prediction_vs_actual.csv复用576条旧记录中571条同定义的baseline接管记录；另5条为2条TD actor、2条TD MPC接管和1条重复重放，明确列于 historical_prediction_statistics.json，不混入同策略排名。H1覆盖18地图、H2/4/10只有6地图，不能直接横向比较总体数字。每行均保留H、split、预测H10分数、同H重算score（可用时）、prefix、Q、实际reward/return/gain和原始路径；缺失字段留空，不补造。

相同候选H1/H10原始Return相关{hist['h1_h10_raw_label_spearman']:.3f}，但去掉起点baseline后的收益差相关只有{hist['h1_h10_within_origin_gain_spearman']:.3f}。原始相关受到起点难度影响，不可拿它反驳历史已报告约.079的收益差相关；数据子集也不同。H1→SAC、H10→SAC与每步重规划MPPI是三种不同策略，均明确分列。

最强未解决问题是TD-Q延续策略与Lattice+Residual SAC及随后重新规划MPPI不一致，且a_H参与优化却不在真实前缀执行。过去policy_prior、Mean/Elite、iterations1/2/4、多个独立学习头均未可靠改善；本轮没有再重复这些尝试。现有证据尚不能把策略定义错位、Q泛化误差、监督状态缺失与环境收益噪声完全分离。

## 4. C：动作转换与执行

所有真实动作均精确等于selected_sequence[0]，车辆收到的归一化steering/throttle控制最大分量差0；实际控制周期=.02×5=.1s。没有首动作偏移、维度颠倒或重复残差映射证据。

可复现问题：Frenet s_d不是世界速度模，但原虚拟future车状态用s_d。36个预定快照差值最高约2.17m/s；对角直线和偏移圆弯单元测试均复现。原实际控制器重放36/36精确复现旧动作，21个快照未来动作改变，最大分量差0.54536；首动作36/36不变。

只修改3文件24行新增/4行删除：time_alignment.py加入可选XY/time速度模；trajectory_adapter.py透传可选参数；coordinator.py仅post-SAC MPPI路径读取显式开关。原coarse535、网络、映射、环境、WM评分、64/8/4、std/rho/beta及Cost=0不变。`cartesian_virtual_speed`默认false，config.yaml无改动。

D预注册在任何D结果前完成（当时B/C完成12/32），假设来自物理量契约而非挑选有利回报；之后只运行这一种D。D_preflight恢复相同采样RNG并验证coarse535、coarse动作、latent、SAC残差、refined参数、baseline首动作逐位不变。该直道检查点未来动作差0，变化证据来自36条真实路径，而不是假称该点也有变化。

新闭环中C/D理想后继状态的速度MAE分别{chain['C']['next_speed_mae']:.3f}/{chain['D']['next_speed_mae']:.3f}m/s，位置误差约{chain['C']['next_position_mae']:.3f}/{chain['D']['next_position_mae']:.3f}m；不支持该修正已经解决实际动力学误差。两组后续状态本来不同，不能作为同状态预测器公平误差对照。端点保持区间约1.1%控制步出现；D让保持位置对应零虚拟速度，但这不是物理停车保证。未增加安全规则。

## 5. 耗时、测试与完整性

48回合采集使用两进程共享GPU、记录候选；该耗时独立列为 shared_native_recording，含冷启动峰值，不据此宣称实时性。随后单进程、第一二预定地图相同B前缀29步到相同真实状态，各配置5暖机+50完整agent.act，共300次正式计时；关闭候选记录、不执行探针输出动作，配对状态及交通/RNG完全一致。

独立完整规划P50/P95：B {isolated.loc['B','p50_ms']:.2f}/{isolated.loc['B','p95_ms']:.2f}ms，C {isolated.loc['C','p50_ms']:.2f}/{isolated.loc['C','p95_ms']:.2f}ms，D {isolated.loc['D','p50_ms']:.2f}/{isolated.loc['D','p95_ms']:.2f}ms。D与C微小差异不能宣称性能优化；两者都不满足100ms。

C平均Lattice约18.96ms、两个adapter合计5.86ms，SAC推理约.47ms；4轮候选+最终复评WM推演（含terminal）约47.92ms GPU时间，完整C约130.63ms。latency.csv保留CPU wall/GPU区分、4轮各次评价、最终复评和Q阶段；包含关系不可相加，CPU异步提交不能冒充GPU同步耗时。固定两状态探针不代表全地图P95合格。

核心及本轮回归46项通过，其中7项为新增的转换/默认行为/原控制器路径重放/诊断RNG保持检查。额外运行的旧test_native_mppi_cost.py有15通过、20失败；干净main上完全相同20个失败（旧参数/旧去重/旧配置迁移期待），未删除测试或降低断言，未将这些既有问题伪报为全套通过。详见3份测试日志与verification.json。

Checkpoint仍为原WM200k+RL300k总500k，SHA256 `{protocol['checkpoint_sha256']}`；不是WM500k+RL500k。每回合前后WM及Actor张量hash相等；源checkpoint、原配置、历史文件重新hash一致，主main干净；无训练、commit或push。

## 6. 最终技术决策

1. 候选不是大量重复，但局部控制扰动与单一nominal路径限制尚不能完全排除；未获取丢弃路径真实收益，所以不重构候选空间。
2. 当前最有证据的损失环节是价值评价与真实滚动执行的对应关系；唯一根因未被因果识别，不能以一次修正命中主因。
3. 本轮只保留默认关闭的虚拟速度语义修正及完整诊断工具。它解决明确的转换定义问题，但未通过驾驶收益验收。
4. 当前不值得扩大当前MPPI配置评测。继续使用B；不训练第六个头、不改采样预算或新增规则。若以后恢复研究，唯一优先问题应是**同状态首动作评价与真正逐步重规划后续策略的可验证一致性**，先有预注册反事实证据再决定算法变化；本轮不新增该在线模型或实验分支。

## 7. 交付与复现

DATA_FLOW_AUDIT.md含完整图；candidate_coverage.csv含每步每轮精确采样统计；path_and_action_comparison.csv比较原Lattice/SAC/Mean/实际首动作；prediction_vs_actual.csv及historical_origin_ranking.csv严格分H；action_alignment.csv逐步控制和物理对照；replanning_action_deviation.csv记录旧未来计划与后来真控制差；paired_closed_loop_episodes.csv和summary.csv是真实原生回合；paired_statistics.json是场景聚类区间；latency.csv含分阶段实测；experiment_manifest.json和verification.json记录所有控制变量；code_changes.patch包括3个core文件及本轮脚本、协议、文档。

所有修改和新数据在隔离worktree，旧实验/main未覆盖。新NPZ为原生轨迹与快照，并非新checkpoint。运行与保存说明见RUNNING.md。
'''
    with (R/'ROOT_CAUSE_AUDIT.md').open('x') as f:f.write(text)
    # Code/doc-only reproducible diff. Generated CSV/NPZ/logs remain separate artifacts.
    write_code_patch()
    dump('artifact_index.json',{str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in str(p)})
    print('REPORT_AND_PATCH_COMPLETE',len(historical),'historical files verified',flush=True)

if __name__=='__main__':main()
