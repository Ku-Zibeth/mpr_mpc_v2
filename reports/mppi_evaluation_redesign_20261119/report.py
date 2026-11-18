"""Final evidence report, provenance and complete source diff. No publishing or model saving."""
import subprocess
import pandas as pd
from study import *

def patch():
    text=subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)
    files=sorted(f for f in R.rglob('*') if f.is_file() and '__pycache__' not in str(f) and (f.suffix in ['.py','.md'] or f.name in ['protocol.json','decision.json','experiment_manifest.json']))
    for file in files:
        result=subprocess.run(['git','diff','--no-index','--','/dev/null',str(file.relative_to(ROOT))],cwd=ROOT,text=True,capture_output=True)
        assert result.returncode==1;text+=result.stdout
    (R/'code_changes.patch').write_text(text)
    subprocess.run(['git','apply','--reverse','--check',str(R/'code_changes.patch')],cwd=ROOT,check=True)

def main():
    p=load(R/'protocol.json');m=load(R/'experiment_manifest.json');h=load(R/'h1_statistics.json');d=load(R/'decision.json');a=load(R/'acceptance.json');closed=load(R/'closed_statistics.json')
    assert load(R/'closed_complete.json')['episodes']==108
    assert '46 passed' in (R/'tests.log').read_text() and load(R/'real_guard_probe.json')['passed']
    assert 'FROZEN_DEFAULT_PARITY default_after' in (R/'default_after.log').read_text()
    assert load(R/'candidate_action_columns_audit.json')['passed']
    changed={name:dict(before=value,after=file_sha(ROOT/name)) for name,value in m['protected_sources'].items() if file_sha(ROOT/name)!=value}
    assert set(changed)=={'planning/local_mppi.py'}
    assert file_sha(ROOT/'config.yaml')==m['original_config_sha256'] and file_sha(Path(p['checkpoint']))==p['checkpoint_sha256']
    assert file_sha(ROOT/'planning/local_mppi.py')==load(R/'core_ready.json')['core_files']['planning/local_mppi.py']
    sources=m['historical_inputs']+load(R/'historical_binary_and_labels_verified.json')
    assert all(file_sha(Path(x['path']))==x['sha256'] for x in sources)
    main_status=subprocess.check_output(['git','status','--porcelain'],cwd=MAIN,text=True);assert not main_status,main_status
    initial_pairs=0;exact_abstention=0
    for scene in p['scenes']:
        for seed in p['seeds']:
            paths=[original_dir(scene,seed),R/'closed'/f'B_{scene}_r{seed}',R/'closed'/f'C_{scene}_r{seed}']
            states=[load(x/'initial.json')['hashes'] for x in paths];assert states[0]==states[1]==states[2];initial_pairs+=1
            if not d['online_options']['replacement_supported']:
                aa=np.load(paths[0]/'trace.npz');cc=np.load(paths[2]/'trace.npz')
                assert all(np.array_equal(aa[k],cc[k]) for k in ['actions','observations','rewards','costs','done','terminated','truncated']);exact_abstention+=1
    dump(R/'verification.json',dict(base_commit=p['base_commit'],remote_main_verified=m['remote_verified_sha'],checkpoint_sha256=p['checkpoint_sha256'],
        protected_core_changes=changed,original_config_unchanged=True,WM_SAC_Lattice_mapping_unchanged=True,checkpoint_unchanged=True,
        historical_sources_rehashed=len(sources),historical_data_unchanged=True,tests_passed=46,new_tests=10,default_action_mean_std_value_rng_bitwise_parity=True,
        sampling_and_update_ast_unchanged=load(R/'core_ready.json')['unchanged_native_sampling_and_update_methods'],real_model_guard_probe=True,
        new_maps=12,seeds=[1,2,3],h1_cases=36,h1_branches=h['branches'],full_closed_episodes=108,native_terminated_episodes_total=108+h['branches'],
        initial_conditions_paired=initial_pairs,B_E_action_and_return_identical=True,unsupported_C_exact_baseline_episodes=exact_abstention,
        margin_frozen=p['margin'],new_map_margin_retuning=False,candidate_action_columns_verified=174,main_status_end=main_status,training=0,new_networks=0,formal100=False,commit=False,push=False))
    def c(metric):return f"{metric['mean']:+.4f}，95%CI [{metric['ci95'][0]:+.4f}, {metric['ci95'][1]:+.4f}]"
    htable='| H1首动作方法（之后全为原SAC） | Gain vs SAC | 地图95%CI | 替换率 |\n|---|---:|---:|---:|\n'
    for role,label in [('B','Native final2'),('C','Uniform Random'),('D','Reward-only all pool'),('G','RQ matched all pool'),('F','Frozen margin')]:
        v=h['selectors'][role];g=v['gain'];htable+=f"| {label} | {g['mean']:+.4f} | [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] | {v['physical_replacement']['mean']:.2%} |\n"
    episodes=pd.read_csv(R/'paired_closed_loop_episodes.csv');means=episodes.groupby('configuration').mean(numeric_only=True)
    ctable='| 完整闭环（每组36回合） | Success | Crash | OOR | Route | Timeout | Discounted Return | 替换率 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for role,label in [('A','A Lattice+SAC'),('B','B 原Native'),('C','C 本轮保护')]:
        v=means.loc[role];ctable+=f"| {label} | {v.success:.2%} | {v.crash:.2%} | {v.out_of_road:.2%} | {v.route_completion:.2%} | {v.timeout:.2%} | {v.discounted_return:.3f} | {v.replacement_rate:.2%} |\n"
    hist=load(R/'historical_statistics.json');f=h['selectors']['F']['gain'];nvr=h['paired']['B-C']['gain'];rq=h['paired']['D-G']['gain'];rnative=h['paired']['D-B']['gain']
    concluded='通过筛选验收，可考虑下一轮真正独立验证' if a['accepted_for_larger_experiment'] else '未达到算法验收，不建议扩大正式100场；保留Lattice+SAC'
    report=f'''# MPPI评价机制重构与闭环筛选结果

**结论：{concluded}。本轮实现的是默认关闭的Baseline保护，不把始终保留SAC算作MPPI成功。**

## 八个问题的直接回答

1. **当前评分优于随机吗？** 新12图×3规划种子H1中，Native−Random为{c(nvr)}。区间包括0，未证明评分提供可靠的额外决策价值；不能仅凭负均值声称所有地图都更差。
2. **Terminal Q导致偏差吗？** 同一258候选池、同一Q成员下，Reward-only−RQ为{c(rq)}，支持终端Q会损害这组最终选择。实际Native仅比较Baseline/Mean；Reward-only−Native为{c(rnative)}，该比较还混有终选资格范围差异，不能单独用它归因Q。高Q不等于已证实训练分布外，当前没有原训练replay latent覆盖用于OOD判定。
3. **Reward-only更适合吗？** 它相对SAC的Gain为{c(h['selectors']['D']['gain'])}，尚未通过正收益及跨地图稳定性门槛。只比RQ少损失不等于真实优于SAC，因此没有移除默认Terminal Q，也没修改冻结WM或SAC特征中的Q。
4. **Baseline-aware有效吗？** 开发集唯一冻结margin={p['margin']:.6f}，旧数据点估计仍负；新地图F为{c(f)}，收益只在{f['positive_maps']}/12图为正，剔除200333后归零，不能视为稳健提升。没有用新地图重选margin。故`replacement_supported=false`时拒绝无证据替换；这是保护，不是学会了H1价值。
5. **候选覆盖不足吗？** 老123分支有正Oracle，证明有限候选中有价值，但不证明整个高斯空间覆盖充分。新图执行方法子集Oracle Gain为{c(h['selectors']['A']['oracle_gain'])}，不是未执行的全部258条轨迹Oracle。没有实际证据支持调std/rho或增加样本，故采样和Elite更新完全未改。
6. **改了哪些代码？** 生产模块仅`planning/local_mppi.py`，新增可选final_selection_policy、冻结margin、replacement_supported和数值诊断字段。旧配置保持legacy原样；保护启用后仅当有支持证据且原生预测差严格超过冻结margin时替换；否则保持原SAC。所有64/8/4计算仍执行，非安全证书、非Recovery、非新价值网络。
7. **修改后真优于Lattice+SAC吗？** 见完整闭环表及验收条件。当前C保护无支持替换，{exact_abstention}/36回合轨迹、观测、reward、cost和终止标志逐位等于A，不能称为驾驶性能提升。
8. **值得正式100场吗？** {concluded}。当前LocalMPPI没有满足同时改善成功率、事故不恶化、正配对收益和实时性的要求；不继续堆网络、样本或任意阈值。

## 1. 历史评分消融：先离线，后新地图

完整读取了用户指定的9份报告/CSV/JSON/补丁，另按上轮artifact manifest核验155份catalog/NPZ/真实标签。全部原123分支保留，不训练模型。

从冻结WM复算r0、H2/H3/H4累计Reward、H10 Reward、Terminal Q和完整H10分数，保留原预测和真实H1标签、控制动作与相对Baseline分数。为精确复现旧预测，保持旧8角色GPU batch再去重，而非换成7行batch后忽略数值差异。初次数值复核失败日志保留，未放宽状态恢复条件。

score_component_ablation.csv中的component argmax是同一有限实测池的诊断，不冒充原Native输出；另有native_actual_final2行复现固定前提−1.0457。所有Spearman/Kendall/Pairwise Accuracy都标为目标不匹配的描述性指标，正式匹配H1/SAC的模型排名仍NOT ESTIMABLE。Baseline-relative只平移分数，不改变起点内排名，更不是真实Advantage。

历史候选动作分布见historical_h1_analysis.csv和figures/historical_positive_action_regions.png。Gaussian真实正收益仅在已执行的分位子集可观察，不把全256条未执行采样标为正/负。Q双成员变更会改变排名，相关诊断在q_rank_stability.csv；没有因此擅自改默认random_pair。

## 2. 新地图与H1公平对照

基础代码/GitHub main为`{p['base_commit']}`。固定地图：{p['scenes']}，规划种子[1,2,3]，每图相同交通seed。排除清单扫描5743份去重历史元数据，并包含WM/SAC训练窗口10000–11999。36个固定pre50状态均可达，没有重选地图、种子或时刻；其中SAC在不同规划种子下可能重复，始终以12图聚类，不当作36张独立地图。

同状态保存全部4轮Gaussian、Baseline及WeightedMean，共258序列；预注册独立随机数均匀选非Baseline的唯一首动作，不改变环境RNG。B/E严格复现原最后Baseline/Mean两个分数的选择，E与B合并同一实际分支；D按同池Reward选最大，G按同池共享Q评分选最大，额外G是为了分离评分与候选范围混杂。F只测试开发期冻结的一个margin，不扫新阈值。7方法去重后实际{h['branches']}条H1分支。

每分支重置同场景并调用策略重建前49步历史，再注入一周期动作，随后完全原SAC直到原生done；逐位检查观测、车辆、交通、全局/引擎RNG、latent/535输入、policy历史、实际上一动作及控制映射。所有Baseline尾轨迹重现源A。原模型训练和网络输入、物理控制顺序、Reward/Cost/Done均未改。

最终复核修正了新诊断工具的一处动作列对齐错误：汇总动作列曾误用最后一个候选循环变量。149行动作诊断列已按candidate_id回填，并逐条核验174条实际执行轨迹；原始表和修复前源码保留于source_versions。真实执行、回报标签、选组统计及预注册decision完全未变，详见candidate_action_columns_audit.json。

{htable}

Native的负平均高度受200465影响：剔除该图的Native Gain变为正，所有图仍完整保留，不以剔除坏图制造正收益。F的平均正收益又由单张200333驱动。所有逐图、逐seed、地图bootstrap2000次和LOMO值均保留于h1_statistics.json，不能把单图信号当成泛化。

## 3. 最小代码变更与验证

按预注册分支选中`{d['branch']}`，没有因为新测试不理想再调margin。默认原文件config.yaml逐字不变。新增保护只改变最后的替换资格，原Gaussian/AR1采样、8Elite与4轮Mean/Std更新、WM reward/dynamics/terminal-Q全部保留；不引入TTC、几何、Emergency或其他状态规则。

46项相关测试通过。真实冻结模型默认路径的动作、Mean、Std、值及CPU/GPU/私有RNG与修改前逐位相同；AST核验采样/更新方法完全相同。实状态探针验证535输入与原SAC首动作不变、无支持时返回SAC、支持开关在零margin时确能产生替换；这两次探针输出均未执行，不能当作收益证据。无支持时也完整计算262候选评价，未跳过搜索伪造速度。

新增数值日志baseline_guard_enabled/blocked/reason_code、replacement_supported和final_selection_margin。原因1是缺少支持证据、原因2是冻结margin未达到，均不宣称SAC动作安全；与原numerical fallback分开。所有模型权重、535维顺序、映射、旧实验文件和checkpoint SHA `{p['checkpoint_sha256']}`保持不变。

## 4. 真正逐步重规划的108回合

H1方法字母与完整闭环A/B/C分开：闭环A=原SAC，B=原Native，C=本轮保护。每周期重新Lattice→SAC→可选MPPI，只执行首动作后env.step，直到原生终止；绝非固定10步重放。36组初始观测、车辆、交通及RNG哈希匹配。

{ctable}

C−A折扣收益：{c(closed['comparisons']['C-A']['discounted_return'])}；C−B：{c(closed['comparisons']['C-B']['discounted_return'])}。地图CI、LOMO及每个seed均见closed_statistics.json。零替换仅说明保护工作，不能宣称MPPI成功；acceptance.json明确检查非零替换与额外成功。

## 5. 计算性能

单LocalMPPI计时直接包围LocalMPPI.plan；完整planner包围同步agent.act；原生control_cycle=agent.act+env.step，均分开记录。闭环采集是6进程共享GPU，包含冷启动，不用它冒充单进程实时性认证。

全部回合完成后，在前两张预定新图同一真实pre50状态，单进程做5暖机+50完整planner调用/方法，共300次测量；没有执行探针输出动作。独立完整planner P95：{a['isolated_planner_p95_ms']}ms。该探针是planner-only，不是完整物理控制周期；完整cycle的实测共享负载数据在latency.csv。无法达到100ms时明确保留瓶颈，不删预算或必要控制计算。

## 6. 验收、交付与限制

验收条件：{a['conditions']}。是否进入更大规模实验：**{a['accepted_for_larger_experiment']}**。本轮没有证明Success>Lattice+SAC。保护实验结果只说明可拒绝无支持替换，不解决缺失的H1/SAC延续价值；默认控制器未切换。

新图对旧研究独立，但H1用于预注册机制分支选择，随后在同图做完整闭环属于筛选验证，不是另一个完全封存的最终验证集。没有原训练latent replay覆盖，不能仅凭乐观Q宣称OOD；没有未执行候选的真实收益，不能伪造全池正收益比例；没有同语义合格H1预测头，不声称正式H1排名准确率。

交付含历史评分分量、Native/Random与Reward/Q对照、开发及冻结margin、新图候选覆盖、108原生回合、summary/latency、完整patch与figures。所有新内容在隔离工作区`{ROOT}`，主目录未覆盖，无训练/新网络/commit/push/正式100场。原WM200k+RL300k仍为总500k，不换权重。复现步骤见RUNNING.md。
'''
    with (R/'MPPI_EVALUATION_REDESIGN_REPORT.md').open('x') as f:f.write(report)
    patch()
    dump(R/'artifact_manifest.json',{str(f.relative_to(R)):dict(sha256=file_sha(f),bytes=f.stat().st_size) for f in sorted(R.rglob('*')) if f.is_file() and '__pycache__' not in str(f)})
    print('COMPLETE_REPORT_AND_DIFF',concluded,flush=True)
if __name__=='__main__':main()
