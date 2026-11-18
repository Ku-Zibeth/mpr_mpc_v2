"""Evidence report, final preservation checks and complete source-only diff."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from analyze import R, OLD, load, dump, table
from tools.first_action_study import file_sha
from tools.first_action_failure import cluster_bootstrap


def fmt(value, percent=False):
    if value['mean'] is None:
        return 'N/A'
    f = 100 if percent else 1
    return f"{value['mean']*f:.3f} [{value['ci95'][0]*f:.3f}, {value['ci95'][1]*f:.3f}]"


def main():
    p = load(OLD/'protocol.json')
    analysis = load(R/'ranking_analysis.json')
    summaries = analysis['summaries']
    state = load(R/'state_oracle_ranking.json')
    dynamics = load(R/'dynamics_error_analysis.json')
    numeric = load(R/'model_numeric_diagnostics.json')
    oracle = load(R/'oracle_stability.json')
    chosen = load(OLD/'training_decision.json')['selected']
    prior = load(OLD/'verification.json')
    tests = (R/'tests.log').read_text()
    assert '82 passed' in tests
    # Existing raw data/checkpoints/configs and prior sources must be byte-identical.
    with (R/'data_sha256.csv').open() as stream:
        inventory = list(csv.DictReader(stream))
    changes = [r['path'] for r in inventory if file_sha(Path(r['path'])) != r['sha256']]
    assert not changes, changes
    assert file_sha(Path(p['checkpoint'])) == p['checkpoint_sha256']
    manifest = load(R/'experiment_manifest.json')
    assert file_sha(Path(p['source_environment_config'])) == manifest['config_sha256']
    assert all(file_sha(ROOT/f)==h for f,h in prior['source_sha256'].items())
    assert file_sha(ROOT/chosen['checkpoint']) == chosen['checkpoint_sha']
    for folder in [ROOT, ROOT.parent/'mpr_mpc_v2']:
        assert subprocess.check_output(['git','diff','--name-only'],cwd=folder,text=True).strip()==''
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=folder,text=True).strip()==p['base_commit']
    with np.load(R/'planner_parity_0.npz') as a, np.load(R/'planner_parity_1.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    with (R/'scene_level_ranking.csv').open() as stream:
        scenes = [r for r in csv.DictReader(stream) if r['horizon']=='1' and r['scorer']=='qb_original']
    with (R/'branch_integrity.csv').open() as stream:
        branches = list(csv.DictReader(stream))
    assert len(branches)==576 and all(r['checks_passed']=='True' for r in branches)
    assert all(r['equal'] for r in state['current_qb_control'])
    # Keep the earlier ranking-only bootstrap artifact, then deliver combined results.
    previous_bootstrap = R/'paired_bootstrap_statistics.json'
    backup = R/'paired_bootstrap_ranking_only.json'
    assert not backup.exists()
    previous_bootstrap.rename(backup)
    combined = load(backup)
    combined.update(state_oracle=state['paired_improvements'], reverse_score=numeric['reverse_paired_comparison'],
                    oracle_stability=oracle, horizon_target_agreement=load(R/'horizon_target_agreement.json'))
    dump('paired_bootstrap_statistics.json',combined)
    # All reported input-state swaps are terminal-Q diagnostics, never a mixed-time Q_B.
    decisions = dict(
        reproduced_validation_pair_accuracy=analysis['old_validation_accuracy_reproduced'],
        sign_or_alignment_bug_found=False, full_prompt_received=True,
        evidence='Map generalization gap and weak action-relative discrimination; no unique causal allocation to state coverage, objective or representation.',
        current_qb_has_candidate_future_latent_input=False,
        future_dynamics_is_proven_primary_cause=False,
        terminal_q_state_oracle_has_some_regret_improvement=True,
        terminal_q_state_oracle_restores_reliable_ranking=False,
        candidate_space_has_positive_measured_actions=True,
        candidate_gain_stability_across_new_maps_proven=False,
        h1_is_target_aligned=True, h1_is_independently_proven_best_horizon=False,
        new_training=0,new_environment_runs=0,formal100=False,
        online_enabled=False,default_planner_changed=False,
        next_primary_direction='Audit same-origin action-advantage supervision and state/context coverage under fixed features; do not simply widen/retrain the same absolute-return Q_B.',
        dynamics_retraining_recommended_now=False,
        new_independent_maps=[],seen_validation_is_retrospective=True)
    dump('decision.json',decisions)
    val=summaries['seen_validation/H1/all_available/qb_original']
    rev=summaries['seen_validation/H1/all_available/qb_reversed_diagnostic_only']
    st1=state['paired_improvements']['all_seen/H1/matched_endpoint_subset/J01_actual_latent_same_predicted_mask']
    lines=['# MPPI First-Action Value Failure Attribution — Phase 2（2026-11-18）',
        '\n## 结论：十个问题',
        f"\n1. **35.64%可准确复现，不是已发现的符号或样本错位。** 原方向地图宏准确率={fmt(val['pair_accuracy'],True)}%；训练68.662%、内部早停59.077%，验证35.638%。可确认跨地图泛化落差和动作相对收益区分偏弱，不能唯一证明是训练覆盖、监督目标还是表示信息缺失。",
        f"\n2. **评分方向没有反转Bug。** 输出是正scale还原的raw Return，argmax；Ranking loss为softplus(-sign(真实差)×预测差/10)，方向正确。反向准确率={fmt(rev['pair_accuracy'],True)}%，但Regret由{val['top1_regret']['mean']:.3f}增至{rev['top1_regret']['mean']:.3f}；反向也没有稳定优于随机的证据，禁止据此取负部署。",
        '\n3. **没有证据证明H1在独立地图上明显优于H2/4/10。** H1与执行首动作的目标定义一致；其他时域只有6张Pilot地图、且都属于该模型fit数据，H1训练模型跨H比较属于目标外诊断。不能把更短时域较低的原始Regret直接解释成规划更优。',
        '\n4. **候选未来WM Dynamics不能直接解释该Q_B的35.64%。** 此模型输入当前535+a0，不接收候选的预测未来latent；当前Encoder表示、共享WM派生上下文仍可能间接影响。未将当前535中的latent替换成未来latent，避免时刻错误和混合状态。',
        f"\n5. **真实末端状态没有可靠恢复原TD终端Q的排序。** H1纯latent替换的准确率变化={fmt(st1['pair_accuracy_improvement'],True)}个百分点，CI跨0；Regret改善={fmt(st1['regret_improvement'])}，说明Dynamics并非毫无影响，但不足以认定它是主要瓶颈。该对照不等于Q_B的输入替换或在线收益。",
        f"\n6. **候选确有正收益，但跨地图稳定潜力尚未充分证明。** 既有验证6图中仅{oracle['seen_validation']['meaningful_positive_maps']}图含>1收益候选；Oracle={fmt(oracle['seen_validation']['summaries']['oracle_gain'])}，减去原固定近等阈值1后={fmt(oracle['seen_validation']['summaries']['oracle_gain_above_fixed_tolerance'])}，CI跨0。Oracle还带有事后最大化偏差，不能冒充可选政策。",
        f"\n7. **跨地图泛化不足。** fit只有{numeric['fit_maps']}图/{numeric['distinct_fit_origins']}个不同当前状态，虽有{numeric['fit_rows']}条动作标签。验证预测候选分差范围均值0.462，真实收益范围4.916；16个状态并不等于126个独立上下文。",
        '\n8. **只值得继续首动作相对收益可辨识性研究，不值得继续原样训练同类绝对Return评分头。** 下一步唯一主方向：冻结现有表示，审计同起点action-advantage监督及状态/道路交互覆盖。先预注册新诊断，不扩网络、不在已看过的验证图上反复选参。本轮未实施新训练。',
        '\n9. **暂不转向整个WM或Dynamics重训。** 原TD终端Q有局部状态误差贡献，但真实状态/Reward oracle仍未形成可靠排名，且不是当前Q_B的未来状态通路。Dynamics只能作为后续受控子问题，不应自动替代本轮主要方向。',
        '\n10. **不具备在线闭环验收条件。** 无新价值函数通过独立验证，无新闭环正收益与安全性证据；保留原默认MPPI与Lattice+SAC，未运行正式100场。',
        '\n## 1. 版本、数据和统计边界',
        f"\n主main、实验worktree和2026-11-18 GitHub main均为`{p['base_commit']}`。实验分支`experiment/mppi-first-action-20261117`；主目录干净，上一轮新增源码/报告/测试仍未跟踪，尚未进入GitHub。新工作仅在`{R}`及独立tools模块，未commit/push/reset/checkout/clean/pull。",
        f"\nCheckpoint SHA256=`{p['checkpoint_sha256']}`，沿用WM200k+RL300k总500k模型；没有悄悄换成WM500k+RL500k。配置SHA与包版本见experiment_manifest.json，逐文件数据SHA见data_sha256.csv；本轮结束已重新校验所有{len(inventory)}个旧输入文件。",
        '\n576分支来自18图、36起点：571条非重复SAC接管分支，另4条原TD Actor/完整MPC对照和1条同seed重放。主要排序排除后两类，但完整索引保留全部576。H1=286条唯一SAC分支；H2/H4/H10各95条，只在6张Pilot图。无新场景或交通随机流重复；旧验证图现已属于“已看过数据”，不是本轮新独立验证。',
        '\n真实Reward取原info.tdmpc2_reward，gamma=.995，近似平局仍为abs(ΔReturn)≤1。先同origin计算，后地图等权聚合，2000次整图Cluster Bootstrap、seed20261118。95%区间与上一轮seed1略不同，但35.637516%的点估计逐位复现。6图、4图的区间可靠性有限，不能用Pair数掩盖。',
        '\n## 2. 动作、收益和早终止审计',
        '\n提交给env.step的首动作及完整固定前缀均与保存候选一致；Baseline完整保留；有限性、边界、起点观测/latent与可记录状态/RNG哈希、gamma回报分解和done=terminated OR truncated均通过。按精确动作去重后的代表候选没有重复，也没有新增clip坍缩；原Gaussian裁剪前噪声未保存，不能反推裁剪比例。',
        '\n原始动作是归一化steer/throttle物理控制命令，而非SAC残差；第0维转向、第1维油门/刹车，当前MetaDrive源码支持这一顺序。记录的是env.step提交值，未保存分支运行时实测轮角/引擎力，SI执行差异标为缺失，不假装已验证。action_mapping_scope.json保存源码位置和SHA。',
        '\nH=1/2/4/10均为固定候选前缀→原Lattice+SAC每步反馈接管到原生done；不是前缀内反复跑MPPI。9条提前原生终止样本均保留在收益和Oracle分析；没有对应z_H时不造状态。纯状态对照标为matched_endpoint_subset，完整样本另报joint actual-state/reward/native-terminal；缺失Baseline时不伪造Baseline指标。具体样本见early_terminal_samples.json。',
        '\n旧JSON actual_continuation_return一律是第一步后的RTG；H>1包括剩余固定候选前缀，不能叫纯SAC延续。本轮索引分列after_first、after_fixed_prefix及prefix_return，未改旧标签。原NPZ没有逐步crash/OOR/success，只记录累计episode标志和最终终止；能定位提前原生终止，但不能补造首个事故的时刻。',
        '\n## 3. 方向、Pair规模与模型实现',
        '\n|Split/H1|原方向Accuracy% [95%CI]|反方向Accuracy%|原Regret|反Regret|\n|---|---:|---:|---:|---:|']
    for split in ['fit','inner_stop','seen_validation']:
        a=summaries[f'{split}/H1/all_available/qb_original'];b=summaries[f'{split}/H1/all_available/qb_reversed_diagnostic_only']
        lines.append(f"|{split}|{fmt(a['pair_accuracy'],True)}|{fmt(b['pair_accuracy'],True)}|{fmt(a['top1_regret'])}|{fmt(b['top1_regret'])}|")
    lines.extend([
        '\n训练/推理F2都由原current535+同一candidate物理a0构造；加载同一个冻结checkpoint，归一化均值/方差从fit数据重算与保存buffer最大误差均为0。正target_scale、不含未来信息、larger-is-better与loss方向均已核实。没有在反向结果上重新选模型或拟合校准。',
        '\n验证336原始Pair中140有效、196近平局；先起点再地图宏平均35.638%，混合所有有效Pair为45.357%（仅作诊断，不替代主指标）。地图94201仅3对有效、94928仅14对有效，准确率分别0%和8.333%，各仍占一张地图权重；这解释了宏/微平均差异，不能因此删掉它们或改平局阈值。',
        '\n|H|原始Pair|有效Pair|近平局Pair|\n|---|---:|---:|---:|'])
    for h,c in analysis['pair_counts'].items():
        lines.append(f"|{h}|{c['raw']}|{c['effective']}|{c['tied']}|")
    lines.extend([
        '\n各图Pair数、分场景/起点、动作幅度、真实收益差、初速、episode碰撞/越界及固定交通density分层见scene_level_ranking.csv、pair_diagnostics.csv与ranking_analysis.json；所有分层Accuracy按地图bootstrap。真实道路曲率未保存，仅能使用原coarse-path曲率特征代理；traffic_density恒定0.1，没有虚构多个密度组。代理与实测道路曲率明确区分。',
        f"\n最终fit归一化Huber={numeric['final_fit_normalized_huber']:.6f}，rank loss={fmt(numeric['final_fit_rank_loss']['ranking_loss'])}，接近无差异log2≈0.693；正确方向pair probability={fmt(numeric['final_fit_rank_loss']['pair_probability'])}。这支持动作分差压缩的观察，但不是“某个loss权重就是唯一根因”的因果实验，不据此调整系数。",
        '\nfeature_shift_and_sensitivity.csv报告距fit当前上下文的标准化距离与a0梯度。fit=8图/16起点，stop=4图/8起点，旧validation=6图/12起点；模型/epoch经过小型stop集选择，存在选择与分布不确定性，不能把地图相关性直接当唯一原因。',
        '\n## 4. 时域：同起点配对，而非混合不同地图',
        '\n下表只比较同6张Pilot地图。原Q_B只训练过H1，且这些图属于fit；其跨H结果是目标外分析。原H10评分列固定同一套旧分数，真实标签随干预长度改变。原始Regret的收益尺度也随H变化，不将其直接当规划优劣。',
        '\n|H|Q_B Accuracy%|原H10评分 Accuracy%|Q_B Regret|原H10 Regret|\n|---|---:|---:|---:|---:|'])
    for h in [1,2,4,10]:
        a=summaries[f'all_seen/H{h}/common_pilot/qb_original'];b=summaries[f'all_seen/H{h}/common_pilot/original_h10_score']
        lines.append(f"|{h}|{fmt(a['pair_accuracy'],True)}|{fmt(b['pair_accuracy'],True)}|{fmt(a['top1_regret'])}|{fmt(b['top1_regret'])}|")
    agreement=load(R/'horizon_target_agreement.json')
    lines.append('\nH10与H1真实收益排序Spearman='+fmt(agreement['H10_vs_H1']['spearman'])+'。这显示干预目标差异，不能把H10标签当成H1；也不能把H1→Baseline接管当作新MPPI每步重规划的π_new闭环表现。')
    lines.extend([
        '\n## 5. Predicted/Actual State Oracle 与Dynamics',
        '\n**Q_B当前输入对照**：36个origin在正确current535+a0接口下A/B输入完全相同，没有候选未来latent可替换。换成z_H并留下原路径特征，会成为混合时刻输入，且评价Q_B(s_H,a0)而非原目标Q_B(s_t,a0)，因此未做这种无效实验。当前535中的WM派生coarse value仍可能间接影响表示，不声称完全排除所有WM因素。',
        '\n**原TD终端Q对照**：同一冻结Encoder/Q、候选前缀、候选末端动作、gamma、固定历史Q成员对。J00=预测Reward+预测mask×Q(z_pred)；J01仅替换z；J10仅替换Reward；J11替换两者但保持原mask。joint另使用原生终止mask，明确不是纯Dynamics效果。所有真实观测只用于离线oracle，未训练或输入在线。TD Q训练涉及编码起点和imagined latent；不能保证新地图真实末端latent与其训练分布同分布，也不武断称其从未见过真实encoded状态。',
        '\n|H|Latent-only Accuracy变化，百分点 [95%CI]|Regret改善|\n|---|---:|---:|'])
    for h in [1,2,4,10]:
        a=state['paired_improvements'][f'all_seen/H{h}/matched_endpoint_subset/J01_actual_latent_same_predicted_mask']
        lines.append(f"|{h}|{fmt(a['pair_accuracy_improvement'],True)}|{fmt(a['regret_improvement'])}|")
    lines.extend([
        '\nH1有18图，其余仅6图且9条缺失末端，不能把跨行变化当纯H因果效应。状态替换有局部Regret贡献，Accuracy提升区间各H均跨0；Q/延续策略与训练覆盖依然存在。J00/J01/J10/J11完整评分、分split和严格同候选配对差异见state_oracle_ranking.json和predicted_vs_actual_state.csv。',
        '\n按请求H统计latent MSE/cosine、rollout Reward误差、用真实前驱latent的teacher-forced Reward误差；差分Dynamics误差与排序错误在同origin内相关，再按地图bootstrap，见dynamics_error_analysis.json。没有仅凭latent距离较大断言因果，也没有把latent L2换算为米；缺少WM物理decoder及逐步真实XY/yaw，不能补造物理预测误差。',
        f"\n数值审计发现：不同8候选batch重算H10相对旧记录最大分数差{dynamics['cached_h10_recomputed_score_max_abs_difference']:.6f}。已归档初版state_oracle_recompute_v1；最终H10精确复用原predicted endpoint/prefix/mask，Actual替换保持原258候选Q batch，同形状Q校验最大差={dynamics['cached_h10_same_shape_q_check_max_abs_difference']:.6f}。这是修复诊断可比性，未修改模型或默认规划评分。H1/2/4因旧文件无对应中间缓存，明确使用冻结模型重算。",
        '\n## 6. Oracle、Random、Model 与影响分析',
        '\n|Split/H1|Oracle Gain|Model Gain|固定Random Gain|>1收益地图数|\n|---|---:|---:|---:|---:|'])
    for split,o in oracle.items():
        s=o['summaries'];lines.append(f"|{split}|{fmt(s['oracle_gain'])}|{fmt(s['model_gain'])}|{fmt(s['random_gain'])}|{o['meaningful_positive_maps']}/{o['total_maps']}|")
    vo=oracle['seen_validation']
    lines.extend([
        f"\n旧验证Model Gain={fmt(vo['summaries']['model_gain'])}；移除最大Oracle图{vo['largest_gain_map']}后，Model Gain={fmt(vo['leave_largest_map_out']['model_gain'])}。这仅是影响诊断，绝不把删图后结果叫新验证。Oracle包含Baseline所以必然非负，raw Oracle CI>0不等于一个已知、可泛化的动作选择政策；固定阈值1与any-positive比率、同seed重复非独立等限制均保留。",
        '\nRandom沿用上一轮预先固定的Gaussian候选角色，未重新抽有利样本；另给均匀非Baseline候选期望。逐origin的Baseline、Random、Oracle、Model Selected和Model Worst均在oracle_candidate_analysis.csv，含正收益比例、至少一个正收益候选、选正/误选负比例、收益差与Regret。候选子集并非完整Gaussian空间。',
        '\n归因判定更接近情况A：确有有用候选，而当前评价不能可靠识别；同时新地图收益潜力较弱且受少数场景影响。不能排除候选覆盖不足，但还没有证据宣布搜索空间全无价值并重构Lattice。情况C（新模型闭环有效/无效）本轮没有新闭环，不冒充已验证。',
        '\n## 7. 代码、测试、实时性和下一步',
        '\n仅新增离线审计/统计、冻结推演、绘图与11项回归测试。原71项全部保留，82 passed。覆盖索引、Reward索引、H分解、早终止/截断、提交动作顺序/边界、同origin匹配、方向/平局、Oracle、地图bootstrap、输入隔离与真实默认规划输出一致性；测试成功不等于模型有效。',
        '\n在原main与实验worktree各加载同一冻结WM与同一保存z0/Baseline，以相同RNG和默认64/8/4运行一次LocalMPPI，不启动MetaDrive；selected序列及标志逐元素一致。Checkpoint、原配置和全部旧数据文件结束时重新SHA核验，原在线文件无diff。没有改WM/SAC/Lattice/535/Action Mapping/Reward/Cost/Cost门控或默认采样/输出。',
        '\n本轮没有新在线方案，故没有新增在线计算负担或新P95结论。旧同硬件完整规划P95：B≈76ms，M1≈104ms，M2≈117ms，M4≈141ms；仍高于100ms参考周期，不因离线Oracle改善而免除闭环收益、安全性和耗时门槛。',
        '\n唯一下一阶段主方向：在冻结表示下研究**同起点首动作相对优势的可辨识性和上下文覆盖**，先明确什么信息能稳定解释动作收益差，再决定是否需要新的监督形式。不得直接加大MLP、翻转分数、切换在线选择、重训WM或复用已看过validation冒充独立验证。本轮不自动实施这个研究方案。',
        '\n## 图表和交付索引',
        '\n以下图均另存可导出的PDF，H时域曲线使用同6图；误差点按origin显示，统计区间按地图计算。'])
    for name,title in [('horizon_accuracy','H排序准确率'),('horizon_regret','H Top-1 Regret'),
                       ('predicted_vs_actual_state','Predicted/Actual终端状态'),('oracle_vs_model_gain','Oracle/Model逐图收益'),
                       ('scene_accuracy','逐图排序准确率'),('dynamics_vs_ranking_error','Dynamics差分误差与排序错误')]:
        lines.append(f'\n![{title}](figures/{name}.png)')
    lines.append('\n所有JSON/CSV/NPZ、完整代码diff、版本SHA、初始失败日志和诊断重算初版均保留在本目录。PHASE0_1_AUDIT.md与初始experiment_manifest.json是收到截断prompt时的阶段快照；本报告及verification.json为收到完整prompt后的最终交付。没有修改上一轮文件或重新运行其环境采集。')
    (R/'MPPI_FIRST_ACTION_FAILURE_REPORT_20261118.md').write_text('\n'.join(lines)+'\n')
    sources=[ROOT/'tools/first_action_failure.py',*sorted(R.rglob('*.py'))]
    patch=[]
    for f in sources:
        diff=subprocess.run(['git','diff','--no-index','--','/dev/null',str(f.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True)
        assert diff.returncode==1
        patch.append(diff.stdout)
    (R/'code_changes.patch').write_text(''.join(patch))
    dump('verification.json',dict(
        complete_prompt_processed=True, head=p['base_commit'], old_files_sha256_checked=len(inventory), old_files_changed=changes,
        checkpoint_sha256=p['checkpoint_sha256'], checkpoint_unchanged=True, config_unchanged=True,
        main_and_experiment_default_planner_output_equal=True, original71_tests_retained=True, new_tests=11, tests='82 passed',
        all576_indexed=True, early_terminal_samples_retained=9, current_QB_validation_accuracy_reproduced=True,
        current535_not_replaced_by_future_latent=True, only_terminal_Q_oracle_uses_true_successors=True,
        metadata_note='Pure state counterpart undefined when native done precedes H; no invented future.',
        original_online_sources_changed=False, new_training=0, new_environment_runs=0, formal100=False, commit=False, push=False,
        source_sha256={str(f.relative_to(ROOT)):file_sha(f) for f in sources},
        required_figures=6, figure_formats=['png','pdf'], models_still_offline=True,
        failed_internal_attempts=['audit_syntax_attempt.log','state_oracle_syntax_attempt.log','planner_parity_serialization_attempt', 'state_oracle_recompute_v1'],
        summary_decision='Do not deploy or invert Q_B. Investigate action-relative supervision/context coverage before more value training; not enough evidence to choose WM Dynamics as primary cause.'))
    print('Phase2 complete. All inputs preserved;82tests;no training/environment runs/online changes.',flush=True)


if __name__=='__main__':main()
