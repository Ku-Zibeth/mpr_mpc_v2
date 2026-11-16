"""Assemble the final factual report from completed artifacts, no experimental selection."""
import json
from pathlib import Path
R=Path(__file__).parent;ROOT=R.parents[1]
def read(name):return json.loads((R/name).read_text())
v=read('validation_summary.json');p=read('value_ranking_probe.json');selection=read('selection.json');protocol=read('protocol.json')
assert len(v['summary'])==6 and len(p['origins'])+len(p['unavailable'])==9
lines=['# SAC-Centered MPPI 价值评估与搜索分布优化（2026-11-16）','',
'基线 `f1f2bcf79089f396281b21a9d967ec737f6c069b`，初始工作区干净。没有训练、修改 checkpoint、提交或推送。历史日志未覆盖。',
'', '## 修改及边界','',
'- `planning/local_mppi.py`：默认 beta=0.5 乘法权重；旧除法仅保留为消融选项。新增终端动作/Q聚合选择、逐轮熵/ESS、收益分解和最佳Elite动作记录。Gaussian、AR(1)、Top-K、加权均值/方差、64/8/4及std边界保留。最终仍以同一批次、同一定义比较 baseline/mean，不增加动作审批或偏差惩罚。',
'- `planning/evaluator.py`：Candidate a_H、Policy Prior 两种定义；H次reward/next，原折扣和termination mask不变。Q(avg)不重复解码；Q(all)显式解码。固定Q对每plan只抽一次。原SAC全Q均值接口保持不变。',
'- `planning/mppi_ablation.py`、`planning/native_config.py`、`config.yaml`：明确D0–D4单因素配置，默认仅D1，不组合其他修改。',
'- `planning/types.py`、`planning/coordinator.py`、`evaluation_diagnostics.py`：追加离线终端动作/latent/Q成员及定义日志，未改变SAC特征构造。',
'- `tests/test_simplified_tdmpc2_mppi.py`保留旧温度断言并参数化；`tests/test_mppi_value_evaluation.py`补充必要回归。新报告目录下脚本负责独立消融、离线探针和分析。',
'', '## 与官方算法的关系','',
'参考 [TD-MPC2 planner](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py)、[World Model](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/world_model.py)、[配置](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/config.yaml)及[论文](https://arxiv.org/html/2310.16828v2)。读取的blob SHA存于 `references.json`。',
'',
'温度改为官方逆温度beta定义；legacy tau=0.5等价beta=2，与新beta=0.5不等价。当前SAC中心AR(1)采样与官方包含prior轨迹的完整planner仍有差异，本轮不引入额外轨迹类别。',
'',
'Policy Prior使用pi输出的已tanh均值，而非官方随机样本：这是为baseline/候选公平复评所作的确定性适配，并非完全复现随机prior。fork_rng恢复未使用pi样本的RNG，避免D2同时改变随机Q对。a_H保留为H+1接口填充，固定为baseline尾动作，std=0；不参与评分/更新。仍绘制相同数量随机数以保持首H步共同噪声。',
'',
'随机两Q、单plan固定两Q、五Q均值只作用于MPPI；五Q本身已在同一次网络前向生成，诊断在离线额外取成员值。没有加入ensemble不确定性惩罚、Cost、Recovery、Emergency或几何筛选。',
'', '## 可复现协议与测试','',
'Checkpoint SHA256：`'+protocol['checkpoint_sha256']+'`。`sac_interface_verification.json`中1714/559的535维状态、残差、refined_actions和首动作与修改前最大误差全部0。`legacy_reproduction.json`中两个优先起点D0动作序列与历史结果完全相同。',
'',
'相关测试：**40 passed**，详见 `tests.txt`。涵盖新旧权重、AR噪声、Top-K、均值/方差、H/H+1、prefix/terminal折扣、固定Q对、五Q解码、batch公平比较、无增益回退、NaN/Inf、原SAC接口及Cost门控相关断言。没有删除有效测试或降低断言。',
'',
'训练场景范围10000–11999已从训练归档核实。验证集10个新ID位于20000–20999，封存测试集20个新ID位于30000–30999，彼此及旧100/训练集不重叠，详见 `protocol.json`。各组checkpoint、环境、seed规则(1+scenario)、1000步horizon、AR rho=0.8和搜索预算相同。旧100只用于选定诊断起点。',
'',
'每个候选版本只改变一个变量：D0=旧除法/candidate/随机两Q；D1=仅beta乘法；D2=仅prior terminal；D3=仅固定Q对；D4=仅五Q均值。未进行组合或反复调参。',
'', '## 独立验证（每组10回合）','',
'|组|Success|Crash|OOR|Route|Return|真实折扣Return|规划ms|','|---|---:|---:|---:|---:|---:|---:|---:|']
for x in v['summary']:lines.append(f"|{x['configuration']}|{x['success']:.0%}|{x['crash']:.0%}|{x['out_of_road']:.0%}|{x['route_completion']:.2%}|{x['reward']:.2f}|{x['actual_discounted_return']:.2f}|{x['planning_time_ms']:.2f}|")
lines+=['','逐场景结果、合并CSV与场景配对bootstrap95% CI在 `validation_run.json` 指向的结果目录。2000次重采样、seed=1，场景作为簇；多重比较未校正，小样本区间只表示不确定性。下列差值均为候选−B。','','|比较|Success差值 [95% CI]|Return差值 [95% CI]|','|---|---:|---:|']
for x in v['paired_bootstrap95']:
 if not x['comparison'].endswith(' - B'):continue
 lines.append(f"|{x['comparison']}|{x['success_delta']:.1%} [{x['success_delta_ci95_low']:.1%}, {x['success_delta_ci95_high']:.1%}]|{x['reward_delta']:.2f} [{x['reward_delta_ci95_low']:.2f}, {x['reward_delta_ci95_high']:.2f}]|")
lines+=['','选择标准预先写入protocol：D1–D4中success>=B、crash<=B、OOR<=B且平均return>B，按success、return排序。验证集上的筛选仅为provisional，不认定推广收益。', '', '选择结果：`'+str(selection['provisional_candidate'])+'`；'+selection['reason'], '', '## 搜索分布与价值分解','',
'以下为每回合步均值再按场景平均；原始逐轮Mean/Std/Elite/多样性/熵/ESS保存于step日志旁的 `.search.jsonl.gz`。',
'','|组|预测ΔJ|prefix ΔR|折扣terminal ΔQ|ESS|末std steer/throttle|首动作油门Δ|基线保留率|','|---|---:|---:|---:|---:|---:|---:|---:|']
for x in v['summary']:
 if x['configuration']=='B':continue
 lines.append(f"|{x['configuration']}|{x['mppi_predicted_return_gain']:.3f}|{x['mppi_prefix_reward_gain']:.3f}|{x['mppi_discounted_terminal_gain']:.3f}|{x['mppi_effective_sample_size']:.3f}|{x['mppi_final_std_steer']:.4f}/{x['mppi_final_std_throttle']:.4f}|{x['mppi_first_throttle_delta']:.4f}|{x['mppi_baseline_selected']:.2%}|")
lines+=['','不同Q定义的预测增益不能直接解释为真实收益。`validation_summary.json`保留动作修正、采样多样性、熵、Q标准差及探索性油门/事故/收益相关性；这些小样本相关性不构成因果证据。未新增偏差惩罚。',
'', '## 固定状态与价值排序探针','',
'1714:250、559:600重放历史逐步控制，速度误差严格<=1e-4。其余七个场景预先固定为未修改SAC行驶至step100后的共同起点，**不是**原D失败瞬间的状态重建。每个起点两次新进程执行：相同前H个SAC/D0 mean动作，再由相同SAC策略延续至环境原生终止。观测、ego/收集到的traffic、动作候选完全匹配；未声称恢复完整隐藏模拟器快照。',
'',
'同一候选池包含baseline、各版本mean/每轮最佳Elite及64个共同初始Gaussian提案。所有Q对在相同pool上比较；Elite与mean复评同批进行，不把不同Q对产生的旧分数直接比较。D2与D0在相同首H步上的reward、末latent和mask逐项相同。',
'',
'下表预测增益为**D0同批复评**，不是搜索最后一轮随机Q对的旧分数。历史探针曾实际执行H+1步，本轮执行H步后SAC，真实回报不要求与旧探针相等；两优先起点动作本身已确认一致。',
'', '|起点|预测prefix增益|预测terminal贡献|预测ΔJ|真实ΔReturn|terminal latent RMSE B/D|Q对排序一致率|','|---|---:|---:|---:|---:|---:|---:|']
for x in p['origins']:
 y=x['definitions']['D0'];rmse=f"{x['baseline_terminal_latent_rmse']:.4f}/{x['mean_terminal_latent_rmse']:.4f}" if x['mean_terminal_latent_rmse'] is not None else '未达到H'
 lines.append(f"|{x['scenario']}:{x['before']}|{y['predicted_prefix_gain']:.3f}|{y['predicted_discounted_terminal_gain']:.3f}|{y['predicted_gain']:.3f}|{x['actual_discounted_gain']:.3f}|{rmse}|{y['ranking']['mean_ranking_agreement']:.2%}|")
lines+=['','原始 `probe_*_{sac,mean}.json` 保存动作、预测/真实末latent、奖励、actual Return-to-Go、同一terminal动作在预测/真实latent上的5Q、实际prior与SAC动作。`value_ranking_probe.json`记录全部10个Q pair之间的ranking agreement、Spearman、Top8 overlap、raw Q std，以及mean与最佳记录Elite的同批比较。',
'',
'归因：1714优先起点五Q都看好D0均值，真实收益反而下降；换成真实latent后多个成员排序改变。Q对间高一致率没有排除共同模型偏差。559处真实latent上的candidate Q仍普遍看好D0，但SAC延续真实回报未改善。latent误差是无量纲RMSE，不是物理位移。Policy Prior Q与真实SAC RTG的差值同时含价值泛化误差及延续策略不匹配，不能全部叫作Q错误。',
'',
'最终mean的预测分数并不保证优于记录Elite；详见每版本same_batch_audit。这次仅测量该现象，保留原输出机制，不趁机增加候选审批规则。仅对D0前缀实际执行；D2在探针中是相同D0前缀的不同terminal定义，D2自己优化动作的真实效果由独立闭环消融评价。',
'', '## 封存独立测试与结论','']
if selection['provisional_candidate']:
 t=read('test_summary.json');lines+=['按预定规则，仅B与验证候选在全部20个封存ID上测试，无再次选择。','','|组|Success|Crash|OOR|Route|Return|真实折扣Return|','|---|---:|---:|---:|---:|---:|---:|']
 for x in t['summary']:lines.append(f"|{x['configuration']}|{x['success']:.0%}|{x['crash']:.0%}|{x['out_of_road']:.0%}|{x['route_completion']:.2%}|{x['reward']:.2f}|{x['actual_discounted_return']:.2f}|")
 by_test={r['configuration']:r for r in t['summary']};bt=by_test['B'];dt=by_test[selection['provisional_candidate']]
 if dt['crash']>bt['crash'] or dt['out_of_road']>bt['out_of_road'] or dt['reward']<=bt['reward'] or dt['success']<bt['success']:
  lines+=['','**封存测试未满足整体驾驶改善目标，不能把验证集候选D4认定为已优于B的最优版本。** 不使用封存集改选D1/D2，不组合或进一步调参。']
 else:
  lines+=['','封存点估计存在改善，但需结合下列配对区间判断不确定性，不能据此认定全面泛化改善。']
 c=t['paired_bootstrap95'][0];lines+=['',f"候选−B：Success {c['success_delta']:.1%}，95% CI [{c['success_delta_ci95_low']:.1%}, {c['success_delta_ci95_high']:.1%}]；Return {c['reward_delta']:.2f}，95% CI [{c['reward_delta_ci95_low']:.2f}, {c['reward_delta_ci95_high']:.2f}]。区间涵盖0时，不能认定可靠正向边际贡献。逐场景与完整CI见 `test_summary.json` 及 `test_run.json` 指向目录。"]
else:lines+=['无候选满足全部预定条件，未打开封存测试集、未组合修改、未运行额外100场。默认D1是指定的数学定义修正，不等于已选出驾驶表现最优版本。']
lines+=['','未重跑正式旧100；本次完成独立单因素小规模验证和固定状态诊断。只有独立真实success/route/return改善且事故指标不恶化，才有依据认定MPPI贡献；预测增益不能替代这些指标。',
'',
'规划耗时来自至多4个消融worker及2个离线探针共享一张RTX3070的运行，不能与历史独占GPU耗时作性能优化比较。D4后启动时其他worker陆续结束，其较低的表中耗时不是五Q更快的证据。本轮不宣称达到100ms。',
'',
'尚未解决：冻结WM在SAC部署分布下的latent与价值泛化、Q训练策略/真实SAC延续不匹配、少样本结论的不确定性。未通过重训、扩大预算、修改成功条件或规则控制掩盖问题。',
'',
'完整代码Diff：`reports/mppi_value_20261116/code_changes.patch`；相关JSON、脚本及测试输出均在同目录。运行源指纹和checkpoint哈希由实验runner起止核验。','']
pos=lines.index('## 分布与收益分解（验证集，逐回合均值再平均）') if '## 分布与收益分解（验证集，逐回合均值再平均）' in lines else lines.index('## 固定状态与价值排序探针')
extra=['## 分布与收益分解（验证集，逐回合均值再平均）','','|组|Entropy|ESS|末std steer/throttle|Prefix增益|折扣Q贡献|首油门改变量|','|---|---:|---:|---:|---:|---:|---:|']
for r in v['summary'][1:]:
 extra.append(f"|{r['configuration']}|{r['mppi_weight_entropy']:.3f}|{r['mppi_effective_sample_size']:.3f}|{r['mppi_final_std_steer']:.4f}/{r['mppi_final_std_throttle']:.4f}|{r['mppi_prefix_reward_gain']:.3f}|{r['mppi_discounted_terminal_gain']:.3f}|{r['mppi_first_throttle_delta']:+.4f}|")
extra+=['','D1提高ESS并稍减std收缩，但Terminal Q仍主导预测增益，且各版本平均首动作仍偏向增加油门。各组对D0的边际Return置信区间均涵盖0，验证集不能可靠区分四项修改。探索性episode相关系数保存在analysis.json的behavior_association，不作因果结论。','','配对Bootstrap以场景为单位、2000次、seed1，完整95%区间在validation_summary.json及逐场景CSV。D0−B在这10个观测上的Success差异全为0，因此经验区间退化到[0,0]；这不证明总体等价。', '']
lines[pos:pos]=extra
neg=sum(r['definitions']['D0']['predicted_gain']>0 and r['actual_discounted_gain']<0 for r in p['origins'])
lines.extend([f'诊断样本中，{neg}/{len(p["origins"])}个起点出现D0同批预测增益为正而实际增益为负。该比例来自预选故障/救回场景，不是独立总体失败率。',''])
(ROOT/'MPPI_VALUE_EVALUATION_REPORT_20261116.md').write_text('\n'.join(lines))
print(ROOT/'MPPI_VALUE_EVALUATION_REPORT_20261116.md')
