"""Concise delivery generated from actual training/validation/outcome artifacts."""
import json
from pathlib import Path
R=Path(__file__).parent;ROOT=R.parents[1]
def read(name):return json.loads((R/name).read_text())
p=read('protocol.json');fit=read('predicted_training.json');oracle=read('actual_training.json');e=read('existing_pool_evaluation.json');d=read('development_comparison.json');cost=read('cost_validation.json');m=read('dataset_manifest.json')
lines=['# MPPI专用SAC延续价值与候选复评','',
'基线 `'+p['base_commit']+'`；修改前工作区干净。原WM、Q、SAC、Lattice及535维输入/动作映射保持不变。**只训练了独立轻量Value Head**，原checkpoint未覆盖，未commit/push/reset。',
'','## 修改内容','',
'- `planning/sac_value.py`：独立512→128→64→1 SiLU网络（73,985个可训练参数），输入/目标归一化单独保存。模型兼容性绑定原WM+SAC checkpoint、gamma、H及环境/Lattice/Residual配置。未经独立预测latent排序验证的文件不能由在线配置加载。真实latent输入对照头不能用于在线。',
'- `planning/evaluator.py`：新增状态价值终端分支，J=预测prefix Reward+原discount/mask×V_SAC；不调用原Q/pi，不使用未执行a_H。SAC原特征路径仍使用原Q。缓存候选统一复评只重算终端模型，复用原Reward、Cost和WM rollout。',
'- `planning/local_mppi.py`：可选audit保留四轮64条实际Gaussian轨迹和当轮分数，统一参考批次复评全部256条实际采样轨迹，得到全局Best Elite及最终轮Best Elite，并与Baseline/Mean同池比较。**默认输出仍只在Baseline/Mean之间选择**，没有因为Elite分数最高就改为执行Elite。',
'- `planning/types.py`、`planning/native_config.py`、`planning/coordinator.py`、`config.yaml`：新增侧模型、审计与连续Cost配置/诊断；默认仍为旧Q、audit=false、cost_weight=0，未验证新头没有启用。',
'- Cost通过现有CostModelValidator的候选排序证据才可作为连续惩罚；没有TTC、几何拦截、Emergency、Recovery或手工动作候选。验证在WM独立副本进行，避免修改原SAC依赖的Cost就绪缓冲区。',
'- Gaussian/AR(1)、64/8/4、std/rho、训练阶段及原始训练入口均保留。audit仍为**262次完整候选WM推演**，额外对缓存的258个末端状态统一算一次终端价值；不增加动作采样或重复H步reward/dynamics。随机两Q模式也在共同终端批次使用同一Q对，不能直接用不同迭代旧分数选Best。',
'','## 独立头训练与数据','',
'12个预先划定的新60000段场景：8个用于优化，4个仅用于内部早停，与开发10场景、指定验证6场景及旧20场/正式测试完全隔离。72条新原生终止轨迹：每场景step50/150，SAC baseline及两个预先固定的AR(1) Gaussian前缀。扰动为原Gaussian池的固定第1/2条，未按预测价值挑选。',
'',
'各分支在新进程严格匹配同一初始观测、车辆/交通及可记录RNG。保存真实观察、动作、tdmpc2_reward、Cost、terminated/truncated、预测latent、真实编码latent和MC Return-to-Go。H=10后由原冻结SAC接管至原生终止；滑动窗口也仅来自实际执行的H步前缀，其后确为同一SAC策略。提前终止保留，不补造未来状态。可记录状态一致不等于完整隐藏模拟器快照。',
'',
'主输入是冻结WM推演的预测z_H，监督目标是实际SAC接管后的RTG，gamma=0.995；不是Q(s_H,a_H)标签。a_H只保留H+1接口填充，采样/均值的有效维度为H，填充末行固定且std=0。没有多算r_H或训练未执行a_H的价值。',
'',
'训练损失：按场景平衡的归一化SmoothL1 + 同起点真实分支回报差的SmoothL1（固定等权）。Adam lr=3e-4，batch256，最多100 epochs、内部早停patience12。未用开发/验证选epoch、网络结构或学习率。',
'',f"训练窗口{m['train_examples']}个、内部早停窗口{m['inner_stop_examples']}个，来自实际轨迹且高度相关，**不是同等数量的独立场景**；训练配对{m['training_pairs']}对。主头实际运行{len(fit['history'])}个epoch，最佳第{fit['best_epoch']}个；真实latent对照头运行{len(oracle['history'])}个epoch，最佳第{oracle['best_epoch']}个。", 
'',
'权重：`models/sac_value_candidate.pt`（预测latent主头）、`models/actual_latent_oracle.pt`（实际latent输入对照）。后者仅用于诊断，不是真正的最优价值oracle，也不证明表示能力的理论上界。原32起点和172分支只读保留；训练文件清单、SHA及分组见dataset_manifest.json。',
'','## 原候选池上的单次独立价值验证','',
'指定原验证6个场景/12个起点，与训练地图隔离；并非新封存测试。开发/验证数据没有用于拟合或内部早停。旧Q(a_H)只作为原评分的诊断基准，不作为新V_SAC监督目标。以下CI按场景聚类bootstrap2000次、seed1，未作多重比较校正。','',
'|模型|终端RTG成对排序准确率 [95% CI]|终端Spearman [95% CI]|总J MAE|总J排序准确率|Top-1 Regret|','|---|---:|---:|---:|---:|---:|']
for x in e['results']['validation']['summary']:
 lines.append(f"|{x['model']}|{x['terminal_pair_accuracy']:.1%} [{x['terminal_pair_accuracy_ci95_low']:.1%},{x['terminal_pair_accuracy_ci95_high']:.1%}]|{x['terminal_spearman']:.3f} [{x['terminal_spearman_ci95_low']:.3f},{x['terminal_spearman_ci95_high']:.3f}]|{x['j_mae']:.2f}|{x['total_pair_accuracy']:.1%}|{x['top1_regret']:.2f}|")
lines+=['',
'**主头没有通过预测latent上的排序门槛。** J的绝对MAE由105.45降到42.26，但排序准确率32.5%、Spearman为负，不能把尺度/偏差缩小视为优化成功。没有生成可在线加载的rank-validated模型，真实文件也已验证会被在线loader拒绝。',
'',
'实际latent输入对照头约46.4%的RTG排序准确率，仍未显示稳定能力；不能把问题全归为WM rollout误差，也不能据此证明512维表示必然不够。当前数据覆盖（仅8个拟合场景）、函数拟合和任务/控制器隐藏状态条件仍需区分。本轮未在看到验证失败后继续调头或反转分数。',
'','## 新V搜索：四候选同起点实际执行','',
'4个预先固定的开发起点，分别执行Baseline、Mean、全局Best Elite、Final Iteration Best Elite的H步动作，然后原SAC接管。16条成功保存的实际分支严格匹配观测/车辆/交通/RNG/候选池。所有候选在同一V、同一Reward/Cost/折扣下统一复评。',
'',
'**此表是起点后的实际SAC延续结果，不是全程逐步重规划MPPI的独立驾驶评测。** Episode Return和Average Safety Cost均统计该后缀；route_completion为原生终止信息。规划ms为LocalMPPI加缓存审计的CUDA同步wall time，不含之前Lattice/SAC上下文构建，且有共享GPU负载，不作为独占性能比较。','',
'|候选|Success|Crash|OOR|Route Completion|后缀Episode Return|实际折扣Return|Average Safety Cost|规划ms|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in d['summary']:lines.append(f"|{x['choice']}|{x['success']:.0%}|{x['crash']:.0%}|{x['out_of_road']:.0%}|{x['route_completion']:.2%}|{x['episode_return']:.2f}|{x['actual_return']:.2f}|{x['average_safety_cost']:.3f}|{x['planning_time_ms']:.1f}|")
lines+=['', '|起点|Mean−Best预测收益|Mean−Best真实收益|动作序列L2|实际轨迹XY RMSE(m)|','|---|---:|---:|---:|---:|']
for x in d['origins']:lines.append(f"|{x['scenario']}:{x['before']}|{x['predicted_mean_minus_best']:+.3f}|{x['real_mean_minus_best']:+.3f}|{x['mean_best_action_l2']:.3f}|{x['mean_best_true_xy_rmse_m']:.3f}|")
lines+=['',
'逐场原始回报、CI、全局/最终轮Elite索引与排名在development_comparison.json；四轮实际256条Gaussian动作、当轮分数、原Top-K及统一复评分数在各起点audit_samples.npz。固定Baseline单独保存，不是新采样候选。所有4个起点的全局Best和最终轮Best恰好相同，重复候选不算独立证据。',
'',
'Elite动作距离和首转向正负比例已记录；符号差异不是多模态驾驶模式的证明。真实轨迹距离使用实际执行位置，不把latent距离解释为米。本轮不因开发池中Best的点估计稍好就切换默认输出：样本只有4个开发起点，且主Value排序验证失败。',
'',
'首次16个离线分支到达原生终止后，保存结果因耗时字段名错误失败；失败日志/候选池另存failed_logging，未计入结果。修正采集器为CUDA同步wall计时后按相同种子补采，采样动作与初次尝试逐项完全一致。未改模型、评分或参数，也未补造丢失的真实回报。',
'','## WM Cost验证','',
f"原checkpoint有{cost['cost_updates']}次Cost更新，但cost_ready={cost['initial_cost_ready']}、验证schema={cost['initial_validation_schema']}、horizon mask={cost['initial_validated_horizons']}。验证使用原CostModelValidator及真实同起点Cost标签，预测/真实Cost分开，不修改阈值。",'',
'|集合|候选/起点|H10 MAE|有效排序对|ranking ready|实际cost_weight|','|---|---:|---:|---|---:|']
for split,x in cost['splits'].items():
 mm=x['metrics'];lines.append(f"|{split}|{x['candidates']}/{x['origins']}|{mm['wm/cost_mae_h10']:.6f}|{int(mm['wm/cost_ranking_pairs_h10'])}|False|0|")
lines+=['',
'具体未通过原因：H=10候选真实Cost差未达到原min_gap=0.05，无有效排序对/可评排序起点；独立验证仅62个候选/12个起点，也小于原min_samples=128、min_origins=20及min_pairs=100门槛。小MAE主要面对近似相同的低Cost，不能证明能识别危险候选。**保持cost_weight=0**；原模型Cost缓冲区校验逐项不变。未来通过验证时只作连续惩罚，不硬拒绝候选。',
'','## 测试、验收与是否扩大验证','',
'**47项相关测试通过，原43项未删除/放宽。** 新4项覆盖：状态V不读a_H/Q/pi、SAC旧特征保持不变；随机不同轮Q不按旧分数选Best、统一末端batch及262预算；Cost只在排序证据有效时连续生效、copy不改原模型；未验证/不兼容的真实格式侧模型禁止在线加载。两个真实旧场景的535维输入、残差、动作最大差均0，原checkpoint SHA保持不变。',
'',
'明确结论：**当前不值得进入更大规模或独立闭环MPPI测试，也不能宣称优于B。** 开发Mean未超过Baseline，独立预测latent Value排序验证失败。未训练原WM/SAC，未启用新头、未启用Cost、未改变默认Mean/Baseline输出、未进行正式100场。',
'',
'下一步应先增加另划新训练场景中的真实SAC与固定扰动/困难候选覆盖，保留新的场景隔离验证；同时用真实完整SAC状态作离线诊断，区分数据覆盖/拟合不足与latent-only条件缺失。未来如需额外可预测任务上下文，只用于独立价值建模，不改变原535维SAC输入，不凭空构造未来状态。不能在本轮6个验证场景上继续挑网络、loss或翻转分数来宣称成功。',
'',
'交付目录 `reports/sac_value_20261117/`：protocol、dataset_manifest、models、existing_pool_evaluation、development_comparison、cost_validation、verification和完整code_changes.patch。无commit/push/reset。','']
path=ROOT/'SAC_CONTINUATION_VALUE_REPORT.md';path.write_text('\n'.join(lines));print(path)
