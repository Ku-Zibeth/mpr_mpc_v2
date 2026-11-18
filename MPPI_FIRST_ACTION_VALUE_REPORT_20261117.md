# MPPI首动作价值对齐与规划能力研究（2026-11-17）

本轮先证明候选是否有用，再检验评分能否选对。实验与修改位于独立worktree/分支，原main、原WM/SAC、535输入、Lattice、Action Mapping、默认LocalMPPI均保留。**没有把H10标签当成H1，也没有把H1→Baseline接管等同于新MPPI闭环。**

本轮停止位置：Stage4 independentranking gates did not pass. 未满足接入门槛，默认在线规划保持不变。候选有用与模型能泛化识别是两件事，见下面实际覆盖/排序结果。

工作目录：`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_first_action_20261117`；分支`experiment/mppi-first-action-20261117`；基线`27f8131d95a39bfe2f4a95d101fef1774903b907`。SSHfetch及HTTPS/proxyfetch失败已记录，通过GitHub连接器核实远端main确为同SHA，没有宣称fetch成功。Checkpoint SHA=`4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a`，模型为WM200k+RL300k，总500k；未切换到另一个500k+500k模型。

## 1. 原生实现审计与对照

完整审计见MPPI_NATIVE_PARITY_REPORT_20261117.md。Candidate、PolicyMean、原生Sample终端动作在相同缓存前缀、末端latent、Reward、折扣及Q组合下比较。原生规划Qavg与训练targetQmin、Phase1固定Q01均值严格区分。枚举当前五Q的10种平均/10种最小、整体mean/median/min，保存100次random_pair分布；这里的min诊断用当前Q，不冒称调用了完整训练TD Target。

已真实补充完整TD-MPC2 MPC接管，原512samples/64elites/6iterations与Actor-only分列；LocalMPPI仍64/8/4。两个新固定起点完整MPC回报100.634/63.218，Baseline为141.514/85.092，Actor-only为44.540/76.481。样本只有两地图，不作泛化结论，也不将其称为单独原版迁移模型成绩。

## 2. 原生反事实协议与覆盖

Pilot场景=[90115, 90261, 90125, 90604, 90443, 90066]，每图固定起点[50, 150]；扩展开发=[92165, 92137, 92041, 92923, 92903, 92704]；新独立验证=[94397, 94928, 94201, 94142, 94057, 94894]。旧6封存图继续封存。所有场景划分、候选角色、近似相等阈值1及统计门槛在完成真实分支前固定；版本manifest记录历史Fit/InnerStop/Phase1 Pilot，绝不将其当作新验证。

每个起点只运行一次原64×4搜索，保存258条Baseline/Mean/真实Gaussian序列与每轮Q成员；按预测分数固定取Baseline、实际MPPI输出、最佳非Baseline、75/50/25分位、最差及固定Gaussian。按精确首动作去重，角色别名与被合并不同tail仍保留。所有模型输入在任何候选环境动作前生成；F3确为实际SAC修正后的轨迹参数。

每条分支新建独立进程，核对初始Observation、完整可记录vehicle/traffic状态及六类RNG。H=1直接执行物理a0一次，然后每步真实重算Lattice+SAC，Residual ON/MPPI OFF，直到原生done。H2/4/10使用同一序列，仅干预长度不同。保留terminated/truncated/native done；提前失败不补未来。EpisodeReturn/Cost为从固定起点至原生终止的后缀，前置共同回放不计。

动作单位是MetaDrive归一化steer/throttle控制[-1,1]，不是残差d/v或SI转角。continuation_after_first与continuation_after_fixed_prefix分列，H>1时不能把第1步后的RTG误称为纯SAC接管RTG。映射次数0仅指固定候选物理前缀；后续Baseline仍正常执行原残差映射。

共完成576条原生分支（含4条Actor/MPC对照及1条同seed重放）；按协议计数={'H10/baseline': 95, 'H1/baseline': 287, 'H2/baseline': 95, 'H4/baseline': 95, 'H1/td_actor': 2, 'H1/td_mpc': 2}。不可达固定起点=0，原样保留不换图。所有覆盖与oracle都是实际测量子集，不能称为完整Gaussian搜索空间oracle。

当前random_traffic=false，Engine与TrafficManager RNG状态实际可读并全量保存。相同策略的精确重放完全一致，没有只换Torchseed冒充交通重复或将其当作独立地图。不同候选改变交互轨迹可能改变后续事件次序，不能宣称所有未来交通事件被强行锁死。

## 3. H1候选收益与现有评分

|数据|Oracle Gain [95%CI]|实际MPPI输出Gain|实际输出Regret|正收益候选比例%|
|---|---:|---:|---:|---:|
|pilot|10.426 [2.511,18.457]|4.277 [-0.992,11.862]|6.148 [1.870,13.008]|37.302 [19.048,53.175]|
|development|12.980 [5.660,22.855]|8.743 [1.919,18.362]|4.238 [1.921,7.801]|46.032 [29.762,62.698]|
|validation|3.260 [0.315,8.576]|0.100 [-0.743,0.980]|3.160 [0.844,7.607]|26.190 [5.952,52.381]|

实际MPPI输出是原weightedMean/Baseline第一步；下表Top1为同一已测候选子集中的评分argmax，两者不能混同。指标先按起点计算，再按场景bootstrap2000次seed1，95%区间；候选对、窗口与重复都不作独立地图。真实收益差≤1为近似相等，排序准确率排除，预测tie记0.5；常数Top1固定Baseline而非事后选真值最优。

Oracle按用户定义的真实Y1最大值计算，未用未来事故筛选候选；它不等于已得到同时满足Success/Crash/OOR约束的可部署政策。事故条件仍需独立闭环检验，不能只因oracle收益高就启用。

|现有评分（开发集）|Pair Accuracy%|Spearman|Top1 Regret|Argmax Actual Gain|
|---|---:|---:|---:|---:|
|current_mppi_h10|55.580 [48.458,61.864]|0.050 [-0.085,0.189]|7.811 [4.077,11.975]|5.170 [-2.557,14.525]|
|current_q_avg01|57.999 [46.915,69.330]|0.107 [-0.042,0.276]|5.509 [2.629,8.621]|7.471 [1.435,15.845]|
|current_q_mean|59.332 [48.595,70.027]|0.155 [0.012,0.313]|4.783 [2.080,7.797]|8.197 [2.134,16.347]|
|current_q_median|51.257 [40.456,63.167]|0.014 [-0.183,0.226]|6.988 [2.771,11.968]|5.993 [0.161,14.465]|
|terminal_policy_mean|55.151 [48.035,61.625]|0.031 [-0.109,0.175]|9.008 [4.613,13.743]|3.972 [-3.328,13.539]|
|terminal_native_policy_sample|55.164 [47.963,61.783]|0.030 [-0.107,0.175]|8.702 [4.307,13.549]|4.278 [-3.034,13.914]|
|constant_baseline|50.000 [50.000,50.000]|N/A|12.980 [5.428,23.188]|0.000 [0.000,0.000]|

terminal_policy_mean相对Candidate的场景配对Accuracy变化=-0.429 [-1.338,0.347]个百分点，Regret改善=-1.198 [-3.275,0.000]，argmax真实收益差=-1.198 [-3.209,0.000]。这是开发集诊断，不能据多组比较修改默认。

terminal_native_policy_sample相对Candidate的场景配对Accuracy变化=-0.415 [-1.413,0.521]个百分点，Regret改善=-0.891 [-3.109,0.702]，argmax真实收益差=-0.891 [-3.093,0.613]。这是开发集诊断，不能据多组比较修改默认。

Baseline Win Rate、负收益比例、预测改善误报率、Kendall、所有10双Q avg/min及Q方差与真收益差关系见ranking_metrics/scenario_bootstrap与逐起点CSV。SAC Critic只保留原残差Baseline诊断值：Gaussian物理动作无唯一反向残差映射，未将物理a0直接塞进残差Critic，故不伪造其候选排序。

## 4. H1/2/4/10与闭环含义

|干预长度|与H1真实收益Spearman|同向配对比例%|Top1一致率%|
|---|---:|---:|---:|
|1|1.000 [1.000,1.000]|100.000 [100.000,100.000]|100.000 [100.000,100.000]|
|2|0.123 [-0.044,0.282]|63.153 [45.462,80.578]|8.333 [0.000,25.000]|
|4|-0.152 [-0.345,0.017]|39.273 [26.833,49.266]|0.000 [0.000,0.000]|
|10|-0.056 [-0.171,0.038]|59.544 [48.635,69.626]|16.667 [0.000,33.333]|

H1严格对应“首动作干预+原Baseline反馈延续”，比H10开环标签在定义上更接近实际单步决策；**它仍不是新规划器每步重复优化后的π_new闭环价值**。H2/4/10是真实独立环境执行，不是序列重放预测。原Baseline的开环H10首序列同样可能偏离后续反馈Baseline，因此分别保留各H对应Baseline，不混用差值。

## 5. 条件式Q_B研究与验证

开发H1数据通过候选覆盖与评分缺口门槛后，才训练独立小模型。实际评分缺口门槛为“原H10排序Accuracy CI下界≤.5，或原输出Gain CI下界≤0”；本轮前一项成立。protocol.training_gate文字简写为无可靠评分收益，含义不够精确，以预先执行的analyze.py公式及逐项decision为准，并非看到开发结果后新增门槛。开发集原MPPI输出Gain CI已大于0，不能声称现有输出没有首动作收益；本轮检验的是剩余排序/泛化缺口。没有使用历史H10标签、旧SAC Value或CandidateValue权重，也不更新原WM/SAC。按预先固定8图fit、4图innerstop；场景/起点平衡，归一化仅fit，Huber+.1Pairwise Ranking，近等收益对排除，固定参数、不做超参搜索。

F1=当前latent512+候选物理a0（514）；F2=当前535+a0（537）；F3=当前535+实际SAC后参数3+Baseline首动作2+a0（542）。线性Ridge初始化后以L2正则Huber+Ranking优化，准确称ridge_huber_rank；不是宣称闭式普通最小二乘Ridge。MLP仅64/32两隐层。原535定义不变，所有输入在线当前时刻可得，没有未来obs/latent/Reward/动作。

|候选离线模型|Best Epoch|InnerStop Accuracy%|InnerStop Regret|
|---|---:|---:|---:|
|F1_ridge_huber_rank|0|55.236 [35.154,79.514]|3.142 [1.776,4.507]|
|F1_mlp_huber_rank|25|53.421 [47.612,62.456]|3.492 [1.658,4.796]|
|F2_ridge_huber_rank|0|55.236 [35.154,79.514]|3.142 [1.776,4.507]|
|F2_mlp_huber_rank|30|59.077 [42.765,79.974]|2.245 [0.395,5.459]|
|F3_ridge_huber_rank|0|55.236 [35.154,79.514]|3.142 [1.776,4.507]|
|F3_mlp_huber_rank|15|49.687 [46.448,52.654]|3.503 [0.467,6.543]|

仅根据innerstop冻结一个模型：`F2_mlp_huber_rank`。是否进入一次新独立验证：True。没在验证集上重选feature/网络/epoch/Q组合。

新独立验证：Accuracy%=35.638 [15.516,54.584]，Spearman=-0.143 [-0.385,0.083]，Kendall=-0.104 [-0.298,0.093]，Top1 Regret=1.124 [0.546,1.828]，相对原H10评分Regret改善=1.262 [-0.990,4.576]，MAE=29.593 [17.591,40.579]，Bias=10.684 [-14.423,31.719]，Calibration=29.593 [18.583,40.571]。全部门槛通过=False。

本轮没有通过全部独立门槛，因而未新增planning/first_action_value.py或二维在线优化器；未运行A/B/C新控制器闭环。旧LocalMPPI和默认terminal_value/terminal_action/random_pair保持原样。也未实施高层Lattice参数搜索重构。

## 6. 耗时、测试、最终决策

逐阶段与P50/P95/P99见timing_breakdown.csv。并发采集时钟只标为concurrent，单个冷启动搜索值不冒称P95。复用旧同核心RTX3070的100次单worker完整规划：B P95≈76ms，原MPPI M1/M2/M4≈104/117/141ms。未创建可验证的新在线规划器，所以不宣称新方法达到100ms，也未通过减少检查/样本/预算换速度。

原62项测试全部保留，新增9项协议/物理动作/去重/特征/终端采样RNG/metadata gamma checkpoint/真实模型加载/Residual开关及分组损失平衡回归，共71 passed。原权重前后SHA、每分支权重hash、所有起点hash、每步物理前缀、Native done及反馈次数都复核。初始新增测试漏写了重复角色worst_nonbaseline，修正为验证所有角色均保留；原62项断言未动。Pairwise损失按有效起点在地图内平衡，避免无有效收益差的起点稀释该地图权重。

A：候选首动作确有真实收益，证据限于实测子集与当前地图。
B：没有一种终端定义在本轮证据下可被称为普遍最可靠，表中差异不授权修改默认。
C：旧审计中Actor-only接管未消除Q误差，但随机Actor、确定性策略与完整MPC并不相同。当前数据不能唯一分离训练目标、状态分布与延续策略贡献；不要继续只归因策略不匹配。
D：H1正确对齐单步干预定义，但不冒充新MPPI完整闭环收益。
E：是否值得训练由已执行的覆盖/缺口门槛决定，训练后的失败不靠增加模型容量挽救。
F：未通过独立排名/Regret门槛，暂不进入独立First-Action MPPI闭环或正式100场。
G：不能说低层搜索空间完全无效；已存在正收益候选，当前瓶颈仍是识别/泛化。不得据此自动重构Lattice搜索变量。

完整数据与diff：`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_first_action_20261117/reports/mppi_first_action_value_20261117`；版本、失败日志、协议及历史split均保留。没有commit或push。
