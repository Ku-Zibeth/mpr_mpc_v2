# MPPI First-Action Advantage — Phase 3（2026-11-18）

## 结论先行

本轮**没有证明 Baseline-relative Advantage 恢复跨地图首动作排序**。按冻结的inner-stop选择规则选中`A2_mlp`：LOMO准确率=53.085 [39.231,68.491]%，Model Gain=-2.057 [-6.930,0.628]；inner-stop Regret=10.207 [1.340,27.040]，旧A0=2.245 [0.395,5.260]。预注册LOMO和inner-stop门槛未通过，按要求停止：**没有新validation、没有在线接入、没有增加网络容量或重训WM/SAC**。

不能简单说“所有方法都没信号”：Linear的LOMO Model Gain正区间、KNN的inner-stop排名强，均保留。但这些结果没有同时满足跨图排序、Regret及模型选择门槛，不能事后把更好看的control换成主模型。

## 1. 版本、历史时点和数据

parent_commit=`7d15cdda92ff434024bb72c5dd324d05fa3fe3d0`；phase2_report_commit=`7d15cdda92ff434024bb72c5dd324d05fa3fe3d0`。Phase2正文27f8131是当时生成报告的基线；该报告及代码之后由7d15cdd提交并推送，历史文字不改，本manifest明确两种时点。2026-11-18 GitHub连接器核实main为7d15cdd。

Checkpoint SHA=`4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a`，仍为WM200k+RL300k总500k；原535、WM/SAC/Lattice权重、Action Mapping、Reward/Cost、gamma=.995和原生done/成功条件均保留。所有强制阅读的Phase2报告/JSON/CSV与旧训练实现均已全量读取，SHA和结构摘要在experiment_manifest.json。

新目录位于main工作区：`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2/reports/mppi_first_action_advantage_20261118`。Git忽略了旧.pt/.npz，故只读原实验worktree `/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_first_action_20261117` 内的真实二进制，不生成替代旧数据。只打开原fit8图126条H1分支与inner-stop4图64条H1分支，共24个origin；**没有为模型训练/选型打开旧seen-validation分支**。历史报告中的已知验证结论仅作为用户指定固定前提。所有输入文件和源代码SHA均记录并在结束复核。

## 2. 严格的same-origin相对目标与输入

Y取真实首Reward+.995×原冻结确定性Lattice+SAC反馈延续至native done。每个origin根据baseline角色定位唯一真实Baseline，ΔY=Y_i−Y_B，Δa=a_i−a_B；不跨origin/地图计算参考，不用预测score或Oracle作Baseline。原生提前失败与所有候选保留，完整Reward/native done向量重算核验。

A0是原冻结F2：current535+绝对物理action2（537），权重完全不变，仅在离线MAE/作图中减去同origin A0 Baseline预测，argmax/排名不变。A1=current535+Δa（537）；主A2=current535+a_B+Δa（539）。没有future latent/obs/Reward/Cost/crash/OOR/隐藏状态作为输入，真实结果只在label/评价侧。

A1/A2输出为`[f_theta(x)-f_theta(same context,Δa=0)] × fit_RMS(ΔY)`，Baseline严格为0；不是另一个绝对Return模型。Linear与32→16→1的固定SiLU MLP各训练一套；无Transformer/Ensemble/更大网络。

**结构限制**：线性f的状态项、Baseline项与bias被相减消去，最终只有w_delta·Δa。因此A1 Linear和A2 Linear表达类完全相同，本次全部LOMO/inner预测也逐元素相同。这不是“Baseline/context没有价值”的证据，只能在MLP控制中比较其额外信息。

## 3. 冻结训练协议及无泄漏LOMO

protocol.json在读取新relative数据前冻结。300固定epoch、seed1、Adam lr=.001，Linear decay=.001、MLP decay=.0001。输入mean/std与label RMS仅取相应fit maps，RMS只缩放不减label均值，保证零参考不被移动。SmoothL1(normalized advantage)+.25×同origin pairwise softplus；pair预测差使用fit尺度归一化，不再raw_difference/10。原abs真实差≤1的Pair排除主排名，tie保留诊断。

每图总权重相同、图内每origin总权重相同、origin内候选均分；实际每图2个起点，因此origin全局权重也相同。全fit及各7图fold的scene/origin/candidate权重在training_weights.csv，Pair权重与fit-scale在advantage_pair_diagnostics.csv。没有只训练大动作或删除不利候选。

8折均只用7张fit图训练，固定300epoch，不用heldout图、inner-stop或旧seen-validation做早停/归一化。预测只来自未见该地图的那一折，不平均多个fold模型。全fit模型也只用8图，训练结束后一次性评价4张inner-stop；选型按Accuracy DESC、Regret ASC、Gain DESC，稳定tie优先Linear。

**A0公平性限制**：旧A0已训练过全部8张fit图，不能提供真实无泄漏的A0 LOMO。遵守“不重训/修改A0、不再训练绝对Return”约束，表中保留它为**seen-fit参考**并显式打标；不把A2对它的LOMO差异当因果比较，也不把泄漏A0纳入LOMO门槛。旧A0还已用inner-stop选过，故本轮inner对比也是开发诊断，不是独立验证。

LOMO门槛：选定A2的Accuracy CI下界>.5、Spearman CI下界>0、ModelGain CI下界≥0。inner门槛：点估计Accuracy>.5、Spearman>0、Regret优于A0、Gain≥0、selected-negative不高于A0。两者同时过才允许冻结模型并预注册新的6图/12origin验证。本轮未通过，不继续采图。

## 4. 结果：真正held-out地图与开发inner-stop

|阶段|模型|Pair Accuracy% [95%CI]|Spearman|Top1 Regret|Model Gain|Selected Negative%|
|---|---|---:|---:|---:|---:|---:|
|lomo|A0_frozen (seen-fit，非LOMO)|68.662 [57.046,81.342]|0.215 [0.049,0.416]|7.987 [1.617,15.537]|2.012 [-0.857,6.516]|18.750 [0.000,37.500]|
|lomo|A1_linear|56.788 [40.065,72.699]|0.115 [-0.117,0.332]|8.082 [2.503,14.144]|1.918 [0.360,3.754]|18.750 [6.250,37.500]|
|lomo|A1_mlp|55.830 [43.065,70.383]|0.074 [-0.129,0.295]|11.199 [4.001,18.003]|-1.200 [-6.732,2.491]|18.750 [6.250,37.500]|
|lomo|A2_linear|56.788 [40.065,72.699]|0.115 [-0.117,0.332]|8.082 [2.503,14.144]|1.918 [0.360,3.754]|18.750 [6.250,37.500]|
|lomo|A2_mlp|53.085 [39.231,68.491]|-0.020 [-0.281,0.275]|12.057 [4.874,19.180]|-2.057 [-6.930,0.628]|12.500 [0.000,25.156]|
|lomo|A2_knn|36.679 [28.540,45.382]|-0.112 [-0.266,0.057]|11.653 [3.611,21.419]|-1.654 [-5.786,0.865]|37.500 [12.500,68.750]|
|inner_stop|A0_frozen|59.077 [42.765,79.974]|0.217 [-0.051,0.631]|2.245 [0.395,5.260]|16.697 [3.798,37.785]|0.000 [0.000,0.000]|
|inner_stop|A1_linear|54.097 [49.851,61.023]|0.101 [-0.024,0.286]|4.670 [3.147,6.193]|14.272 [1.620,36.868]|0.000 [0.000,0.000]|
|inner_stop|A1_mlp|53.199 [35.014,76.573]|0.116 [-0.190,0.542]|2.631 [1.316,4.253]|16.311 [2.708,38.979]|0.000 [0.000,0.000]|
|inner_stop|A2_linear|54.097 [49.851,61.023]|0.101 [-0.024,0.286]|4.670 [3.147,6.193]|14.272 [1.620,36.868]|0.000 [0.000,0.000]|
|inner_stop|A2_mlp|54.448 [40.559,75.628]|0.128 [-0.143,0.536]|10.207 [1.340,27.040]|8.735 [2.829,16.254]|0.000 [0.000,0.000]|
|inner_stop|A2_knn|77.900 [61.431,89.632]|0.497 [0.173,0.729]|1.756 [0.288,4.221]|17.186 [4.008,38.802]|0.000 [0.000,0.000]|

每个候选/起点/地图层级保留。先起点后图等权，整图2000次Cluster Bootstrap、seed20261118；Pair不当独立地图。8图和4图的区间较宽，单图/单origin不代表总体。Kendall、逐fold Model/Baseline Gain、selected-positive/negative、实际候选选择在对应CSV及evaluation_results.json。

A1按同一规则选中Linear；A2选中MLP。A2 inner Accuracy仅略高于Linear，而Regret更差，仍严格沿用已固定的字典序规则；没有看到Regret后再换一个模型来过门槛。KNN仅诊断，不因为inner结果较好就临时改成部署候选。

|真正LOMO比较（后者−前者）|Accuracy变化百分点|Regret改善|Gain改善|
|---|---:|---:|---:|
|lomo/A2_mlp_minus_A1_mlp|-2.745 [-11.007,7.531]|-0.857 [-2.356,0.310]|-0.857 [-2.356,0.310]|
|lomo/A2_linear_minus_A2_mlp|3.703 [-14.860,19.886]|3.975 [0.056,9.266]|3.975 [0.056,9.266]|
|lomo/A2_knn_minus_A2_mlp|-16.406 [-36.484,1.912]|0.403 [-5.466,6.732]|0.403 [-5.466,6.732]|

## 5. Oracle、动作幅度与简单controls

Oracle是实际测量代表候选子集中的最大真实ΔY，非完整Gaussian空间或可部署policy。Random沿用旧fixed Gaussian角色，不重新抽有利样本。Baseline=0；逐origin的A0/A1/A2各variant、Random、Oracle和选正/选负都保留在oracle_comparison.csv。

|数据|Oracle Gain|至少含>1收益候选的origin比例%|Random Gain|
|---|---:|---:|---:|
|lomo|9.999 [3.540,17.519]|50.000 [25.000,75.000]|2.474 [0.227,5.291]|
|inner_stop|18.942 [4.767,43.045]|100.000 [100.000,100.000]|14.004 [0.857,38.139]|

开发数据Oracle明显存在正收益，故不能把本轮主模型失败说成“整个candidate space无价值”。但Oracle事后最大化含Baseline，必然非负；既有Phase2新地图Oracle不稳定的固定前提也没有被推翻。本轮无新独立地图，不能宣称潜力已广泛泛化。

KNN使用与A2 Linear相同539维当前输入，fit-only标准化后按current535、Baseline2、Δa2分块等权距离，固定k8与距离倒数×训练层级权重；再减去同context零Δa的预测，Baseline仍0。没有试不同k或事后调距离。它在inner很强但LOMO失败，提示邻域/地图覆盖不稳，不能仅凭一个开发split指认MLP容量为唯一问题。

动作L2固定层<.01、.01–.05、.05–.1、≥.1，全部样本保留。为可比较的Gain，分层内额外保留Baseline参考；子集Top1是限制候选池的离线描述，不是原在线实际选中动作。缺乏有效Pair的层记N/A，不填0；分层Accuracy/Spearman/Regret/Gain区间见action_magnitude_analysis.csv与evaluation_results.json。

## 6. 计算开销（仅离线head，不是完整MPPI）

CPU单线程、GPU同设备输入，warmup20/repeats200；CPU/GPU均报告含两次baseline-centered推理的wall时间，GPU另列CUDA kernel时间。参数量、batch1、batch258、tensor峰值显存和CPU/GPU数值差异完整在inference_benchmark.csv。数据传输未包括，tensor显存不是驱动全部进程显存。

|模型|参数量|CPU batch258 P50/P95 ms|GPU batch258 wall P50/P95 ms|GPU模型tensor bytes|
|---|---:|---:|---:|---:|
|A0_frozen|36545|0.707/0.837|0.435/0.678|154624|
|A1_linear|538|0.194/0.244|0.212/0.295|8704|
|A1_mlp|17761|0.446/0.520|0.389/0.538|78848|
|A2_linear|540|0.198/0.265|0.211/0.318|8704|
|A2_mlp|17825|0.435/0.780|0.393/0.657|78848|
|A2_knn|0|1.872/2.593|0.637/0.844|281088|

Centered Linear虽有538/540个参数槽，纯线性context/Baseline项不可识别，实际动作表达为2个系数。KNN无神经参数但保存fit数据。小head变快不等于完整规划变快；原完整规划P95 B≈76ms、M1≈104ms、M2≈117ms、M4≈141ms仍为历史基准，本轮没有改变原采样预算或在线时间。

## 7. 十一个必须回答的问题与下一步

1. **Absolute→relative是否改善？** 没有形成同时通过LOMO与inner-stop的改善；A2 inner Regret明显较差。A0 LOMO看过fit，禁止据其直接证明目标因果优劣。

2. **state-value干扰是否被证实？** 未证实。主模型失败不能单独反证state-value假设；本轮同时改变相对目标、局部表示、中心化、归一化/排序loss和小网络，不能把结果唯一归因某一个因素。

3. **Δaction是否更易泛化？** 尚无可靠证据。相对Linear有正LOMO Gain，但Accuracy区间仍跨50%；与旧A0的不同结构/目标存在混杂，不能叫单变量结论。

4. **Baseline action有额外信息？** A2 vs A1 MLP差异不稳定；Linear中该项数学上必然抵消。不能从Linear相同断言Baseline信息无用。

5. **MLP优于Linear？** 没有；LOMO Linear Gain为正，主A2 MLP为负，配对差异上表保留，不扩网络。

6. **MLP优于KNN？** 依split而变。KNN inner排名强但LOMO更差，既不支持稳定部署KNN，也不足以唯一归因容量。

7. **LOMO是否仍失败？** 主A2未过三项预注册LOMO门槛；Linear有部分积极信号，不把“未过门槛”夸大为“所有低层动作都无用”。

8. **主要问题是什么？** 当前可观测瓶颈仍是action-advantage跨地图/context泛化，训练/容量有辅助迹象，不能唯一归因。开发Oracle正收益不支持立即否定candidate空间；本轮没有证据把Dynamics升级为主要原因。

9. **Oracle是否稳定正收益？** 本轮fit/inner开发数据有积极区间；跨新地图的稳定性未获得新证据，不能把开发Oracle冒充泛化。

10. **是否进入新validation？** 否，开发门槛已失败，新的6图/12origin没有采集，旧seen-validation没有复用来重新选模型。

11. **是否具备在线条件？** 否；所有新head保持离线，没有修改MPPI/WM/SAC/Lattice/原配置。

唯一下一阶段方向：若继续研究，应预注册增加不同地图/道路交互条件下的origin覆盖，检验当前特征下的局部优势可辨识性；不要继续增加同类网络宽度/深度，也不要因本次结果直接重训WM或重构candidate空间。本轮按门槛停止，不自动新增采集。

## 8. 测试、保留和交付

原82项在保有历史二进制fixture的旧worktree独立进程全部通过；新增16项在main新目录另一个进程通过，合计98项。首次把两个不同工作区放同一pytest进程触发了原有mpr_mpc导入保护，日志保留；改为独立进程，没有移除保护或放宽任何原断言。测试通过不等于算法有效。

新增回归覆盖Baseline Δa/优势精确0、raw差值、same-origin真实target、原gamma/Reward/done、同origin Pair与原tol、场景/origin平衡、fit-only normalization、LOMO无泄漏、Linear/KNN相同输入、无未来输入、A0 SHA/旧inner结果一致、原在线源码不变、seen-validation不选型、Linear中心化消去context。

结束复核：Checkpoint/A0 SHA不变，所有受保护在线源码与config逐项SHA不变，全部旧Phase2输入2631项及本轮fit/inner来源文件SHA复核；无tracked在线diff、无commit/push、无新环境回合或正式100场。所有新模型只存新目录，不覆盖旧head或WM/SAC。

图均另存PDF：

![absolute_vs_relative_accuracy](figures/absolute_vs_relative_accuracy.png)

![leave_one_map_out](figures/leave_one_map_out.png)

![oracle_vs_model_gain](figures/oracle_vs_model_gain.png)

![action_delta_accuracy](figures/action_delta_accuracy.png)

![regret_comparison](figures/regret_comparison.png)
