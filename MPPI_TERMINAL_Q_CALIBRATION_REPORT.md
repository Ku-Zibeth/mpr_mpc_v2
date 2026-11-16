# MPPI Terminal Q 校准与真实收益分析

基线 `fae10f34e15ffe987a7527272cb1693637a9f488`，修改前工作区干净。未训练、未改模型权重、未修改535维SAC输入/动作映射、未改环境Reward/终止条件，未commit/push/reset。历史结果保持原样。

## 代码与测试

- `planning/evaluator.py`增加MPPI专用 `terminal_q_weight`，默认1。只缩放已经解码且包含原discount/termination mask的终端贡献。原SAC特征wrapper始终按原λ=1计算，不读取MPPI权重。Q计算次数、reward/next次数不变。
- `planning/local_mppi.py`将同一λ用于全部迭代及最终baseline/mean复评，增加weighted_terminal_q_gain诊断；Gaussian、AR(1)、Elite、Mean/Std公式、H/H+1和采样预算不变。
- `config.yaml`、`planning/native_config.py`新增默认1.0及兼容字段；`planning/mppi_ablation.py`新增Q1/Q05/Q025/Q0四个单因素配置。
- `tools/value_calibration.py`为独立离线工具，记录pre-action reward、post-action latent及同一终端动作的Q比较，不进入在线选择。`tests/test_terminal_q_weight.py`增加3个针对性回归，原40项保留。

**43项相关测试通过**，详见 `tests_final.txt`。λ=1旧分数/分布精确相同；λ仅改终端贡献；baseline/mean与四轮定义一致；SAC特征不受影响；Teacher-Forced/open-loop索引、tdmpc2_reward标签、终端动作、起点配置/RNG匹配均有回归。两个真实场景的535维特征、残差和动作与修改前最大误差全部0，见 `sac_interface_verification.json`。

原D4的除法tau=0.5等价逆温度beta=2。四个λ实验统一写为inverse beta=2，未改为beta=0.5。默认主配置其余选项未改；实验D4不是已验证的最优方案。64/8/4、rho=0.8、原std、candidate a_H、五Q均值及checkpoint均固定。λ=0只作短时域诊断，不自动部署。λ会同时改变prefix/terminal的相对贡献及CEM权重集中度，温度本身未变；不能把全部变化直接归为Q偏差已修复。共同候选池的静态重评分与各λ自身mean的真实执行分别报告。

## 实验协议与可复现性

`protocol.json`在采集前确定：16个新场景×step50/150共32个计划起点，按场景划分开发10个/验证6个。原训练10000–11999、旧100、前次验证20000段/旧20测试30000段均不重叠。开发完整回合为4个场景×B和4个λ，验证为6个独立场景×相同5组。

每个反事实分支独立启动进程，严格匹配初始观测、ego/交通状态、记录到的Python/NumPy/Torch/CUDA/MetaDrive RNG、候选动作及episode配置。不是完整Bullet/隐藏控制器快照，不能据此声称所有隐藏状态完全相同。每个分支执行相同H=10长度的对应候选前缀，再统一SAC延续到环境原生终止；此干预不同于在线每步只执行首动作再重规划，因此反事实结果不代替完整闭环性能；提前终止保留为真实失败，不补造H时刻状态。

试运行发现同进程重复reset无法精确复现baseline，因此正式分支采用新进程；首轮诊断中的CPU/GPU动作设备错误已修复。失败试运行单独保留，不参与统计。起点前SAC已结束的情况记录为unreachable，不按分支真实收益挑选或替换。额外在预定4个场景的起点实际执行Q1最佳记录Elite。旧1714:250/559:600仅用于诊断，原9个起点不用于λ选优。

真实Reward统一为当前任务 `info[tdmpc2_reward]`，另存原始MetaDrive reward作核对，不混用。闭环Episode Return沿用环境float32返回的同一奖励；实际折扣Return与探针使用info中同一标签（仅序列化精度略有差异）。反事实Crash/OOR为起点后的事件，完整回合表统计整个episode。

所有区间按场景聚类bootstrap 2000次、seed1。起点/候选/Q成员不视为独立场景。未作多重比较校正，小样本区间不构成总体泛化证明。共享GPU负载下的耗时不可当作独占硬件优化对比。

## 三层误差分解（独立验证）

|分支|末latent RMSE|Teacher Reward MAE|Open-loop MAE|Reward Head折扣误差|rollout Reward偏移|mask影响|
|---|---:|---:|---:|---:|---:|---:|
|B|0.0796|0.0803|0.1243|-0.0769|0.3912|-0.0000|
|Q1|0.0825|0.0616|0.1059|-0.1411|0.4075|0.0000|
|Q05|0.0826|0.0580|0.1029|-0.0756|0.4103|-0.0000|
|Q025|0.0829|0.0558|0.1011|-0.0453|0.4071|-0.0000|
|Q0|0.0838|0.0578|0.1029|0.0145|0.4222|0.0000|
|elite_Q1|0.0964|0.0259|0.0311|-0.0067|-0.0045|-0.0000|

开环与Teacher-Forced都在执行a_t之前预测reward；真实奖励来自该次step。先model.next，再编码o_{t+1}比较latent，保存每一步L2和RMSE。上表末latent误差是无量纲特征距离，绝非车辆位移。按步长的Reward误差趋势、位置/速度/航向误差及原始latent均保留在JSON中。物理误差复用现有SAC几何近似，4/6m/s²配置不视为已校准车辆真值，不用于在线筛选。

折扣前缀误差严格拆为：Teacher Reward−真实Reward（Reward Head项）、Open-loop−Teacher（latent rollout影响）、模型masked prefix−Open-loop（终止mask项，另含微小batch数值舍入差异）。它们是加性误差分解，不把相关性直接解释为唯一因果来源。

以下是**Q成员平均一致率**（各起点五个成员的正确比例再平均），与下方五Q均值的排序统计分开。

|λ|预测末latent Q对RTG排序一致率|真实末latent+同一a_H|真实末latent+下一SAC动作|pred/real latent排序翻转率|
|---|---:|---:|---:|---:|
|Q1|35.0%|41.7%|45.0%|26.7%|
|Q05|33.3%|26.7%|28.3%|23.3%|
|Q025|25.0%|28.3%|30.0%|20.0%|
|Q0|51.7%|41.7%|45.0%|33.3%|

以下为**五Q均值**对真实SAC延续RTG的排序准确率及场景聚类95% CI：

|λ候选|预测末latent|真实末latent+同一a_H|真实末latent+实际下一SAC动作|
|---|---:|---:|---:|
|Q0|50.0% [25.0%, 75.0%]|50.0% [25.0%, 75.0%]|50.0% [25.0%, 75.0%]|
|Q025|25.0% [0.0%, 58.3%]|33.3% [16.7%, 50.0%]|33.3% [16.7%, 50.0%]|
|Q05|33.3% [8.3%, 66.7%]|25.0% [8.3%, 41.7%]|25.0% [8.3%, 41.7%]|
|Q1|33.3% [8.3%, 66.7%]|41.7% [25.0%, 50.0%]|50.0% [25.0%, 75.0%]|

换成真实末端latent及真实下一SAC动作后，五Q均值仍未形成稳定正确排序。说明仅修复Dynamics未必足够，下一阶段应独立验证SAC-continuation value/FPE与Dynamics适配，而不是仅改变Q成员数量或继续缩小λ。样本量有限，不能据此断言所有场景均不可修复。

Q比较在预测/真实z_H上保持相同候选a_H；第三项用真实下一SAC动作，仅为诊断。各成员原始值与真实SAC Return-to-Go、尺度差异在cases及原始分支文件中。终端Q对RTG排序与包含prefix的总J排序分别计算。Q训练延续策略不同，因此Q−SAC RTG不能全部归咎于Q拟合错误。绝对Q尺度偏大本身不证明错排：同一discount/mask下共同常数偏移会在候选差值中抵消，须检查相对排序。

## λ消融：反事实候选排序

### 开发集

|λ|有效起点/场景|Mean相对B真实增益 [95% CI]|Mean/B排序准确率 [95% CI]|全候选对排序准确率|Spearman|Top-1 Regret|
|---|---:|---:|---:|---:|---:|---:|
|Q1|20/10|-9.925 [-26.868, -0.739]|30.0% [15.0%, 45.0%]|51.3%|0.016|3.904|
|Q05|20/10|-0.815 [-3.639, 2.749]|50.0% [35.0%, 65.0%]|52.3%|0.023|3.904|
|Q025|20/10|0.536 [-2.473, 4.157]|65.0% [50.0%, 80.0%]|51.8%|0.020|3.596|
|Q0|20/10|-0.164 [-2.460, 1.824]|55.0% [40.0%, 70.0%]|53.8%|0.090|4.238|

不可到达起点：0个，详见JSON；未按结果进行替换。

### 按场景隔离的验证集

|λ|有效起点/场景|Mean相对B真实增益 [95% CI]|Mean/B排序准确率 [95% CI]|全候选对排序准确率|Spearman|Top-1 Regret|
|---|---:|---:|---:|---:|---:|---:|
|Q1|12/6|0.491 [-1.196, 3.027]|50.0% [16.7%, 83.3%]|51.1%|0.038|2.615|
|Q05|12/6|-0.979 [-8.144, 5.747]|33.3% [8.3%, 66.7%]|50.6%|0.010|3.445|
|Q025|12/6|-3.076 [-12.561, 5.104]|50.0% [16.7%, 83.3%]|47.8%|-0.068|5.616|
|Q0|12/6|-1.339 [-8.489, 4.769]|58.3% [33.3%, 83.3%]|55.6%|0.138|3.844|

不可到达起点：0个，详见JSON；未按结果进行替换。

预测ties给半分，真实return ties不计方向准确率。Top-1 regret相对已实际执行的共同候选池，不是全动作空间最优值。每个λ自己的mean、B及预定Elite都有真实分支；候选池仅用于离线排序分析，不是新增在线多模态规划器。

## 两个旧故障起点（不参与选参）

|起点|真实ΔReturn|Mean末latent RMSE|Reward Head折扣误差|rollout Reward偏移|预测末Q差|真实末Q差（同a_H）|实际SAC RTG差|
|---|---:|---:|---:|---:|---:|---:|---:|
|1714:250|-6.141|0.0785|+0.127|+1.498|+5.597|-0.447|-6.395|
|559:600|-0.117|0.1151|+2.648|+0.987|+3.472|+0.644|-0.090|

此处沿用旧D0动作序列，补算三层误差，并非重跑λ优化或使用旧9起点选参。

## 小规模完整回合配对评测

### 开发：每组4回合

|组|Success|Crash|OOR|Route|Episode Return|实际折扣Return|预测增益|prefix增益|加权terminal增益|规划ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B|100.0%|25.0%|0.0%|98.94%|346.44|143.74|NA|NA|NA|91.6|
|Q1|75.0%|25.0%|25.0%|82.54%|298.43|124.47|5.109|0.639|4.470|168.7|
|Q05|100.0%|25.0%|0.0%|98.95%|334.51|144.77|3.021|0.748|2.273|173.7|
|Q025|100.0%|25.0%|0.0%|98.77%|315.78|130.83|2.158|0.980|1.178|171.4|
|Q0|100.0%|0.0%|0.0%|98.65%|344.93|144.88|1.058|1.058|0.000|168.7|

逐场配对CSV和所有指标的bootstrap95% CI保存在对应的 `*_paired.csv`、`*_closed_analysis.json`。

### 独立验证：每组6回合

|组|Success|Crash|OOR|Route|Episode Return|实际折扣Return|预测增益|prefix增益|加权terminal增益|规划ms|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B|100.0%|0.0%|0.0%|98.84%|361.75|142.38|NA|NA|NA|101.1|
|Q1|83.3%|0.0%|16.7%|95.47%|328.45|137.19|4.935|0.570|4.365|186.8|
|Q05|100.0%|16.7%|0.0%|98.76%|315.47|142.43|3.115|0.959|2.155|188.8|
|Q025|100.0%|16.7%|0.0%|98.89%|353.81|144.31|2.146|0.808|1.338|189.6|
|Q0|66.7%|0.0%|16.7%|92.94%|319.87|136.36|1.038|1.038|0.000|187.5|

逐场配对CSV和所有指标的bootstrap95% CI保存在对应的 `*_paired.csv`、`*_closed_analysis.json`。

## 动作分布偏移与Mean/Elite

原始候选中保存SAC、四种mean及预定Q1最佳记录Elite动作。cases给出首steer/throttle、完整H步L2/RMS偏移、预测/真实Reward增益、latent误差、Crash/OOR；associations给出动作偏移−latent误差、油门−事故、预测−真实增益及latent误差−排序错误关联。它们是探索性相关性，未加动作惩罚或硬信任域。

- Q1：predicted_vs_actual_gain_spearman=-0.343, latent_error_vs_ranking_error=0.193, sequence_shift_vs_latent_error=-0.084, throttle_vs_crash=0.193, throttle_vs_offroad=NA, throttle_vs_actual_gain=0.091
- Q05：predicted_vs_actual_gain_spearman=0.077, latent_error_vs_ranking_error=0.307, sequence_shift_vs_latent_error=-0.280, throttle_vs_crash=NA, throttle_vs_offroad=NA, throttle_vs_actual_gain=0.301
- Q025：predicted_vs_actual_gain_spearman=0.378, latent_error_vs_ranking_error=0.579, sequence_shift_vs_latent_error=-0.238, throttle_vs_crash=-0.160, throttle_vs_offroad=NA, throttle_vs_actual_gain=0.406
- Q0：predicted_vs_actual_gain_spearman=0.462, latent_error_vs_ranking_error=0.563, sequence_shift_vs_latent_error=0.028, throttle_vs_crash=0.123, throttle_vs_offroad=NA, throttle_vs_actual_gain=0.573

|起点|集合|Mean−Elite预测值|Mean−Elite真实折扣Return|
|---|---|---:|---:|
|40113:150|开发|-0.087|-28.186|
|40113:50|开发|-0.148|+0.088|
|40330:150|开发|-0.018|-3.496|
|40330:50|开发|-0.084|-7.585|
|40511:150|开发|-0.017|-0.470|
|40511:50|开发|-0.340|+0.030|
|40765:150|验证|-0.066|+0.229|
|40765:50|验证|-0.016|+8.381|

正值表示Mean更高。8个起点中预测均偏好记录Elite，实际Mean与Elite各有4个起点更高。不能据此认定总是应选Mean或Elite，也未改变在线输出规则。验证集动作偏移与latent误差关联为负或接近0，油门−事故关联事件样本很少，未证实‘偏移越大误差必越高’或‘加油门必然致事故’，没有据此添加惩罚。

覆盖32个可复现起点；道路类型['CircularLane', 'StraightLane']，速度范围6.19–14.82m/s。交通数量及横向偏差、10m参考航向变化保存在coverage。

## 选参、独立结论与尚未解决的问题

开发集选择：`None`。只允许Q05/Q025作为候选，需开发反事实实际收益和完整回合均超过B/Q1且事故不恶化；Q0只诊断。选择只读取开发文件，验证集不用于改选第二个λ。

No independent, accident-preserving improvement evidence. No new test/full100, no extra coefficients and no retraining.

未取得预定的独立、事故不恶化的改善证据，未启动额外测试或正式100场。代码和参数消融可运行，不等于已实现真实驾驶收益提升。

**仅靠不修改权重的MPPI搜索与超参数调整，可能无法解决冻结WM的价值排序问题。** 同一动作在predicted/actual latent上的Q翻转提示Dynamics误差；在actual latent和真实SAC下一动作上仍不匹配RTG，则提示Q泛化、策略分布偏移或状态表示不足。Reward Teacher-Forced与开环分解用于判断Reward Head是否也需适配。有限验证样本不能证明不可修复，但不支持继续堆叠评分系数。λ=0.25在完整验证回合中实际折扣Return略高于B，但未折扣Episode Return下降且出现1次碰撞；这同时说明折扣目标与成功/事故/未折扣收益存在取舍，不能把所有安全指标退化都解释成Q拟合错误。本轮未改Reward、gamma或事件定义。后续适配应先改善真实折扣候选排序，再独立检验整体驾驶指标，不能承诺仅拟合Value就改善安全。

## 下一阶段最小适配方案（本轮未执行）

1. **Dynamics适配单因素实验**：另采训练场景中的真实SAC及预先固定Gaussian扰动前缀，保留真实后继观测/Reward。冻结encoder与SAC，使用MPPI评估器专用WM副本做短多步transition适配；以未见场景候选相对排序、真实前缀误差为验收，不能只看平均MSE。若Teacher Reward误差主导，再单独验证Reward Head适配，勿与Dynamics同时改而无法归因。原WM给535维SAC提供特征的路径继续冻结，避免更改其输入分布。
2. **SAC延续价值适配**：先在真实轨迹上做冻结SAC对应的离线FPE/MC诊断，使用本任务相同Reward、gamma和原生done，分别检验SAC分布内和MPPI扰动分布。真实535维状态可用于离线FPE；若在线只预测latent，必须另外验证latent-only SAC value近似能否表达Lattice/SAC策略信息，不能凭空补造未来535维特征。经独立验证后，才考虑用评估器专用的SAC-continuation terminal value替换统一目标中的旧Q。
3. 训练/开发/验证按场景分离，禁止使用旧20场封存集、本轮验证及正式测试场景训练。原checkpoint只读，适配副本另存新checkpoint；两个适配方向先各自消融，不加入Recovery、安全审批或规则候选。本轮没有创建或训练这些新模型。

证据/代码：`reports/mppi_calibration_20261116/`；完整Diff为 `code_changes.patch`，输入不可变校验为 `verification.json`。未自动训练、提交或推送。
