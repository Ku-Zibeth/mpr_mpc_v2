# SAC-Centered TD-MPC2 MPPI

基线：`f619b91696d7d005ff49513a894c0cea3c1a4d63`，开始修改时工作区干净。原checkpoint、旧报告和历史CSV均保留；未训练、未提交/推送/reset/checkout。

## 核心算法与官方对应

已读取[官方planner](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py)、[WorldModel](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/world_model.py)、[配置](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/config.yaml)及[论文§3.2、式(6)](https://arxiv.org/html/2310.16828v2#S3.SS2)。实际读取的源码blob SHA记录在`reports/sac_diffusion_20261112/official_sources.json`。

1. 每次规划独立以SAC `refined_actions[H+1,2]`初始化mean，std由原initial_std初始化；没有历史轨迹混合、移位Warm Start或模式切换。
2. 每轮64条Gaussian，转向/油门独立创新噪声。先生成标准化AR(1)：`eta[t]=rho*eta[t-1]+sqrt(1-rho²)*xi[t]`，`xi~N(0,1)`；再采样`mean[t]+std[t]*eta[t]`。默认两个rho均0.8，支持分别配置。各时刻std相同时与prompt的epsilon公式一致；Elite使不同时刻std不同时，本实现保持各时刻边际标准差为std[t]、相邻标准化噪声相关系数为rho。只clamp动作范围[-1,1]；max_delta不再是搜索约束。
3. 按prompt给出的拼接方式，另外加入1条SAC baseline，因此每轮评价65条，而不是偷偷增加Gaussian预算。Top-8、4轮，最后baseline/mean两条共同复评；正常每步总计262次候选评价。
4. 复用现有WM evaluator：前H步Reward/next，a_H只进入Terminal Q，保留discount/termination mask。MPPI调用官方`Q(return_type='avg')`，随机抽两个Q并对解码后的数值取平均；一次调用中的所有候选共享该对Q，最后mean与baseline也在同一批次。不会对已经解码的avg Q再次two_hot_inv。
5. SAC输入所用的`evaluate_trajectory_consequence()`仍使用原全Q均值及原Cost/几何特征接口；没有把新的Q reducer灌进535维输入。
6. Top-K只按`J=sum(discounted rewards)+discounted terminal Q`。按本次明确公式使用`softmax((elite-Jmax)/tau)`，tau=0.5；加权mean/variance更新，std限制仍为原向量min/max。没有额外排序或安全分数。
7. 最终mean与baseline复评后，只有mean价值严格更高才选其首动作，否则精确保留SAC。评估输出确定，不再从Elite抽样。NaN/Inf明确发出数值异常警告并记录原因，返回有限SAC baseline，不触发额外终止、Recovery或Emergency。

**温度定义的差异必须明确：** 官方源码使用`beta*(J-Jmax)`，官方beta默认0.5；本prompt要求分母tau=0.5，等价于官方beta=2，锐度是官方默认的4倍。本轮按明确指定的分母公式实现，并未把两个“0.5”误称为相同超参数。另有三个明确的项目适配：SAC初始化而非零初始化、候选a_H而非policy-prior terminal action、最终mean与baseline比较而非官方Elite抽样。

## 清理与调用链

现在只有：`Lattice → Residual SAC → refined_actions → LocalMPPI → evaluator → Top-K/mean/std → selected_sequence[0] → MetaDrive`。

从LocalMPPI实际删除了Safety Intervention、Tail Viability Repair、True Stuck、固定减速/滑行/加速候选、结构化恢复、停车认证、Center/Legacy筛选、资格门控和历史轨迹状态。Coordinator不再构造/传入独立安全上下文，Agent不再检查证书或请求人工中止。`env/env.py`的`request_safety_failure`、pending/failure状态及其step分支实际删除，正常step仍沿用原Reward、Cost与MetaDrive的Crash/OOR/Success/Timeout判定。

`planning/mppi_safety.py`独立保留，供原离线几何/制动辨识及其测试使用；不导入或控制在线LocalMPPI。现有SAC的legacy几何服务保持不变。没有另一套并行MPPI类。

配置只保留采样预算、std、rho、温度、种子及SAC依赖的3个几何fallback。删除max_delta、安全/认证配置和无效Cost评分开关；旧checkpoint metadata经native_mppi_config过滤，不会恢复旧状态机。Cost Head结构/权重、训练阶段和训练超参数均未改；本轮仅纯Reward+Q，无Cost罚项。

入口`tools/run_trajectory_pair.sh`去掉上一轮制动资格验收阻断，仍防止覆盖旧结果。原认证回放命令转发到同一个当前Smoke入口；制动辨识工具改为显式构建离线checker，不依赖LocalMPPI。

## 诊断与测试

每步保存baseline/optimized/selected预测回报、优化/接受增益、初末std、动作修正、采样diversity、迭代次数/候选数及各阶段耗时。每轮mean/std、Elite mean/max、diversity记录在轻量`.search.jsonl.gz`中，不保存全部候选张量。原Step Logger的compact/stride机制保持。

额外记录真实完整回合折扣return、初始预测减真实折扣return、逐步return-to-go偏差和单步Reward误差。它们描述部署中的预测偏差：WM的terminal Q延续策略与实际SAC/滚动MPPI不同，不能把此偏差直接视为同起点候选排序验证，也不能唯一归因于Q Head而排除latent dynamics。

31项相关测试通过：11类12项当前MPPI回归，以及原SAC/Checkpoint与WM/Cost索引/排序测试。覆盖AR相关性、仅物理边界、64+1候选/8 Elite/4轮、Top-K、分母温度Softmax、mean/std公式、共同最终评分、无改善保留、NaN/Inf、H/H+1与已解码avg Q、SAC全Q聚合保持、Agent首动作一致及诊断日志。

按新prompt退役旧在线认证断言：保留`tests/test_sac_centred_repair.py`中的3项独立离线几何测试，删除其中已失效的在线模式/资格/认证Elite断言；删除`test_trajectory_selection.py`中仅要求人工证书中止的两项测试。`RolloutModel`测试stub补齐真实Q(avg)已解码的API，原WM数值断言未放宽。未删除SAC网络、动作映射和WM算术断言。没有声称整个历史模式测试套件都适用于本轮。

1714与559：修改前后535维输入、残差输出、refined_actions和SAC首动作逐位一致，误差均0，见`sac_interface_verification.json`。

## 固定状态Smoke

使用原checkpoint，原完整控制前缀1714:250、559:600，回放速度误差不超过原1e-4标准，各完成60个真实控制步：

| 状态 | 路线增量 | 均速 | Crash/OOR/人工SF | 完整规划均值 | 预测接受增益 |
|---|---:|---:|---:|---:|---:|
| 1714:250 | 0.06536 | 4.42m/s | 0/0/0 | 137.89ms | +4.7385 |
| 559:600 | 0.14786 | 6.93m/s | 0/0/0 | 139.41ms | +4.6836 |

每步实际4轮，未绕过搜索，无证书中止。此结果只证明局部可执行和搜索发生，不能替代100场成功率；完整规划仍未达100ms。

## 正式B100/D100

本轮B、D均重新运行，不复用旧B。相同checkpoint、原固定100个ID、test seed=1、每回合RNG种子`(1+scenario)%2**32`、相同环境配置、1000步上限及同一个`evaluate_model_suite._evaluate_checkpoint()`。每组4个相同场景分片，最多4个进程并发。原始输出和代码/权重指纹见`paired_run.json`指向的新日志目录。

两组各100场均完成，8个分片退出码全部0，均为100个唯一原始场景，按原顺序保留，不选择最佳子集。B逐场长度、Return、Success、Crash、OOR、Route Completion均与历史基准完全一致（`B_reproduction.json`），本轮仍使用新跑的B参与统计。

| 指标 | B：Lattice+SAC | D：AR(1) MPPI |
|---|---:|---:|
| Success | 76% | 71% |
| Crash（回合任一步） | 16% | 17% |
| Out of Road（回合任一步） | 24% | 29% |
| Route Completion | 89.42% | 87.25% |
| Timeout | 0% | 0% |
| Mean episode Return（未折扣） | 305.11 | 291.85 |
| Mean actual discounted Return | 131.69 | 129.23 |
| Mean episode Speed | 8.58m/s | 8.44m/s |
| Mean episode Planning Time | 95.80ms | 164.58ms |

**本轮代码重构和完整验证已完成，D优于B的研究目标未达到。** 场景配对Bootstrap 2000次、seed=1的D−B差值及95%区间：

- Success：−5个百分点 [−11,+2]；救回3场，新增失败8场。
- Crash：+1个百分点 [−7,+9]；OOR：+5个百分点 [−2,+11]。
- Route Completion：−2.169个百分点 [−5.134,+0.581]。
- 未折扣Return：−13.251 [−27.533,−0.050]；真实折扣Return：−2.459 [−6.053,+0.854]。

成功率、碰撞和越界区间包含0，不能将点估计下降/上升表述为已证明的总体显著变化；点估计没有满足成功率或风险目标。Reward负向区间接近0且为多个指标之一，不作夸大推断。碰撞按整回合事件统计，可以与成功并存；没有改变环境判定或人工中止。

结果目录：`logs/benchmark/sac_diffusion_BD_fixed100_20261112_204847/`。`episodes.csv`包含200条新记录；`scenario_contributions.csv`逐场配对；`paired_comparisons.csv`保留置信区间。当前实验与旧75%或4%的D属于不同代码版本，不合并统计。

### 搜索与耗时

以下均为先算每回合均值、再对100场平均：

- 接受的WM预测增益+5.815，优化mean的预测增益+5.814；baseline保留率0.318%，模型NaN/Inf率0。每步4轮、262条评价，没有跳过搜索。
- 首动作修正L2均值0.16837；整条序列内最大steer/throttle修正的逐步统计均值0.08887/0.28090。后者已超过旧油门max_delta=0.25，旧信任域不再拦截。
- 采样diversity按候选横向标准差定义：steering 0.02431、throttle 0.06177，合成RMS 0.04725。std从0.04/0.12收缩到约0.01187/0.03712，说明后轮接近min_std；已记录每轮分布，没有隐瞒方差收缩或自动扩大预算。
- WM推演总计48.13ms，Gaussian采样2.89ms，Elite更新1.57ms，最终两条复评10.40ms。最后复评包含在WM推演总耗时内，不能重复相加。
- 完整规划164.58ms，尚未达到100ms。RTX3070 8GB上的并发耗时是共享负载记录；不将95.80/164.58ms差值或与历史实验的差值称为隔离硬件的算法速度提升。

### 预测与实际收益分解

全100场D的初始选中轨迹预测回报平均比实际完整回合折扣回报高72.17，单步Reward MAE约0.1565。部署延续策略与Q的训练延续策略不同，这些数值是部署预测偏差，不是单独对Critic的因果归因。

额外在两个**预先固定**起点，各做SAC序列/优化mean两个真实分支，执行H+1个候选动作后接相同SAC直到原环境终止。初始观测、记录到的ego/障碍物状态、两条候选及WM预测逐位相等；回放误差标准未放宽。不声称所有隐藏模拟器状态完全相等，也不把两例作为整体排序资格验证。

| 起点 | WM预测增益 | 真实折扣增益 | 预测中的Terminal贡献增益 | 实际Terminal贡献增益 |
|---|---:|---:|---:|---:|
| 1714:250 | +7.0485 | −6.0904 | +6.4996 | −6.0311 |
| 559:600 | +3.4148 | +0.4840 | +3.2343 | +0.5154 |

1714发生反序，559同向但幅度被夸大；不能说“两例全部反序”。两例约92%/95%的预测增益来自Terminal Q，而不是前H步Reward。五个Q成员在预测末端状态上均偏好mean，因此1714的反序不是仅由随机抽取两个Q产生。

在1714的**真实编码末端状态**上，同一Q对的mean−SAC约−0.283，与预测末端状态上约+6.834方向不同，表明latent rollout会改变这对候选的末端排序。两例末端latent RMSE约0.079–0.125，为无量纲潜变量误差，不是米。真实状态上的Q绝对值仍约212–216，而同a_H+SAC延续真实值约84–97；这同时提示价值标定/续接策略错配，不能仅修正latent dynamics就宣称问题解决。

完整原始四条分支及分解在本轮`probe_*.json`、`value_ranking_probe.json`。分解恒等式已核对：真实Return=前H步折扣Reward+gamma^H×真实后续Return。探针不是闭环MPPI整回合收益，也不构成大样本排序验证。

### 仍存在的问题

搜索确实发生且提高了模型自身评分，但未改善100场实际收益。索引测试通过，当前未发现接口门控或人工中止；不把这些旧问题作为本轮已确认的差距来源。优先排查冻结WM的latent预测、末端价值/部署续接策略一致性，以及对Gaussian扰动动作的分布外泛化。SAC还包含Cost约束，纯Reward+Q优化不自动继承它；环境已有risk reward处罚也不等同于SAC的Lagrange成本目标。

本轮还同时改变了温度定义、Q聚合、相关噪声和搜索域，不能把全部差距唯一归因于其中一项。后续应在独立数据上分解这些因素并检查部署策略下的价值目标；本轮没有重新训练、调整测试参数挑选结果、增加安全规则、关闭搜索或强开未经验证Cost。

## 修改文件与交付

核心：`planning/local_mppi.py`、`evaluator.py`、`coordinator.py`、`types.py`、`native_config.py`、`agent.py`、`env/env.py`、`config.yaml`。

诊断/入口：`evaluation_diagnostics.py`、`evaluation_statistics.py`、`tools/smoke_mppi_v3.py`、`run_trajectory_pair.sh`、`replay_sac_centred_repair.py`、`calibrate_mppi_braking.py`。评测/探针完成后，仅将历史`tools/smoke_v2.py`中的旧Cost/信任域断言适配为纯WM数值和物理动作范围断言；保留其原有WM训练/Checkpoint断言，**本轮未运行这个会做训练更新的旧工具**。该CLI未被当前评测导入，精确源码差异及指纹重建证明见`post_evaluation_tool_compatibility.json`；模型、规划器和环境的评测源码完全未变。

测试：`tests/test_simplified_tdmpc2_mppi.py`、`test_sac_centred_repair.py`、`test_trajectory_selection.py`、`test_native_mppi_cost.py`。

辅助脚本及结构化证据：`reports/sac_diffusion_20261112/`。完整Diff：`code_changes.patch`；最终核验：`final_verification.json`。核心代码：`planning/local_mppi.py`。后续同协议前台命令为`bash tools/run_trajectory_pair.sh`，每次写入新目录。没有修改`mpr_mpc/`、SAC/TD-MPC2网络、Lattice生成、模型权重或训练流程。
