# Simplified TD-MPC2 MPPI

基线：main `618eebfe6c15123134ee18a04e0a83762b06acc0`；工作区开始时干净。

## 官方实现与本轮对应

已直接阅读[官方 `_plan()` / `_estimate_value()`](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py)、[WorldModel](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/world_model.py)、[配置](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/config.yaml)，以及[论文 §3.2、式(6)](https://arxiv.org/html/2310.16828v2#S3.SS2)。源码 blob SHA 保存在 `reports/simplified_tdmpc2/official_sources.json`。

| 核心环节 | 本轮实现 |
|---|---|
| `_prev_mean` | 首次使用SAC；之后只做上次mean平移一拍并投影回当前SAC信任域。每步std从保留的initial_std初始化。 |
| Gaussian | 64条，含原始SAC baseline；动作范围[-1,1]及原max_delta双约束。 |
| `_estimate_value` | 复用现有批量evaluator，H次reward/next，候选a_H只做terminal Q；保留termination mask、discount、全Q ensemble聚合。 |
| Elite | 只按WM总回报Top-8，无几何、风险、Cost、进度或Pareto重排。 |
| 加权更新 | exp(temperature×(value−max))，temperature=0.5，归一化后更新同一mean/variance/std，共4轮。 |
| 输出 | 最终mean经边界/信任域投影，与baseline在同一批次、同一Q reducer下重新比较；只有严格更高才执行mean[0]，否则精确保留SAC。 |
| 异常 | 任一WM价值NaN/Inf，直接保留SAC；没有Recovery/Emergency/Safety Failure分支。 |

本项目遵循用户要求的三个明确适配，不能误称逐行照搬官方：官方当前代码最后从Elite抽样，本轮评估输出mean；官方terminal action来自policy prior，本轮使用候选a_H；官方Q avg随机抽两个，本轮保留项目既有全Q均值以统一比较。官方无效值nan_to_num(0)也替换为要求的SAC保留。没有复制官方512/64/6或horizon=3；当前仍64/8/4、horizon=10。

## 清理与接口

`planning/local_mppi.py` 已重写为单一优化器，旧Recovery、Emergency、Anti-Stall、Latch、Pareto、进度门槛、Preview/Exact、Center/Legacy候选检查、完整证书筛Elite、多初始化模式及所有附加评分代码实际删除，没有留一套旧MPPI并行运行。`value_ranking_ready`运行时阻断也已删除；候选WM evaluation显式compute_cost=False。

Coordinator只向MPPI传latent、SAC refined_actions、eval_mode；不再创建/传递MPPI安全/恢复/多几何上下文。删除Agent因缺证书而终止环境的分支。MetaDrive实际Crash/OOR/Reward/Success及1000步终止规则没有修改。

SAC已有几何服务和Cost Head均保留：前者继续产生原535维输入所需特征，后者继续服务原训练/状态接口。选定动作的可视化仅复用已有SAC几何rollout，不筛候选、不参与value、也不作为证书。独立历史物理校准工具保留为离线分析，不连接新MPPI。

配置删除retired recovery/trajectory_selection/proposal_init/cost/deviation/smoothness/long-risk等项。保留3个车辆几何fallback字段，因为全局搜索确认Coordinator的SAC特征仍读取它们。E0/E2仅删MPPI评分项，其Cost Head训练开关未动。旧checkpoint参数通过native_mppi_config过滤迁移，不能复活旧模式；网络/维度/输入顺序/动作映射的校验保留。

相关入口和profiler已适配，新旧Smoke入口都指向同一新实现。历史日志和checkpoint未覆盖；未训练、未commit/push/reset/checkout。

## 已完成的验证

- 六类核心单元测试，NaN/Inf各一例；连同原有SAC、WM、checkpoint相关测试共27项通过。原测试文件及断言保留。本轮未宣称旧Recovery/Cost评分接口测试仍适用，也未运行整个旧功能套件。
- 真实checkpoint在1714、559：前后535维输入、残差输出、refined_actions及首动作逐位一致，误差均0，见`sac_interface_verification.json`。
- 固定状态：1714:250、559:600，各60步。均无碰撞、越界、Safety Failure。每步实际64/8/4，WM预测增益均值分别3.772、3.955；原基线保留比例约1.67%与0%。这是预测优化发生的证据，不是实际成功率提高的证据。
- Smoke单进程平均完整规划耗时约135.70/136.30ms（含完整Lattice/SAC/WM），不能宣称已达100ms。

B使用之前已完成且固定的同协议100场基准，来源CSV及SHA在`paired_run.json`明确记录，没有将其描述为本轮新重跑。D在相同checkpoint、100个固定ID、test seed=1、环境配置、Reward/Success、1000步上限下新跑100场。SAC前后接口一致性记录作为复用B的回归证据。场景顺序不按成功率重排，所有场景保留。

## 完整100场配对结果

代码修改和测试已完成，但**D优于B的实验目标未实现，碰撞点估计恶化，不能认定达到安全性能目标**。

| 指标 | B：Lattice+SAC（复用） | D：Simplified MPPI（新跑） |
|---|---:|---:|
| Episodes / Unique Scenarios | 100 / 100 | 100 / 100 |
| Success | 76% | 75% |
| Crash（episode任一步） | 16% | 26% |
| OOR（episode任一步） | 24% | 25% |
| Route Completion | 89.42% | 89.28% |
| Timeout | 0% | 0% |
| Safety Failure | 0% | 0% |
| Mean episode return（未折扣） | 305.11 | 286.48 |
| Mean episode speed | 8.58m/s | 8.28m/s |
| Mean episode planning time | 115.32ms | 166.22ms |

同一场景配对bootstrap 2000次，seed=1。D−B的95%区间：Success −1个百分点 [−7,+6]；Crash +10个百分点 [0,+20]；OOR +1个百分点 [−6,+7]；Route Completion −0.137个百分点 [−2.543,+2.297]；Return −18.62 [−35.84,−1.72]。成功率没有提高的证据；Crash区间下界为0，不将其表述为严格显著，但点估计和Return均不支持性能改善。救回5场、退化6场；未挑选场景或按成功率重新排列。

Crash是全回合事件，可能与Success并存；D的结束原因是75 arrive_dest、23 out_of_road、1 crash_vehicle、1 crash_sidewalk。不要将23次OOR终止误当作25%的全回合OOR事件率。Safety Failure为0只说明删除了旧规划器证书终止路径，不代表实际安全保证。

输出：`logs/benchmark/simplified_tdmpc2_BD_fixed100_20261111_213522/`。`episodes.csv`保留200条记录；`scenario_contributions.csv`按原100个ID顺序逐场对照；`paired_comparisons.csv`保留区间；`summary.json`包含来源与完整统计。四个D分片退出码均0，每个实际配置与对应B存档的环境配置完全相等，见`protocol_checks.json`。B复用的局限是未在本轮硬件负载下重新运行，不能将历史B耗时与新D耗时之差作为严格隔离的性能归因。

## 优化幅度与耗时

D每步固定4轮，64 samples、8 elites；每步258次候选评价（4×64+最终baseline/mean两条），NaN/Inf回退率0。以下均按episode均值再对100场平均：

- WM score gain 4.587；精确保留baseline比例0.626%。优化确实运行，但预测提升不能代替真实回报验证。
- 首动作修正L2均值0.16050，整条序列RMS修正0.11412。序列内最大steer/throttle修正的逐步统计均值0.08635/0.20418；硬边界仍为原0.10/0.25。
- 采样1.06ms，Elite更新1.83ms，WM批量推演总计47.61ms；最后两条价值复评10.44ms。复评包含在WM推演总耗时中，不能再次相加。
- 完整规划166.22ms包含Lattice/SAC/WM等流程，四进程共享硬件；单进程Smoke约136ms，也未达到100ms。没有以更少采样预算换速度，没有声称实时目标达成。

## WM排序探针与尚未解决的问题

在预先固定的1714:250、559:600状态各做两次独立进程回放。比较记录到的观测、ego/已收集障碍物状态以及baseline/mean/预测值，全部逐位相等。真实环境执行各自H+1动作，再接相同SAC策略直到原环境终止；没有新增规划规则或排序门控。

| 初始状态 | WM预测mean−SAC | 真实折扣return mean−SAC |
|---|---:|---:|
| 1714:250 | +3.0094 | −5.8133 |
| 559:600 | +2.7670 | −0.5051 |

两例均出现排序反向，见`reports/simplified_tdmpc2/value_ranking_probe.json`及四个原始`probe_*.json`。这不是100个状态的排序验证，也不是完整隐藏模拟器状态相同的证明。探针评价的是候选序列+共同SAC后续策略，并非闭环MPPI策略价值；不能用它直接量化WM各子模块误差。

已经确认的问题是：纯WM目标能稳定提高自身预测，却没有改善这100场的实际结果。需要进一步分解短时reward预测误差、latent rollout误差和terminal Q误差；训练Q所对应的后续策略与部署SAC后续策略不一致、局部动作分布外偏差及优化器利用模型误差均是待验证假设，不是本轮已证实的唯一根因。保留项目既有全Q均值及termination mask，没有为使结果变好而更换Q聚合、加不确定性罚项、恢复控制器或Cost门控。当前不建议将D作为优于SAC的方案发布。

## 修改文件与交付

- `planning/local_mppi.py`：1478行简化为203行；保留唯一Gaussian→WM→Elite→Mean/Std→平移Warm Start链路。
- `planning/coordinator.py`、`planning/types.py`、`agent.py`：删除旧候选安全上下文、恢复字段、证书中断；适配精简结果和旧checkpoint配置迁移，保留SAC特征服务。
- `planning/evaluator.py`：仅去掉无用排序资格接口；原H/H+1、discount、termination及Q聚合数学未改。
- `config.yaml`、`planning/native_config.py`、`experiment/E0.yaml`、`experiment/E2.yaml`：删除退出路径的配置；保留训练和SAC依赖字段。
- `evaluate_model_suite.py`、`evaluate_ablation.py`、`evaluation_diagnostics.py`：不恢复旧MPPI模式，适配诊断；退役Cost评分消融入口明确报错，不能静默误标实验。
- `tools/run_trajectory_pair.sh`、`tools/planning_profiler.py`、`tools/benchmark_mppi.py`、`tools/smoke_mppi_v3.py`、`tools/smoke_v32_recovery.py`、`tools/calibrate_mppi_motion.py`：调用方同步，旧Smoke入口只转发新实现，历史物理分析与优化器分离。
- `tests/test_simplified_tdmpc2_mppi.py`：仅六类针对性测试。`reports/simplified_tdmpc2/`：官方来源、SAC前后接口、Smoke、27项测试记录、评测/分析辅助脚本及JSON。

完整代码Diff：`reports/simplified_tdmpc2/code_changes.patch`。可重新启动两组新评测：`bash tools/run_trajectory_pair.sh`；该命令创建新目录并前台显示进度。本轮已完成100场D，交付时没有继续运行正式评测或训练。

Checkpoint SHA256：`4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a`。评测前后runtime源码指纹一致；模型权重、SAC输入/动作映射、Lattice及MetaDrive判定未改。没有Git提交、推送或覆盖历史实验。
