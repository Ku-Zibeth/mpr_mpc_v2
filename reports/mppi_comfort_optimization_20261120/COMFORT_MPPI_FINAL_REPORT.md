# Comfort-aware Local MPPI 最终报告

**结论：未通过联合验收，保持默认Lattice+SAC，不建议部署Comfort MPPI。** 这是舒适性实验，不再以WM预测Return增加作为成功依据。

## 直接回答

- C相对A的主要Comfort Index配对变化：**-17.307%，地图95%CI [-26.032, -10.032]%**。工程目标是降低至少10%，不是挑一个改善的分量。
- C相对简单因果滤波B：**-2.524%，地图95%CI [-11.028, +4.734]%**。不能仅因MPPI复杂或有非零替换，就认定优于滤波。
- C相对A的Success差：+0.000%，地图95%CI [+0.000, +0.000]%个百分点；Crash：+0.000%，地图95%CI [-16.667, +16.667]%个百分点；OOR：+0.000%，地图95%CI [+0.000, +0.000]%个百分点；路线完成：-0.330%，地图95%CI [-1.051, +0.072]%个百分点。
- C相对A的实际折扣回报变化：-0.756%，地图95%CI [-4.651, +3.423]%；转向rate：-30.181%，地图95%CI [-44.925, -17.181]%；纵向jerk：+2.635%，地图95%CI [-2.069, +7.813]%；横向jerk：-21.053%，地图95%CI [-29.584, -13.023]%。
- 以地图CI检查仍未满足的条件：['longitudinal_jerk_not_worse', 'lateral_oscillation_not_worse', 'crash_noninferior', 'route_noninferior', 'return_noninferior', 'beats_simple_filter_comfort_or_task']。区间过宽时结论是证据不足，不以工程点估计替代非劣效证据。

## 1. 历史开发审计与模型限制

完整读取上一轮要求的6份报告/CSV/JSON，以及36条真实Baseline轨迹。旧12图现在只作为开发数据，没有重跑H1或Native/Reward/Q大量消融。真实轮角映射rate RMS的地图中位数约0.2872rad/s，纵向jerk RMS约25.997m/s³。完整分布见baseline_comfort_audit.md及baseline_comfort_metrics.csv。这些是操作性诊断，不是医学舒适性认证。

旧日志没有逐步lane ID/refined_path，因此旧变道、参考路径误差为NA，不能用world-Y伪造车道行为。新实验逐步记录lane、SAC参考目标和真实参考误差；道路接续与同一道路上的横向换道分别统计，未把所有换道归为不必要行为。转向方向/转向rate反转、横向误差波动等均单列，不能自动推断其任务必要性。

MetaDrive实车DefaultVehicle为1100kg、宽1.852m、轴距2.46894m、最大转角40deg；油门/制动对应四轮engine/brake force，原模型4/6m/s²并非经验证的真实能力边界。未拟合或篡改参数。原rollout每窗口从真实状态启动，H1/2/5/10误差如下（各图P50/P95/RMSE的平均，非声称所有窗口独立）：

```
                          p50       p95      rmse
horizon metric                                   
1       heading_rad  0.001967  0.012474  0.007580
        position_m   0.016919  0.044841  0.024739
        speed_mps    0.105901  0.737633  0.264471
2       heading_rad  0.003674  0.021420  0.013516
        position_m   0.045230  0.108254  0.060433
        speed_mps    0.213082  1.081788  0.463469
5       heading_rad  0.008363  0.046418  0.030326
        position_m   0.192964  0.518558  0.270271
        speed_mps    0.565478  2.127348  1.019021
10      heading_rad  0.017466  0.107597  0.061484
        position_m   0.711567  1.829750  0.966865
        speed_mps    1.388689  3.595527  1.886934
```

因此**不启用预测纵向jerk、横向jerk或横向加速度的在线评分**，不在H10上做虚假的物理精确评价。只优化由动作、控制周期和实车转角映射可以准确计算的转向rate/二阶差分。复用原TrajectorySafetyEvaluator.rollout，仅H1相对参考一致性；原Safety模块及SAC几何输入保持原样。

## 2. 最小代码实现

生产改动仅planning/local_mppi.py、planning/coordinator.py与新增planning/comfort_evaluator.py。前者增加objective_mode分派、comfort CEM及独立诊断；Coordinator在原SAC完成之后传递真实车辆状态、实际历史动作及refined_path，重置comfort历史，显示仅H1轨迹；新增模块承担归一化动作品质评分和短时任务保持检查。evaluator.py/types.py/trajectory_adapter.py/structured_proposal.py/safety.py及网络未改。

默认objective_mode=legacy，原config.yaml未变。显式comfort模式以refined_actions初始化；原Gaussian/AR1、64/8/4、std/rho/temperature、Elite更新不改。候选转向局部投影在Baseline±0.043423归一化量内，幅度是开发P95单步转向变化的一半；终端填充不优化。所有候选的油门序列严格等于SAC，避免不可靠纵向模型为舒适性压制必要制动。

成本为0.5×mean[(steering_rate/0.606295)²+(steering_second_difference/8.519899)²]。尺度为开发真实物理角导数的绝对P95，不把不同量纲直接相加，不进行权重网格搜索。只选这两个可靠控制指标，不机械堆叠所有示意目标。

任务条件包括：短时相对refined_path额外误差不超0.044841m、相对参考进度损失不超0.044841m、相对航向差不超0.012474rad，容差来自开发H1模型误差；实际要提交的首动作归一化舒适性成本也不能变差。它们是近似参考意图保持条件，**不是安全证书，也不能保证真实避障不退化**；最终仍以真实Crash/OOR及任务非劣效判断。

Baseline始终在池内。先选重评后可行且改善的Weighted Mean，否则选择实际采样的最佳可行Elite，否则返回原Baseline。没有Oracle、Reward/Q新权重、Recovery或新规则安全拦截。WM仍服务原SAC特征，但所有comfort候选的WM/Q调用数为0。独立舒适性分数不伪装成WM Return；相关WM别名为不可用，comfort诊断另列。

简单滤波B仅用当前SAC动作与实际上一动作，alpha=0.584241，tau=0.071162s；时间尺度由开发转向rate/jerk P95之比计算，采用相同局部修正幅度，油门不滤波，未利用未来状态。

## 3. 真实闭环实验

基线GitHub/main：a9e84b55f4d373cad11979e70eceed149db59a6d。新地图为[300910, 300881, 300853, 300588, 300200, 300832, 300410, 300794, 300751, 300883, 300497, 300776]。排除训练区间和全部历史图，扫描记录见exclusions.json；每张实际生成的道路几何SHA独立，不是把旧图改名。3个seed是配对规划/global种子，同地图环境/交通seed固定，故统计独立单位为12地图，不是36独立地图。

2张旧图A/B/C共6次100步短测通过；原A的动作、观测和reward与上轮记录逐位一致。核心及配置随后冻结，再运行每组36次完整原生回合，三组共108次。每周期Lattice→SAC→可选滤波或Comfort MPPI→执行首动作→环境反馈，未重放整条10步序列。36组初始观测/车辆/交通/RNG哈希相同，没有根据结果更换地图、seed或缩短失败回合。

| 每组36回合 | Comfort Index | 转向rate RMS rad/s | 纵向jerk RMS m/s³ | 横向jerk RMS m/s³ | 横向加速度RMS m/s² |
|---|---:|---:|---:|---:|---:|
| A 原Lattice+SAC | 1.2639 | 0.5475 | 23.9147 | 3.1751 | 1.3759 |
| B 因果低通 | 1.0530 | 0.3459 | 24.1909 | 2.7423 | 1.3703 |
| C Comfort MPPI | 1.0189 | 0.3502 | 24.3494 | 2.4539 | 1.2997 |


| 配置 | Success | Crash | OOR | Route | Timeout | 折扣回报 | 平均速度m/s | 实际动作替换率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A 原Lattice+SAC | 83.33% | 8.33% | 16.67% | 94.82% | 0.00% | 120.489 | 7.827 | 0.00% |
| B 因果低通 | 83.33% | 8.33% | 16.67% | 95.05% | 0.00% | 123.807 | 7.879 | 99.81% |
| C Comfort MPPI | 83.33% | 8.33% | 16.67% | 94.49% | 0.00% | 119.464 | 7.761 | 93.37% |


Comfort Index是raw转向rate RMS、统一SG滤波后的真实纵向jerk RMS、横向jerk RMS、横向加速度RMS各除以开发地图中位尺度后等权平均。SG为固定5点poly2，原始数据全部保留，离线滤波没有进入在线控制。横向加速度使用真实v×航向变化率，属于运动学测量代理而非加速度传感器。硬制动事件阈值为开发负加速度P95=11.336m/s²，不是通用舒适阈值或真实紧急状态标签。

主统计先按相同map/seed配对，再按地图聚类2000次bootstrap，报告95%CI、LOMO、逐seed、各分量及raw/filtered结果。相对变化以原方法绝对值作分母，避免失败负回报的符号歧义。详见paired_statistics.json、paired_comparisons.csv、map_level_statistics.csv及每回合episode.json。

## 4. 实时性

原生108回合是6进程共享GPU负载，包含冷启动。另在前两张预定地图各方法单进程运行105个真实控制周期，前5步暖机，600个剩余实际周期用于独立延迟测量，不加入驾驶结果样本量。

| 单进程，前5步暖机后 | LocalMPPI P95 | 完整Planner P50 / P95 / P99 | 实际Cycle P95 |
|---|---:|---:|---:|
| A 原Lattice+SAC | 0.00ms | 71.86 / 86.50 / 89.04ms | 91.49ms |
| B 因果低通 | 0.00ms | 71.38 / 86.36 / 90.97ms | 91.53ms |
| C Comfort MPPI | 25.35ms | 86.65 / 95.91 / 99.48ms | 100.79ms |


完整Planner同步包围agent.act及B组实际滤波；LocalMPPI单独包围plan；Control Cycle=Planner+env.step，均没有减少必要控制步骤或改变0.1s物理周期。诊断日志处理和磁盘写入不包含在控制计时内。所有P50/P95/P99/max及阶段计时在latency.csv，硬件状态在hardware_timing_start.txt。目标是Planner P95≤100ms，另预注册不超过A的1.20倍；未满足时不部署。

## 5. 验收、测试和交付

工程点估计：{'comfort_at_least10': True, 'success_noninferior': True, 'crash_noninferior': True, 'oor_noninferior': True, 'route_noninferior': True, 'return_noninferior': True}。

地图95%条件：{'comfort_at_least10': True, 'steering_better': True, 'longitudinal_jerk_not_worse': False, 'lateral_jerk_not_worse': True, 'lateral_oscillation_not_worse': False, 'success_noninferior': True, 'crash_noninferior': False, 'oor_noninferior': True, 'route_noninferior': False, 'return_noninferior': False, 'beats_simple_filter_comfort_or_task': False, 'across_maps': True, 'nonzero_replacement': True, 'isolated_planner_under100': True, 'isolated_planner_not_materially_above_A': True}。

最终联合通过：**False**。不把Baseline fallback、预测成本下降、单张地图改善或测试通过当成算法成功。如果简单低通已能达到相同或更好的舒适性且计算更省，本轮证据不支持继续保留复杂MPPI。

59项相关测试通过（原46项覆盖保留，新增10项comfort核心及3项实际指标/地图统计测试）。真实冻结WM legacy动作/Mean/Std/Q/RNG逐位一致。原WM/SAC535维输入、Residual映射、Lattice、Reward/Cost/Done、checkpoint SHA 4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a均保持不变。WM200k+RL300k仍为原总500k模型，未换成另一checkpoint、未训练网络。

交付包含protocol、manifest、开发舒适性审计、物理模型验证、108回合原始记录、实际舒适性指标、配对统计、延迟、图表、验证与完整代码Diff。正式输出目录按prompt为reports/mppi_comfort_optimization_20261120；若采集期间使用短目录名，所有任务结束后仅迁移目录，原始内容哈希保持不变，见directory_migration.json。文件夹日期为任务标识，实际完成时间记录在verification。

尚未解决：不可靠的长时域纵/横向物理预测仍未用于优化；没有传感器舒适度主观验证；12张地图/固定地图交通种子的外推范围有限；没有自动识别所有必要/不必要变道；不能根据本轮测试再调权重后将相同地图称为新独立验证。默认模式未启用，未commit/push。操作与复现见RUNNING.md。
