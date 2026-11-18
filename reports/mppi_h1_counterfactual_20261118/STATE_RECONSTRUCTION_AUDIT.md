# 状态恢复审计

全部16起点和123个分支通过严格恢复。源数据目录：`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_root_cause_20261118/reports/mppi_root_cause_20261118`。

| scene | step | prefix_controls | baseline_tail_exact | exact_observation_vehicle_traffic_rng | exact_latent_residual535_policy | all_extended_environment_hashes_equal | unique_executed_branches | formal_h1_ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 130719 | 50 | 49 | 281 | True | True | True | 8 | NOT ESTIMABLE |
| 130719 | 150 | 149 | 181 | True | True | True | 7 | NOT ESTIMABLE |
| 130132 | 50 | 49 | 406 | True | True | True | 8 | NOT ESTIMABLE |
| 130132 | 150 | 149 | 306 | True | True | True | 8 | NOT ESTIMABLE |
| 130620 | 50 | 49 | 385 | True | True | True | 7 | NOT ESTIMABLE |
| 130620 | 150 | 149 | 285 | True | True | True | 8 | NOT ESTIMABLE |
| 130467 | 50 | 49 | 292 | True | True | True | 8 | NOT ESTIMABLE |
| 130467 | 150 | 149 | 192 | True | True | True | 8 | NOT ESTIMABLE |
| 130806 | 50 | 49 | 155 | True | True | True | 7 | NOT ESTIMABLE |
| 130806 | 150 | 149 | 55 | True | True | True | 8 | NOT ESTIMABLE |
| 130343 | 50 | 49 | 176 | True | True | True | 8 | NOT ESTIMABLE |
| 130343 | 150 | 149 | 76 | True | True | True | 7 | NOT ESTIMABLE |
| 130605 | 50 | 49 | 559 | True | True | True | 8 | NOT ESTIMABLE |
| 130605 | 150 | 149 | 459 | True | True | True | 8 | NOT ESTIMABLE |
| 130018 | 50 | 49 | 343 | True | True | True | 7 | NOT ESTIMABLE |
| 130018 | 150 | 149 | 243 | True | True | True | 8 | NOT ESTIMABLE |

## 核验内容

- replay49/149而非50/150；每个过去周期调用冻结原B并执行保存动作，核验全部观测与reward。
- 原快照在该周期agent.act后/env.step前；新边界与原snapshot_*_origin.json的observation、vehicle、traffic、Python/NumPy/Torch/引擎/traffic-manager RNG哈希逐位相等。
- 新指纹对照当前latent、535状态、SAC残差、coarse/refined序列、原TD dormant prev_mean、私有RNG及Lattice历史；所有分支逐位相同。
- 扩展指纹还包括全部manager RNG、对象物理和RNG、交通policy的数值/PID历史。未初始化的Gym RNG不经getter创建。
- 每个A0整段尾轨迹精确重现原B，累计4394周期；每个分支实际收到的第一动作与提交值一致。
- 后续每步上一实际动作由真实vehicle.current_action进入SafetyContext；第一后继latent重编码核验，无递归隐藏latent需要手工覆盖。下一planner缓存全部由新真实状态计算。
- 候选准备过程不执行env.step，不改变物理/交通/RNG；执行器为独立进程，目标状态后MPPI调用守卫保证0。

## 诊断工具问题的处理

初次失败只有Gym未使用的懒初始化RNG两字段不同；inspect已确认其getter会使用系统熵创建RNG。保留diagnostic_attempts、state_diff.log及原始失败日志。新增工具修正为只读现有_np_random，未放宽任一关键比较。正式分支均在修正后执行。

## 证据范围

确定性前缀重放实际重建Bullet/车辆与策略状态，而非手工设坐标。原历史文件没有保存旧policy指纹：新policy指纹是本轮所有分支间比较；旧latent由相同冻结Encoder+相同观测确定，并由完整A0尾轨迹复现补充验证。哈希是记录到的状态域一致性证据，不声称对未观测物理引擎内存逐字节序列化。
