# MPR-MPC V3.1 增量修改与验证

基线为 `main` / `9492872f94d33d294f0f75a84e5d961612e2005f`；开始时工作区干净，远端 main 与本地一致。没有 checkout、reset、commit、push，没有启动训练或正式百场评测，没有写入已有 checkpoint。保留既有训练进程。

## 五项任务

1. **Recovery Search**：正常安全基线仍使用 64 samples / 8 elites / 4 iterations 的单高斯 MPPI。不安全基线在首轮正常搜索后转入有预算的恢复分支，不再浪费剩余轮次反复搜索同一 SAC Trust Region。先检查原始 Lattice，安全时以它为中心进行最多 32 samples × 2 iterations 的同类 Gaussian/elite/WM 搜索；独立范围 `[0.20, 0.50]`，初始 std `[0.06, 0.20]`。Lattice 不安全时仅检查 3 个回正比例 × 3 个减速水平，共 9 条受控候选。所有候选保留动作边界、运动学、道路车身和连续碰撞筛选，之后才进行 WM 评价。几何与 WM 缓存仍在同一 plan 内复用。失败后保留已验证/未验证制动回退。恢复成功时更新最终 mean/std 和 warm start，避免延续旧的不安全搜索中心。没有重建多模态分布或 Emergency Controller。
2. **Cost 双重验证和门控**：保留原 Horizon Validation 及 SAC 的原 mask 语义，新增独立 Candidate Ranking Validation 状态。默认 `allow_unvalidated_cost=false`，未获排序批准的 Cost 继续预测、记录，但实际评分惩罚为 0。实验开关打开时才能使用未经验证的 Cost，并明确告警。没有把 MAE、RMSE 或跨状态相关系数当成排序批准依据。
3. **诊断**：最终执行第一步相对 SAC baseline 的变化，分别计入 `normal_search_delta`、`recovery_search_delta`、`emergency_fallback_delta`，同时记录 steer/throttle 分量；仅原样 Lattice 回退另记 `lattice_fallback_delta`，避免误称为搜索增益。受控候选中的全程紧急制动即使由 WM 选中，也归入制动而非恢复搜索。记录正常/恢复安全比例、Lattice 检查状态、模式/原因、Cost 两类批准状态、实际惩罚和恢复阶段耗时。JSONL 有可读模式名，CSV/汇总用数值模式和每模式占比。未检查的 Lattice 安全性为 NaN，`lattice_baseline_checked=0`，不会伪造安全结论。
4. **安全检查**：保留连续车身道路/碰撞检查，不放宽现有阈值。增加可配置 CV 位置误差包络 `velocity_error*t + 0.5*acceleration_bound*t²`；扫掠每一时间区间使用区间末的最大包络，包括递归叶节点。增加停车时间、含一控制周期反应距离的停车距离、预测时域是否覆盖停车时间的诊断。可选的同向前方障碍停车距离约束需要显式提供经过标定的保守减速度下界。新物理边界均默认 0/关闭，未捏造标定值。TTC/clearance 硬阈值保持原默认关闭。
5. **耗时**：Cost 不具有逐步递推依赖，把同一批 H 步 latent/action 对一次送入 Cost Head，代替 H 次小 MLP 调用，保持折扣、终止后 Cost 累计和 Terminal Q 语义。复用无残差时相同的 coarse/refined corridor context。正常采样规模与必要安全检查不变。

## Cost 排序批准条件和生命周期

复用 `CostModelValidator.set_candidates()`，不另建验证框架。每条记录仍包含 `executed=True`、`origin_id`、`observation`、实际执行 `actions` 和环境 `costs`；批准还需要：

- 同一 origin 的 `initial_state_id` 一致，标识恢复了交通参与者、仿真器与 RNG 等完整初始状态；只有 observation 一致时仍可输出统计，但不能批准。
- `cost_definition` 与 `cost_definition(agent.cfg)` 一致，涵盖 Cost 折扣、环境 Cost 配置、风险场和相关越界/事件配置。
- 当前时域的基本误差验证通过；默认至少 20 个不同快照、100 个合格可区分候选对，origin 平均排序准确率至少 0.8。
- 对“该 origin 排序准确率达到 0.8”的 origin 计数，95% Wilson 下界至少 0.6；预测最安全候选的平均真实 Cost regret 不超过 0.05。重复快照不能充当多个独立 origin；缺少来源证明的候选对不能凑足合格样本数。

这些是可配置的初始验收规则，不是已经标定的通用可靠性结论。记录 MAE、RMSE、相关性、排序正确率、合格 pairs/origins、置信下界和 top-1 regret。

批准以非持久化推理元数据保存，不增加网络参数或 checkpoint buffer。任意模型参数版本变化、Cost 更新计数变化、基本验证撤销、Cost 定义变化、重新设置/撤销验证数据、数据过期都会使排序批准失效。有效期从候选数据设置时计起，重复 validate 不会刷新旧数据年龄；默认一天，可配置。加载旧 checkpoint 不能凭旧 `cost_ready` 获得新排序批准。

**未完成真实排序验证数据采集**：仓库没有现成的、经验证的完整 MetaDrive 状态分支恢复采样入口，本轮没有伪造候选真实 Cost，也没有启动新数据采集评测。`initial_state_id` 及真实执行来源必须由调用方的可靠快照重放流程提供，验证器不能从一个 observation 反推完整隐藏状态。已有 200k+300k 权重当前仍为排序未验证，默认 MPPI 实际 Cost 惩罚为 0。

使用已有接口接入合格数据后：

```python
agent.cost_validator.set_candidates(candidate_rows)
agent.validate_cost_model()
# 撤销数据：agent.cost_validator.invalidate_data()
```

## 预测与实际结果的对应关系

分别保留 H 步预测碰撞/越界和**第一控制周期**预测标签。`EpisodeDiagnostics.observe()` 与 Step Logger 在环境 step 返回后记录实际碰撞/越界，并输出 `predicted_safe_actual_collision/offroad` 和标签可用性。这避免拿 H 步某个未来事故去比较下一步环境结果。数据进入 JSONL、episode CSV、summary 和 paired comparisons；没有预测时保持未知，不当成“预测安全”。

模式代码：0 baseline，1 normal，2 recovery_lattice，3 recovery_controlled，4 lattice_fallback，5 emergency_verified，6 emergency_unverified。恢复原因代码：0 none，1 unsafe_sac_baseline，2 no_usable_normal_candidate。CSV 中的模式均值只作原始诊断，比较时优先看 `mppi_mode_*_rate`。

仍需注意以下误差来源：CV 不预测车辆变道、加减速和转向；位置包络只有在提供可信界时才有约束意义，固定朝向也不是完整转向预测。运动学点估计不等于 Bullet 的实际轮胎、质量、制动力学。当前 H=10、dt=0.1 s 只覆盖 1 秒，高速时未覆盖完整停车过程。若当前车身已经在道路有效区域外，严格安全筛选可能无法认证任何从该状态出发的恢复轨迹，仍必须诚实输出未验证回退。没有为提高候选比例而放宽边界。

实际 MetaDrive 源码确认：`enable_reverse=false` 时负 action[1] 清零引擎力并调用 `setBrake`。探针车辆 max_steering=40°，max_engine_force≈773.26、max_brake_force≈103.26、wheel_friction≈0.9；这些参数不能直接当作已验证的减速度下界。本轮没有新增未经标定的 TTC、clearance 或制动距离默认硬阈值，也没有重新标定原 V3 的运动学参数。

## 修改文件

| 文件 | 增量修改 |
|---|---|
| `planning/local_mppi.py` | 恢复分支、独立预算/范围、评分门控、回退与动作归因 |
| `planning/coordinator.py` | 传递恢复所需 Lattice、复用同 corridor、模式/原因元数据 |
| `planning/cost_validation.py` | 同状态排序批准、真实标签来源检查、失效机制 |
| `planning/evaluator.py` | Cost 批量计算、分离两类验证状态 |
| `planning/mppi_safety.py` | 首控制周期安全标签、停车诊断、可选约束/包络 |
| `planning/swept_collision.py` | 连续碰撞中的时域不确定性膨胀 |
| `planning/safety.py` | 仅给共享结果类型补充首步标签可选字段；SAC 几何计算不变 |
| `planning/types.py` | 排序状态、规划模式和恢复原因字段 |
| `planning/native_config.py` | 新配置白名单，继续过滤退休搜索模块 |
| `agent.py` | 给现有验证器传入环境 Cost 定义 |
| `config.yaml` | 新增 recovery、Cost 门控/排序规则、默认关闭的物理可选约束；训练配置不变 |
| `evaluation_diagnostics.py` | 第一控制周期预测/真实对照、日志/汇总归因 |
| `evaluate_model_suite.py` | 传入 step info；历史 MPPI 恢复不误启用新 recovery |
| `tools/run_v3_ablation.sh` | V3.1 输出名、显式安全/恢复/Cost 门控开关 |
| `tools/smoke_mppi_v3.py` | 可选真实环境不安全基线注入探针、实际结果记录 |
| `tests/test_mppi_v31.py`（新增） | 8 个关键回归测试 |
| `tests/test_mpr_mpc.py` | 正常搜索专用 fixture 显式禁用 recovery |
| `tests/test_native_mppi_cost.py` | 原无门控对照测试显式打开实验开关；fake evaluator 区分排序状态 |
| `tests/test_mppi_v3.py` | fake evaluator 声明排序状态；验证旧 Lattice 回退不算恢复搜索 |
| `V31_MPPI_REPORT.md`（新增） | 修改、证据和局限说明 |

TD-MPC2/Residual SAC 网络、训练损失和权重、535 维特征顺序/归一化/mask/Action Mapping、Lattice 生成、环境成功/终止/真实 Cost 规则均未修改。默认仍为 0–20k WM、20k–50k SAC，训练期间 MPPI 关闭。历史运行恢复的训练阶段参数仍按 checkpoint 元数据处理。

## 验证结果

- 修改前 `259 passed`；修改后全套 `267 passed in 16.62s`，新增 8 个，未删除原有测试。最后的纯诊断归因调整另跑相关测试，`90 passed in 4.43s`。
- 覆盖恢复搜索超出 SAC Trust Region 但仍在 Lattice 独立范围内、受控恢复/未验证制动区分、安全基线不消耗恢复预算、全制动不计搜索增强、Cost 门控、排序反例、参数/定义/数据/时间失效、Cost 批量时域对齐、不确定性包络、逐步/汇总/配对指标。
- CUDA/MetaDrive：2 场景 × B/C/D 各 5 步，共 30 个真实物理步；20 次 MPPI 调用中 17 次非零第一步修正，没有制动回退；535 维状态检查通过。
- 另在 seed=539 下先执行 20 步 Lattice，再注入全 steering=1/throttle=1 的**人工不安全基线**。它不是 SAC 输出，也不是成功率实验。真实几何拒绝基线，Lattice baseline 通过，选择 `recovery_lattice`，恢复第一动作 delta norm≈0.954；normal/emergency delta 都为 0，实际执行后该步无碰撞/越界。恢复阶段约 46 ms，不代表完整规划或尾部延迟。
- 完整证据：`reports/v31/smoke_verified.json`。检查点 SHA256 前后一致：`4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a`。

## 完整规划耗时

采用 `tools/benchmark_mppi.py` 原协议：相同 checkpoint、RTX 3070、seed=1、场景 1714/539、各 20 步 Lattice 前缀后的相同实时位置/速度/朝向、H=10、64/8/4、各场景 2 次预热，总共 100 次同步完整规划；不执行测试候选动作。逐一核对状态、checkpoint hash、dt、配置和工作负载，两版都是每次 252 个唯一 WM 候选、安全比例 1。

正式对比使用 `before_verified.json` 和最终的 `after_final.json`；最早 `before.json` 与测试有过资源重叠，不作为主要对比。基线由 main 的临时源码导出运行，源码指纹与最初干净 main 一致，没有切换或覆盖工作目录。基线导出清单在 `baseline_source.json`。前后串行测量时，用户原有训练进程仍共享 GPU，因此只是本机条件下观测值，不能当作隔离硬件实时性认证。

最终数值见 `reports/v31/performance_comparison.csv`：

| 完整规划统计 | V3 基线 | V3.1 |
|---|---:|---:|
| 平均 | 161.27 ms | 150.23 ms |
| P50 | 160.02 ms | 148.87 ms |
| P95 | 166.07 ms | 158.01 ms |
| P99 | 178.26 ms | 174.06 ms |
| 最大 | 264.72 ms | 244.91 ms |

平均降低约 **6.85%**。主要收益来自 Cost MLP 批量化：单独 profiler pass 中 Cost 调用 CPU wall 总计约 13.0 → 1.4 ms；完整 WM 评价约 72.0 → 56.0 ms。阶段嵌套，不能相加；CUDA event 也不是纯 kernel 时间。最终运行源码指纹为 `314f03a3c5742461fdb29a65b89533264bef5cf530771492b49cd7d5f78d1ae9`。中间的 `after.json`/`after_verified.json` 留作审计，不替代最终数据。

**100 ms 目标尚未达到。** SAC 特征/几何和 coarse 上下文准备、串行 latent dynamics、独立安全层仍占主要时间。此基准状态没有触发 recovery，不能据此保证恢复分支最坏延迟。没有减少正常采样规模或跳过安全检查。

## 后续正式实验

本轮没有启动百场 A/B/C/D，也没有结果可证明成功率提升或紧急制动比例下降。建议首先固定同一 checkpoint/100 场景/配对种子，Cost 排序批准前保持正式 Cost 惩罚关闭，比较 C/D 的 recovery 开/关，重点看 success/offroad/crash/timeout、各模式占比、未验证制动、预测安全但实际事故和规划 P95/P99；随后扩展场景/重复种子。

可直接使用已有入口（此命令未由本轮自动启动）：

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
ENABLE_WANDB=true bash tools/run_v3_ablation.sh
```

默认加载已有 `wm200k_rl300k_20261103_195237/models/milestone_500k_full_mpr_mpc.pt`，固定 `scenarios/fixed100.json`，种子 `[1]`，A/B/C/D 共 400 回合，恢复/独立安全启用，未验证 Cost 不评分。日志会直接显示，结果写入新 `logs/benchmark/mpr_mpc_v31_ablation_fixed100_时间戳/`。Ctrl+C 只停止日志查看，后台评测继续；完整命令保存在结果目录 `command.sh`，也可先 `bash tools/run_v3_ablation.sh --dry-run` 检查。

仅开展明确标注的未经验证 Cost 对照实验时使用 `ALLOW_UNVALIDATED_COST=true`，不要混入正式排序已验证组。真实排序数据收集和安全/制动参数标定仍需后续完成；这些缺口通过默认关闭评分/可选物理阈值和显式未验证回退暴露，不会被误报为已完成安全保证。
