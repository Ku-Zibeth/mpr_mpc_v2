# Phase 2：阶段性完整性审计（2026-11-18）

本轮 prompt 在“4.3 样本独立性：检查：同一地图”处截断，已请求补齐。仅完成已明确授权的版本核实和原始数据审计；未预设后续归因/实验门槛，未训练、采集新环境轨迹或修改在线代码。

主目录 main、上一轮 worktree 与 GitHub main 都为 `27f8131d95a39bfe2f4a95d101fef1774903b907`。主目录干净；上一轮新增脚本、报告、测试和数据仍未跟踪，**尚未进入 GitHub**。相同 HEAD 不代表实验代码已发布。

读取了全部 576 条原生分支及配套 NPZ/JSON、36 个固定起点、18 个地图。575 个不同协议分支，另 1 条精确重放；这些分支不是 576 个独立地图。分支计数：`{'H10/baseline': 95, 'H1/baseline': 287, 'H2/baseline': 95, 'H4/baseline': 95, 'H1/td_actor': 2, 'H1/td_mpc': 2}`。原 18 项交付齐全，上一轮所记录的脚本 SHA 均匹配；checkpoint SHA 与原记录一致。

所有可复核的记录项通过：提交给 env.step 的首动作和完整干预前缀、Baseline 保留、起点观测和哈希、535 状态的前512维与初始 latent、有限性和动作边界、gamma=.995 的完整 RTG/前缀回报分解、done=terminated OR truncated、原生终止位置和反馈 SAC 步数。保留当前实际 Reward，不重标标签。

H=1/2/4/10 的定义均为“先提交固定前 H 步候选动作（提前原生终止则停止），之后每步由原 Lattice+SAC 重新规划”；并非 H 步内每步重新运行 MPPI。4 条 TD Actor/完整 MPC 接管和1条重放有单独标记，不并入主要 SAC 策略标签。

**标签语义警示**：旧 JSON 的 `actual_continuation_return` 始终是第一步后的 RTG。H>1 时包含尚未完成的强制候选前缀，不能称为纯 SAC 接管价值。新 branch_index.csv 分列 `actual_continuation_after_first`、`actual_continuation_after_fixed_prefix`、`actual_prefix_discounted_return`，旧文件不变。

**证据边界**：NPZ 记录的是传给 env.step 的归一化 steer/throttle 命令，不是实测轮角或轮胎力。可验证记录命令完全一致，不能据此声称所有物理执行响应已验证；SI 差异记为空而不补造。MetaDrive 当前源码中 action[0] 是转向、action[1] 是油门/刹车，车辆前处理会 clip；保存候选已经有界，候选前的 Gaussian 原始越界采样未保存，不能反推裁剪来源。详细见 action_mapping_scope.json。

每步 crash/OOR/success 标志及真实车身姿态没有保存在旧分支 NPZ 中，仅有 episode 累积 crash/OOR 和最终 success；因此无法从现有记录确定事故发生的精确时间。不能用最终成功反推过程中无碰撞，也不能把 episode crash=1 当作前缀早期碰撞。

H2/H4/H10 只有6张 Pilot 地图，不能与 H1 全18张地图直接横比。上一轮6张 validation 已看过，后续只可作为既有失效归因数据，不能继续称为全新独立验证。repeat_seed 是记录的策略 seed；相同场景的重放不是新交通流，也不是独立地图样本。

文件：experiment_manifest.json（代码、checkpoint/config、运行包版本）、data_sha256.csv（原实验文件逐项 SHA256）、branch_index.csv（576分支统一索引与数组引用）、candidate_integrity.csv、integrity_audit.json、action_mapping_scope.json。原71项测试记录已核实，本次只运行数据审计，没有无关回归或重复上一轮环境实验。

待补齐 prompt 后，再确定如何解释35.64%的排序准确率、比较时域及分解 Dynamics/Value；当前审计本身不证明任一归因。
