# SAC-centred MPPI refinement

基线 main：`3e404e439d1b07aff3979bd5e3f1781c40233629`。
修改前审查见 `reports/sac_refinement/ROOT_CAUSES.md`。

**结论：代码逻辑修正已完成，但成功率提升目标没有达成。固定100场景重跑，B=76%，D=4%；不得宣称MPPI带来正向贡献。**


## 实现

- `planning/local_mppi.py`：正常搜索始终围绕传入的 SAC `refined_actions`；统一使用 SAC 参考、原信任域及同一 J。移除生产路径中的 Lattice 高斯重定心、Recovery Pareto 排序、无改善触发恢复、强制取消 baseline 和绝对进度门槛。旧辅助函数保留兼容已有独立测试，但不再决定实际规划。
- 候选流程：Preview → WM → Top-2×elites shortlist → 完整/同前缀制动延续验证 → 合格 Elite → 分布更新。合格数量不足时保留上一轮分布。正常替换须超过原改善门槛、保持相对进度并在 SAC 信任域内；未知路线投影不当成零。
- `NORMAL_REFINEMENT`、`SAFETY_INTERVENTION`、`TRUE_STUCK_RECOVERY` 分离。只有近端硬失败或真实低速、实际进度不足、车身边界困境且没有前车阻塞时，才允许相应异常处理。安全干预不拿不安全 SAC 的高预测进度否决安全减速。分数不可用时先保留已验证的小幅修正，之后才考虑独立回退。
- `planning/evaluator.py`：复用现有同起点配对检查和排序统计；真实终止回报、场景留出和足够有效配对才可批准 Value 排序，且重算当前权重预测、随权重版本变化失效。短时域奖励、相关系数和调用者提供的旧预测不能直接批准。当前没有合格证据，可认证 SAC 优先。
- Value 未获批准且 SAC 已有完整执行证书时，跳过不能被接受的性能搜索，明确记录 `search_skipped_unvalidated_value=1`。这是证据门控，不是降低启用搜索的样本数：真正启用的搜索仍是 64 samples / 8 elites / 4 iterations，原有 Gaussian、Elite、Mean/Std 与 WM 机制保留。D 的这一状态不能声称产生了 WM 性能优化收益。
- 继续保留 Center/Legacy 安全标准、连续碰撞与车身道路检查、同前缀制动延续、已证明的圆弧车道内部接缝修复及最终序列证书绑定。模型分歧单独记录为不确定性，不能冒充真实碰撞或越界证明。
- 诊断新增状态、选择原因、SAC/最终首动作、基线/候选分数、性能修改/安全修改、模型分歧与有效 Elite 数。`progress_rejected_candidates` 现在仅计实际的相对进度拒绝；Emergency 不计正常性能修正。

TD-MPC2、SAC 535维输入/顺序/映射、Lattice 生成、训练超参数与阶段、环境终止条件和模型权重均未修改。Cost 验证器及 `allow_unvalidated_cost=false` 保持原样。

`tools/calibrate_mppi_motion.py` 仅增加 B/D 执行日志的显式来源选择及溯源；`tools/run_trajectory_pair.sh` 改用本轮验收文件和新结果目录，保留固定100场景协议。正式评测不保存全候选表，只保存原有轻量逐步诊断。

## 证据与局限

`reports/sac_refinement/b_safety_confusion.json`：实际 B 控制重放的115个窗口；fit=1525/539、heldout=1714/559，分别统计近端/完整时域、Center/Legacy、速度/弯道/边界分层。两模型同时拒绝但窗口内无真实事件：1/115；模型可行性分歧：1/115。留出62窗口未观察到事故正样本，不能估计召回率，也不能据此取消 Legacy 否决。该小样本与旧 D 的状态分布不同，不能把通过率直接作全100场景的前后提升。

`reports/sac_refinement/value_ranking.json`：559:600 与1714:250，各两条同观测、同物理起点的真实轨迹；候选H步后用同一SAC策略继续至环境终止。559预测增益+1.474、实际折扣回报差−0.187；1714预测+2.058、实际−4.426。仅2个起点且出现反序，不足以批准排序；不能外推整体成功率或把相关系数当验证。两种部署延续都可能碰撞后到达终点，Collision & Success 仍单独统计。该实验检查SAC延续下的部署回报，不声称原TD-MPC2 Q是为该延续策略训练的。

实际物理起点、动作、终止、回报和预测均保存在 `value_*_baseline.json` / `value_*_candidate.json`。复现脚本及原始来源在同目录，未修改或覆盖历史实验。

## 验证

相关测试：60 passed；六个新增回归覆盖精确保留基线、有效局部改善/相对进度、无改善不恢复、模型分歧、完整失败不能更新Elite、交通等待不加速。另覆盖不安全参考不能以进度否决安全干预，以及缺失WM分数时小幅修正优先。保留旧测试断言；Cost算术测试仅显式声明包含其固定候选的测试信任域，生产默认不变。

最终固定状态闭环、正式B/D100结果与配对置信区间由下方结构化报告补充。短场景通过不代表成功率已提高；并发正式评测中的规划耗时不能用作隔离硬件时延优化结论。

确定性受控恢复自身不是一次Gaussian优化：其`recovery_search_delta`以所选受控参考计算为0，真实SAC动作变化仍在`sac_to_selected_delta`。运行模块与交付源码仅有这一纯诊断赋值差异，动作选择及其他运行源码一致性已经校验；原始日志不改写，汇总保留`recorded_*`并按修正后的定义统计。对应源码差异与校验见`diagnostic_attribution_only.patch`和`diagnostic_attribution_fix.json`。60项相关测试通过，归因修正后其中6项关键回归再次通过（不是新增66项测试）。

## 完整100场景配对结果

| 配置 | Success | Crash | Out of road | Route completion | Timeout | Safety failure |
|---|---:|---:|---:|---:|---:|---:|
| B: Lattice + SAC | 76% | 16% | 24% | 89.42% | 0% | 0% |
| D: Lattice + SAC + MPPI | 4% | 2% | 10% | 72.27% | 4% | 82% |

100个固定独立场景、相同Checkpoint、test seed=1、场景seed规则相同，原1000控制步上限。各组恰好100行，无重复或缺失。以本轮B为基线；历史75%不是本轮对照数值。四分片并发，按原始场景顺序合并，未选取最佳子集。

- D−B成功率差：−72个百分点，场景配对bootstrap 95% CI [−80, −63]；没有挽救任何B失败回合，新增72个B成功而D失败的回合。
- 路线完成率差：−17.15个百分点，95% CI [−21.93, −12.57]。平均速度B=8.58m/s，D=7.94m/s，仍不能替代成功率结论。
- 记录的平均完整规划耗时B=115.32ms，D=218.53ms；属于本轮并发负载下的记录，不是隔离硬件的前后性能基准，不能宣称达到100ms目标。
- D实际执行36084个物理控制步，未验证动作执行数为0。82次Safety Failure都是失败终止，不能当成成功避险；4次Timeout，10次越界终止，4次到达终点。碰撞按整回合任一步事件统计，未修改环境判定。
- D基线保留率约91.07%（回合均值），但这一比例没有保住整回合成功率。性能优化修改比例为0：Value排序未获批准；修改主要来自安全干预，Cost罚项始终为0。

## 主要剩余问题

82个Safety Failure的最终步分类：56个近端通过但完整/制动延续认证失败；24个预测近端越界；1个预测近端碰撞；1个其他近端硬约束/上下文失败。2个终止状态记录到模型不确定性。分类是**模型拒绝原因**，不是实际事故或误报的证明。B、D在干预后状态不同，不能仅用同场景B成功就把D当前状态标成安全。

这说明主问题尚未解决：原有安全认证及回退在真实评测中仍使大量可行性无法得到证明，尤其是近端通过却被尾部证书要求阻断。不能通过取消双模型检查、允许未认证动作、直接放宽道路/碰撞阈值来“修好”成功率。需要更有代表性的同状态B动作/恢复动作重放与曲线、边界、动态交互正负样本校准；现有115窗口、缺少事故正样本的数据不足以批准更改主从安全判据。Value的两对真实回报也未支持放开性能排序门控。**本轮没有完成安全模型的有效标定，也没有证明Normal MPPI具有正向性能贡献。**

Timeout下降主要被Safety Failure替代，不能解释为脱困能力已经解决。两处固定状态Smoke通过只说明这些局部状态可以保留SAC驾驶，不能替代全量结果。

## 交付位置

- 合并结果、置信区间、逐场景贡献：`logs/benchmark/sac_refinement_BD_final100_20261111_154509/`，重点看`summary.csv`、`paired_comparisons.csv`、`scenario_contributions.csv`。
- 完整摘要和终止状态诊断：`reports/sac_refinement/summary.json`、`safety_failure_diagnosis.json`。
- 559与1714的最终60步记录：`accepted_559.json`、`accepted_1714.json`。均无碰撞/越界/Emergency/Safety Failure，路线增量分别0.128886、0.064480。
- 相关测试：`tests.txt`；纯归因修正后6项复验：`attribution_tests.txt`。
- 交付代码Diff：`code_changes.patch`；实际评测加载源码：`frozen_evaluation.patch`。唯一纯诊断差异见`diagnostic_attribution_only.patch`，动作选择与其他运行源码一致性见`diagnostic_attribution_fix.json`。
- 后续同协议完整启动入口：`bash tools/run_trajectory_pair.sh`。当前默认仍保守关闭未经验证的Value性能优化与Cost排序；不能把此D配置误称为已经验证的WM优化器。

本轮未训练、未修改Checkpoint、未变更535维SAC输入或Action Mapping，也未执行Git commit/push/reset/checkout。

原版TD-MPC2仍是独立研究基准，不混作本轮B/D。下一轮不宜直接扩大评测或训练；先对上述56个近端通过但认证失败的状态做同起点可行动作重放和独立场景标定。达到资格后再做A/B/C/D同场景、多随机种子的比较，并继续报告配对置信区间，不能强开未验证Cost来制造改善。
