# 策略匹配型终端价值验证：Phase1（2026-11-17）

结论：**现有Residual SAC Critic没有表现出比TD-MPC2 Q更好的真实候选排序，不能直接替换MPPI终端价值。** 它是残差参数动作下的soft Q，通过实际Lattice/轨迹再生/跟踪路径学习组合系统收益，不能等同于任意物理动作Q。其gamma=.99、熵项、残差无效惩罚与当前MPPI gamma=.995原始环境回报不一致。本轮没有修改任何原有网络、权重、输入、映射或在线评分。

本轮新增8个真实末端状态，来自4个预先固定新场景；每状态两种接管策略各2个随机种子，32条有效原生后续轨迹，加1条同seed精确重放。另复用197条历史完整535末端记录。新场景ID=[83964, 83060, 83571, 83797]；不属于训练10000–11999、既有开发场景或封存6场景。它们是本轮独立的小样本验证，不因结果补图、选参或训练；不能称为充分泛化证明。

## 1. Residual SAC完整审计

- Actor：535→256→256→4，ReLU；输出2均值及2logStd，tanh后为归一化[d残差,v残差]。训练随机rsample，部署评测用tanh(mean)。
- Reward Q1/Q2：独立537→256→256→1，输入concat(535状态,2残差)。另有独立Cost Q，不混成Reward Q。
- Target Q1/Q2：分别保存、无梯度，Polyak tau=.05；next action来自**当前**随机Actor，无target actor。Target=`r_res +.99*(1-done)*(min(targetQ1,targetQ2)-alpha*logpi_next)`。
- 状态顺序：latent512、路径12、coarse WMreturn1、可行性2、风险值/有效位8。含target_d/target_speed与路径摘要，**不含显式Lattice控制动作/完整序列/全部内部跟踪状态**。
- Replay action存requested normalized residual；executed steer/throttle只在诊断字段。状态转移由这条残差经实际组合路径产生，这是合法的残差MDP语义，并非单凭数值不等就判定Bug。

Checkpoint已有59000次SAC更新；alpha=0.017679。原始训练Replay未序列化，训练源码SHA未附：本审计能验证当前实现与Checkpoint配置，不能证明训练时每个样本的具体映射、无效比例或历史代码完全相同。

六项审计回答：1）Critic面向Lattice诱导环境中的当前随机Residual SAC soft策略。2）含Lattice路径摘要，不含显式完整基准控制。3）在535充分表征状态的前提下可近似组合策略收益，但不能断言完整Markov或评价任意MPPI动作。4）训练action是残差，真实执行action是steer/throttle，两者数值不同；Critic动作必须使用残差定义。5）不与当前WM原始回报/折扣完全兼容。6）Target使用当前随机SAC策略，但不等于deterministic评测的raw Return。

## 2. 统一离线数据和评价定义

value_comparison.csv逐状态记录场景/seed/起点/候选/H10、Actor残差、Q1/Q2/targetQ、原TD Q及真实RTG。完整向量在historical_terminal_dataset.npz和fresh_terminal_dataset.npz，按split及array_index连接，含predicted latent、actual observation/latent、original535、12Lattice特征、Actor残差与Qensemble；fresh另存完整coarse控制与原始Lattice特征。旧缓存缺少完整rawLattice状态，明确不补造。

主比较A：真实末端z与**实际Lattice+SAC首个物理执行动作**上的原TD Q，固定成员01均值；B：同一真实535及确定性SAC残差均值上的min(Q1,Q2)，其残差对应同一个物理动作。因而没有把未执行候选a_H或残差直接当物理动作。预测latent下TDQ、Actor mean Q及未执行a_H只作独立敏感性诊断，不混进主结果。C是SAC实际延续RTG oracle；D=0。

本轮MAE是绝对末端RTG误差；前轮8.46是候选相对Baseline的延续收益差误差，两者定义不同，不能直接横比。真实535和由真实末端规划得到的动作都只用于离线验证，没有成为MPPI在线未来输入。

所有标签使用真实info.tdmpc2_reward、原生done、H10后的后续轨迹。主标签统一gamma=.995；另对同一真实奖励重算gamma=.99作为敏感性对照，**不重标或缩放任何Critic参数**。现成两个Critic固有训练gamma不同，无法在不改权重的条件下同时严格兼容；这是被审计出的混杂，而非宣称已消除。TDQ只由原stochastic TD Actor bootstrap，并无SAC式soft Q entropy项。

## 3. 四组价值结果（新独立4场景）

所有排序先同起点候选内计算，再场景平均/bootstrap2000次seed1；两个起点候选、重复seed不当作地图。真实收益差≤1排除准确率计数，预测tie计0.5；Constant Top1固定原Baseline顺序，不用真实收益打破tie。Spearman/Kendall对常数预测未定义，记N/A。Calibration用预先固定值域bins，未拟合校准器。C的完美指标只是标签自比较，不能部署。

|模型|Pair Accuracy% [95%CI]|Spearman|Kendall|Top1 Regret|MAE|Calibration Error|
|---|---:|---:|---:|---:|---:|---:|
|A_tdmpc2_q|33.33 [0.00,100.00]|-0.50 [-1.00,0.50]|-0.50 [-1.00,0.50]|2.10 [0.04,5.34]|104.01 [84.90,133.54]|104.01 [84.91,133.54]|
|B_sac_critic|0.00 [0.00,0.00]|-0.50 [-1.00,0.50]|-0.50 [-1.00,0.50]|5.12 [0.61,9.62]|99.56 [69.70,117.03]|99.56 [69.70,117.03]|
|C_actual_oracle|100.00 [100.00,100.00]|1.00 [1.00,1.00]|1.00 [1.00,1.00]|0.00 [0.00,0.00]|0.00 [0.00,0.00]|0.00 [0.00,0.00]|
|D_zero|50.00 [50.00,50.00]|N/A|N/A|5.14 [0.65,9.62]|133.30 [106.01,151.67]|133.30 [106.01,151.72]|

B相对A的配对准确率改善=-33.33 [-100.00,0.00]个百分点，Top1 Regret改善=-3.02 [-9.12,0.05]（正值才改善）。同时考虑场景和两次重复标签噪声，Regret改善nested95%CI=[-9.115583063126625, 0.054495241778660386]；Pair Accuracy改善nested95%CI=[-1.0, 0.0]。

两项排序门槛是否同时有统计支持：**False**。不能因为MAE下降就判定通过。独立4地图配对CI很粗，尤其有效收益差候选对不足时不应夸大确定性；数值与有效数量详见ranking_metrics.json。

主排序只有3个场景的候选收益差>1，SAC为0/3、TDQ为1/3；第4场景属于近似相等。经验bootstrap对0/3给出0–0的退化区间，**不表示总体准确率确定为0**。为展示小样本不确定性，SAC的三场景Bernoulli精确95%区间约[0%,70.8%]；这不改变“没有优势证据”的结论。

历史数据与同奖励gamma=.99敏感性（按split严格分开；历史不能冒称独立验证）：

- fresh_independent_pilot/gamma995/A_tdmpc2_q：准确率 33.33 [0.00,100.00]%，Regret 2.10 [0.04,5.34]，MAE 104.01 [84.90,133.54]。

- fresh_independent_pilot/gamma995/B_sac_critic：准确率 0.00 [0.00,0.00]%，Regret 5.12 [0.61,9.62]，MAE 99.56 [69.70,117.03]。

- historical_fit_innerstop/gamma995/A_tdmpc2_q：准确率 55.96 [45.74,65.31]%，Regret 4.55 [1.75,8.45]，MAE 106.44 [94.77,119.23]。

- historical_fit_innerstop/gamma995/B_sac_critic：准确率 44.60 [32.87,55.49]%，Regret 5.59 [2.81,9.20]，MAE 110.97 [99.00,122.51]。

- fresh_independent_pilot/gamma99/A_tdmpc2_q：准确率 50.00 [0.00,100.00]%，Regret 0.37 [0.02,1.01]，MAE 155.71 [148.13,168.04]。

- fresh_independent_pilot/gamma99/B_sac_critic：准确率 0.00 [0.00,0.00]%，Regret 3.23 [0.04,8.66]，MAE 47.87 [33.82,58.14]。

- historical_fit_innerstop/gamma99/A_tdmpc2_q：准确率 60.49 [49.63,70.50]%，Regret 2.44 [1.16,4.30]，MAE 149.09 [143.00,155.84]。

- historical_fit_innerstop/gamma99/B_sac_critic：准确率 48.18 [34.53,61.17]%，Regret 3.54 [1.77,5.85]，MAE 68.05 [61.44,74.12]。

进一步仅在新真实轨迹上，离线减去原训练定义中的0.2×无效残差标志、使用gamma=.99：SAC Critic准确率 0.00 [0.00,0.00]%，Regret 3.28 [0.06,8.67]，MAE 47.40 [33.80,56.80]。这没有恢复排序优势。该辅助标签明确不替换环境rawReturn，仍缺随机SACsoft目标的熵/期望，不能称为已彻底消除训练目标混杂。

## 4. 策略接管与随机波动

每条分支均用独立进程重建同一起点、固定H10动作前缀，并核对末端观测、ego/交通可记录状态及RNG。SAC继续原确定性评测路径；TD Actor直接执行原model.pi(z)[0]随机动作，不运行TD-MPC2 MPC，也不宣称是确定性Actor mean。两者首动作可能不同，因此策略接管对照包含首动作及后续策略的共同影响，不能唯一分离“相同首动作下的tail不匹配”。

这里的TD Actor和TD Q均来自同一个WM200k+RL300k Checkpoint，绝非另一个单独训练/迁移的原版TD-MPC2完整MPC模型。该Checkpoint配置是Lattice采集StageA、随后冻结WM/Q；内置Actor的表现不能代表原版TD-MPC2规划器表现，也不能据此宣称所有TD-MPC2 Actor都失效。

|接管策略|Discounted Return|Episode Return|Success%|Collision%|OOR%|Route Completion%|Safety Cost|
|---|---:|---:|---:|---:|---:|---:|---:|
|lattice_sac|133.30 [106.01,151.67]|264.58 [178.03,337.16]|75.00 [25.00,100.00]|12.50 [0.00,37.50]|25.00 [0.00,75.00]|89.49 [70.06,99.40]|1.46 [0.42,2.74]|
|td_actor|22.30 [-0.09,37.86]|23.88 [-0.26,41.34]|0.00 [0.00,0.00]|43.75 [6.25,81.25]|100.00 [100.00,100.00]|27.48 [23.73,34.33]|2.91 [1.51,5.65]|

TD Actor−SAC的配对折扣回报差=-111.00 [-139.67,-77.00]。更关键的是，TDQ在其自身真实Actor首动作上平均预测237.63，真实Actor RTG仅22.30，MAE=215.33。因此，**不能把TDQ失准全部解释成“换成SAC延续策略了”**：Q对原Actor接管也严重失准，仍有价值校准/泛化/数据覆盖问题。首动作不同的限制仍然存在，不能据此定量分配唯一因果贡献。

两个suffix seeds在**相同**末端状态上施加，交通内部RNG保留场景前缀状态。审计进一步确认：SAC状态中的coarse WMreturn使用完整Q ensemble mean，**不是MPPI的random_pair**。故SAC确定性评测没有这项随机性，8个末端的两次SAC RTG样本方差实际均为0；TD Actor存在采样波动。预注册protocol中的“coarse-Q随机性”设想已在protocol_implementation_notes.json更正，不改历史protocol或结果。重复并非独立交通重采样，不能把TDActor采样波动归为Critic或纯环境误差。repeat_rollout_variance.csv及candidate_effect_vs_rollout_noise.csv分别记录重复方差和候选效果。每末端只有2次，不足以识别纯环境噪声分布。另一次相同seed重放动作/观测/奖励/Cost/物理状态完全一致。

## 5. 最终A—F与下一阶段

- A：现有SAC Critic可近似残差诱导组合系统收益，但不是已验证的完整Lattice+SAC确定性raw V，也不是仅latent可调用的末端价值。
- B：在本轮新场景中**未优于TDQ**，SAC Critic Top1 Regret约5.12，TDQ约2.10；已有历史数据亦无优势。
- C：**没有正向统计支持**；配对Regret改善CI包含0且点估计为负。SAC gamma=.99下MAE更低，但排序和Regret仍未改善。
- D：能确认训练目标和策略定义不一致，但仅靠“策略匹配”不足以解释：TDQ对自身Actor也失准。优先审计已有Q训练目标、泛化及状态—动作覆盖；535状态遗漏、gamma/entropy及Replay分布各自贡献尚不能唯一分离。
- E：当前数据支持否定“直接使用现成SAC Critic”的方案，**不足以支持马上再训练新的策略匹配价值模型**；需先明确目标并定位真实状态下Q为什么失准。
- F：**暂不值得推进MPPI在线动作选择或正式100场**；保留Lattice+SAC Baseline，不改在线架构。

## 6. 交付与限制

新增离线tools/policy_value_diagnosis.py、3项针对性测试及本目录数据采集/分析脚本。原59项相关测试保留，总计**62 passed**。原WM/SAC/535/Action Mapping/Lattice/采样预算/默认Q聚合/在线选择/环境定义全部不改，Checkpoint前后SHA及每条分支权重hash复核通过。没有新价值网络训练、没有100场、没有commit/push。

采集工具的初始问题及日志保留：Path配置序列化、TensorDict非tensor元数据hash曾在reset前退出；之后同进程reset在2条SAC后续轨迹完成后被严格起点比较拒绝。两个未配对pilot保留在failed_same_process_reset，未混入33条有效分支；改为freshprocess，未改任何行为策略/场景/参数。最终有效轨迹和起点配对检查见rollout_pair_verification.json；真实执行总数另有这2条pilot，不隐瞒失败尝试。

数据文件：critic_architecture_audit.json、value_comparison.csv、continuation_returns.csv、ranking_metrics.json、paired_rollout_results.csv、scenario_bootstrap.json、data_coverage.json、verification.json。结构化数组与CSV通过split/array_index连接；所有旧报告、日志和权重保留。
