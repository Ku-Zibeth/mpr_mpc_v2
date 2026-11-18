# Native TD-MPC2 parity audit — 2026-11-17

本轮默认LocalMPPI没有修改。原生TD-MPC2采样Actor终端动作与当前policy_prior均值动作是不同定义。

独立worktree `/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_first_action_20261117`，分支`experiment/mppi-first-action-20261117`，基线`27f8131d95a39bfe2f4a95d101fef1774903b907`。Gitfetch网络超时，通过GitHub连接器确认main SHA相同。源main/历史数据未覆盖。

规划使用随机双Q **平均**，训练TD Target用随机双targetQ **最小值**；Phase1是固定Q01平均。三个定义已分别枚举，未把min误称avg。

离线对照来自4个已见Fit起点、40条原生H10前缀→SAC真实后果。Candidate、PolicyMean、NativePolicySample共享完全相同的前缀/预测latent/Reward/折扣/枚举Q成员组合，仅末端动作变动。原生Sample确实调用原model.pi(z)[0]，Mean使用info.mean；没有改默认terminal_action。

逐候选保存Q0…Q4、10对平均、10对最小、五成员mean/median/min；100次随机双Q抽样共享一对成员评分整个候选池，分布存random_pair_100_distributions.npz及q_random_pair_100.csv。100次只抽Q成员，不是100次独立Actor或环境轨迹。

q_calibration.json报告各定义的候选内Spearman/Kendall/PairAccuracy/Top1Regret、Q MAE/Bias/Calibration与Top1一致率。历史Fit的描述性最好定义不构成独立验证，也未用于选择后续默认Q或终端模式。

## 小规模真实接管（与Actor-only严格区分）

从两个新Pilot地图的固定起点，都先执行同一个Baseline物理动作，再由对应控制器接管至原生终止。此处Q(s,a_baseline)的第一步相同，延续控制器分别为π_B、原随机Actor、完整原TD-MPC2 MPC。完整MPC实际调用tdmpc_agent.act，cfg.mpc=true；保留其原512/64/6预算，这不是将LocalMPPI的64/8/4预算加大。

|场景|控制器|Discounted Return|Success|Crash|OOR|Route|
|---|---|---:|---:|---:|---:|---:|
|90115|baseline|141.514|1|0|0|0.986|
|90115|td_actor|44.540|0|1|1|0.649|
|90115|td_mpc|100.634|0|0|1|0.630|
|90261|baseline|85.092|0|0|1|0.414|
|90261|td_actor|76.481|0|1|1|0.575|
|90261|td_mpc|63.218|0|0|1|0.402|

完整MPC可以运行；不能继续把Actor-only结果称为完整TD-MPC2性能。这里也仅是同一个WM200k/RL300k Checkpoint内的控制器对照，不是另一个原版迁移模型的评估。2个地图不足以宣称任何控制器普遍优劣。

依旧没有证据据此修改默认终端动作或Q聚合。下一阶段的主要标签是新的真实H1→反馈SAC收益，与本节H10历史对照严格分开。

全部源表、脚本、checkpoint SHA及历史Fit/InnerStop/Phase1 Pilot/Sealed Validation的区分见version_manifest.json；旧6个封存场景未触碰。
