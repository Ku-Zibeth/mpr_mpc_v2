# H1 首动作原生反事实实验方案

本轮固定基线 `12da6bbfd40138b839eab35b0bd6d36dbb964eac`，GitHub main与本地一致。独立worktree `mpr_mpc_v2_h1_counterfactual_20261118`，仅新增本目录工具/报告。禁止训练、修改在线算法、默认配置、checkpoint、Reward/Cost/gamma或环境终止条件，不commit/push。

## 冻结样本与研究目标

地图依次为130719、130132、130620、130467、130806、130343、130605、130018；各取原B_r1轨迹第50、150周期开始前的状态，共16起点、8张地图。地图已经用于开发，本轮不是新独立验证。不重复计算B_r2相同轨迹，不根据候选分数或真实结果更换状态。checkpoint仍为原WM200k+RL300k总500k，SHA见protocol.json与manifest。

唯一干预：首动作执行一次0.1秒，随后立即由冻结原Lattice+SAC每步反馈控制至原生done。真实折扣回报从干预动作开始，以gamma=.995累计info.tdmpc2_reward，并逐步核验其与env reward一致。不加入任何 imagined 或补造回报；提前成功、越界和截断全部保留。

## 状态重建边界

原B快照物理时刻是第50/150步执行前，但保存时该步agent.act已经计算完。故从相同seed重置，逐步调用原B策略重建内部状态，同时执行原先前49/149条真实控制；每一条重放动作、观测和reward均与原记录核验。然后只计算一次目标步A0，匹配原快照全部observation/vehicle/traffic/RNG哈希，保存此“决策已准备、动作未执行”边界。

新增policy指纹涵盖实际latent、535状态、coarse/refined动作和参数、残差、原TD dormant prev_mean、私有MPPI RNG、Lattice引用路径及几何状态。扩展环境指纹涵盖各manager RNG、对象物理状态/RNG、交通policy numeric/PID历史。未初始化的Gym RNG保持None，不通过带副作用getter新建随机状态。

原MPR评测链路是feed-forward：Encoder每步编码真实观测，SAC是当前535状态MLP；没有递归latent或已执行上一物理动作的额外神经缓存。原TD `_prev_mean`只在另一条native-TD规划入口使用，此处保持不变。last_plan/last_result是诊断缓存，不作为下一周期控制输入。上一实际动作来自vehicle.current_action，安全特征构造必须逐步等于执行器上一步实际提交的动作；下一步latent与冻结Encoder对真实观测的输出逐位相等。不会伪造或改写训练用residual replay。

## 候选冻结与隔离

候选准备器与每个执行分支运行于不同进程。C/D在同一重建B状态，以相同目标决策前Torch/Python/NumPy状态及native私有MPPI seed产生真实在线输出；D仅启用已有Cartesian virtual speed开关。计算完恢复执行边界RNG并证明环境/交通状态未变，准备器的诊断缓存不传入执行进程。

- A0：原SAC baseline。
- A1：原C实际选中序列的首动作，可能是baseline或最终weighted mean。
- A2：已有D实际选中序列的首动作。
- A3–A7：C最后一轮实际Gaussian候选的最高、75%、50%、25%、最低预测分数对应首动作。实际trace确认4轮，每轮65×11×2，row0为baseline；选择只在row1:65中进行，按稳定排序取round(q×63)。

L2≤1e-5的首动作仅执行一个代表，保留全部来源别名及各自原分数/索引。最终Mean与实际Gaussian严格区分；不把Mean替换成最接近的采样动作。选择和去重只基于动作及预测值，发生在任何分支真实结果之前。

记录每个原score的Q成员和评价批次。另以C最终同一双Q组合对8条原序列进行H10复评，仅作目标不匹配的描述性对照，不改变实际选中动作、采样或终端Q定义。

## 分阶段放行

1. Phase0：版本、源数据哈希、checkpoint、接口与时间/状态审计。
2. Phase1：130719步50，先跑A0至原生done；不仅前10周期，整个余下轨迹的动作/观测/物理状态/reward/done均应重现原B。
3. Phase2：同一起点跑实际C选择及至少3个不同Gaussian分支，核验起点、只有1步干预、后续全为SAC、真实上一动作同步、RTG和终止。其结果保留在最终16起点数据中，不重复计数。若候选因去重与baseline重合，保留别名并说明实际数量。
4. Phase3：只有前述全部通过才运行剩余起点；对每个新起点先验证完整A0再放行其余候选。最多4个独立环境进程，不把共享GPU采集耗时当成实时性测试。
5. Phase4/5：统计、图表和报告。遇到关键起点或执行核验失败即停止放行，保留日志；不以挑换地图、放宽误差或删样本绕过。

## 统计与解释边界

每起点与A0配对，计算原C/D实际选中动作增益、有限候选Oracle Gain、Top1 Regret、随机候选平均增益及安全/路线结果差。地图内两起点平均，以完整地图聚类bootstrap2000次（seed20261118），95%CI；报告8次Leave-One-Map-Out敏感性。正收益同时报告严格>0及预设>1的实际回报差，不用结果调整阈值。

**正式H1 Ranking Accuracy=NOT ESTIMABLE**：当前没有通过独立验证、与本Lattice+SAC延续一致的可信V_SAC；不得把原TD-Q或Residual Critic硬当作该价值。原H10分数与真实H1收益的相关性只报告为目标不匹配的描述性诊断，不给它正式Pairwise/Kendall验收意义。即时WM Reward误差独立报告，不冒充长时域排序能力。

已有48回合是持续MPPI闭环，本轮是一次动作后B接管。若H1选中动作有正收益而持续MPPI没有收益，只能为滚动交互提供线索，不能证明某一后续重规划步骤抵消了全部收益；本轮没有相同起点“此后永远MPPI”的第三组，不夸大此项因果识别范围。

不以少数正/负起点宣布MPPI有效/无效。开发证据不足时停止扩大样本、网络与参数搜索，只提出唯一下一实验方向。
