# 研究假设与资料核对

本轮只验证一个已观测到的问题：道路并集的几何覆盖与MetaDrive原生实线接触终止存在差异，是否会使MPPI修正方向错误。独立比较有无原生接触代价及是否只在名义风险出现时搜索，不扩大网络和搜索预算。

历史证据来源为上一轮的37份报告索引及72个闭环回合。本轮不把上周负结果删掉，也不把新场景当作无需验证的“干净地图”。保留旧的24张封存图；新的12图只是开发集。条件控制保留SAC当前油门，避免上轮纵向搜索增加碰撞的问题。

- [MetaDrive车辆源码](https://github.com/metadriverse/metadrive/blob/main/metadrive/component/vehicle/base_vehicle.py)：原生白/黄实线flag来自物理世界contactTest。
- [MetaDrive道路构造源码](https://github.com/metadriverse/metadrive/blob/main/metadrive/component/block/base_block.py)：实际道路标线被构造成Bullet box ghost。读取真实构造结果比根据所有lane边界重新绘制更能避免给本来没有实线的路口加上虚假约束。实际使用的是本地安装版本并保存源码SHA，而不是假定最新版源码与本地相同。
- [Nav2 MPPI官方说明](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/controller_plugins/mppi_controller/configuring_mppic/)：车身轮廓占用与路径跟踪是不同critic。这里只借鉴分别核对目标的原则，不移植移动机器人阈值或权重。
- [When to Trust Your Model / MBPO](https://arxiv.org/abs/1906.08253)：模型使用应考虑模型偏差，短预测是可研究方向。这是不同于本项目的RL训练工作，不为本轮控制器提供理论保证。
- [Robust MPPI](https://arxiv.org/abs/2102.09027)：反馈跟踪能力、名义状态和真实状态的区别需要进入分析。本轮没有实现该文的完整算法、重要性采样或证明，不能引用其安全/性能界作为本轮保证。

新native项使用实际车身与真实标线/路沿box的有向分离轴距离；平面接触模型对旧7003状态复现3次白线接触、无假阳性/假阴性，但正例很少，不能证明覆盖所有曲线/路沿/俯仰状态。方向变换用xformVec，避免大坐标float32位置相减导致轴向精度损失。异常形状或非平面几何明确报错，不当作“没有风险”。

误差包络采用旧24图物理误差的H5位置P95加半车长乘航向P95，并非统计独立的联合95%置信包络。它定义连续近线代价的尺度，不是允许越线的阈值。不改变原生压线宽度、reward、cost或终止条件。

验证判据、地图及样本量先保存于protocol/preregistration。若12张开发图未通过门槛，本轮停止该假设，不运行封存验证，也不按结果反向调整包络、权重或地图。
