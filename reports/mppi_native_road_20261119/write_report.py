"""Fill report from completed paired data, with no optimistic assumption."""
from study import *
import pandas as pd

def md(f):
    return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |',*['| '+' | '.join(map(str,x))+' |' for x in f.itertuples(index=False,name=None)]])

def main():
    assert load(R/'study_complete.json')['holdout_episodes']==96
    p=load(R/'protocol.json');dev=pd.read_csv(R/'development_all_summary.csv').set_index('configuration');hold=pd.read_csv(R/'holdout_all_summary.csv').set_index('configuration')
    statistics=load(R/'holdout_map_cluster_statistics.json')['results']['C'];devstats=load(R/'development_map_cluster_statistics.json')['results']['Tconditional']
    actual=hold.loc['C']-hold.loc['A'];return_ci=statistics['discounted_return']['ci95']
    majority=sum(v>0 for v in statistics['discounted_return']['maps'].values());seed_summary=load(R/'holdout_seed_summaries.json')
    both=all(v['C']['discounted_return']>v['A']['discounted_return'] for v in seed_summary.values())
    clock=pd.read_csv(R/'timing_breakdown.csv');isolated=clock[(clock.split=='latency')&(clock.phase=='mpr/total_planning_ms')].set_index('configuration')
    performance=load(R/'performance_integration.json');fast_passed=load(R/'fast_full_trace_parity.json')['all_actions_observations_states_rewards_costs_termination_exact']
    fast_latency=float(isolated.loc['Tfast','p95']) if fast_passed else float(isolated.loc['Tconditional','p95'])
    criteria=dict(independent_discounted_return_CI_lower_positive=return_ci[0]>0,success_improved=actual.success>0,crash_not_increased=actual.crash<=0,out_of_road_not_increased=actual.out_of_road<=0,
        route_completion_not_lower=actual.route_completion>=0,majority_maps_gain=majority>12,both_traffic_seeds_return_improve=both,full_planning_P95_under100ms=fast_latency<100.,contemporary_lowpass_comparator_available=False)
    criteria={k:bool(v) for k,v in criteria.items()}
    accepted=all(criteria.values());dump(R/'deployment_gate.json',dict(accepted=accepted,criteria=criteria,independent_maps=24,paired_episodes=48,strict_gate=p['joint_acceptance'],default_enabled=False))
    def table(frame):
        rows=[]
        for name,q in frame.iterrows():rows.append(dict(method=name,Success=f'{q.success*100:.2f}%',Crash=f'{q.crash*100:.2f}%',OOR=f'{q.out_of_road*100:.2f}%',Route=f'{q.route_completion*100:.2f}%',RTG=f'{q.discounted_return:.3f}',Episode_Return=f'{q.episode_return:.3f}',Safety_Cost=f'{q.safety_cost:.3f}'))
        return md(pd.DataFrame(rows))
    conclusion=f'独立Success由{100*hold.loc["A","success"]:.2f}%提高到{100*hold.loc["C","success"]:.2f}%，OOR由{100*hold.loc["A","out_of_road"]:.2f}%降到{100*hold.loc["C","out_of_road"]:.2f}%；实际折扣回报区间是否支持正改善：{return_ci[0]>0}'
    report=f'''# 原生道路语义与条件MPPI：2026-11-19

**{conclusion}；是否达到全部启用门槛：{accepted}。** 原WM/SAC、535输入、Lattice、Reward/Cost/gamma/done、checkpoint和默认控制器未改，没有训练、commit或push。不能把成功率改善等同于安全、收益与实时性均已达标。

## 结论和实际数据

独立24张预先封存地图、每图两组真实交通seed、A/C严格配对，共48对/96回合。C−A实际折扣回报均值`{actual.discounted_return:+.6f}`，地图聚类bootstrap95%CI`{return_ci}`；`{majority}/24`张图平均正收益；两组交通seed均正向：`{both}`。Success差`{100*actual.success:+.3f}`个百分点，Crash差`{100*actual.crash:+.3f}`，OOR差`{100*actual.out_of_road:+.3f}`。事故不恶化的点估计不构成普遍安全保证。

{table(hold)}

验收逐项结果：`{criteria}`。完整规划实时性使用单进程真实闭环测量，不拿六进程采集的资源竞争耗时冒充独占benchmark。默认启用保持false。

## 这一周证据如何约束本次修改

上周终端Q、独立Value/Ranking Head和Mean/Elite迭代没有可靠任务收益；上轮校准物理模型改善H5预测，但72回合仍未提升驾驶。这表明只降低预测误差或毫米级跟踪误差不足。本轮没有再训练评分头或增加Recovery/Emergency，而是验证一个具体缺项：原native实线接触不等于全图road union覆盖。

新增`planning/native_road.py`读取实际Bullet白/黄实线和路沿box，不按所有lane边界虚构标线；使用真实shape/net transform、车身尺寸和有向分离轴距离。7003个旧实际状态复现3个白线接触、0误报/漏报，但正例稀少，不能证明未来动态预测或所有路沿/立交场景安全。变换方向用xformVec避免大坐标float32位置相减丢精度。

旧图物理误差包络约0.254665m来自H5位置P95加半车长×航向P95，不是联合95%安全置信界；它只定义连续近线代价的尺度，没有放宽真实线宽/碰撞/终止条件。评分加同样的连续系数20，64 proposals+baseline、8elite、4轮、std/rho/temperature、当前SAC油门都保留。无模型风险周期执行精确SAC；活跃搜索保持完整预算。

## 新开发图与消融

预先固定12张图：`{p['development_scenes']}`。所有不利地图保留，不能因几何修复或原生OOR而删除。开发48回合：A、Ttrack、Tnative、Tconditional各12。不是反复使用旧559/1714，也没有用封存图选权重。开发条件方案真实回报差`{devstats['discounted_return']['mean']:+.6f}`，95%CI`{devstats['discounted_return']['ci95']}`。

{table(dev)}

Tnative−Ttrack的RTG差约+0.063，两者仍未过门槛；不能把组合收益全部归功于添加实线项。开发数据更支持限制不必要的模型动作修正。条件优化略微增加了实际一步路径误差，仍可提高真实回报，再次说明跟踪误差不是任务收益的替代指标。

跳过搜索也改变该优化器内部RNG序列推进；保持采样分布和episode seed，不声称始终搜索与条件搜索逐周期噪声相同。独立验证检验的是条件算法整体。见`MECHANISM_AND_LIMITS.md`，不夸大单个子模块的因果贡献。

## 配对、冻结与统计

起点obs/车辆/交通/RNG摘要和首个535/SAC残差/轨迹指纹完全匹配。在首次实际动作干预前，观测和动作前缀完全一致，第一次干预时SAC原建议也一致；不是只核对reset。每周期真实Lattice→SAC→可选MPPI→首动作→环境更新到原生终止，非开环H步重放。

map seed保留；TrafficManager原生reset前单独seed(scene*1009+repeat)，两个repeat真实交通差异见`independent_traffic_diversity.csv`。统计以24张地图聚类，两个seed整组重采样2000次、固定seed20261119，并保存LOMO。96回合不是96张独立地图；开发结果和独立结果分开。验证期间源码/配置哈希冻结，没有根据结果再调参。

## 性能与等价优化

未优化条件方案的单进程完整轨迹：A规划P50/P95/P99=`{isolated.loc['A',['p50','p95','p99']].to_list()}`ms；C=`{isolated.loc['Tconditional',['p50','p95','p99']].to_list()}`ms。重复回合完整动作、观测、状态、Reward/Cost/done与原开发轨迹一致，4回合不算新独立样本。活跃搜索的评分开销显著，不能只报告未触发搜索的快周期。

验证期间另外准备了未接入冻结控制的数学等价broad phase：初始距离超过自车最大行程+车身半径+障碍CV最大行程+障碍半径的物体，在同一模型内对原overlap成本严格为0。仅剔除这些零贡献物体，保留所有可能接触检查。3项针对测试、8个真实当前状态×65候选及完整MPPI动作/mean/std逐元素一致，未用真实未来收益选择动作。冻结验证结束后才接入显式加速开关，完整轨迹等价和实测见`performance_integration.json`。

## 文件、版本和测试

基线main `{p['base_commit']}`，工作区`{ROOT}`、分支`experiment/mppi-native-road-20261119`。使用历史共同权重WM200k+RL300k总500k，非WM500k+RL500k；SHA256 `{p['checkpoint_sha256']}`。原main和上轮tracking工作区保留。

新增native接触几何、tracking evaluator的可选连续代价/上下文、LocalMPPI的可选条件调用和诊断；不改原采样/elite/update，未增加控制器或评分网络。原legacy/comfort/上轮tracking关闭新功能时，12条件92输出数组完全一致。最终相关59测试通过，保存日志；不能把通过测试当作驾驶提升。

所有步骤/实际执行动作、Reward、Cost、原生终止、车辆参数、完整NPZ和配置保存在development/holdout/latency，旧结果不覆盖。现有.gitignore仍忽略NPZ/log/模型；完整原始记录保留本地并有最终SHA索引。

## 当前决策

本轮已经获得独立Success/OOR改善证据，但真实折扣回报的可信正向改善、事故鲁棒性和实时性尚未全部过关。保留实验配置，不设为默认；不能再将这24张图称作未来新版本的未见验证集。下一优先级是三个新增碰撞的近端风险表达与静态对象几何问题，需新协议和新场景验证，不继续盲调权重或训练评分头。等价加速可以保留，但不代替安全与收益验证。

资料核对：[MetaDrive车辆接触](https://github.com/metadriverse/metadrive/blob/main/metadrive/component/vehicle/base_vehicle.py)、[真实标线构造](https://github.com/metadriverse/metadrive/blob/main/metadrive/component/block/base_block.py)、[Nav2车身轮廓与路径critics](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/controller_plugins/mppi_controller/configuring_mppic/)、[MBPO模型偏差](https://arxiv.org/abs/1906.08253)、[RMPPI反馈建模](https://arxiv.org/abs/2102.09027)。参考思想不等于实现其算法或继承其理论保证。本地MetaDrive源码SHA记录于`native_source_sha256.json`。
'''
    tradeoff=load(R/'success_return_tradeoff.json')
    report+=f'''\n## 配对安全细节与已完成加速\n\n48对回合中6次由失败转成功、0次由成功转失败，均为避免原生OOR；来自5张地图，其中一图两个seed均改善。总体碰撞8/48对8/48，但避免3次并新增另3次。不能将总体均值相同写成“没有新增碰撞”。Crash差的地图bootstrap95%CI为`{statistics['crash']['ci95']}`；需要保留安全不确定性。\n\n安全成本每回合由2.14549增加到2.32414，不能因成功率提高而隐去。成功改善与Return不是同一标签：救回的6回合中1回合折扣回报反而下降；完整分组见success_return_tradeoff.json。此分解只是描述，不修改Reward或重新选择方案。\n\n加速在冻结96回合完成且核对原源码SHA之后才接入显式exact_geometry开关；新增helper在验证期间未被调用，冻结源码保存在frozen_validation_sources/。除零贡献障碍物culling外，静态标线SAT通过复用自/交叉点积避免重复巨型axis张量；保持原浮点点积而非把近似正交当成精确0/1。8个真实当前状态的65候选全部输出、完整优化动作/均值/方差精确一致，另有2条完整闭环轨迹与原版本逐元素一致。\n\n优化后Tfast单进程全闭环P50/P95/P99=`{isolated.loc['Tfast',['p50','p95','p99']].to_list()}`ms；P95从{float(isolated.loc['Tconditional','p95']):.3f}降至{fast_latency:.3f}ms。依然高于100ms；这只是两张预定开发图的同轨迹测量，不是独立地图实时保证。新增2个性能重复回合不算新独立样本，神经权重和候选排序均不变。\n'''
    section=R/'collision_audit_report_section.md'
    if section.exists():report+=section.read_text()
    with (R/'MPPI_NATIVE_ROAD_REPORT_20261119.md').open('x') as f:f.write(report)
    print('FINAL_CONCLUSION',conclusion,criteria,flush=True)
if __name__=='__main__':main()
