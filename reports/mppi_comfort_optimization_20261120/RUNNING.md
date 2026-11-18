# Comfort MPPI 本轮工作区与复现

工作区：`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_comfort_20261120`。
基线：`a9e84b55f4d373cad11979e70eceed149db59a6d`。
默认配置、主目录、WM/SAC checkpoint保持原样，未自动commit或push。

## 查看本轮结果

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_comfort_20261120
cat reports/mppi_comfort_optimization_20261120/COMFORT_MPPI_FINAL_REPORT.md
cat reports/mppi_comfort_optimization_20261120/acceptance.json
cat reports/mppi_comfort_optimization_20261120/verification.json
git diff -- planning/local_mppi.py planning/coordinator.py
```

报告中的舒适性结果均由真实车辆速度、航向和实际动作计算，未使用在线预测Cost冒充真实改善。每回合保留raw及SG滤波信号NPZ、逐步CSV和原生终止结果。

## 显式配置接口

本轮已冻结的独立配置为 `comfort_config.json`，由实验执行器叠加到 `cfg.mpr_mpc.mppi`，同时设置 `restore_mppi_runtime=false` 防止旧checkpoint运行配置覆盖实验选项。

```python
import json
from omegaconf import OmegaConf
options = json.load(open("reports/mppi_comfort_20261120/comfort_config.json"))
for key, value in options.items():
    cfg.mpr_mpc.mppi[key] = OmegaConf.create(value) if isinstance(value, dict) else value
```

其中 `objective_mode=comfort` 和 `comfort.enabled=true` 必须显式设置；关闭模式使用 `objective_mode=legacy`。配置缺少开发集校准尺度时会报错，不会偷偷采用任意物理阈值。新模式仅优化转向rate/jerk，沿用64/8/4及原Gaussian/AR1采样和Elite更新；所有候选油门严格等于当前SAC参考序列，保持必要制动决策。独立配置尚未通过验收时不得接入默认评测或部署。

`ComfortTrajectoryEvaluator` 的H1参考一致性条件不是道路或碰撞安全保证。没有重新加入Recovery/Emergency/TTC规则。WM仅继续服务原SAC特征，不参与舒适性候选排序。

## 测试

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_comfort_20261120
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -m pytest -q \
  tests/test_simplified_tdmpc2_mppi.py \
  tests/test_mppi_value_evaluation.py \
  tests/test_mppi_final_selection.py \
  tests/test_mppi_diagnosis.py \
  tests/test_first_action_study.py \
  tests/test_comfort_mppi.py \
  reports/mppi_evaluation_redesign_20261119/test_redesign.py \
  reports/mppi_comfort_optimization_20261120/test_comfort_metrics.py
```

另有真实冻结WM的legacy动作/Mean/Std/Q/RNG逐位对照，见 `legacy_parity.json`。

## 已执行的实验顺序

1. 读取上一轮6份报告/数据及36条完整Baseline轨迹，审计实际舒适性与未拟合的运动学模型。
2. 扫描历史manifest及日志，预注册12张新地图与3个规划种子；开发数据确定唯一舒适性配置及因果低通参数，没有网格搜索。
3. 两张旧开发地图各运行A/B/C共6次100步短测；原A与旧轨迹前100步动作、观测和奖励逐位一致。
4. 冻结核心文件哈希后，运行12图×3seed×3方法=108次原生完整闭环回合；每周期重新规划，只执行第一动作。
5. 所有共享负载采集完成后，前两张预定测试地图各方法单进程跑105个实际控制周期，前5步暖机，余下100步用于延迟测量。这些重复只用于性能测量，不增加驾驶统计样本量。
6. 场景聚类bootstrap2000次、LOMO、逐seed统计、图表、报告、Diff与哈希验证。

A为原SAC；B为因果转向低通，alpha由开发数据导数特征时间尺度确定；C为Comfort MPPI。B/C均不修改SAC油门，但后续SAC会反馈真实状态，不能宣称整回合速度或碰撞必然不变。

工具独占创建输出，不应在现有结果目录直接重新启动pipeline。独立复现应使用新的隔离工作区和空结果目录，保留本轮预注册参数、地图、种子及checkpoint，并生成新的manifest。旧开发轨迹原始NPZ位于上轮实验工作区，checkpoint位于原训练目录，路径及SHA见protocol/manifest。

代码补丁包含核心修改、实验工具、测试、配置和报告；较大的CSV/NPZ原始数据保存在实验目录，不内嵌代码Diff。现有.gitignore忽略NPZ/运行日志及权重，日后Git上传不会自动包含它们。
