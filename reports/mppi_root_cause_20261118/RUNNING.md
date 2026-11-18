# 查看与复现

本轮目录：`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_root_cause_20261118`，分支 `experiment/mppi-root-cause-20261118`。主目录 `mpr_mpc_v2` 未改动；直接在那里 git add 不会提交此 worktree 的未提交修改。本轮没有自动 commit、合并或 push。

查看结果：

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_root_cause_20261118
cat reports/mppi_root_cause_20261118/summary.csv
cat reports/mppi_root_cause_20261118/paired_statistics.json
cat reports/mppi_root_cause_20261118/verification.json
git diff --stat
```

相关回归测试（不启动训练或环境评测）：

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_root_cause_20261118
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -m pytest -q \
  tests/test_simplified_tdmpc2_mppi.py \
  tests/test_mppi_value_evaluation.py \
  tests/test_mppi_final_selection.py \
  tests/test_mppi_diagnosis.py \
  tests/test_first_action_study.py \
  tests/test_mppi_failure_attribution.py \
  reports/mppi_root_cause_20261118/test_alignment_fix.py
```

对应46项。额外的旧 `test_native_mppi_cost.py` 已在本轮和干净main各执行一次，均为15通过/20失败；完整日志已保存，没有修改旧测试。

## 实验脚本与顺序

1. `study_io.py`：只加载已冻结的WM200k+RL300k checkpoint；B/C读干净main，D读隔离worktree。无训练入口。
2. `batch.py` / `closed_loop.py`：两进程收集8图×2种子×B/C=32回合。每周期原生重规划，不重放预存H步动作。
3. `test_alignment_fix.py` + `speed_preflight.py`：测试及真实同状态535/首动作保持检查。
4. `batch_D.py`：仅一个预注册速度修正，16回合；等待B/C完成和D_preflight通过。
5. `run_probes.py` / `latency_probe.py`：等待D完成，单进程按地图/方法依次重建同一物理状态，5暖机+50计时，共300调用。记录各轮采样/评价/更新及完整规划；不执行探针输出动作。
6. `historical_analysis.py`：复用571条可比较历史原SAC接管记录，明确排除另外5条不同接管/重复重放；不运行环境。
7. `replay_adapter.py`：36个原实际控制器路径快照重放；`snapshot_comparison.py`比较原Lattice/SAC/MPPI动作。
8. `summarize.py`：配对初始条件、原生done与RTG校验；以8张完整地图聚类bootstrap2000次；分离共享负载与独立探针耗时。
9. `finalize.py`：验证原checkpoint/配置/历史文件未变，生成报告及代码Diff。

这些脚本对输出采用独占创建：不要在当前已完成目录重新启动整套任务。需要复现实验时，应复制源码和冻结协议到新的报告目录，再按上述顺序运行；使用同一批地图只能称为复现，不能称为新独立验证。`history_review.json`及原 `protocol.json`/`D_protocol.json`保留本轮预注册证据，不根据已看到的结果更换地图或D方案。

核心修正需显式设置 `mpr_mpc.mppi.cartesian_virtual_speed=true`；默认不设置即为旧行为。D不是已通过部署验收的方案。

历史NPZ/模型等未跟踪二进制位于原实验worktree，脚本按manifest绝对路径读取。新trace/snapshot NPZ也受仓库通用ignore规则影响，仍保存在磁盘；`git add -A`不一定包含这些原始数据。代码补丁只打包本轮源码、协议和文档，不打包旧模型或实验二进制。
