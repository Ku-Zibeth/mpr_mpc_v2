# 查看、复核与复现

工作区：`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_h1_counterfactual_20261118`。
分支：`experiment/mppi-h1-counterfactual-20261118`。
所有改动均在独立报告目录，主目录没有合并、commit或push。

## 查看当前进度/结果

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_h1_counterfactual_20261118
tail -n 12 reports/mppi_h1_counterfactual_20261118/pipeline.log
cat reports/mppi_h1_counterfactual_20261118/paired_statistics.json
```

最终报告：FINAL_REPORT.md；状态核验：STATE_RECONSTRUCTION_AUDIT.md、state_reconstruction_checks.csv；完整逐步数据：counterfactual_steps.csv；分支NPZ在branches/；候选及去重在origins/*/catalog.json/npz。原B trace来自之前root_cause工作区的未跟踪NPZ；manifest记录绝对路径及SHA。仓库通用ignore可能不包含这些NPZ，不能把git上传等同于原始数据备份。

## 相关测试（不启动环境或训练）

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_h1_counterfactual_20261118
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -m pytest -q \
  tests/test_simplified_tdmpc2_mppi.py \
  tests/test_mppi_value_evaluation.py \
  tests/test_mppi_final_selection.py \
  tests/test_first_action_study.py \
  reports/mppi_h1_counterfactual_20261118/test_protocol.py
```

42项，其中9项新增。最初pytest目录导入失败日志、初期41项通过日志及后补地图统计回归的日志都保留，不修改原测试。

## 实验执行顺序

1. setup.py：冻结protocol.json及输入哈希；不根据结果改地图或候选规则。
2. prepare.py --scene 130719 --step 50：恢复首个状态、生成真实C/D及末轮Gaussian候选。
3. branch.py --scene 130719 --step 50 --candidate A0：完整复现原B余下轨迹。
4. pipeline.py：检查Phase1和测试通过；先A1+三个Gaussian单状态分支，才放行其余状态；每状态先验证A0整段，再运行其他动作。最多4个独立进程。
5. finish.py：等待全部成功收集后，按audit_data.py→analyze.py→plots.py→report.py顺序生成结果。

输出采用独占创建，重复复现实验请用新的报告目录，不在本已完成目录重复运行整套流程。复用相同地图只能称为复现，不称为新独立验证。原有失败前诊断记录保留于diagnostic_attempts；没有进行失败样本替换或按结果重选地图。

源动作仅干预一次0.1s，此后所有控制为原冻结Lattice+SAC；严禁将其重写为H步固定动作重放或持续MPPI。当前没有匹配H1/SAC接管的独立合格价值预测器，正式Ranking Accuracy必须保持NOT ESTIMABLE。
