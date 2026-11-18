# 本轮结果与复现入口

工作区：`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_evaluation_redesign_20261119`。
基线：`75b8a212a62f50ca76e29e37085fe1cd639ac718`。
生产目录 `../mpr_mpc_v2` 保持原样；这里没有 commit、merge 或 push。

## 查看现有结果

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_evaluation_redesign_20261119
cat reports/mppi_evaluation_redesign_20261119/MPPI_EVALUATION_REDESIGN_REPORT.md
cat reports/mppi_evaluation_redesign_20261119/verification.json
cat reports/mppi_evaluation_redesign_20261119/acceptance.json
git diff -- planning/local_mppi.py
```

`summary.csv` 是完整闭环 A/B/C，各组36回合；`h1_role_results.csv` 是单步干预之后原SAC接管，字母含义见 `protocol.json`，不可混用。原始记录保存在 `baselines/`、`closed/`、`branches/` 和 `origins/`。

## 核心开关

仅在本工作区的 LocalMPPI 配置中显式传入：

```yaml
final_selection_policy: baseline_guard
final_selection_margin: 13.2349853515625
replacement_supported: false
```

当前没有证明替换SAC具有稳定收益，因此支持标记为false。这会保留SAC动作，但仍执行完整64 samples、8 elites、4 iterations搜索；它是保护机制，不是收益改进或安全保证。不能为了增加替换率自行把支持标记改为true。

字段缺省时 `final_selection_policy=legacy`，旧动作选择完全保留，原 `config.yaml` 未修改。不要额外指定 `selection_mode`，因为原实现显式设置该字段会触发额外Elite复评，与本实验的旧默认 final2 接口不同。

## 相关回归测试

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_evaluation_redesign_20261119
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/kzb/anaconda3/envs/tdmpc2/bin/python -m pytest -q \
  tests/test_simplified_tdmpc2_mppi.py \
  tests/test_mppi_value_evaluation.py \
  tests/test_mppi_final_selection.py \
  tests/test_mppi_diagnosis.py \
  tests/test_first_action_study.py \
  reports/mppi_evaluation_redesign_20261119/test_redesign.py
```

原测试断言未删除或降低。本轮另有真实冻结模型默认路径逐位复现和独立保护探针，见 `default_before/after.npz`、日志及 `real_guard_probe.json`。

## 实验执行顺序

已实际执行：`setup.py`（预注册、历史排除及冻结margin）→ `historical.py` / `historical_summary.py`（只读冻结WM）→ `pipeline.py`（严格pilot、A36、候选36、H1共174分支、B36、C36）→ `finish.py`（单进程延迟探针、汇总、图表）→ `report.py`（哈希验证、报告和Diff）。

`h1_analysis.py` 在所有H1标签完成后生成预注册决策；`core_ready.json` 记录对应实现与测试，C组在此之后运行。A/B从未修改的生产目录导入，C从本工作区导入。A采集时的执行器源码另存 `source_versions/closed_baseline.py`；后续仅增加LocalMPPI计时包装，不改控制或RNG。

`audit_candidate_actions.py` 修正了汇总表动作列的循环变量对齐错误，核对全部174条实际首动作；原表及修复前源码保留在source_versions。此修正不改变真实执行、标签或决策。修复已完成，勿在原结果目录重复运行一次性修复脚本。

实验工具使用独占创建（`open('x')` / `mkdir(exist_ok=False)`），不会覆写已有结果。不要在此非空目录重新启动pipeline；独立复现须使用新的隔离工作区和空结果目录，保留本次预注册场景、种子、checkpoint及冻结margin，记录新的完整manifest。

本轮checkpoint为WM200k+RL300k、总500k的原权重，SHA256见 `protocol.json`。所有逐回合配置、初始状态哈希和原生终止记录均已保存。新图H1用于预注册机制选择，同图闭环属于筛选验证，不能再次称为完全未触碰的最终验证集。

`code_changes.patch` 包含核心修改、新实验工具、报告及预注册元数据；体积较大的CSV/NPZ原始数据另以本目录和 `artifact_manifest.json` 保留，不嵌入代码补丁。
