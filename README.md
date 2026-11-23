# MPR-MPC 保留版

本目录开源产生下表结果的算法、配置、固定场景和实验记录；训练好的模型仅保留在本地，不随源码发布。旧实验工作目录、历史调参脚本及非选定检查点已清理；原始兄弟项目 `mpr_mpc`、原版TD-MPC2不在清理范围内。

| 测试组 | 原到达成功率 | 碰撞后仍到达回合 | 成功率：到达且全程无碰撞 | 失败回合 |
|---|---:|---:|---:|---:|
| A：Lattice | 83% | 8 | 75% | 25 |
| B：Lattice＋SAC | 92% | 11 | 81% | 19 |
| C：Lattice＋MPPI | 89% | 16 | 73% | 27 |
| D：Lattice＋SAC＋MPPI | 96% | 6 | 90% | 10 |

这是同一批100个场景上的既有实验重统计，不是新独立验证。原始400回合记录和完整配置保存在 [reports/selected_run](reports/selected_run)。

## 保留内容

- 模型：200k世界模型＋冻结世界模型后训练300k Residual SAC。
- MPPI：实测Tfast native-road tracking版本，原单高斯/AR(1)、64 samples、8 elites、4 iterations预算；无风险时保留基准动作。
- 原535维SAC输入、动作映射、Lattice生成、TD-MPC2及SAC权重均保持原样。
- 周车密度 `0.075`，事故障碍概率 `0.8`，越界容忍固定 `0.1m`。
- 测试种子 `20261127`；四组共享 [selected100.json](scenarios/selected100.json)。
- 唯一正式评测配置：[profiles/selected.yaml](profiles/selected.yaml)。旧核心兼容接口保留，未再提供历史实验启动配置。

## 模型与环境

```bash
conda activate tdmpc2
python scripts/check_environment.py
```

精确重跑需要原 WM200k＋Residual SAC300k 完整权重。权重不随源码发布；持有原权重的用户应将其放置在 `models/wm200k_rl300k.pt`。对应 SHA256 为：

```text
4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a
```

没有原权重时，可以复查已有400回合数据及统计，但不能直接重现该检查点的驾驶表现。重新训练不会保证得到完全相同的成功率。

依赖清单见 `requirements.txt`、`environment.yml`、`environment-lock.txt`。评测需要可用CUDA；不依赖任何已删除的工作树或外部TD-MPC2源码目录。

## 运行四组评测

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
bash scripts/evaluate_selected.sh
```

默认保存到 `logs/selected_evaluation/`。同一命令可以续跑；使用新目录可重新评测：

```bash
bash scripts/evaluate_selected.sh --output-dir logs/my_selected_evaluation
```

- `summary.csv`：正式成功率按“原到达且整回合无碰撞”统计。
- `episodes.csv`：`native_success`保留原到达标志；`final_success`/`strict_success`为正式新口径。
- `failures/all_failed_seeds.json`：四组的新口径失败场景种子。
- `steps/`：逐步规划诊断。

碰撞后继续行驶至原生终止，以保持原实验的控制路径和奖励定义；回报、路线完成率、成本、耗时使用实际完整回合记录。一次碰撞与任务失败重叠时只计一个失败回合。

## 重建已有结果与检查

```bash
python tools/summarize_selected.py
python -m pytest -q
bash scripts/evaluate_selected.sh --smoke-steps 2 --output-dir logs/technical_smoke
```

短测会人为截断回合，仅用于启动与接口检查，不代表驾驶性能。

## 文件布局

```text
profiles/selected.yaml       唯一保留的评测配置
models/                     权重说明及SHA256清单；权重文件仅在本地
scenarios/selected100.json   固定100场景
reports/selected_run/        原始记录、新口径汇总、失败清单及状态证据
planning/                   被选版本的原规划核心与必要兼容实现
residual_rl/                 原Residual SAC
vendor/                     独立运行所需TD-MPC2依赖
scripts/evaluate_selected.sh 正式启动入口
```

`reports/selected_run/source_*`是历史来源记录，其中旧路径仅作溯源。运行时只使用项目内的 `profiles/selected.yaml`、`models/` 和 `scenarios/`。

训练框架 `train.py`、`trainer.py` 及必要运行依赖保留；本次整合没有训练、微调或覆盖模型权重。
