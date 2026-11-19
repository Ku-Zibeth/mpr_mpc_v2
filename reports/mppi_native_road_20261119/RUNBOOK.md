# 本轮工作区与复现

实际代码工作区：`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_native_road_20261119`
分支：`experiment/mppi-native-road-20261119`。原main和上轮tracking工作区保留，未commit/push。

`protocol.json` / `preregistration.json` 固定12张开发图、三个MPPI对照和门槛；`frozen_before_validation.json`固定一次性的24图×2交通seed验证。读取`research_notes.md`和`MECHANISM_AND_LIMITS.md`了解证据边界。测试通过不等于驾驶提升，开发CI不等于独立验证。

所有完整回合在`development/`、`holdout/`；`latency/`为同一开发轨迹的单进程完整重现，不计作新独立地图。`.npz`和`.log`按既有规则忽略但保留本地，不上传模型。

以下仅重现本轮已注册开发地图中的一组对照，写入新的结果目录，不覆盖本轮实验：

```bash
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2_native_road_20261119
set -o pipefail
out="$PWD/logs/benchmark/native_road_reproduction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
for method in A Tconditional Tfast; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
  /home/kzb/anaconda3/envs/tdmpc2/bin/python -u \
    reports/mppi_native_road_20261119/closed.py \
    --split development --scene 600009 --seed 1 --method "$method" \
    --output-root "$out" \
    2>&1 | tee "$out/$method.log" || break
done
```

不重新执行已完成的`assess.py`/`validate.py`写入相同目录；它们的独占创建保护用于防止覆盖历史或反复看验证结果选参。另起实验必须创建新的结果目录和协议。

本轮未改变W&B、训练、checkpoint和默认规划器。它使用历史共同基准的WM200k+RL300k权重，总500k；不是WM500k+RL500k权重。最终是否满足启用门槛以报告和`verification.json`为准，不能因单个正收益回合将其设为默认。

Tfast为数学等价加速开关：8个真实状态评分/优化输出及2条完整闭环轨迹精确一致。独立96回合在原冻结Tconditional上完成，最终报告分别标记驾驶验证和加速测量，不能把2条性能重复轨迹算成独立样本。
