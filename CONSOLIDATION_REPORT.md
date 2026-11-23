# 保留版整合记录（2026-11-23）

保留的是原表格对应的WM200k＋RL300k完整模型、Tfast native-road MPPI及实测配置。权重未修改，SHA256为4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a。当前项目路径是`/home/kzb/kzb_code/tdmpc2/mpr_mpc_v2`。

正式评测成功定义已统一为“到达终点且整回合无碰撞”。保留原生到达标志`native_success`，正式`final_success`/`strict_success`使用新口径；环境终止和完整回合奖励、成本、执行动作不变。四组原始400回合重统计仍为75/81/73/90个成功、25/19/27/10个失败。

完整模型在`models/wm200k_rl300k.pt`，使用Git LFS；唯一正式评测配置在`profiles/selected.yaml`，100场景在`scenarios/selected100.json`。周车密度0.075、事故障碍概率0.8、越界容忍固定0.1m、种子20261127。启动入口为`bash scripts/evaluate_selected.sh`，输出至`logs/selected_evaluation/`，支持续跑。

清理了7个实验工作目录及7个已合并的本地实验分支、主目录历史调参报告/脚本、无生产引用的旧模块及非选定检查点。原始兄弟项目`mpr_mpc`和原版TD-MPC2未修改。依赖源码和选定模型所需兼容接口保留。未提交的旧源码差异在本地`.git/consolidation-backup-20261123`留有小型备份，不会上传。

保留的测试覆盖跟踪MPPI、native几何、绝对越界容忍、配对重置/续跑、Residual SAC、535维输入、WM Cost验证及新成功口径。旧版本专用的过时Cost评分/多模式测试随旧版本清理；有效的Cost验证与535维断言原样迁入独立测试，未为了通过而更改规划核心。

验证结果：106项单元/回归测试通过；环境依赖检查通过；原400回合统计复现；整合前后四组短测均通过；删除实验目录后的新进程12条动作与原记录完全相同，误差为0。源码控制核心哈希及模型哈希校验通过。没有重新训练，也没有启动正式400回合重跑。

未自动commit或push。完整同步指令见`GITHUB_SYNC.md`。
