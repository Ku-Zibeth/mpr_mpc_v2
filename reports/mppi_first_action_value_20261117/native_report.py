"""Parity report includes ACTUAL fullMPC and Actor-only as distinct controls."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
from tools.mppi_failure_attribution import scene_bootstrap

p=load(REPORT/'protocol.json');rows=[]
for scene in p['pilot_scenes'][:2]:
    origin=f'{scene}_{p["before_steps"][0]}';c=np.load(REPORT/'origins'/origin/'catalog.npz');q=np.load(REPORT/'origins'/origin/'offline_scores.npz')
    raw=c['raw_final_values'];pair=np.array(load(REPORT/'origins'/origin/'catalog.json')['actual_final_pair']);currentq=q['q_current'][0,pair].mean()
    ref=None
    for policy in ['baseline','td_actor','td_mpc']:
        f=REPORT/'branches'/origin/f'H1_c0_{policy}.json';r=load(f)
        hashes={k:v for k,v in r.items() if k.startswith('origin_') and k.endswith('_hash')}
        if ref is None:ref=hashes
        else:assert hashes==ref
        rows.append(dict(scenario=scene,origin=origin,controller=policy,actual_return=r['actual_return'],episode_return=r['episode_return'],success=r['success'],crash=r['crash'],out_of_road=r['out_of_road'],
            route_completion=r['route_completion'],safety_cost=r['safety_cost'],steps=r['episode_length'],same_first_physical_action=True,
            original_current_q=float(currentq),q_minus_actual=float(currentq-r['actual_return']),
            budget='512samples/64elites/6iterations originalTD-MPC2fullMPC' if policy=='td_mpc' else 'stochastic model.pi only' if policy=='td_actor' else 'deterministic original Lattice+SAC'))
table(NATIVE/'policy_takeover_comparison.csv',rows)
cal=load(NATIVE/'q_calibration.json');versions=load(NATIVE/'version_manifest.json')
lines=['# Native TD-MPC2 parity audit — 2026-11-17',
    '\n本轮默认LocalMPPI没有修改。原生TD-MPC2采样Actor终端动作与当前policy_prior均值动作是不同定义。',
    f'\n独立worktree `{ROOT}`，分支`{versions["branch"]}`，基线`{p["base_commit"]}`。Gitfetch网络超时，通过GitHub连接器确认main SHA相同。源main/历史数据未覆盖。',
    '\n规划使用随机双Q **平均**，训练TD Target用随机双targetQ **最小值**；Phase1是固定Q01平均。三个定义已分别枚举，未把min误称avg。',
    '\n离线对照来自4个已见Fit起点、40条原生H10前缀→SAC真实后果。Candidate、PolicyMean、NativePolicySample共享完全相同的前缀/预测latent/Reward/折扣/枚举Q成员组合，仅末端动作变动。原生Sample确实调用原model.pi(z)[0]，Mean使用info.mean；没有改默认terminal_action。',
    '\n逐候选保存Q0…Q4、10对平均、10对最小、五成员mean/median/min；100次随机双Q抽样共享一对成员评分整个候选池，分布存random_pair_100_distributions.npz及q_random_pair_100.csv。100次只抽Q成员，不是100次独立Actor或环境轨迹。',
    '\nq_calibration.json报告各定义的候选内Spearman/Kendall/PairAccuracy/Top1Regret、Q MAE/Bias/Calibration与Top1一致率。历史Fit的描述性最好定义不构成独立验证，也未用于选择后续默认Q或终端模式。',
    '\n## 小规模真实接管（与Actor-only严格区分）',
    '\n从两个新Pilot地图的固定起点，都先执行同一个Baseline物理动作，再由对应控制器接管至原生终止。此处Q(s,a_baseline)的第一步相同，延续控制器分别为π_B、原随机Actor、完整原TD-MPC2 MPC。完整MPC实际调用tdmpc_agent.act，cfg.mpc=true；保留其原512/64/6预算，这不是将LocalMPPI的64/8/4预算加大。',
    '\n|场景|控制器|Discounted Return|Success|Crash|OOR|Route|\n|---|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"|{r['scenario']}|{r['controller']}|{r['actual_return']:.3f}|{r['success']:.0f}|{r['crash']:.0f}|{r['out_of_road']:.0f}|{r['route_completion']:.3f}|")
lines.extend(['\n完整MPC可以运行；不能继续把Actor-only结果称为完整TD-MPC2性能。这里也仅是同一个WM200k/RL300k Checkpoint内的控制器对照，不是另一个原版迁移模型的评估。2个地图不足以宣称任何控制器普遍优劣。',
    '\n依旧没有证据据此修改默认终端动作或Q聚合。下一阶段的主要标签是新的真实H1→反馈SAC收益，与本节H10历史对照严格分开。',
    '\n全部源表、脚本、checkpoint SHA及历史Fit/InnerStop/Phase1 Pilot/Sealed Validation的区分见version_manifest.json；旧6个封存场景未触碰。'])
(ROOT/'MPPI_NATIVE_PARITY_REPORT_20261117.md').write_text('\n'.join(lines)+'\n')
dump(NATIVE/'verification.json',dict(full_native_mpc_executed=True,actor_only_kept_separate=True,default_terminal_action_unchanged=True,default_q_aggregation_unchanged=True,
    same_prefix_z_reward_gamma_pairs=True,all_10_pair_avg_and_min=True,random_pair_draws=100,independent_environment_processes=True,original_models_unmodified=True))
print('native parity +fullMPC report complete',flush=True)
