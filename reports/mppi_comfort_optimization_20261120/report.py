"""Final evidence report and complete reviewable source diff; no Git publication."""
from datetime import datetime
import ast
import pandas as pd
from comfort_study import *

def build_patch():
    result=subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)
    new=[ROOT/'planning/comfort_evaluator.py',ROOT/'tests/test_comfort_mppi.py']
    new+=sorted(f for f in R.rglob('*') if f.is_file() and '__pycache__' not in str(f) and
        (f.suffix in ['.py','.md'] or f.name in ['protocol.json','comfort_config.json','experiment_manifest.json','acceptance.json','verification.json']))
    for f in new:
        diff=subprocess.run(['git','diff','--no-index','--','/dev/null',str(f.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True)
        assert diff.returncode==1;result+=diff.stdout
    (R/'code_changes.patch').write_text(result)
    subprocess.run(['git','apply','--reverse','--check',str(R/'code_changes.patch')],cwd=ROOT,check=True)

def main():
    p=load(R/'protocol.json');manifest=load(R/'experiment_manifest.json');frozen=load(R/'core_frozen.json')
    assert load(R/'closed_complete.json')['episodes']==108 and load(R/'legacy_parity.json')['passed']
    assert '59 passed' in (R/'tests_final_v2.log').read_text()
    assert all(file_sha(ROOT/name)==sha for name,sha in manifest['protected_sources'].items())
    assert file_sha(ROOT/'config.yaml')==manifest['original_config_sha256']
    assert all(file_sha(ROOT/name)==sha for name,sha in frozen['files'].items())
    assert file_sha(R/'protocol.json')==manifest['protocol_sha256']==frozen['protocol_sha256']
    assert file_sha(R/'comfort_config.json')==manifest['comfort_config_sha256']==frozen['comfort_config_sha256']
    assert file_sha(Path(p['checkpoint']))==p['checkpoint_sha256']==manifest['checkpoint_sha256']
    before=ast.parse(subprocess.check_output(['git','show','HEAD:planning/local_mppi.py'],cwd=ROOT,text=True))
    after=ast.parse((ROOT/'planning/local_mppi.py').read_text())
    unchanged=['reset','initial_mean','_candidate_batch','_weights','_elite_weights','_update_distribution']
    def method(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    assert all(ast.dump(method(before,n),include_attributes=False)==ast.dump(method(after,n),include_attributes=False) for n in unchanged)
    for row in manifest['prior_required_inputs_read']+manifest['baseline_traces']:assert file_sha(Path(row['path']))==row['sha256']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=MAIN,text=True)
    stats=load(R/'paired_statistics.json')['comparisons'];accept=load(R/'acceptance.json')
    episodes=pd.read_csv(R/'paired_closed_loop_episodes.csv');summary=pd.read_csv(R/'summary.csv').set_index('configuration')
    times=pd.read_csv(R/'latency.csv');isolated=times[times.scope=='isolated_single_worker_warm']
    for scene in p['scenes']:
        for seed in p['seeds']:
            initial=[]
            for group in ['A','B','C']:
                out=R/'episodes'/f'{group}_{scene}_r{seed}';d=load(out/'episode.json');assert d['native_done']
                initial.append(d['initial_hashes']);trace=np.load(out/'trace.npz');step=pd.read_csv(out/'steps.csv')
                assert discounted(trace['rewards'])==d['discounted_return']
                assert np.array_equal(trace['actions'][:,1],step.sac_throttle.to_numpy(dtype=np.float32))
                if group=='C':assert (step['mppi/wm_evaluated_candidates']==0).all()
            assert initial[0]==initial[1]==initial[2]
    dump(R/'verification.json',dict(base_commit=p['base_commit'],current_main_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=MAIN,text=True).strip(),
        actual_completion_time=datetime.now().astimezone().isoformat(),date_in_folder_is_task_identifier=True,
        core_frozen_hashes=frozen['files'],original_config_unchanged=True,original_checkpoint_sha256=p['checkpoint_sha256'],checkpoint_unchanged=True,
        WM_SAC535_action_mapping_Lattice_reward_cost_done_unchanged=True,legacy_real_model_bitwise_parity=True,
        prior_required_files_and36_baseline_traces_rehashed=True,tests_passed=59,development_smoke_episodes=6,independent_closed_episodes=108,
        independent_maps=12,paired_planning_seeds=[1,2,3],same_traffic_within_map=True,initial_pairings=36,road_geometries_unique=True,
        sampling_budget=[64,8,4],comfort_candidate_WM_Q_calls=0,executed_throttle_equals_SAC=True,no_test_retuning=True,
        native_sampling_update_methods_AST_unchanged=unchanged,
        actual_Bullet_wheel_angle_mapping_verified=True,
        model_physical_prediction_horizon=1,unvalidated_longitudinal_lateral_jerk_objective_disabled=True,
        protection_not_comfort_gain='Comfort scores are predictions; all reported improvements use actual closed-loop states',
        networks_trained=0,commit=False,push=False,default_controller_changed=False))
    def gain(comparison,key,percent=False):
        x=stats[comparison][key];scale=100 if percent else 1;unit='%' if percent else ''
        return f"{scale*x['mean']:+.3f}{unit}，地图95%CI [{scale*x['ci95'][0]:+.3f}, {scale*x['ci95'][1]:+.3f}]{unit}"
    comfort='| 每组36回合 | Comfort Index | 转向rate RMS rad/s | 纵向jerk RMS m/s³ | 横向jerk RMS m/s³ | 横向加速度RMS m/s² |\n|---|---:|---:|---:|---:|---:|\n'
    driving='| 配置 | Success | Crash | OOR | Route | Timeout | 折扣回报 | 平均速度m/s | 实际动作替换率 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    latency='| 单进程，前5步暖机后 | LocalMPPI P95 | 完整Planner P50 / P95 / P99 | 实际Cycle P95 |\n|---|---:|---:|---:|\n'
    labels=dict(A='A 原Lattice+SAC',B='B 因果低通',C='C Comfort MPPI')
    for g in ['A','B','C']:
        s=summary.loc[g];comfort+=f"| {labels[g]} | {s.comfort_index:.4f} | {s.steering_rate_rms_raw:.4f} | {s.longitudinal_jerk_rms_filtered:.4f} | {s.lateral_jerk_rms_filtered:.4f} | {s.lateral_acceleration_rms_filtered:.4f} |\n"
        driving+=f"| {labels[g]} | {s.success:.2%} | {s.crash:.2%} | {s.out_of_road:.2%} | {s.route_completion:.2%} | {s.timeout:.2%} | {s.discounted_return:.3f} | {s.average_speed:.3f} | {s.replacement_rate:.2%} |\n"
        data=isolated[isolated.configuration==g].set_index('stage');lp=data.loc['local_mppi_ms'];pl=data.loc['planner_ms'];cy=data.loc['control_cycle_ms']
        latency+=f"| {labels[g]} | {lp.p95_ms:.2f}ms | {pl.p50_ms:.2f} / {pl.p95_ms:.2f} / {pl.p99_ms:.2f}ms | {cy.p95_ms:.2f}ms |\n"
    calibration=load(R/'comfort_config.json')['comfort'];scale=load(R/'development_scales.json');physics=pd.read_csv(R/'kinematic_prediction_errors.csv').groupby(['horizon','metric'])[['p50','p95','rmse']].mean()
    failed=[k for k,v in accept['map95_conditions'].items() if not v]
    conclusion='通过本轮筛选，但仍需封存新地图进一步确认' if accept['accepted'] else '未通过联合验收，保持默认Lattice+SAC，不建议部署Comfort MPPI'
    text=f'''# Comfort-aware Local MPPI 最终报告

**结论：{conclusion}。** 这是舒适性实验，不再以WM预测Return增加作为成功依据。

## 直接回答

- C相对A的主要Comfort Index配对变化：**{gain('C-A','comfort_index_relative',True)}**。工程目标是降低至少10%，不是挑一个改善的分量。
- C相对简单因果滤波B：**{gain('C-B','comfort_index_relative',True)}**。不能仅因MPPI复杂或有非零替换，就认定优于滤波。
- C相对A的Success差：{gain('C-A','success',True)}个百分点；Crash：{gain('C-A','crash',True)}个百分点；OOR：{gain('C-A','out_of_road',True)}个百分点；路线完成：{gain('C-A','route_completion',True)}个百分点。
- C相对A的实际折扣回报变化：{gain('C-A','discounted_return_relative',True)}；转向rate：{gain('C-A','steering_rate_rms_raw_relative',True)}；纵向jerk：{gain('C-A','longitudinal_jerk_rms_filtered_relative',True)}；横向jerk：{gain('C-A','lateral_jerk_rms_filtered_relative',True)}。
- 以地图CI检查仍未满足的条件：{failed}。区间过宽时结论是证据不足，不以工程点估计替代非劣效证据。

## 1. 历史开发审计与模型限制

完整读取上一轮要求的6份报告/CSV/JSON，以及36条真实Baseline轨迹。旧12图现在只作为开发数据，没有重跑H1或Native/Reward/Q大量消融。真实轮角映射rate RMS的地图中位数约{scale['comfort_index']['steering_rate_rms_raw']:.4f}rad/s，纵向jerk RMS约{scale['comfort_index']['longitudinal_jerk_rms_filtered']:.3f}m/s³。完整分布见baseline_comfort_audit.md及baseline_comfort_metrics.csv。这些是操作性诊断，不是医学舒适性认证。

旧日志没有逐步lane ID/refined_path，因此旧变道、参考路径误差为NA，不能用world-Y伪造车道行为。新实验逐步记录lane、SAC参考目标和真实参考误差；道路接续与同一道路上的横向换道分别统计，未把所有换道归为不必要行为。转向方向/转向rate反转、横向误差波动等均单列，不能自动推断其任务必要性。

MetaDrive实车DefaultVehicle为1100kg、宽1.852m、轴距2.46894m、最大转角40deg；油门/制动对应四轮engine/brake force，原模型4/6m/s²并非经验证的真实能力边界。未拟合或篡改参数。原rollout每窗口从真实状态启动，H1/2/5/10误差如下（各图P50/P95/RMSE的平均，非声称所有窗口独立）：

```
{physics.to_string()}
```

因此**不启用预测纵向jerk、横向jerk或横向加速度的在线评分**，不在H10上做虚假的物理精确评价。只优化由动作、控制周期和实车转角映射可以准确计算的转向rate/二阶差分。复用原TrajectorySafetyEvaluator.rollout，仅H1相对参考一致性；原Safety模块及SAC几何输入保持原样。

## 2. 最小代码实现

生产改动仅planning/local_mppi.py、planning/coordinator.py与新增planning/comfort_evaluator.py。前者增加objective_mode分派、comfort CEM及独立诊断；Coordinator在原SAC完成之后传递真实车辆状态、实际历史动作及refined_path，重置comfort历史，显示仅H1轨迹；新增模块承担归一化动作品质评分和短时任务保持检查。evaluator.py/types.py/trajectory_adapter.py/structured_proposal.py/safety.py及网络未改。

默认objective_mode=legacy，原config.yaml未变。显式comfort模式以refined_actions初始化；原Gaussian/AR1、64/8/4、std/rho/temperature、Elite更新不改。候选转向局部投影在Baseline±{calibration['steering_delta_bound']:.6f}归一化量内，幅度是开发P95单步转向变化的一半；终端填充不优化。所有候选的油门序列严格等于SAC，避免不可靠纵向模型为舒适性压制必要制动。

成本为0.5×mean[(steering_rate/{calibration['steering_rate_scale']:.6f})²+(steering_second_difference/{calibration['steering_jerk_scale']:.6f})²]。尺度为开发真实物理角导数的绝对P95，不把不同量纲直接相加，不进行权重网格搜索。只选这两个可靠控制指标，不机械堆叠所有示意目标。

任务条件包括：短时相对refined_path额外误差不超{calibration['path_extra_tolerance_m']:.6f}m、相对参考进度损失不超{calibration['progress_loss_tolerance_m']:.6f}m、相对航向差不超{calibration['heading_tolerance_rad']:.6f}rad，容差来自开发H1模型误差；实际要提交的首动作归一化舒适性成本也不能变差。它们是近似参考意图保持条件，**不是安全证书，也不能保证真实避障不退化**；最终仍以真实Crash/OOR及任务非劣效判断。

Baseline始终在池内。先选重评后可行且改善的Weighted Mean，否则选择实际采样的最佳可行Elite，否则返回原Baseline。没有Oracle、Reward/Q新权重、Recovery或新规则安全拦截。WM仍服务原SAC特征，但所有comfort候选的WM/Q调用数为0。独立舒适性分数不伪装成WM Return；相关WM别名为不可用，comfort诊断另列。

简单滤波B仅用当前SAC动作与实际上一动作，alpha={p['lowpass']['alpha']:.6f}，tau={p['lowpass']['tau_seconds']:.6f}s；时间尺度由开发转向rate/jerk P95之比计算，采用相同局部修正幅度，油门不滤波，未利用未来状态。

## 3. 真实闭环实验

基线GitHub/main：{p['base_commit']}。新地图为{p['scenes']}。排除训练区间和全部历史图，扫描记录见exclusions.json；每张实际生成的道路几何SHA独立，不是把旧图改名。3个seed是配对规划/global种子，同地图环境/交通seed固定，故统计独立单位为12地图，不是36独立地图。

2张旧图A/B/C共6次100步短测通过；原A的动作、观测和reward与上轮记录逐位一致。核心及配置随后冻结，再运行每组36次完整原生回合，三组共108次。每周期Lattice→SAC→可选滤波或Comfort MPPI→执行首动作→环境反馈，未重放整条10步序列。36组初始观测/车辆/交通/RNG哈希相同，没有根据结果更换地图、seed或缩短失败回合。

{comfort}

{driving}

Comfort Index是raw转向rate RMS、统一SG滤波后的真实纵向jerk RMS、横向jerk RMS、横向加速度RMS各除以开发地图中位尺度后等权平均。SG为固定5点poly2，原始数据全部保留，离线滤波没有进入在线控制。横向加速度使用真实v×航向变化率，属于运动学测量代理而非加速度传感器。硬制动事件阈值为开发负加速度P95={p['hard_braking_threshold_mps2']:.3f}m/s²，不是通用舒适阈值或真实紧急状态标签。

主统计先按相同map/seed配对，再按地图聚类2000次bootstrap，报告95%CI、LOMO、逐seed、各分量及raw/filtered结果。相对变化以原方法绝对值作分母，避免失败负回报的符号歧义。详见paired_statistics.json、paired_comparisons.csv、map_level_statistics.csv及每回合episode.json。

## 4. 实时性

原生108回合是6进程共享GPU负载，包含冷启动。另在前两张预定地图各方法单进程运行105个真实控制周期，前5步暖机，600个剩余实际周期用于独立延迟测量，不加入驾驶结果样本量。

{latency}

完整Planner同步包围agent.act及B组实际滤波；LocalMPPI单独包围plan；Control Cycle=Planner+env.step，均没有减少必要控制步骤或改变0.1s物理周期。诊断日志处理和磁盘写入不包含在控制计时内。所有P50/P95/P99/max及阶段计时在latency.csv，硬件状态在hardware_timing_start.txt。目标是Planner P95≤100ms，另预注册不超过A的1.20倍；未满足时不部署。

## 5. 验收、测试和交付

工程点估计：{accept['engineering_point_conditions']}。

地图95%条件：{accept['map95_conditions']}。

最终联合通过：**{accept['accepted']}**。不把Baseline fallback、预测成本下降、单张地图改善或测试通过当成算法成功。如果简单低通已能达到相同或更好的舒适性且计算更省，本轮证据不支持继续保留复杂MPPI。

59项相关测试通过（原46项覆盖保留，新增10项comfort核心及3项实际指标/地图统计测试）。真实冻结WM legacy动作/Mean/Std/Q/RNG逐位一致。原WM/SAC535维输入、Residual映射、Lattice、Reward/Cost/Done、checkpoint SHA {p['checkpoint_sha256']}均保持不变。WM200k+RL300k仍为原总500k模型，未换成另一checkpoint、未训练网络。

交付包含protocol、manifest、开发舒适性审计、物理模型验证、108回合原始记录、实际舒适性指标、配对统计、延迟、图表、验证与完整代码Diff。正式输出目录按prompt为reports/mppi_comfort_optimization_20261120；若采集期间使用短目录名，所有任务结束后仅迁移目录，原始内容哈希保持不变，见directory_migration.json。文件夹日期为任务标识，实际完成时间记录在verification。

尚未解决：不可靠的长时域纵/横向物理预测仍未用于优化；没有传感器舒适度主观验证；12张地图/固定地图交通种子的外推范围有限；没有自动识别所有必要/不必要变道；不能根据本轮测试再调权重后将相同地图称为新独立验证。默认模式未启用，未commit/push。操作与复现见RUNNING.md。
'''
    with (R/'COMFORT_MPPI_FINAL_REPORT.md').open('x') as f:f.write(text)
    build_patch()
    dump(R/'artifact_manifest.json',{str(f.relative_to(R)):dict(sha256=file_sha(f),bytes=f.stat().st_size) for f in sorted(R.rglob('*')) if f.is_file() and '__pycache__' not in str(f)})
    print('COMFORT_REPORT_COMPLETE',conclusion,flush=True)
if __name__=='__main__':main()
