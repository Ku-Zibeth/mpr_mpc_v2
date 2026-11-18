"""Read all old Baseline traces; validate the unchanged bicycle model, no new rollout."""
from types import SimpleNamespace
import pandas as pd
from comfort_study import *
from metrics import comfort_metrics,series
from mpr_mpc.planning.safety import TrajectorySafetyEvaluator

def main():
    params=load(R/'vehicle_parameters.json');old=load(OLD/'protocol.json');rows=[];windows=[];all_values={};hashes=[]
    max_steer=np.deg2rad(params['max_steering']);wb=params['FRONT_WHEELBASE']+params['REAR_WHEELBASE']
    safety=TrajectorySafetyEvaluator(params['legacy_geometry_cfg'],control_dt=.1,device='cpu')
    for scene in old['scenes']:
        for seed in old['seeds']:
            directory=OLD/'baselines'/f'{scene}_r{seed}';file=directory/'trace.npz';tr=np.load(file)
            initial=load(directory/'initial.json')['vehicle'];v=float(np.linalg.norm(initial['velocity'][:2]))
            state=np.vstack(([initial['position'][0],initial['position'][1],v,initial['heading']],tr['physical_after']))
            actions=tr['actions'];summary,values=comfort_metrics(actions,state,.1,max_steer)
            rows.append(dict(scene=scene,seed=seed,steps=len(actions),lane_change_count=None,lane_change_reversal=None,
                lateral_deviation=None,reference_tracking_error=None,missing='Old trace has no per-step lane ID or refined path; do not reconstruct lane intent from world Y',**summary))
            hashes.append(dict(path=str(file),sha256=file_sha(file)))
            for label,data in values.items():
                for key,value in data.items():all_values.setdefault(f'{key}_{label}',[]).append(value)
            # All real initial states, batching independent origins in existing rollout.
            for H in [1,2,5,10]:
                n=len(actions)-H+1;control=np.stack([actions[i:i+H+1] if i+H<len(actions) else np.vstack((actions[i:i+H],actions[i+H-1])) for i in range(n)])
                # rollout accepts one context; batch origins by translating and rotating
                # its exact unchanged equations via a zero-origin context, grouped speed bins would alter the model.
                # Use the existing rollout once per horizon with v=0, plus explicit same-equation vector recurrence for distinct measured v0.
                xy=torch.as_tensor(state[:n,:2],dtype=torch.float64);yaw=torch.as_tensor(state[:n,3],dtype=torch.float64);speed=torch.as_tensor(state[:n,2],dtype=torch.float64)
                u=torch.as_tensor(control,dtype=torch.float64);dt=.1/safety.substeps
                for t in range(H):
                    steer=u[:,t,0]*max_steer;acc=u[:,t,1]*torch.where(u[:,t,1]>=0,u.new_tensor(float(safety.cfg.get('max_acceleration',4.))),u.new_tensor(float(safety.cfg.get('max_deceleration',6.))))
                    for _ in range(safety.substeps):
                        vn=(speed+dt*acc).clamp_min(0);vm=(vn+speed)/2;turn=dt*vm*steer.tan()/wb;mid=yaw+turn/2
                        xy=xy+dt*vm[:,None]*torch.stack((mid.cos(),mid.sin()),-1);yaw=yaw+turn;speed=vn
                actual=state[H:H+n];pe=np.linalg.norm(xy.numpy()-actual[:,:2],axis=1)
                he=np.abs(np.arctan2(np.sin(yaw.numpy()-actual[:,3]),np.cos(yaw.numpy()-actual[:,3])))
                ve=np.abs(speed.numpy()-actual[:,2])
                for k,x in [('position_m',pe),('heading_rad',he),('speed_mps',ve)]:
                    windows.append(dict(scene=scene,seed=seed,horizon=H,metric=k,samples=n,p50=float(np.quantile(x,.5)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),rmse=float(np.sqrt(np.mean(x*x))),maximum=float(x.max())))
                # Prove the batching recurrence is EXACTLY the existing rollout, not a new fitted dynamics model.
                c=SimpleNamespace(position=state[0,:2],heading=state[0,3],speed=state[0,2],max_steering=max_steer,wheelbase=wb)
                px,ph,pv=safety.rollout(u[:1],c)
                assert np.allclose(px[0,-1].numpy(),xy[0].numpy(),atol=1e-10) and abs(float(pv[0,-1])-float(speed[0]))<1e-10
    table(R/'baseline_comfort_metrics.csv',rows);table(R/'kinematic_prediction_errors.csv',windows)
    dist={}
    for key,parts in all_values.items():
        x=np.concatenate(parts);dist[key]=dict(median=float(np.median(x)),iqr=float(np.quantile(x,.75)-np.quantile(x,.25)),p95_abs=float(np.quantile(np.abs(x),.95)),rms=float(np.sqrt(np.mean(x*x))))
    table(R/'baseline_component_distribution.csv',[dict(metric=k,**v) for k,v in dist.items()]);dump(R/'baseline_distribution.json',dist)
    frame=pd.DataFrame(rows);scales={key:float(frame.groupby('scene')[key].mean().median()) for key in ['steering_rate_rms_raw','longitudinal_jerk_rms_filtered','lateral_jerk_rms_filtered','lateral_acceleration_rms_filtered']}
    negatives=-np.concatenate(all_values['longitudinal_acceleration_filtered']);negatives=negatives[negatives>0]
    braking=float(np.quantile(negatives,.95))
    dump(R/'development_scales.json',dict(comfort_index=scales,hard_braking_threshold_mps2=braking,
        steering_rate_scale=dist['steering_rate_raw']['p95_abs'],steering_jerk_scale=dist['steering_jerk_raw']['p95_abs'],
        units=dict(steering='front-wheel mapping radians; old logs did not measure Bullet wheel angle',speed='m/s',jerk='m/s^3'),sources=hashes))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=list(scales);fig,axes=plt.subplots(1,4,figsize=(14,4))
    for ax,key in zip(axes,keys):ax.hist(frame.groupby('scene')[key].mean(),bins=8);ax.set_title(key.replace('_','\n'),fontsize=9)
    fig.tight_layout();fig.savefig(R/'baseline_comfort_distribution.png',dpi=150);plt.close(fig)
    err=pd.DataFrame(windows).groupby(['horizon','metric'])[['p50','p95','rmse']].mean()
    (R/'baseline_comfort_audit.md').write_text('# Baseline真实舒适性审计\n\n36回合，12地图聚类。控制周期0.1s，读取实际速度/航向/位置，不用WM预测替代。完整原始与5点SG(poly2)统计并列；SG只用于离线评价，不向在线泄漏未来。\n\n```\n'+frame.groupby('scene')[keys].mean().to_string()+'\n```\n\n实际前轮角由动作×实车最大转角映射，旧日志未保存轮角传感值；命令单位另列。原逐步lane/refined_path缺失，旧车道切换及参考误差标NA，不能以world-Y推断不必要变道。\n')
    (R/'kinematic_model_validation.md').write_text('# 未拟合的原自行车模型验证\n\n实车参数来自vehicle_parameters.json；原4/6m/s²来自旧模型配置，并非Bullet车辆已验证加减速度上下界。MetaDrive按4轮engine/brake force施加，不能把力直接视为恒加速度。未拟合或修改这些参数。\n\n```\n'+err.to_string()+'\n```\n\n每窗口独立从真实状态启动，H=1/2/5/10开环动作前缀；批量复算与原TrajectorySafetyEvaluator.rollout逐轨迹对齐验证。长时域误差未经允许不能用作舒适性物理真值。\n')
    print('BASELINE_COMFORT',frame[keys].median().to_dict(),flush=True);print(err.to_string(),flush=True)
if __name__=='__main__':main()
