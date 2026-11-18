"""Export actual-environment comfort/task figures, never predicted costs as outcomes."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from comfort_study import *
def save(fig,name):
    (R/'figures').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(R/'figures'/name,dpi=160);plt.close(fig)
def main():
    e=pd.read_csv(R/'paired_closed_loop_episodes.csv');means=e.groupby('configuration').mean(numeric_only=True)
    keys=['comfort_index','steering_rate_rms_raw','longitudinal_jerk_rms_filtered','lateral_jerk_rms_filtered','lateral_acceleration_rms_filtered']
    fig,axes=plt.subplots(1,5,figsize=(16,4))
    for ax,k in zip(axes,keys):ax.bar(means.index,means[k]);ax.set_title(k.replace('_','\n'),fontsize=9)
    save(fig,'actual_comfort_components.png')
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    for ax,k in zip(axes,['success','crash','out_of_road','route_completion']):ax.bar(means.index,means[k]);ax.set_ylim(0,1);ax.set_title(k)
    save(fig,'actual_driving_outcomes.png')
    s=load(R/'paired_statistics.json');fig,ax=plt.subplots(figsize=(10,4))
    for comp in ['B-A','C-A','C-B']:
        d=s['comparisons'][comp]['comfort_index_relative']['maps'];ax.plot(range(len(d)),list(d.values()),'-o',label=comp)
    ax.axhline(-.1,color='gray',ls='--',label='10% engineering target');ax.axhline(0,color='black',lw=.7)
    ax.set_xticks(range(12),[str(x) for x in d],rotation=45);ax.set_ylabel('Paired comfort-index relative difference');ax.legend();save(fig,'map_comfort_improvement.png')
    lat=pd.read_csv(R/'latency.csv');isolated=lat[(lat.scope=='isolated_single_worker_warm')&lat.stage.isin(['planner_ms','local_mppi_ms','control_cycle_ms'])]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,k in zip(axes,['planner_ms','local_mppi_ms','control_cycle_ms']):
        z=isolated[isolated.stage==k];ax.bar(z.configuration,z.p95_ms);ax.axhline(100,color='red',ls='--');ax.set_title(k+' P95');ax.set_ylabel('ms')
    save(fig,'isolated_latency.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,k in zip(axes,['steering_rate_rms','longitudinal_jerk_rms','lateral_jerk_rms']):
        for label in ['raw','filtered']:ax.plot(means.index,means[k+'_'+label],'-o',label=label)
        ax.set_title(k.replace('_','\n'));ax.legend()
    save(fig,'raw_filtered_sensitivity.png')
    first=load(R/'protocol.json')['scenes'][0];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for group in ['A','B','C']:
        f=pd.read_csv(R/'episodes'/f'{group}_{first}_r1'/'steps.csv');axes[0].plot(f.x,f.y,label=group)
        axes[1].plot(f.step*.1,f.steering,label=group);axes[2].plot(f.step*.1,f.speed,label=group)
    for ax,title in zip(axes,['Actual XY (separate closed loops)','Normalized actual steering','Actual speed m/s']):ax.set_title(title);ax.legend()
    save(fig,'example_closed_trajectories.png')
if __name__=='__main__':main()
