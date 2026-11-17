"""Standalone scientific figures for paired diagnostic results."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).parent
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def line(ax,x,rows,key,label,color,style='-'):
 y=np.array([r[key] for r in rows],float);low=np.array([r[key+'_ci95_low'] for r in rows],float);high=np.array([r[key+'_ci95_high'] for r in rows],float)
 ax.plot(x,y,style+'o',label=label,color=color,linewidth=1.7,markersize=4);ax.fill_between(x,low,high,color=color,alpha=.12)


def main():
 p=json.loads((R/'protocol.json').read_text());it=json.loads((R/'iteration_results.json').read_text());dy=json.loads((R/'relative_dynamics.json').read_text());info=json.loads((R/'state_information.json').read_text());figs=R/'figures';figs.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
 for kind,color in [('mean','#1261a0'),('elite','#c55a11')]:
  curves=[it['curves']['A'][kind][str(k)] for k in range(5)];line(axes[0],range(5),curves,'predicted_gain',kind,color)
  for col,protocol in [(1,'A'),(2,'B')]:line(axes[col],range(5),[it['curves'][protocol][kind][str(k)] for k in range(5)],'actual_gain',kind,color)
 for ax,title in zip(axes,['Predicted gain (one shared Q pair)','Actual gain: H10 then SAC','Actual gain: first action then SAC']):ax.set(xlabel='MPPI iteration k',ylabel='Discounted return gain vs own baseline',title=title,xticks=range(5));ax.axhline(0,color='gray',lw=.8);ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Four previously-seen development maps; scenario-clustered 95% CI');fig.savefig(figs/'iteration_gain.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
 for col,protocol in enumerate(['A','B']):
  for kind,color in [('mean','#1261a0'),('elite','#c55a11')]:line(axes[col],range(5),[it['curves'][protocol][kind][str(k)] for k in range(5)],'actual_return',kind,color)
  base=it['curves'][protocol]['mean']['0']['actual_return'];axes[col].axhline(base,color='gray',ls=':',label='baseline');axes[col].set(xlabel='MPPI iteration k',ylabel='Actual discounted suffix return',xticks=range(5),title='A: H10 then SAC' if protocol=='A' else 'B: first action then SAC');axes[col].legend();axes[col].grid(alpha=.2)
 fig.savefig(figs/'iteration_actual_return.png',dpi=180);plt.close(fig)
 h=p['horizons'];rr=[dy['by_horizon'][str(x)] for x in h];fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
 for ax,key,title in [(axes[0],'cosine','Direction cosine'),(axes[1],'relative_error_l2','Relative latent error L2'),(axes[2],'q_flip_common_action','Q ranking flip (common action)')]:line(ax,h,rr,key,'paired candidates','#1261a0');ax.set(xlabel='H (actions)',ylabel=title,xticks=h);ax.grid(alpha=.2)
 axes[0].axhline(0,color='gray',lw=.8);fig.suptitle('Fit + Inner Stop only; latent units are not meters');fig.savefig(figs/'relative_dynamics.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
 for ax,key,title in [(axes[0],'kinematic_relative_lateral_error_m','Lateral error (m)'),(axes[1],'kinematic_relative_yaw_error_rad','Yaw error (rad)'),(axes[2],'kinematic_relative_speed_error_mps','Speed error (m/s)')]:line(ax,h,rr,key,'existing bicycle model','#c55a11');ax.set(xlabel='H (actions)',ylabel=title,xticks=h);ax.grid(alpha=.2)
 fig.suptitle('Existing kinematic predictor; not a decoded WM physical prediction');fig.savefig(figs/'kinematic_errors.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,3.6),layout='constrained');names=['real512','real535','imagined512'];x=np.arange(3);means=[info['results'][n]['summary']['accuracy'] for n in names];low=[info['results'][n]['summary']['accuracy_ci95_low'] for n in names];high=[info['results'][n]['summary']['accuracy_ci95_high'] for n in names];ax.errorbar(x,means,yerr=[np.array(means)-low,np.array(high)-means],fmt='o',capsize=5,color='#1261a0');ax.axhline(.5,color='gray',ls=':');ax.set(xticks=x,xticklabels=names,ylabel='Matched continuation ranking accuracy',ylim=(0,1),title='28-map grouped-CV, fixed effective-df16 linear probes');ax.grid(alpha=.2);fig.savefig(figs/'state_information.png',dpi=180);plt.close(fig)
 physical=json.loads((R/'physical_readout.json').read_text());rr=[physical['by_horizon'][str(x)] for x in h];fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
 for ax,name,unit in zip(axes,['lateral_m','yaw_rad','speed_mps'],['m','rad','m/s']):
  line(ax,h,rr,name+'_real_latent_readout_error','real latent readout floor','#1261a0')
  line(ax,h,rr,name+'_imagined_latent_readout_error','imagined latent readout','#c55a11')
  ax.set(xlabel='H (actions)',ylabel='Relative '+name.split('_')[0]+' MAE ('+unit+')',xticks=h);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Ephemeral grouped-CV linear physical readout; no online decoder');fig.savefig(figs/'wm_physical_readout.png',dpi=180);plt.close(fig)
 for name,rows in [('iteration_curves',it['rows']),('iteration_ablation',it['iteration_rows']),('relative_dynamics',dy['rows'])]:
  with (R/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
