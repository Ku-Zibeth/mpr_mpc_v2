import json,csv
from pathlib import Path
R=Path(__file__).parent
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=json.loads((R/'protocol.json').read_text());data=json.loads((R/'stage2_analysis.json').read_text());lat=json.loads((R/'isolated_latency.json').read_text());figs=R/'figures';figs.mkdir(exist_ok=True);groups=['M1','M2','M4','P','E','PE'];labels=['M1','M2','M4','Prior+Mean','Candidate+Elite','Prior+Elite'];means=np.array([data['comparisons'][g]['actual_discounted_return_difference'] for g in groups]);lo=np.array([data['comparisons'][g]['actual_discounted_return_difference_ci95_low'] for g in groups]);hi=np.array([data['comparisons'][g]['actual_discounted_return_difference_ci95_high'] for g in groups]);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10,4),layout='constrained');x=np.arange(len(groups));ax.bar(x,means,color=['#2d6da3']*3+['#b56a25']*3,alpha=.8);ax.errorbar(x,means,yerr=[means-lo,hi-means],fmt='none',ecolor='black',capsize=4);ax.axhline(0,color='gray',lw=1);ax.set(xticks=x,xticklabels=labels,ylabel='Actual discounted return difference vs SAC',title='True native closed-loop: 4 fresh development maps x 2 planner seeds');ax.grid(axis='y',alpha=.2);fig.savefig(figs/'paired_return.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');allgroups=['B',*groups];xt=np.arange(len(allgroups));names=['SAC',*labels]
for key,color in [('success','#2d6da3'),('crash','#b42d35'),('out_of_road','#c08b32')]:axes[0].plot(xt,[data['summary'][g][key]*100 for g in allgroups],'o-',label=key,color=color)
axes[0].set(xticks=xt,xticklabels=names,ylabel='Episode fraction (%)',title='Development driving outcomes');axes[0].tick_params(axis='x',rotation=25);axes[0].legend();axes[0].grid(alpha=.2)
for key,label in [('mean_ms','mean'),('p95_ms','P95')]:axes[1].plot(xt,[lat['summary'][g][key] for g in allgroups],'o-',label=label)
axes[1].axhline(100,color='#b42d35',ls=':',label='100ms control period');axes[1].set(xticks=xt,xticklabels=names,ylabel='Complete planning wall time (ms)',title='Single-worker, matched fixed-state timing');axes[1].tick_params(axis='x',rotation=25);axes[1].legend();axes[1].grid(alpha=.2);fig.savefig(figs/'driving_and_latency.png',dpi=180);plt.close(fig)
with (R/'summary.csv').open('w',newline='') as f:
 cols=['configuration',*data['summary']['B'].keys()];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for group,row in data['summary'].items():w.writerow(dict(configuration=group,**row))
