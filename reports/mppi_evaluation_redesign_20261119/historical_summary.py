"""Map-level CIs for descriptive old component scores; no simulation, fitting or new threshold choice."""
import pandas as pd
from study import *
frame=pd.read_csv(R/'score_component_ablation.csv');answer={}
for method,group in frame.groupby('method'):
    values={}
    for key in ['selected_gain','top1_regret','spearman','kendall','descriptive_pair_accuracy','positive_candidate_recall','negative_selection_rate','score_margin']:
        maps=group.groupby('scene')[key].mean().dropna();v=maps.to_numpy(float)
        if len(v):
            rng=np.random.default_rng(20261119);boot=v[rng.integers(len(v),size=(2000,len(v)))].mean(1)
            values[key]=dict(mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]),effective_maps=len(maps),maps=maps.to_dict())
        else:values[key]=dict(mean=None,ci95=None,effective_maps=0)
    answer[method]=values
raw=pd.read_csv(R/'historical_h1_analysis.csv');nonbase=raw[raw.candidate_id!='A0'];stability=pd.read_csv(R/'q_rank_stability.csv')
dump(R/'historical_statistics.json',dict(methods=answer,interpretation='ALL model-vs-H1 ranking diagnostics are target-mismatched, not formal H1 validation',
    unique_true_actions=len(raw),nonbaseline_true_actions=len(nonbase),positive_fraction_all_nonbaseline=float((nonbase.actual_h1_gain>0).mean()),
    mean_unique_Q_top1_choices_per_origin=float(stability.groupby(['scene','step']).top1.nunique().mean()),
    training_latent_support_available=False,OOD_claim='Not identifiable from high Q alone; no training replay latent coverage supplied',
    baseline_relativization='Subtracting same baseline score cannot change within-origin ordering, not a learned or verified H1 Advantage'))
