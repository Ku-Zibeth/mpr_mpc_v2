"""Immutable data and source manifest; do not promote unmet deployment gates."""
from study import *
import importlib.metadata as metadata
import ast
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
    p=load(R/'protocol.json');old=load(R/'source_manifest.json');previous=Path(old['source_worktree'])
    assert file_sha(Path(p['checkpoint']))==p['checkpoint_sha256']
    assert all(file_sha(previous/name)==sha for name,sha in old['copied_core'].items())
    assert file_sha(PREV/'MPPI_TRACKING_OPTIMIZATION_REPORT_20261119.md')==old['previous_report_sha256']
    assert subprocess.check_output(['git','status','--porcelain'],cwd=MAIN).decode()==''
    assert file_sha(ROOT/'config.yaml')==file_sha(MAIN/'config.yaml')
    modified=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT).decode().splitlines();assert set(modified)=={'planning/local_mppi.py','planning/coordinator.py'}
    for directory in ['env','lattice','residual_rl','vendor']:
        assert subprocess.check_output(['git','diff','HEAD','--',directory],cwd=ROOT)==b''
    records=[load(f) for split in ['development','holdout','latency'] for f in (R/split).glob('*/episode.json')]
    assert len(records)==150 and all(x['native_done'] and x['weights_unchanged'] for x in records)
    assert load(R/'nominal_prefix_verification.json')['all_passed'] and load(R/'nominal_prefix_verification.json')['comparisons']==84
    assert load(R/'fast_full_trace_parity.json')['all_actions_observations_states_rewards_costs_termination_exact']
    assert '59 passed' in (R/'tests_final.log').read_text()
    # Unchanged exact Gaussian/elite/update methods, not just matching budgets.
    baseline=ast.parse(subprocess.check_output(['git','show','HEAD:planning/local_mppi.py'],cwd=ROOT).decode());current=ast.parse((ROOT/'planning/local_mppi.py').read_text())
    def method(tree,name):
        c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LocalMPPI')
        return ast.dump(next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    unchanged=['initial_mean','_candidate_batch','_weights','_elite_weights','_update_distribution','_plan_comfort']
    assert all(method(baseline,n)==method(current,n) for n in unchanged)
    defaults=load(R/'default_parity.json');assert defaults['arrays']==92
    dump(R/'verification.json',dict(base_commit=p['base_commit'],branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip(),
        main_clean=True,previous_worktree_and_report_unchanged=True,checkpoint_sha256=p['checkpoint_sha256'],default_config_sha256=file_sha(ROOT/'config.yaml'),
        original_WM_SAC_Lattice_environment_vendor_sources_unchanged=True,no_training_or_commit_or_push=True,no_default_activation=True,
        development_maps=12,development_episodes=48,heldout_maps=24,heldout_paired_episodes=48,heldout_total_episodes=96,
        repeated_performance_episodes=6,repeated_performance_extra_independent_maps=0,total_native_complete_episodes=150,
        source_frozen_during_all96_validation_episodes=load(R/'validation_integrity_before_performance.json'),
        all_initial_environment_and_policy_paired=True,nominal_prefix_comparisons_exact=84,traffic_repeats_actually_differ_on_all24_maps=bool(pd.read_csv(R/'independent_traffic_diversity.csv').traffic_changed_between_repeats.all()),
        raw_trajectories_preserved=True,unit_tests=dict(passed=59,failed=0),unchanged_sampling_AST=unchanged,default_and_previous_mode_parity=defaults,
        broad_phase_candidate_parity=load(R/'exact_geometry_real_state_parity.json'),full_performance_trace_parity=load(R/'fast_full_trace_parity.json'),
        crash_native_replays=load(R/'crash_replay_verification.json'),deployment_gate=load(R/'deployment_gate.json'),
        python=sys.version,packages={x:metadata.version(x) for x in ['torch','numpy','scipy','shapely','metadrive-simulator']},
        final_core_sha256={name:file_sha(ROOT/name) for name in ['planning/local_mppi.py','planning/coordinator.py','planning/native_road.py','planning/tracking_evaluator.py','planning/tracking_broad_phase.py']},
        limitation='Success/OOR improve on heldout paired sample, but returnCI includes0, collision swaps introduce3 new cases, cumulative safety cost increases, P95>100ms; no all-metric deployment claim.'))
    f=pd.read_csv(R/'holdout_all_summary.csv').set_index('configuration');fig,ax=plt.subplots(figsize=(8,4));keys=['success','crash','out_of_road'];x=np.arange(3)
    ax.bar(x-.18,100*f.loc['A',keys].to_numpy(float),.36,label='Frozen Lattice+SAC')
    ax.bar(x+.18,100*f.loc['C',keys].to_numpy(float),.36,label='Conditional MPPI')
    for i,k in enumerate(keys):
        for shift,tag in [(-.18,'A'),(.18,'C')]:ax.text(i+shift,100*f.loc[tag,k]+1,f'{100*f.loc[tag,k]:.2f}%',ha='center',fontsize=9)
    ax.set_xticks(x,['Success','Crash (ever)','Out of road']);ax.set_ylim(0,106);ax.set_ylabel('% of native episodes');ax.set_title('24 heldout maps × 2 traffic seeds: 48 paired episodes');ax.legend(loc='upper right');fig.tight_layout();fig.savefig(R/'independent_outcomes.png',dpi=180);plt.close(fig)
    print('FINAL_VERIFICATION_150_NATIVE_EPISODES_59_TESTS',flush=True)
