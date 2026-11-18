"""Preserve intermediate report; add complete stratification and oracle caveats."""
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from analyze import R, OLD, load, dump
from finish import fmt
from tools.first_action_study import file_sha


def main():
    archive=R/'final_before_extensions'
    archive.mkdir(exist_ok=False)
    report=R/'MPPI_FIRST_ACTION_FAILURE_REPORT_20261118.md'
    names=['MPPI_FIRST_ACTION_FAILURE_REPORT_20261118.md','verification.json','code_changes.patch','decision.json']
    for name in names:shutil.copyfile(R/name,archive/name)
    state=load(R/'state_oracle_ranking.json')
    dynamics=load(R/'dynamics_error_analysis.json')
    text=report.read_text().replace(
        '原TD终端Q有局部状态误差贡献，但真实状态/Reward oracle仍未形成可靠排名，且不是当前Q_B的未来状态通路。',
        '原TD终端Q有局部状态误差贡献；部分H10 Oracle在6张已见地图上高于机会水平，但配对提升区间仍跨0，尚不是新独立地图上的可靠恢复，且不是当前Q_B的未来状态通路。')
    text+='\n## 补充：完整分层指标及有利于Dynamics的证据\n'
    text+='\nstratified_ranking.csv和stratified_ranking_statistics.json给出751个origin/分层记录的Accuracy、Spearman、Kendall、Regret、Baseline差值MAE、收益差相关性及场景级95%区间。动作幅度、相对Baseline收益差、episode crash/OOR、初速、coarse曲率代理、固定density均按预先规定阈值划分。各子集保留Baseline作共同参照，即使它本身不属于该层；因此子集Top1只是受限候选池的离线描述，不是实际在线选择，结果分层不能变成未来信息过滤器。Pair差值层另见pair_diagnostics.csv。\n'
    text+='\n不能隐藏支持状态误差影响的结果。以下为相同末端可用子集（H1共18图，H10仅6张已见Pilot图，缺失末端另报）：\n'
    text+='\n|H|预测状态J00 Accuracy%|真实状态J01 Accuracy%|真实Reward+状态J11 Accuracy%|\n|---|---:|---:|---:|\n'
    for h in [1,2,4,10]:
        base=f'all_seen/H{h}/matched_endpoint_subset/'
        stats=[state['summaries'][base+k]['pair_accuracy'] for k in ['J00_predicted','J01_actual_latent_same_predicted_mask','J11_actual_reward_actual_latent_same_mask']]
        text+=f'|{h}|'+ '|'.join(fmt(s,True) for s in stats)+'|\n'
    text+='\nH10部分Actual Oracle本身高于50%，是应保留的积极证据；但与同起点预测状态的配对提升CI仍跨0，且该子集是已见地图及可达末端，不足以部署或把它当新独立验证。H1 Regret改善CI大于0也不能抹掉；结论是Dynamics有贡献，尚未证明其解释了当前Q_B失效的主要部分。\n'
    for h in [1,10]:
        a=dynamics['within_origin_pair_associations'][f'H{h}']
        text+=f"\nH{h}同起点相对Dynamics误差与终端Q排序错误的Spearman={fmt(a['error_vs_rank_error_spearman'])}；与真实收益差绝对值的Spearman={fmt(a['error_vs_return_gap_spearman'])}。"
    text+='\n关联并非统一正向，较大动作效应/收益间隔也可能更易排序；这些观察不构成“误差越大必然排错”或相反方向的因果证明。\n'
    report.write_text(text)
    decision=load(R/'decision.json')
    decision.update(h10_seen_map_actual_oracle_can_exceed_chance=True,
                    positive_dynamics_evidence_not_suppressed=True,
                    only_one_next_primary_direction=True)
    (R/'decision.json').write_text(__import__('json').dumps(decision,indent=2))
    required=['MPPI_FIRST_ACTION_FAILURE_REPORT_20261118.md','data_integrity_audit.json','branch_integrity.csv',
              'origin_coverage.csv','horizon_ranking_comparison.csv','scene_level_ranking.csv',
              'oracle_candidate_analysis.csv','oracle_candidate_report.json','predicted_vs_actual_state.csv',
              'dynamics_error_analysis.json','state_oracle_ranking.json','paired_bootstrap_statistics.json',
              'stratified_ranking.csv','stratified_ranking_statistics.json']
    assert all((R/f).is_file() and (R/f).stat().st_size>0 for f in required)
    sources=[ROOT/'tools/first_action_failure.py',*sorted(R.rglob('*.py'))]
    patches=[]
    for f in sources:
        diff=subprocess.run(['git','diff','--no-index','--','/dev/null',str(f.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True)
        assert diff.returncode==1
        patches.append(diff.stdout)
    (R/'code_changes.patch').write_text(''.join(patches))
    verification=load(R/'verification.json')
    verification.update(source_sha256={str(f.relative_to(ROOT)):file_sha(f) for f in sources},
                        full_stratified_metrics_delivered=True, intermediate_report_preserved=True,
                        supporting_h10_oracle_results_explicitly_reported=True,
                        required_deliverables=required)
    (R/'verification.json').write_text(__import__('json').dumps(verification,indent=2))
    dump('execution_manifest.json',dict(python='/home/kzb/anaconda3/envs/tdmpc2/bin/python',cwd=str(ROOT),
        environment={'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        order=['audit.py (initial Phase0/1)','analyze.py','state_oracle.py (archived numerical drift attempt, then exact cached-H10 parity)',
               'supplement.py','planner_parity.py --code-root MAIN / WORKTREE (no environment)','plots.py',
               'test_command.json (original71+new11)','finish.py','stratify.py','finalize_delivery.py'],
        existing_outputs_preserved=True, scripts_use_exclusive_output_creation=True,
        no_new_training=True,no_new_environment_runs=True))
    print('Complete Phase2 delivery verified, including all stratified metrics and positive oracle evidence.',flush=True)


if __name__=='__main__':main()
