"""Clarify the evidence-ranked next step without changing any model/gate."""
import shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from study_io import R,ROOT,load,sha,dump
from report import fmt


def main():
    archive=R/'before_decision_clarification';archive.mkdir(exist_ok=True)
    name='MPPI_FIRST_ACTION_ADVANTAGE_REPORT_20261118.md'
    for f in [name,'decision.json','verification.json','code_changes.patch']:
        if not (archive/f).exists():shutil.copyfile(R/f,archive/f)
    pairs=load(R/'paired_statistics.json')
    signal=pairs['lomo/A2_linear_minus_A2_mlp']['model_gain_improvement']
    assert signal['ci95'][0]>0
    manifest=load(R/'experiment_manifest.json')
    for record in manifest['prior_reports_read']:assert sha(Path(record['path']))==record['sha256']
    d=load(R/'decision.json')
    d.update(primary_observed_failure='Cross-map action-advantage generalization; nonlinear model/training has a simplification signal, not a uniquely proven capacity cause.',
        evidence_branch='B partially supported, with context coverage still unresolved; NOT all models/Oracle devoid of signal.',
        linear_vs_selected_mlp_lomo_gain_improvement=signal,
        next_primary_direction='Stop larger/nonlinear heads; retain simplest Linear diagnostic as the fixed control for any separately preregistered future map/origin-coverage study. No new collection in this round.',
        selection_changed_after_evaluation=False,new_validation_performed=False)
    (R/'decision.json').write_text(__import__('json').dumps(d,indent=2))
    report=R/name;text=report.read_text()
    text=text.replace(
        '8. **主要问题是什么？** 当前可观测瓶颈仍是action-advantage跨地图/context泛化，训练/容量有辅助迹象，不能唯一归因。开发Oracle正收益不支持立即否定candidate空间；本轮没有证据把Dynamics升级为主要原因。',
        '8. **主要问题是什么？** 可观测失败是action-advantage跨地图泛化不足。Linear相对A2 MLP的配对Gain/Regret改善有正95%区间，因此本轮比“所有方法都失败”更支持情况B中的模型/训练简化方向；context覆盖仍是未分解因素，不能唯一指认容量。开发Oracle正收益不支持立即否定candidate空间，没有证据把Dynamics升级为主要原因。')
    text=text.replace(
        '唯一下一阶段方向：若继续研究，应预注册增加不同地图/道路交互条件下的origin覆盖，检验当前特征下的局部优势可辨识性；不要继续增加同类网络宽度/深度，也不要因本次结果直接重训WM或重构candidate空间。本轮按门槛停止，不自动新增采集。',
        '唯一下一阶段方向：先停止复杂MLP扩展，保留最简单的Linear诊断作为固定control；若另行继续，应在预注册的新map/origin覆盖中检验其局部优势能否稳定泛化，而非继续堆网络。当前Linear的LOMO Accuracy区间仍跨50%，不满足本轮门槛，不能事后替换主A2去采新validation。本轮停止，不自动新增采集，不重训WM或重构candidate空间。')
    if '## 补充：不掩盖简单Linear的积极信号' not in text:
        text+='\n## 补充：不掩盖简单Linear的积极信号\n'
        text+=f"\n在相同8个LOMO地图/起点上，A2 Linear相对被选中A2 MLP的Model Gain改善={fmt(signal)}，同样等于Top1 Regret改善。该95%区间下界略大于0，属于支持模型/训练简化的证据，不能把本轮总结成‘所有relative模型完全失败，所以必然只有context coverage问题’。\n"
        text+='\n同时，Linear的Accuracy改善区间跨0，它自身LOMO Accuracy区间跨50%，数据仅8图、单一固定seed，存在多个诊断比较；这不是已经证明Linear满足新独立验证/部署要求，也不能唯一证明参数量是因果根源。KNN的inner/LOMO方向相反，进一步说明只看一个开发split不够。最终A2选型、所有训练超参数、门槛与不采新validation的决定均未改变。\n'
    report.write_text(text)
    sources=sorted(R.glob('*.py'))+[R/'protocol.json',R/'feature_schema.json']
    changes=[]
    for f in sources:
        result=subprocess.run(['git','diff','--no-index','--','/dev/null',str(f.relative_to(ROOT))],cwd=ROOT,capture_output=True,text=True)
        assert result.returncode==1;changes.append(result.stdout)
    (R/'code_changes.patch').write_text(''.join(changes))
    v=load(R/'verification.json')
    v.update(source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},
        phase2_historical_reports_unchanged=True,posthoc_model_reselection=False,
        positive_linear_signal_explicitly_preserved=True,
        scientific_decision='Simplify rather than expand MLP; context generalization remains unresolved. Development gates failed, no validation/online changes.')
    (R/'verification.json').write_text(__import__('json').dumps(v,indent=2))
    dump('execution_manifest_final.json',dict(cwd=str(ROOT),binary_source=str(ROOT.parent/'mpr_mpc_v2_first_action_20261117'),
        environment={'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        scripts=['prepare.py','train.py','evaluate.py','benchmark.py','test_advantage.py','report.py','finalize.py'],
        testing='Original82 under prior worktree; new16 under main, --import-mode=importlib, each fresh process.',
        protocol_sha256=sha(R/'protocol.json'),all_new_outputs_in_phase3_directory=True))
    print('Final report preserves Linear signal and unchanged failed gates;98 tests.',flush=True)


if __name__=='__main__':main()
