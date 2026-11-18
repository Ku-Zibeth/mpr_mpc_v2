"""Single frozen model/feature schema on six NEW validation maps, once."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from tools.first_action_study import *
from train_offline import dataset,evaluate
from offline_model import load_model

@torch.no_grad()
def main():
    assert not (REPORT/'validation.json').exists()
    p=load(REPORT/'protocol.json');chosen=load(REPORT/'training_decision.json');assert chosen['proceed_to_one_validation'];m=chosen['selected']
    assert file_sha(ROOT/m['checkpoint'])==m['checkpoint_sha'];assert file_sha(REPORT/'training_manifest.json')==m['context']['training_manifest_sha']
    model,artifact=load_model(ROOT/m['checkpoint'],m['context']);rows,x=dataset(p['validation_scenes']);pred=model(torch.tensor(x[m['schema']])).numpy()
    result=evaluate(pred,rows);s=result['summary'];required=[s['pair_accuracy']['ci95'][0] is not None and s['pair_accuracy']['ci95'][0]>.5,
        s['spearman']['ci95'][0] is not None and s['spearman']['ci95'][0]>0,s['regret_improvement']['ci95'][0]>0]
    support=bool(s['pair_accuracy']['scenes']>=2 and s['spearman']['scenes']>=2)
    passed=bool(all(required) and support and len(set(r['scenario'] for r in rows))>=6)
    dump(REPORT/'validation.json',dict(passed=passed,selected_checkpoint=m['checkpoint'],selected_checkpoint_sha=m['checkpoint_sha'],context=m['context'],evaluation=result,
        tests=dict(pair_accuracy_ci_above_half=required[0],spearman_ci_above_zero=required[1],regret_improvement_ci_above_zero=required[2],not_single_effective_scene=support),
        scenes=p['validation_scenes'],single_frozen_model=True,models_reselected_after_validation=False,online_ready=False,
        reason='Allofflinegatespassed; isolatedfirst-actionplanner muststillbeimplemented/testedandclosedloopvalidated.' if passed else 'One or more independentrankinggates failed. NO onlinefirst-actionplanner.'))
    print('Independentvalidation pass',passed,s,flush=True)

if __name__=='__main__':main()
