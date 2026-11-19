"""One frozen, paired validation pass. No retuning or re-selection after this."""
from study import *
from run_study import run,verify
if __name__=='__main__':
    assert load(R/'selection.json')['allow_holdout']
    assert load(R/'development_nominal_prefix_verification.json')['all_passed']
    assert load(R/'closed_latency_complete.json')['original_trace_reproduced_exactly']
    p=load(R/'protocol.json');frozen=load(R/'frozen_before_development.json')
    assert all(file_sha(ROOT/name)==sha for name,sha in frozen['source_sha'].items())
    assert file_sha(Path(p['checkpoint']))==p['checkpoint_sha256']
    selected=load(R/'selection.json')['selected']
    dump(R/'frozen_before_validation.json',dict(selected=selected,selected_configuration=load(R/'variant_configs.json')[selected],source_sha=frozen['source_sha'],
        protocol_sha256=file_sha(R/'protocol.json'),checkpoint_sha256=p['checkpoint_sha256'],independent_maps=p['holdout_scenes'],actual_traffic_seeds=p['actual_traffic_seed'],repeats=[1,2],no_further_selection_or_tuning=True))
    run([('holdout',scene,seed,m) for scene in p['holdout_scenes'] for seed in p['holdout_seeds'] for m in ['A','C']]);verify(p['holdout_scenes'],['A','C'],'holdout',p['holdout_seeds'])
    dump(R/'study_complete.json',dict(development_episodes=48,holdout_episodes=96,independent_validation_used=True,frozen_configuration=selected))
