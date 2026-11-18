"""Compare frozen real-WM legacy selection and all RNG against clean main, no environment."""
import argparse,importlib.util
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['main','current'],required=True);a=ap.parse_args()
from comfort_study import *
source=MAIN if a.source=='main' else ROOT
os.environ['MPR_CODE_ROOT']=str(source)
spec=importlib.util.spec_from_file_location('old_study',OLD/'study.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from planning.local_mppi import LocalMPPI
from omegaconf import OmegaConf
with torch.no_grad():
    evaluator=old.load_offline_evaluator();cfg=OmegaConf.to_container(OmegaConf.load(source/'config.yaml').mpr_mpc.mppi)
    planner=LocalMPPI(evaluator,cfg,horizon=10,action_dim=2,device='cuda')
    fixture=np.load(old.HIST/'origins/130719_50/catalog.npz');torch.manual_seed(20261119);planner.reset(1)
    result=planner.plan(torch.tensor(fixture['latent'],device='cuda'),torch.tensor(fixture['baseline_sequence'],device='cuda'),eval_mode=True)
    path=R/f'legacy_{a.source}.npz';assert not path.exists()
    np.savez_compressed(path,selected=result.selected_sequence.cpu().numpy(),mean=result.final_mean.cpu().numpy(),std=result.final_std.cpu().numpy(),
        cpu_rng=torch.random.get_rng_state().numpy(),cuda_rng=torch.cuda.get_rng_state().cpu().numpy(),private_rng=planner.generator.get_state().cpu().numpy(),selected_value=result.selected_wm_value.cpu().numpy())
    if a.source=='current':
        before=np.load(R/'legacy_main.npz');after=np.load(path)
        assert all(np.array_equal(before[k],after[k]) for k in before.files)
        dump(R/'legacy_parity.json',dict(passed=True,real_frozen_model=True,action_mean_std_value_all_rng_bitwise_exact=True))
    print('LEGACY_REAL_MODEL_PARITY',a.source,flush=True)
