"""Real frozen-WM default-output parity fixture; no simulator and no training."""
import argparse
from study import *
ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);args=ap.parse_args()
from planning.local_mppi import LocalMPPI
from omegaconf import OmegaConf
with torch.no_grad():
    evaluator=load_offline_evaluator();cfg=OmegaConf.to_container(OmegaConf.load(CODE_ROOT/'config.yaml').mpr_mpc.mppi)
    planner=LocalMPPI(evaluator,cfg,horizon=10,action_dim=2,device='cuda')
    source=np.load(HIST/'origins/130719_50/catalog.npz');torch.manual_seed(20261119);planner.reset(1)
    result=planner.plan(torch.tensor(source['latent'],device='cuda'),torch.tensor(source['baseline_sequence'],device='cuda'),eval_mode=True)
    np.savez_compressed(R/f'{args.name}.npz',selected=result.selected_sequence.cpu().numpy(),mean=result.final_mean.cpu().numpy(),std=result.final_std.cpu().numpy(),
        cpu_rng=torch.random.get_rng_state().numpy(),cuda_rng=torch.cuda.get_rng_state().cpu().numpy(),private_rng=planner.generator.get_state().cpu().numpy(),
        selected_value=result.selected_wm_value.cpu().numpy())
    if args.name!='default_before':
        old=np.load(R/'default_before.npz');actual=np.load(R/f'{args.name}.npz')
        assert all(np.array_equal(old[k],actual[k]) for k in old.files),'Default action/distribution/value/RNG changed'
    print('FROZEN_DEFAULT_PARITY',args.name,flush=True)
