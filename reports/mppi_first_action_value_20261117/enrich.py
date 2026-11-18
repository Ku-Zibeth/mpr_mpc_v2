"""Frozen current-state and terminal Qs for cached actualsearch pools; noenvcalls."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common import math as td_math
from common.world_model import WorldModel
from tdmpc2 import TDMPC2
from types import SimpleNamespace
from residual_rl.networks import GaussianActor,Critic
import evaluate_model_suite as suite

@torch.no_grad()
def main():
    p=load(REPORT/'protocol.json');raw=OmegaConf.load(p['source_environment_config']);raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False
    _,cfg=suite._make_cfg(raw,Path(p['checkpoint']).parent.parent.name,Path(p['checkpoint']),[1]);cfg.obs_shape={'state':(259,)};cfg.action_dim=2;cfg.episode_length=1000
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');model=WorldModel(cfg).cuda().eval().requires_grad_(False)
    ck=torch.load(p['checkpoint'],map_location='cpu',weights_only=False);TDMPC2.load(SimpleNamespace(model=model,cfg=cfg),{'model':ck['model']})
    actor=GaussianActor(535,2,[256,256]).cuda().eval().requires_grad_(False);actor.load_state_dict(ck['mpr_mpc']['residual_sac']['actor'])
    critics=[Critic(535,2,[256,256]).cuda().eval().requires_grad_(False) for _ in range(2)]
    for m,n in zip(critics,['q1','q2']):m.load_state_dict(ck['mpr_mpc']['residual_sac'][n])
    for folder in sorted((REPORT/'origins').iterdir()):
        if not (folder/'catalog.npz').exists() or (folder/'offline_scores.npz').exists():continue
        c=np.load(folder/'catalog.npz');a=torch.tensor(c['pool'],device='cuda');z=torch.tensor(c['terminal_latents'],device='cuda');z0=torch.tensor(c['z0'],device='cuda').reshape(1,512).expand(len(a),-1)
        data={};qs=td_math.two_hot_inv(model.Q(z0,a[:,0],None,return_type='all'),cfg).reshape(5,-1).T.cpu().numpy();data['q_current']=qs
        modes=terminal_policy_actions(model,z,a[:,-1],20261117+int(folder.name.split('_')[0]))
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(20261117);pairs=np.stack([torch.randperm(5,device='cuda')[:2].cpu().numpy() for _ in range(100)])
        data['random_pairs']=pairs
        for mode,controls in modes.items():
            q=td_math.two_hot_inv(model.Q(z,controls,None,return_type='all'),cfg).reshape(5,-1).T.cpu().numpy();data['q_'+mode]=q;data['terminal_action_'+mode]=controls.cpu().numpy();data['random100_'+mode]=np.stack([q[:,v].mean(1) for v in pairs])
        state=torch.tensor(c['state535'],device='cuda').reshape(1,535);residual=actor(state,deterministic=True)[0];sac=[float(m(state,residual).reshape(())) for m in critics]
        np.savez_compressed(folder/'offline_scores.npz',**data)
        dump(folder/'offline_scores.json',dict(sac_actor_residual=residual.cpu().numpy(),sac_q1=sac[0],sac_q2=sac[1],
            critic_candidate_applicability='ONLY baselineparameteraction known; arbitraryphysicalGaussian actions have nouniqueinverse residual. Do NOT feed physicalcontrol directly into residualcritic.',
            candidate_critic_values=None,full_ensemble_independent_of_random_pair_default=True))
        print('scored',folder.name,flush=True)

if __name__=='__main__':main()
