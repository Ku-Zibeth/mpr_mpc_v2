"""Observational sampling diagnostics; outputs/RNG of the planner are untouched."""
import numpy as np
import torch


class SamplerAudit:
    def __init__(self,planner):
        self.planner=planner;self.events=[];self.elites=[]
        sample,elite=planner._candidate_batch,planner._elite_weights
        def sampled(mean,std,baseline):
            state=planner.generator.get_state().clone()
            batch=sample(mean,std,baseline)
            self.events.append((state,mean.detach().clone(),std.detach().clone(),baseline.detach().clone(),batch.detach().clone()))
            return batch
        def selected(values):
            result=elite(values);self.elites.append(result[0].detach().clone());return result
        planner._candidate_batch=sampled;planner._elite_weights=selected
    def reset(self):self.events=[];self.elites=[]
    @torch.no_grad()
    def rows(self,metadata,values,members):
        rows=[];p=self.planner
        original_state=p.generator.get_state().clone()
        for iteration,(state,mean,std,baseline,batch) in enumerate(self.events):
            side=torch.Generator(device=p.device);side.set_state(state)
            innovations=torch.randn(p.num_samples,p.horizon+1,p.action_dim,device=p.device,generator=side)
            noise=innovations.clone();scale=(1-p.noise_rho.square()).sqrt()
            for t in range(1,p.horizon+1):noise[:,t]=p.noise_rho*noise[:,t-1]+scale*innovations[:,t]
            raw=mean[None]+std[None]*noise
            reconstructed=raw.clamp(-1,1)
            if p.terminal_action=='policy_prior' or p.terminal_value=='sac_value':reconstructed[:,-1]=baseline[-1]
            assert torch.equal(reconstructed,batch[1:]),'Raw-draw reconstruction differs'
            a=batch[1:].cpu().numpy();b=baseline.cpu().numpy();delta=a-b
            elite=a[:0] if iteration>=len(self.elites) else batch[self.elites[iteration]].cpu().numpy()
            first=np.linalg.norm(delta[:,0],axis=1);flat=a.reshape(len(a),-1)
            eliteflat=elite.reshape(len(elite),-1)
            distances=np.linalg.norm(eliteflat[:,None]-eliteflat[None,:],axis=-1)
            clip=(raw.abs()>1).cpu().numpy()
            rows.append(dict(metadata,iteration=iteration,unique_candidate_ratio=len(np.unique(flat,axis=0))/len(flat),
                unique_first_action_ratio=len(np.unique(a[:,0],axis=0))/len(a),first_action_distance_mean=float(first.mean()),
                first_action_distance_p95=float(np.quantile(first,.95)),full_sequence_rmse=float(np.sqrt(np.mean(delta**2))),
                steering_variance=float(a[:,:,0].var()),throttle_variance=float(a[:,:,1].var()),
                clipping_ratio=float(clip.mean()),steer_clipping=float(clip[:,:,0].mean()),throttle_clipping=float(clip[:,:,1].mean()),
                sac_exact_first_overlap=float(np.all(a[:,0]==b[0],axis=1).mean()),sac_near_first_overlap=float((first<=.01).mean()),
                elite_diversity=float(distances[np.triu_indices(len(elite),1)].mean()) if len(elite)>1 else 0.,
                input_std_steer=float(std[:,0].mean()),input_std_throttle=float(std[:,1].mean()),
                q_pair=list(members[iteration]),predicted_value_range=float(np.ptp(values[iteration])),
                exact_preclip_reconstruction=True,near_overlap_threshold=.01))
        assert torch.equal(original_state,p.generator.get_state())
        return rows
