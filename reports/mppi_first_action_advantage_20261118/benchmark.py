"""Head-only CPU/GPU timings; never presented as complete MPPI latency."""
import copy,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from study_io import R,OLD,load,dump,table
from evaluate import load_relative
from models import features
import numpy as np
import torch
from torch import nn
from offline_model import load_model


class A0Centered(nn.Module):
    def __init__(self,model,baseline):
        super().__init__();self.model=model;self.register_buffer('baseline',torch.tensor(baseline))
    def forward(self,x):
        b=x.clone();b[:,-2:]=self.baseline
        return self.model(x)-self.model(b)


@torch.no_grad()
def main():
    torch.set_num_threads(1);p=load(R/'protocol.json')
    meta=load(R/'model_manifest.json');c=np.load(OLD/'origins/90115_50/catalog.npz')
    a0,_=load_model(p['A0_checkpoint'],p['A0_context'])
    models={'A0_frozen':A0Centered(a0,c['baseline'][0])}
    for name in ['A1_linear','A1_mlp','A2_linear','A2_mlp','A2_knn']:
        models[name]=load_relative(next(m for m in meta if m['name']=='full_'+name))
    inputs={s:np.stack([features(c['state535'],c['baseline'][0],a,s) for a in c['pool'][:,0]]) for s in ['A0','A1','A2']}
    rows=[]
    for name,base in models.items():
        schema=name[:2]
        for device in ['cpu','cuda']:
            if device=='cuda' and not torch.cuda.is_available():continue
            before=torch.cuda.memory_allocated() if device=='cuda' else 0
            model=copy.deepcopy(base).to(device).eval().requires_grad_(False)
            footprint=torch.cuda.memory_allocated()-before if device=='cuda' else 0
            for batch in p['benchmark']['batches']:
                x=torch.tensor(inputs[schema][:batch],device=device)
                for _ in range(p['benchmark']['warmup']):model(x)
                if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                times=[];kernel=[]
                for _ in range(p['benchmark']['repeats']):
                    if device=='cuda':
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize();start=time.perf_counter();begin.record();model(x);end.record();end.synchronize()
                        times.append((time.perf_counter()-start)*1000);kernel.append(begin.elapsed_time(end))
                    else:
                        start=time.perf_counter();model(x);times.append((time.perf_counter()-start)*1000)
                peak=torch.cuda.max_memory_allocated() if device=='cuda' else 0
                cpu=base(torch.tensor(inputs[schema][:batch])).numpy()
                output=model(x).cpu().numpy()
                rows.append(dict(model=name,device=device,batch=batch,parameters=sum(t.numel() for t in model.parameters()),
                    wall_p50_ms=float(np.quantile(times,.5)),wall_p95_ms=float(np.quantile(times,.95)),wall_p99_ms=float(np.quantile(times,.99)),
                    cuda_kernel_p50_ms=float(np.median(kernel)) if kernel else None,model_tensor_bytes=footprint,
                    peak_process_tensor_bytes=peak,cpu_gpu_max_abs_difference=float(np.max(np.abs(cpu-output))),
                    repeats=p['benchmark']['repeats'],scope='Head-only including centering; inputs already on device. NOT whole Lattice/SAC/MPPI time.'))
            del model
    table('inference_benchmark.csv',rows)
    dump('inference_benchmark.json',dict(rows=rows,cpu_threads=1,device_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        transfer_included=False,CUDA_tensor_memory_not_full_driver_process_memory=True,not_complete_planner_latency=True))
    print('Inference benchmark complete:',len(rows),'configurations.',flush=True)


if __name__=='__main__':main()
