"""Freeze all16 origins and sampling/statistical definitions before H1 outcomes."""
import importlib.metadata as md,subprocess,sys
from study import *

def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert commit=='12da6bbfd40138b839eab35b0bd6d36dbb964eac'
    previous=load(SOURCE/'protocol.json');sources={}
    for scene in previous['scenarios']:
        old=original_dir(scene)
        for name in ['trace.npz','config.yaml','episode.json','snapshot_50.npz','snapshot_150.npz','snapshot_50_origin.json','snapshot_150_origin.json']:
            f=old/name;assert f.is_file();sources[str(f)]=file_sha(f)
    ck=Path(previous['checkpoint']);assert file_sha(ck)==previous['checkpoint_sha256']
    p=dict(base_commit=commit,remote_main_commit=commit,source=str(SOURCE),checkpoint=str(ck),checkpoint_sha256=file_sha(ck),
        source_environment_config=previous['source_environment_config'],scenes=previous['scenarios'],steps=[50,150],origins=16,
        global_seed='scene+1',repeat_seed=1,horizon=1,control_dt=.1,gamma=.995,continuation='original frozen Lattice+SAC to native done',
        decision_boundary='Rebuild prefix step-1 with agent calls and recorded actions; compute target Baseline once; match old post-decision/pre-execution snapshot. Branch starts at this exact boundary.',
        candidate_roles=dict(A0='Baseline SAC',A1='C native selected first action',A2='D Cartesian virtual speed selected first action',
            A3='C last iteration Gaussian max',A4='C last iteration Gaussian q75',A5='C last iteration Gaussian q50',A6='C last iteration Gaussian q25',A7='C last iteration Gaussian min'),
        gaussian_quantiles=[1.,.75,.5,.25,0.],quantile_method='stable ascending raw last-iteration values; nearest index round(q*63), exclude baseline row0',
        dedup_l2=1e-5,pair_tie_return=1.,formal_h1_ranking='NOT ESTIMABLE: no independently validated value for the deployed SAC continuation',
        descriptive_prediction='Original raw H10 scores retain Q batch indices; same-Q H10 re-evaluation only descriptive, not H1 ranking accuracy',
        mppi_budget=[64,8,4],candidate_rng='fresh native private generator at scene+1; C/D share pre-target Torch RNG, branch RNG is restored post-Baseline decision',
        phases=['one origin restoration','one origin A0/A1 plus >=3 Gaussian branches','remaining origins'],pilot_origin=[previous['scenarios'][0],50],
        bootstrap=dict(unit='whole map, both origins together',samples=2000,seed=20261118),
        worker_limit=4,training=0,formal100=False,online_changes=False,commit=False,push=False)
    dump(R/'protocol.json',p)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    protected={name:file_sha(ROOT/name) for name in tracked if name and not name.startswith('reports/') and (ROOT/name).is_file()}
    dump(R/'experiment_manifest.json',dict(commit=commit,branch='experiment/mppi-h1-counterfactual-20261118',
        main_status_start=subprocess.check_output(['git','status','--porcelain'],cwd=MAIN,text=True),checkpoint_sha256=p['checkpoint_sha256'],
        python=sys.version,packages={k:md.version(k) for k in ['torch','numpy','metadrive-simulator','omegaconf']},
        sources=sources,protected_sources=protected,protocol_sha256=file_sha(R/'protocol.json'),
        notes='Eight previously used development maps explicitly requested; not an independent validation set.'))
    print('PROTOCOL_FROZEN',commit,len(sources),flush=True)
if __name__=='__main__':main()
