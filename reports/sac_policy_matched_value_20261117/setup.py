"""Predeclare evaluation-only states and all value definitions before new outcomes."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np
from tools.benchmark_mppi import source_fingerprint
old=json.loads((ROOT/'reports/mppi_selection_20261117/protocol.json').read_text())
scenes=np.random.default_rng(2026111711).choice(np.arange(83000,84000),4,replace=False).tolist()
p=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],source_environment_config=old['source_environment_config'],
    runtime_fingerprint=source_fingerprint(ROOT),fresh_scenarios=scenes,before=30,horizon=10,
    candidates=['baseline','gaussian1'],candidate_selection='Fixed baseline and first of64 originalGaussian draws; NEVER chosen by Return',
    primary_gamma=.995,secondary_gamma=.99,repeat_seeds=[1,2],
    takeover_policies=['lattice_sac','td_actor'],
    td_actor='Original stochastic model.pi(z)[0] direct physical action; noTD-MPC2 planner. Not deterministic mean; NOT TD-MPC2 fullMPC.',
    lattice_sac='Original deterministic eval_mode=True; MPPI OFF. Use alreadyprepared first action then recompute eachstep.',
    paired_rng='Same scene/pre-origin/prefix, same recorded terminalstate/observation/traffic/RNG. Suffix global seeds1/2 applied identically to both policies AFTER terminalstate is recorded. Traffic internalRNG remains scenario/prefix fixed.',
    variance='Two suffix seeds measure policy/coarse-WM-feature stochasticity and any environment interaction, NOT isolated traffic noise. Exact same-seed SAC replay for firstscene baseline separately checks reproducibility.',
    terminal_value_primary='Original TDQ01 at ACTUAL z_H and actual deterministic Lattice+SAC first physical action. SACmin(Q1,Q2) at actual535 and deterministic residual actor action mapping to that physical action. SAMEphysical first action; native continuation policy/gamma/entropy differences remain explicit.',
    q_indices=[0,1],terminal_comparison='continuation RTG starting immediately afterH, no prefix rewards mixed in',
    controls='All originalweights/535/features/mapping/config/reward/native termination frozen. No onlineMPPI, no training, no heads, no rescaling critics.',
    historical_data='197 Fit/InnerStop cachedprefix-replay states, retrospective only. Preserve originalseen validation and sealed6map set; do not train or select onfreshmaps.',
    acceptance='Existing frozen SACcritic considered promising only if freshscene pairedaccuracy improvement CI95low>0 AND top1regret improvement CI95low>0; no MAE-only approval; independent4scene pilot insufficient for online deployment.',
    bootstrap_samples=2000,bootstrap_seed=1,return_tolerance=1.,
    calibration_edges=['-inf',-200,-100,0,50,100,150,200,300,500,'inf'],
    new_native_suffixes=33,formal100=False,training=False,default_changes=False)
assert not (R/'protocol.json').exists();(R/'protocol.json').write_text(json.dumps(p,indent=2));print('fresh predeclared scenes',scenes,flush=True)
