import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
from tools.benchmark_mppi import source_fingerprint
old=json.loads((ROOT/'reports/mppi_value_20261116/protocol.json').read_text())
scenes=np.random.default_rng(2026111601).choice(np.arange(40000,41000),16,replace=False).tolist()
p=dict(base_commit='fae10f34e15ffe987a7527272cb1693637a9f488',checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],source_environment_config=old['source_environment_config'],
 source_snapshot_sha256=source_fingerprint(ROOT),dev_scenarios=scenes[:10],validation_scenarios=scenes[10:],origin_steps=[50,150],
 elite_scenarios=scenes[::4],groups={'Q1':1.,'Q05':.5,'Q025':.25,'Q0':0.},seed=1,episode_horizon=1000,horizon=10,
 effective_beta=2.,temperature_definition='inverse',terminal_action='candidate',q_aggregation='ensemble_mean',
 closed_loop_dev_scenarios=scenes[:4],closed_loop_validation_scenarios=scenes[10:],
 additional_test_scenarios=np.random.default_rng(2026111602).choice(np.arange(50000,51000),10,replace=False).tolist(),
 selection='Select only Q05 or Q025 on development origins AND development closed-loop, requiring actual mean return > Q1 and B, no higher crash/OOR, closed-loop success >= both. Rank eligible by actual discounted return, then success. Q0 is diagnostic only. Validate single selected candidate against B and Q1 on held-out scenes. No selection on old9 or old20.',
 validation_gate='New final-test scenes only if held-out validation shows success >= B/Q1, return > both, crash/OOR <= both and paired return CI95 lower bound > 0 against both. Otherwise stop; no extra coefficients or full100.',
 scope='32 planned origins across16 new scenes, split by scene:20 development/12 validation. Unreachable SAC prefixes recorded, not replaced based on outcomes. Four predeclared scenes also execute Q1 best recorded elite. Observed+traffic+RNG matching is not a complete hidden simulator snapshot.',no_training=True)
assert hashlib.sha256(Path(p['checkpoint']).read_bytes()).hexdigest()==p['checkpoint_sha256']
assert not set(scenes)&set(old['validation_scenarios']+old['sealed_test_scenarios'])
assert min(scenes)>11999
path=R/'protocol.json';assert not path.exists();path.write_text(json.dumps(p,indent=2))
print(scenes)
