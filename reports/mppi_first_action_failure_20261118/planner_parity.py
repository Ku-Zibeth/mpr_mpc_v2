"""Run unchanged default LocalMPPI once, without MetaDrive or policy fitting."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ap = argparse.ArgumentParser()
ap.add_argument('--code-root', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
sys.path.insert(0, str(args.code_root))
import numpy as np
import torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.world_model import WorldModel
from tdmpc2 import TDMPC2
from planning.evaluator import TrajectoryWorldModelEvaluator
from planning.local_mppi import LocalMPPI
import evaluate_model_suite as suite

old = Path(__file__).resolve().parents[2] / 'reports/mppi_first_action_value_20261117'
p = json.loads((old/'protocol.json').read_text())
raw = OmegaConf.load(p['source_environment_config'])
raw.mpr_mpc.mppi = OmegaConf.load(args.code_root/'config.yaml').mpr_mpc.mppi
raw.restore_checkpoint_runtime, raw.restore_mppi_runtime = True, False
ck = Path(p['checkpoint'])
_, cfg = suite._make_cfg(raw, ck.parent.parent.name, ck, [1])
cfg.obs_shape, cfg.action_dim, cfg.episode_length = {'state': (259,)}, 2, 1000
torch.set_num_threads(1)
torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    model = WorldModel(cfg).to(device).eval().requires_grad_(False)
    agent = SimpleNamespace(model=model, cfg=cfg, device=device, discount=.995)
    TDMPC2.load(agent, {'model':torch.load(ck, map_location='cpu', weights_only=False)['model']})
    evaluator = TrajectoryWorldModelEvaluator(agent, 10)
    planner = LocalMPPI(evaluator, dict(raw.mpr_mpc.mppi), horizon=10, action_dim=2, device=device)
    c = np.load(old/'origins/90115_50/catalog.npz')
    torch.manual_seed(20261118)
    planner.reset(1)
    result = planner.plan(torch.tensor(c['z0'], device=device), torch.tensor(c['baseline'], device=device), eval_mode=True)
    assert not args.output.exists()
    np.savez_compressed(args.output, selected=result.selected_sequence.cpu().numpy(),
                        baseline_selected=bool(result.baseline_selected), numerical_error=bool(result.numerical_error),
                        baseline=c['baseline'])
    print('Default planner offline parity output saved:', args.code_root, flush=True)
