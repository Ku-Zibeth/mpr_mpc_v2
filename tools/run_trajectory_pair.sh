#!/usr/bin/env bash
# Foreground B/D evaluation. Smoke acceptance is required by the task protocol.
set -euo pipefail
root="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
cd "$root"
python="${PYTHON:-/home/kzb/anaconda3/envs/tdmpc2/bin/python}"
checkpoint="$root/logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm200k_rl300k_20261103_195237/models/milestone_500k_full_mpr_mpc.pt"
name="${EXP_NAME:-trajectory_filter_BD_fixed100_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
out="$root/logs/benchmark/$name"
dry_run=false
if [[ $# -gt 0 ]]; then
    [[ $# -eq 1 && "$1" == --dry-run ]] || { printf 'Usage: bash %s [--dry-run]\n' "$0" >&2; exit 2; }
    dry_run=true
fi
[[ -x "$python" && -f "$checkpoint" ]] || { printf 'Python or checkpoint missing.\n' >&2; exit 1; }
[[ ! -e "$out" ]] || { printf 'Refusing to overwrite %s\n' "$out" >&2; exit 1; }
command=("$python" -u evaluate_ablation.py +experiment=E5 "checkpoint=$checkpoint"
    restore_checkpoint_runtime=true restore_mppi_runtime=false
    seed=1 eval_episodes=100 enable_wandb=false save_video=false
    metadrive.simulator.start_seed=300 metadrive.simulator.num_scenarios=2000
    metadrive.simulator.horizon=1000 metadrive.simulator.use_render=false
    mpr_mpc.mppi.allow_unvalidated_cost=false
    '+benchmark.variants=[B_lattice_sac,D_lattice_sac_mppi]'
    +benchmark.cost_ablation=false +benchmark.episodes=100 '+benchmark.repeat_seeds=[1]'
    "+benchmark.scenarios_file=$root/scenarios/fixed100.json"
    +benchmark.scenario_start=300 +benchmark.scenario_pool_count=2000 +benchmark.scenario_seed=1
    +benchmark.bootstrap_samples=2000 +benchmark.bootstrap_seed=1 +benchmark.mppi_cost_coef=1.0
    +benchmark.save_step_diagnostics=true "+benchmark.output_dir=$out")
if "$dry_run"; then
    printf 'Preview only; no evaluation or acceptance bypass is executed.\n'
    printf '%q ' "${command[@]}"; printf '\n'
    exit 0
fi
"$python" - "$checkpoint" <<'PY'
import hashlib, json, sys
from pathlib import Path
from tools.benchmark_mppi import source_fingerprint
report = Path('reports/round3/acceptance_final.json')
if not report.is_file():
    raise SystemExit('No smoke acceptance report. Run and validate the prescribed unit/closed-loop tests first.')
data = json.loads(report.read_text())
if not data.get('formal_evaluation_ready', False):
    raise SystemExit('Formal evaluation blocked by the prompt acceptance criteria:\n- ' + '\n- '.join(data['blocking_reasons']))
if source_fingerprint(Path.cwd()) != data['source_snapshot_sha256']:
    raise SystemExit('Runtime source changed since acceptance; rerun the bounded regression before formal evaluation.')
checkpoint = Path(sys.argv[1])
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != data['checkpoint_sha256']:
    raise SystemExit('Checkpoint differs from the accepted smoke checkpoint.')
PY
mkdir -p "$out"
printf '%q ' "${command[@]}" > "$out/command.sh"; printf '\n' >> "$out/command.sh"
printf 'Checkpoint: %s\nResults: %s\n' "$checkpoint" "$out"
MUJOCO_GL=egl PYTHONUNBUFFERED=1 "${command[@]}" 2>&1 | tee "$out/run.log"
