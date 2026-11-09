#!/usr/bin/env bash
set -euo pipefail
root="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
cd "$root"
python="${PYTHON:-/home/kzb/anaconda3/envs/tdmpc2/bin/python}"
checkpoint="${CHECKPOINT:-$root/logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm200k_rl300k_20261103_195237/models/milestone_500k_full_mpr_mpc.pt}"
name="${EXP_NAME:-mpr_mpc_v31_ablation_fixed100_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
out="$root/logs/benchmark/$name"
dry_run=false
if [[ $# -gt 0 ]]; then
    [[ $# -eq 1 && "$1" == --dry-run ]] || { printf 'Usage: bash %s [--dry-run]\n' "$0" >&2; exit 2; }
    dry_run=true
fi
[[ -x "$python" && -f "$checkpoint" ]] || { printf 'Python or checkpoint missing.\n' >&2; exit 1; }
[[ ! -e "$out" ]] || { printf 'Output already exists: %s\n' "$out" >&2; exit 1; }
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 WANDB_CONSOLE=off WANDB__REQUIRE_CORE=true
if [[ "${ENABLE_WANDB:-false}" == true ]]; then
    proxy="${MPR_PROXY-http://127.0.0.1:7897}"
    if [[ -n "$proxy" ]]; then
        export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy" http_proxy="$proxy" https_proxy="$proxy"
    fi
    export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
fi
command=("$python" -u evaluate_ablation.py +experiment=E5 "checkpoint=$checkpoint"
    restore_checkpoint_runtime=true restore_mppi_runtime=false
    seed=1 eval_episodes=100 enable_wandb=false save_video=false
    mpr_mpc.mppi.safety.enabled=true
    mpr_mpc.mppi.recovery.enabled=true
    "mpr_mpc.mppi.allow_unvalidated_cost=${ALLOW_UNVALIDATED_COST:-false}"
    wandb_entity=shihuangggggg-southeastern-university wandb_project=mpr_mpc_metadrive
    +benchmark.cost_ablation=false +benchmark.episodes=100
    "+benchmark.repeat_seeds=${REPEAT_SEEDS:-[1]}"
    "+benchmark.scenarios_file=$root/scenarios/fixed100.json"
    +benchmark.scenario_start=300 +benchmark.scenario_pool_count=2000 +benchmark.scenario_seed=1
    +benchmark.bootstrap_samples=2000 +benchmark.bootstrap_seed=1 +benchmark.mppi_cost_coef=1.0
    +benchmark.save_step_diagnostics=true "+benchmark.enable_wandb=${ENABLE_WANDB:-false}"
    "+benchmark.output_dir=$out")
if "$dry_run"; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
mkdir -p "$out"
printf '%q ' "${command[@]}" > "$out/command.sh"
printf '\n' >> "$out/command.sh"
printf '[launcher] V3.1 A/B/C/D: same frozen checkpoint, fixed 100 scenes, no training.\n' > "$out/run.log"
nohup "${command[@]}" </dev/null >>"$out/run.log" 2>&1 &
printf '%s\n' "$!" > "$out/evaluation.pid"
printf 'Evaluation PID: %s\nResults: %s\nCtrl+C stops this log viewer, not evaluation.\n' "$(<"$out/evaluation.pid")" "$out"
tail -n 40 -F "$out/run.log"
