#!/usr/bin/env bash
set -euo pipefail

root="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
cd "$root"
python="${PYTHON:-/home/kzb/anaconda3/envs/tdmpc2/bin/python}"
checkpoint="${WM_CHECKPOINT:-$root/logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm500k_rl500k_20261105_215414/models/milestone_500k_world_model.pt}"
name="${EXP_NAME:-mpr_mpc_v2_1.0_wm500k_rl500k_from_wm_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
out="$root/logs/launch/$name"
dry_run=false
if [[ $# -gt 0 ]]; then
    [[ $# -eq 1 && "$1" == --dry-run ]] || { printf 'Usage: bash %s [--dry-run]\n' "$0" >&2; exit 2; }
    dry_run=true
fi
[[ -x "$python" ]] || { printf 'Python not found: %s\n' "$python" >&2; exit 1; }
[[ -f "$checkpoint" ]] || { printf 'Checkpoint not found: %s\n' "$checkpoint" >&2; exit 1; }
[[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'Invalid experiment name.\n' >&2; exit 1; }

export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online WANDB__REQUIRE_CORE=true
export WANDB_CONSOLE=off WANDB_INIT_TIMEOUT=60
proxy="${MPR_PROXY-http://127.0.0.1:7897}"
if [[ -n "$proxy" ]]; then
    export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy" http_proxy="$proxy" https_proxy="$proxy"
fi
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

command=("$python" -u train.py +experiment=E5 +schedule=frozen_500k_1m
    "exp_name=$name" seed=1 checkpoint=null "resume_checkpoint=$checkpoint" resume_replay_warmup=true
    enable_wandb=true wandb_log_artifacts=false
    wandb_entity=shihuangggggg-southeastern-university wandb_project=mpr_mpc_metadrive
    "wandb_run_name=$name" save_agent=true save_csv=true save_video=false
    metadrive.simulator.start_seed=10000 metadrive.simulator.num_scenarios=2000)
if "$dry_run"; then
    printf 'WANDB_CONSOLE=%s\n' "$WANDB_CONSOLE"
    printf '%q ' "${command[@]}"
    printf '\nOutput: %s/run.log (direct file, no pipe)\n' "$out"
    exit 0
fi

for pid in $(pgrep -f '[p]ython.*train.py' || true); do
    if [[ "$(readlink "/proc/$pid/cwd" 2>/dev/null || true)" == "$root" ]]; then
        printf 'Another training process is active (PID %s). Stop it before launching a new run.\n' "$pid" >&2
        exit 1
    fi
done
[[ ! -e "$out" && ! -e "$root/logs/metadrive-risk/1/$name" ]] || {
    printf 'Experiment already exists: %s\n' "$name" >&2; exit 1;
}
models="$root/logs/metadrive-risk/1/$name/models"
mkdir -p "$out" "$models"
cp --reflink=auto -- "$checkpoint" "$models/milestone_500k_world_model.pt"
printf '%q ' "${command[@]}" > "$out/command.sh"
printf '\n' >> "$out/command.sh"
printf '[launcher] Starting Python runtime; world model: %s\n[launcher] Frozen WM 500k + fresh SAC 500k; global target 1000000.\n' "$checkpoint" > "$out/run.log"
nohup "${command[@]}" </dev/null >>"$out/run.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$out/train.pid"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
    tail -n 50 "$out/run.log"
    printf 'Training exited during startup. Check %s/run.log\n' "$out" >&2
    exit 1
fi
printf 'Started PID: %s\nCheckpoint: %s\nModels: %s/logs/metadrive-risk/1/%s/models\n' "$pid" "$checkpoint" "$root" "$name"
printf 'Log: %s/run.log\nMonitor: tail -n 40 -F %q\n' "$out" "$out/run.log"
