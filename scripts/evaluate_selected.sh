#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
if [[ -x /home/kzb/anaconda3/envs/tdmpc2/bin/python ]]; then
  selected_python=/home/kzb/anaconda3/envs/tdmpc2/bin/python
else
  selected_python=python
fi
printf '保留版：WM200k＋RL300k；周车密度0.075；越界容忍0.1米；到达且无碰撞才算成功\n'
"$selected_python" -u evaluate.py --resume "$@"
