# 源码发布：以本地 main 为准

算法、配置、场景和历史结果公开；训练好的权重保留在本地，不纳入当前版本。

```bash
(
set -euo pipefail
cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
test "$(git branch --show-current)" = main
git rm --cached --ignore-unmatch -- models/wm200k_rl300k.pt
git add -A
python - <<'CHECK'
import subprocess
from pathlib import Path
blocked={'.pt','.pth','.ckpt','.safetensors','.onnx'}
paths=subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
weights=[p for p in paths if p and Path(p).suffix.lower() in blocked]
if weights: raise SystemExit('停止上传，仍有已跟踪权重：'+repr(weights))
assert Path('models/wm200k_rl300k.pt').is_file(), '本地原权重缺失'
print('检查通过：当前提交不含权重，本地原权重存在。')
CHECK
git diff --cached --shortstat
if ! git diff --cached --quiet; then
  git commit -m "Publish reproducible source and keep trained checkpoints local"
fi
git push -u origin main
publish_local_sha="$(git rev-parse HEAD)"
publish_remote_sha="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
test "$publish_local_sha" = "$publish_remote_sha"
printf '源码同步成功：%s\n' "$publish_local_sha"
)
```

当前远端 main 与本地同步，正常推送即可。若远端在此期间出现新提交导致拒绝推送，应先核对差异，不使用强制覆盖。

`git rm --cached` 仅移除版本跟踪，不删除本地权重。已上传的旧提交和 LFS 对象不会随当前版本删除自动清除，本流程不改写历史。
