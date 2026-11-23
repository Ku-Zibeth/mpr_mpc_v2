# 同步保留版到GitHub

本地整合已完成；下列命令由你执行，才会创建提交并上传。模型通过Git LFS同步；原实验目录已移除，所有操作都在主目录执行。

```bash
(
  set -euo pipefail
  cd /home/kzb/kzb_code/tdmpc2/mpr_mpc_v2
  test "$(git branch --show-current)" = main

  git lfs install --local
  git add -A
  git diff --cached --shortstat
  git lfs ls-files

  if ! git diff --cached --quiet; then
    git commit -m "Consolidate selected MPR-MPC model and collision-free evaluation"
  fi

  git pull --rebase origin main
  git push -u origin main

  release_local_sha="$(git rev-parse HEAD)"
  release_remote_sha="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
  test "$release_local_sha" = "$release_remote_sha"
  printf '同步完成，本地与GitHub main一致：%s\n' "$release_local_sha"
  git log -1 --oneline
)
```

本机已安装Git LFS3.4.1，权重已通过本地clean过滤器校验：

```text
version https://git-lfs.github.com/spec/v1
oid sha256:4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a
size 113814831
```

SSH远程沿用现有 `ssh://git@ssh.github.com:443/Ku-Zibeth/mpr_mpc_v2.git`。Git LFS的SSH服务发现与认证流程见[官方说明](https://github.com/git-lfs/git-lfs/blob/main/docs/api/server-discovery.md)。
