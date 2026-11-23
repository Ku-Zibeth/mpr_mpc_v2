# Retained checkpoint

`wm200k_rl300k.pt` is the exact original WM200k + frozen-WM Residual SAC300k full checkpoint (113814831 bytes).
SHA256: `4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a`.

The checkpoint is tracked with Git LFS. After cloning, run `git lfs install` and `git lfs pull`.
No model tensor, optimizer state, SAC535 input, action mapping or Lattice generation was changed by consolidation.
`manifest.json` identifies the retained configuration, source runtime hashes and expected reference results.
