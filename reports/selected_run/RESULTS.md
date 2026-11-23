# Retained model: collision-free completion

Success = native destination reached AND no collision at any point in the episode. An episode that both collides and fails is counted once.
Native termination, complete-episode return, safety cost and controls are unchanged. Native arrival is retained as `native_success`; `final_success` is the strict endpoint.
Environment: traffic_density=0.075; accident_prob=0.8; body departure tolerance=0.1m. Same100 scenarios and seed20261127 for all groups.

| Group | Native arrival | Collision then arrival | Strict success | Failure |
|---|---:|---:|---:|---:|
| A_lattice | 83% | 8 | 75% | 25 |
| B_lattice_sac | 92% | 11 | 81% | 19 |
| C_lattice_mppi | 89% | 16 | 73% | 27 |
| D_lattice_sac_mppi | 96% | 6 | 90% | 10 |

This is the retained result on previously used maps, not a new independent validation. The source checkpoint and native400-episode CSV are retained for reproducibility.

Rebuild this report: `python tools/summarize_selected.py`. Run a fresh evaluation: `bash scripts/evaluate_selected.sh`.
