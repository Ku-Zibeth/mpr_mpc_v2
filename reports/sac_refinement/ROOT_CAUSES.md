# SAC-centred refinement: audit before editing

Baseline: main `3e404e439d1b07aff3979bd5e3f1781c40233629`; clean worktree.

| Actual path | Problem | Required correction |
|---|---|---|
| coordinator.refine_and_plan -> LocalMPPI.plan(refined_actions) | SAC enters correctly but stops being the final reference after a recovery trigger. | Preserve this input, including exact baseline bytes, through final acceptance. |
| plan: normal loop -> elite_update -> verified_choice | Preview-only high scorers update the distribution before full verification. | Full-check a bounded shortlist first; only certified elites update it. |
| recovery trigger: not baseline_safe OR best is None OR stall/latch | Tail disagreement/search failure becomes an independent driving-mode trigger. | Separate normal near-safe refinement, imminent-safety correction, and measured trapped state; no best=None trigger. |
| recovery lattice Gaussian; recovery_scoring/Pareto | Changes the search centre and objective, forcibly disables baseline selection. | Remove normal recentering/Pareto; one SAC-relative J, meaningful improvement gate. |
| min_progress_m and recovery latch | Absolute short-horizon progress and missing projections can override low-speed SAC. | Relative progress only; missing projection never means stopped. |
| Center + Legacy safety guard | AND may falsely reject physically executed trajectories; recall evidence is insufficient. | Keep both standards; log disagreement as model uncertainty, not proven incident. |
| evaluator + Cost validator | Cost ranking has no approval; prediction error is not candidate ranking. | Keep allow_unvalidated_cost=false; do not change Cost gate. Audit Value with real paired outcomes separately. |

Historical B/D100 (20261109_202823): success 75%/8%, route completion
88.15%/38.32%, D timeout 90%. These are old results, not this revision's results.
`reports/round3/acceptance_final.json` is short regression evidence only, not success-rate evidence.
Existing motion holdout has no real positive accident events: recall cannot be established,
and it does not justify replacing the AND guard with OR. Preserve analytic circular-lane
seam repair (source-proven internal tessellation holes; no boundary expansion).

Architecture, checkpoint, SAC state/action mapping and training settings remain outside this change.
