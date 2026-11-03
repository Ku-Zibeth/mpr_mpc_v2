"""Separate diagnostic pass. CUDA events measure stream elapsed, not kernel time."""

from contextlib import ExitStack, nullcontext
from functools import wraps
import time
from unittest.mock import patch

import torch


class PlanningProfiler:
    def __init__(self, planner, *, record_ranges=False):
        self.planner = planner
        self.records = {}
        self.stack = ExitStack()
        self.record_ranges = record_ranges
        self.phases = set()

    def wrap(self, owner, method, phase):
        self.phases.add(phase)
        original = getattr(owner, method)

        @wraps(original)
        def measured(*args, **kwargs):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            try:
                with torch.profiler.record_function(phase) if self.record_ranges else nullcontext():
                    return original(*args, **kwargs)
            finally:
                end.record()
                self.records.setdefault(phase, []).append(((time.perf_counter()-started)*1000, begin, end))

        self.stack.enter_context(patch.object(owner, method, measured))

    def __enter__(self):
        from mpr_mpc.planning.multimodal import MultiModalProposals
        p = self.planner
        self.wrap(p, "plan", "total_planning")
        self.wrap(p, "prepare", "context_construction")
        self.wrap(p.local_mppi, "_candidate_batch", "candidate_generation")
        self.wrap(p.safety, "check", "safety_check")
        for method, phase in (("_road_metrics", "road_geometry"), ("_swept_check", "continuous_collision")):
            if hasattr(p.safety, method):
                self.wrap(p.safety, method, phase)
        self.wrap(p.evaluator, "evaluate", "world_model_evaluation")
        self.wrap(p.evaluator.agent.model, "cost", "cost_evaluation")
        self.wrap(p.local_mppi, "_score_candidates" if hasattr(p.local_mppi, "_score_candidates") else "_penalties", "candidate_scoring")
        self.wrap(MultiModalProposals, "update", "distribution_update")
        return self

    def report(self):
        torch.cuda.synchronize()
        phases = ("context_construction", "candidate_generation", "safety_check", "world_model_evaluation",
                  "cost_evaluation", "candidate_scoring", "distribution_update", "total_planning",
                  "road_geometry", "continuous_collision")
        result = {name: dict(cpu_wall_ms=sum(r[0] for r in self.records.get(name, [])),
                            cuda_stream_elapsed_ms=sum(r[1].elapsed_time(r[2]) for r in self.records.get(name, [])))
                  for name in phases if name in self.phases}
        self.records.clear()
        return result

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)
