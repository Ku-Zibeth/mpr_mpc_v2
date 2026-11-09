"""Separate diagnostic pass. CUDA events measure stream elapsed, not kernel time."""

from contextlib import ExitStack, nullcontext
from functools import wraps
import time
from unittest.mock import patch

import torch


def kernel_trace_summary(payload, phases):
    """Attribute actual kernels via launch IDs, never GPU annotation spans."""
    events = payload["traceEvents"]
    operators = {event["args"]["External id"]: event for event in events
                 if event.get("cat") == "cpu_op" and event.get("args", {}).get("External id") is not None}
    launches = {event["args"]["correlation"]: event for event in events
                if event.get("cat") == "cuda_runtime" and event.get("args", {}).get("correlation") is not None}
    kernels = [event for event in events if event.get("cat") == "kernel"]
    producers = [operators.get(event.get("args", {}).get("External id"))
                 or launches.get(event.get("args", {}).get("correlation")) for event in kernels]
    result = {}
    for phase in sorted(phases):
        ranges = [event for event in events if event.get("cat") == "user_annotation" and event.get("name") == phase]
        milliseconds = sum(event.get("dur", 0) for event, producer in zip(kernels, producers) if producer is not None
            and any(producer.get("pid") == region.get("pid") and producer.get("tid") == region.get("tid")
                    and region["ts"] <= producer["ts"] < region["ts"]+region["dur"] for region in ranges))/1000
        result[phase] = dict(cpu_range_ms=sum(region["dur"] for region in ranges)/1000, cuda_kernel_ms=milliseconds)
    return result, dict(all_cuda_kernel_ms=sum(event.get("dur", 0) for event in kernels)/1000,
                       unattributed_kernels=sum(producer is None for producer in producers))


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
        p = self.planner
        self.wrap(p, "plan", "total_planning")
        self.wrap(p, "prepare", "context_construction")
        self.wrap(p.local_mppi, "_candidate_batch", "candidate_generation")
        candidate_safety = getattr(p.local_mppi, "candidate_safety", None)
        if candidate_safety is not None and p.local_mppi.safety_enabled:
            self.wrap(candidate_safety, "check", "mppi_independent_safety")
            self.wrap(candidate_safety.geometry, "_swept_check", "mppi_continuous_collision")
            self.wrap(candidate_safety.geometry, "rollout", "mppi_kinematic_rollout")
        if p.safety is not None:
            self.wrap(p.safety, "check", "sac_feature_geometry")
        for method, phase in (("_road_metrics", "road_geometry"), ("_swept_check", "continuous_collision")):
            if hasattr(p.safety, method):
                self.wrap(p.safety, method, phase)
        self.wrap(p.evaluator, "evaluate", "world_model_evaluation")
        self.wrap(p.evaluator.agent.model, "cost", "cost_evaluation")
        self.wrap(p.local_mppi, "_score_candidates" if hasattr(p.local_mppi, "_score_candidates") else "_penalties", "candidate_scoring")
        return self

    def report(self):
        torch.cuda.synchronize()
        phases = ("context_construction", "candidate_generation", "sac_feature_geometry", "world_model_evaluation",
                  "cost_evaluation", "candidate_scoring", "distribution_update", "total_planning",
                  "road_geometry", "continuous_collision")
        phases += ("mppi_independent_safety", "mppi_continuous_collision", "mppi_kinematic_rollout")
        result = {name: dict(cpu_wall_ms=sum(r[0] for r in self.records.get(name, [])),
                            cuda_stream_elapsed_ms=sum(r[1].elapsed_time(r[2]) for r in self.records.get(name, [])))
                  for name in phases if name in self.phases}
        self.records.clear()
        return result

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)
