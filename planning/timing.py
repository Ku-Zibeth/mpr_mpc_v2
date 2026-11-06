"""Per-phase CPU wall and CUDA-stream timings without per-phase synchronizations."""

from collections import defaultdict
from contextlib import contextmanager
import time

import torch


class PhaseTimings:
    def __init__(self, device):
        self.device = torch.device(device)
        self.cpu = defaultdict(float)
        self.events = []

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        pair = None
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                pair[0].record()
        try:
            yield
        finally:
            if pair is not None:
                pair[1].record()
                self.events.append((name, *pair))
            self.cpu[name] += 1000 * (time.perf_counter() - started)

    def finish(self):
        result = {f"{name}_cpu_wall_ms": value for name, value in self.cpu.items()}
        if self.events:
            torch.cuda.synchronize(self.device)
            elapsed = defaultdict(float)
            for name, start, end in self.events:
                elapsed[name] += start.elapsed_time(end)
            result.update({f"{name}_ms": value for name, value in elapsed.items()})
        else:
            result.update({f"{name}_ms": value for name, value in self.cpu.items()})
        return result
