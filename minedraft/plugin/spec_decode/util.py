import time

import torch
from vllm.spec_decode.util import Timer

from minedraft.patching import MinePatch


class TimerPatch(MinePatch[Timer]):

    _orig___enter__ = Timer.__enter__
    _orig___exit__ = Timer.__exit__

    def __init__(self, is_score=False):
        self.is_score = is_score

    def __enter__(self):
        if self.is_score:
            torch.cuda.synchronize()
            self.start_time_perf = time.perf_counter()
        return self._orig___enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_score:
            torch.cuda.synchronize()
            self.end_time_perf = time.perf_counter()
            self.elapsed_perf_time = self.end_time_perf - self.start_time_perf
        self._orig___exit__(exc_type, exc_value, traceback)