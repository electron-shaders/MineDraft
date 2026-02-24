from typing import Optional

import torch
from vllm.model_executor.layers.spec_decode_base_sampler import SpecDecodeBaseSampler
from vllm.platforms import current_platform
from vllm.spec_decode.metrics import (
    AsyncMetricsCollector,
    SpecDecodeWorkerMetrics,
    Timer,
)
from vllm.utils import is_pin_memory_available

from minedraft.patching import MinePatch


class MineSpecDecodeWorkerMetrics(SpecDecodeWorkerMetrics):
    # The empirical good draft tokens until the first rejection / all drafts
    good_draft_rate: Optional[int] = None


class AsyncMetricsCollectorPatch(MinePatch[AsyncMetricsCollector]):
    _orig_init = AsyncMetricsCollector.__init__

    def __init__(self,
                 spec_decode_sampler: SpecDecodeBaseSampler,
                 timer: Optional[Timer] = None,
                 collect_interval_s: float = 5.0):

        self._orig_init(spec_decode_sampler, timer, collect_interval_s)
        pin_memory = is_pin_memory_available()
        self._aggregate_num_good_draft_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_verification_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregrate_num_reqs = 0

    def _copy_rejsample_metrics_async(self: AsyncMetricsCollector) -> torch.cuda.Event:

        assert self._copy_stream is not None
        self._copy_stream.wait_stream(current_platform.current_stream())

        with current_platform.stream(self._copy_stream):
            self._aggregate_num_accepted_tokens.copy_(
                self.spec_decode_sampler.num_accepted_tokens,
                non_blocking=True)
            self._aggregate_num_good_draft_tokens.copy_(
                self.spec_decode_sampler.num_good_draft_tokens,
                non_blocking=True)
            self._aggregate_num_verification_tokens.copy_(
                self.spec_decode_sampler.num_verification_tokens,
                non_blocking=True)
            self._aggregate_num_emitted_tokens.copy_(
                self.spec_decode_sampler.num_emitted_tokens, non_blocking=True)
            # Number of draft tokens is calculated on CPU, so no copy is
            # required.
            self._aggregate_num_draft_tokens = (
                self.spec_decode_sampler.num_draft_tokens)
            self._aggregrate_num_reqs = self.spec_decode_sampler.num_req

        aggregate_metrics_ready = current_platform.Event()
        aggregate_metrics_ready.record(self._copy_stream)

        return aggregate_metrics_ready

    def _collect_rejsample_metrics(
            self: AsyncMetricsCollector,
            k: int,
            ready_event: torch.cuda.Event) -> MineSpecDecodeWorkerMetrics:

        ready_event.synchronize()

        # update time of last collection
        self._last_metrics_collect_time = self._timer()

        accepted_tokens = self._aggregate_num_accepted_tokens.item()
        good_draft_tokens = self._aggregate_num_good_draft_tokens.item()
        verification_tokens = self._aggregate_num_verification_tokens.item()
        emitted_tokens = self._aggregate_num_emitted_tokens.item()
        draft_tokens = self._aggregate_num_draft_tokens
        num_reqs = self._aggregrate_num_reqs

        max_num_emitted_tokens = verification_tokens + num_reqs

        if draft_tokens > 0:
            draft_acceptance_rate = accepted_tokens / draft_tokens
        else:
            draft_acceptance_rate = float("nan")

        if verification_tokens > 0:
            good_draft_rate = good_draft_tokens / verification_tokens
        else:
            good_draft_rate = float("nan")

        if max_num_emitted_tokens > 0:
            system_efficiency = emitted_tokens / max_num_emitted_tokens
        else:
            system_efficiency = float("nan")

        return MineSpecDecodeWorkerMetrics(
            num_spec_tokens=k,
            draft_acceptance_rate=draft_acceptance_rate,
            system_efficiency=system_efficiency,
            accepted_tokens=accepted_tokens,
            draft_tokens=draft_tokens,
            emitted_tokens=emitted_tokens,
            good_draft_rate=good_draft_rate,
        )