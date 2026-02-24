import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.core import scheduler
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import (
    PartialPrefillMetadata,
    PreemptionMode,
    ScheduledSequenceGroup,
    Scheduler,
    SchedulerOutputs,
    SchedulerPrefillOutputs,
    SchedulerRunningOutputs,
    SchedulerSwappedInOutputs,
    SchedulingBudget,
    logger,
)
from vllm.sequence import (
    SequenceData,
    SequenceGroup,
    SequenceGroupBase,
    SequenceGroupMetadataDelta,
    SequenceStatus,
)

import minedraft.benchmarks.trace as CTrace
from minedraft.benchmarks.trace import TRACER, Step
from minedraft.patching import MinePatch
from minedraft.plugin.sequence import MineSequenceGroupMetadata

rid_tid_map: Dict[str, str] = {}


@dataclass
class MineSchedulerOutputs(SchedulerOutputs):

    # Request IDs of preempted requests.
    preempted_requests_ids: List[str] = field(default_factory=list)


class SchedulerOutputsPatch(MinePatch[SchedulerOutputs]):

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an MineSchedulerOutputs instead of a
        # SchedulerOutputs when creating a new instance of the class.
        if cls is SchedulerOutputs:
            return MineSchedulerOutputs.__new__(
                MineSchedulerOutputs, *args, **kwargs)
        return super(SchedulerOutputs, cls).__new__(cls)


@dataclass
class MineSchedulerRunningOutputs(SchedulerRunningOutputs):

    # Request IDs of preempted requests.
    preempted_requests_ids: List[str]


class SchedulerRunningOutputsPatch(MinePatch[SchedulerRunningOutputs]):

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an MineSchedulerRunningOutputs instead of a
        # SchedulerRunningOutputs when creating a new instance of the class.
        if cls is SchedulerRunningOutputs:
            return MineSchedulerRunningOutputs.__new__(
                MineSchedulerRunningOutputs, *args, **kwargs)
        return super(SchedulerRunningOutputs, cls).__new__(cls)

    @classmethod
    def create_empty(cls) -> MineSchedulerRunningOutputs:
        return MineSchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            decode_seq_groups_list=[],
            prefill_seq_groups_list=[],
            preempted_requests_ids=[],
        )


class SchedulerPatch(MinePatch[Scheduler]):
    _orig_init = Scheduler.__init__
    _orig_add_seq_group = Scheduler.add_seq_group
    _orig_schedule = Scheduler._schedule
    _orig_free_finished_seq_groups = Scheduler.free_finished_seq_groups
    _orig_preempt = Scheduler._preempt
    _orig_preempt_by_recompute = Scheduler._preempt_by_recompute

    def __init__(self, *args, **kwargs):
        self._orig_init(*args, **kwargs)
        self._is_psd = False
        self._is_pearl = False

    def set_use_pearl(self, use_pearl: bool) -> None:
        self._is_pearl = use_pearl

    def set_max_num_seqs_for_psd(self: Scheduler, max_num_seqs: int) -> None:
        self._is_psd = True
        # [Parallel SD] Maximum number of sequences in a batch for decoding.
        self.scheduler_config.max_num_seqs = max_num_seqs
        # [Parallel SD] Balance calculated by the difference of number of
        # requests assigned to False and True flags. (#False - #True)
        self._balance_counter: int = 0
        # [Parallel SD] Whether we have scheduled decoding requests at least once.
        self._has_scheduled_decoding: bool = False
        # [Parallel SD] Whether we have deferred skipping memory allocation
        # of KV blocks for the sequence groups with given request_id.
        self._has_deferred_skip: Set[str] = set()
        # [Parallel SD] The batch flag to skip appending slots.
        self._skip_batch_flag: bool = False
        # [Parallel SD] Whether we have seen one sub-batch empty.
        self._after_first_one_batch_empty: bool = False

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        tid = TRACER.add(CTrace.Request)
        request_trace = TRACER.get(tid)
        request_trace.start_us = time.perf_counter() * 1e6
        rid_tid_map[seq_group.request_id] = tid
        self._orig_add_seq_group(seq_group)

    def _get_next_batch_flag(self, is_mixed_running_batch: bool = False) -> bool:
        """[Parallel SD] Get the batch flag to be assigned to a running request."""
        if self._has_scheduled_decoding and not is_mixed_running_batch:
            self._balance_counter += 1 if self._skip_batch_flag else -1
            return not self._skip_batch_flag
        if self._balance_counter >= 0:
            self._balance_counter -= 1
            return True
        else:
            self._balance_counter += 1
            return False

    def _recycle_batch_flag(self, batch_flag: Optional[bool]) -> None:
        """[Parallel SD] Recycle the batch flag of a request that exits from RUNNING state.
        """
        if batch_flag is None:
            return
        if batch_flag:
            self._balance_counter += 1
        else:
            self._balance_counter -= 1

    @property
    def _is_one_batch_empty(self) -> bool:
        if not self.running:
            return False
        has_scoring = has_alt = False
        for seq_group in self.running:
            if seq_group.batch_flag != self._skip_batch_flag:
                has_scoring = True
            else:
                has_alt = True
            if has_scoring and has_alt:
                return False
        return True

    def abort_seq_group(
        self: Scheduler,
        request_id: Union[str, Iterable[str]],
        seq_id_to_seq_group: Optional[Dict[str, SequenceGroupBase]] = None,
    ) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        seq_id_to_seq_group = seq_id_to_seq_group or {}
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                # When n>1, seq_group.request_id looks like
                # foo_parallel_sample_0, while request_ids is just foo, and we
                # should resolve it as real_request_id to match.
                if seq_group.request_id in seq_id_to_seq_group:
                    real_request_id = seq_id_to_seq_group[
                        seq_group.request_id].group_id
                else:
                    real_request_id = seq_group.request_id
                if real_request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    # We can't remove real_request_id in request_ids here,
                    # because there may be other seq groups sharing the same
                    # real_request_id
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                # Remove the aborted request from the Mamba cache.
                self._finished_requests_ids.append(aborted_group.request_id)
                if self._is_psd:
                    # [Parallel SD] Recycle the batch flag.
                    self._recycle_batch_flag(aborted_group.batch_flag)
                    aborted_group.batch_flag = None
                    # [Parallel SD] Remove from deferred skip set.
                    self._has_deferred_skip.discard(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)
                if aborted_group.request_id in seq_id_to_seq_group:
                    del seq_id_to_seq_group[aborted_group.request_id]

                self._free_seq_group_cross_attn_blocks(aborted_group)

    def _schedule_running(
        self: Scheduler,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerRunningOutputs:
        ret: SchedulerRunningOutputs = self._scheduler_running_outputs_cache[
            self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()
        ret.preempted_requests_ids.clear()

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[
            ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        running_queue = self.running
        if self._is_psd:
            # [Parallel SD] Corner case: One sub-batch is empty, never skip
            # appending slots.
            is_one_batch_empty = self._is_one_batch_empty

        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group = running_queue[0]
            # We discard the cached tokens info here because we don't need it
            # for running sequence:
            #   1. If a sequence is running with chunked prefill, the cached
            #      tokens info was already used for the first prefill.
            #   2. If a sequence is running with non-chunked prefill, then
            #      there it's a decoding sequence, and the cached tokens info is
            #      irrelevant.
            num_uncached_new_tokens, _ = \
                self._get_num_new_uncached_and_cached_tokens(
                seq_group,
                SequenceStatus.RUNNING,
                enable_chunking,
                budget,
                partial_prefill_metadata,
            )

            num_running_tokens = num_uncached_new_tokens
            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # With async postprocessor, an extra decode run is done
            # to process the final tokens. The check below avoids this extra
            # decode run when the model max len is reached, in order to avoid
            # a memory overflow.
            if (self.use_async_output_proc and seq_group.seqs[0].get_len()
                    > self.scheduler_config.max_model_len):
                self._async_stopped.append(seq_group)
                continue

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # It may be that the async pending "victim_seq_group"
                    # becomes finished, in which case we simply free it.
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # Do preemption
                if do_preempt:
                    # [Parallel SD] Update step trace and output with preempted
                    # request ID.
                    cur_step_trace: Step = TRACER.current_step
                    if cur_step_trace.preempted_requests is None:
                        cur_step_trace.preempted_requests = []
                    cur_step_trace.preempted_requests.append(
                        rid_tid_map[victim_seq_group.request_id]
                    )
                    ret.preempted_requests_ids.append(
                        victim_seq_group.request_id
                    )

                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                is_prefill = seq_group.is_prefill()
                if self._is_psd:
                    # [Parallel SD] If one sub-batch is empty (corner case) and this
                    # is not the first time we encounter this situation, allocate a slot.
                    # If the sequence group is decoding and assigned to the skip batch flag
                    # and we have deferred skipping this group, do not allocate a slot.
                    assert seq_group.batch_flag is not None, \
                        "Batch flag must be set for Parallel SD scheduling."
                    should_append_slot = (is_one_batch_empty and self._after_first_one_batch_empty or
                                          is_prefill or
                                          seq_group.request_id not in self._has_deferred_skip or
                                          seq_group.batch_flag != self._skip_batch_flag)
                    if should_append_slot:
                        self._append_slots(seq_group, blocks_to_copy, enable_chunking)

                    if not is_prefill:
                        self._has_deferred_skip.add(seq_group.request_id)
                else:
                    self._append_slots(seq_group, blocks_to_copy, enable_chunking)

                scheduled_seq_group: ScheduledSequenceGroup = (
                    self._scheduled_seq_group_cache[
                        self.cache_id].get_object())
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        if self._is_psd:
            if self._has_scheduled_decoding:
                # [Parallel SD] Flip _skip_batch_flag to alternate between
                # skipping appending slots of the two halves of the running
                # requests.
                self._skip_batch_flag = not self._skip_batch_flag

                if is_one_batch_empty:
                    # [Parallel SD] Mark that we have seen one sub-batch empty
                    # after the first time.
                    self._after_first_one_batch_empty = True
                else:
                    self._after_first_one_batch_empty = False
            elif len(ret.decode_seq_groups_list) > 0:
                # [Parallel SD] If this is the first scheduling step that 
                # schedules decoding sequences, we will start alternating 
                # _skip_batch_flag from the next scheduling step.
                self._has_scheduled_decoding = True
        return ret

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        infeasible_seq_groups: List[SequenceGroup] = []

        swapped_queue = self.swapped

        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group = swapped_queue[0]

            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group,
                self._get_num_lookahead_slots(is_prefill, enable_chunking))
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id,
                )
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group, SequenceStatus.SWAPPED, enable_chunking,
                    budget))

            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.SWAPPED)
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(
                        seq_group,
                        token_chunk_size=num_new_tokens_uncached +
                        num_new_tokens_cached,
                    ))
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            if self._is_psd:
                # [Parallel SD] Assign a batch flag to the sequence group.
                seq_group.batch_flag = self._get_next_batch_flag(enable_chunking and is_prefill)
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        swapped_queue.extendleft(leftover_swapped)

        return SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False, enable_chunking=enable_chunking),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerPrefillOutputs:

        if self._is_psd and enable_chunking and self._has_scheduled_decoding:
            # [Parallel SD] Flip _skip_batch_flag back so that chunked 
            # prefills are assigned to the correct subbatch.
            self._skip_batch_flag = not self._skip_batch_flag

        if budget.remaining_token_budget() == 0:
            if self._is_psd and enable_chunking and self._has_scheduled_decoding:
                # [Parallel SD] Flip _skip_batch_flag again to restore.
                self._skip_batch_flag = not self._skip_batch_flag

            # Do nothing: Can't add any more prefill anyway
            return SchedulerPrefillOutputs(
                seq_groups=[],
                ignored_seq_groups=[],
                num_lookahead_slots=self._get_num_lookahead_slots(
                    is_prefill=True, enable_chunking=enable_chunking),
            )
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []
        using_prompt_embeds: bool = False

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            if (partial_prefill_metadata is not None
                    and not partial_prefill_metadata.can_schedule(seq_group)):
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue
            num_new_tokens_uncached, num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group,
                    SequenceStatus.WAITING,
                    enable_chunking,
                    budget,
                    partial_prefill_metadata=partial_prefill_metadata,
                ))
            num_new_tokens = num_new_tokens_uncached + num_new_tokens_cached

            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    num_new_tokens,
                    prompt_limit,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.FINISHED_IGNORED)
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(
                    True, enable_chunking)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            if can_allocate == AllocStatus.LATER:
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.WAITING)
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens,
                    num_lookahead_slots,
                )
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.FINISHED_IGNORED)
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # We cannot mix sequence groups that use prompt embeds and
            # those that do not.
            if len(seq_groups) == 0:
                using_prompt_embeds = seq_group.uses_prompt_embeds()
            if using_prompt_embeds != seq_group.uses_prompt_embeds():
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.WAITING)
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    self.remove_seq_from_computed_blocks_tracker(
                        seq_group, SequenceStatus.WAITING)
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens
                    >= self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.WAITING)
                break

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if num_new_tokens_uncached == 0 or not budget.can_schedule(
                    num_new_tokens=num_new_tokens_uncached,
                    num_new_seqs=num_new_seqs,
            ):
                self.remove_seq_from_computed_blocks_tracker(
                    seq_group, SequenceStatus.WAITING)
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if partial_prefill_metadata is not None:
                partial_prefill_metadata.maybe_increment_partial_prefills(
                    seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking,
                )

            if self._is_psd:
                # [Parallel SD] Assign a batch flag to the sequence group.
                seq_group.batch_flag = self._get_next_batch_flag(enable_chunking)
            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(
                seq_group.request_id,
                num_batched_tokens=num_new_tokens_uncached,
                num_cached_tokens=num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        if self._is_psd and enable_chunking and self._has_scheduled_decoding:
            # [Parallel SD] Flip _skip_batch_flag again to restore.
            self._skip_batch_flag = not self._skip_batch_flag

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking),
        )

    def _schedule(self) -> SchedulerOutputs:
        if self._is_psd and not self.running and self._has_scheduled_decoding:
            # [Parallel SD] Reset Parallel SD scheduler state if there are no
            # running requests and we have scheduled decoding requests at least once.
            assert len(self._has_deferred_skip) == 0
            if self._balance_counter != 0:
                logger.warning(
                    "The balance of sub-batch splitting was broken (#False - "
                    "#True = %d). It may be attributed to the high number of "
                    "preempts. Please consider decreasing max_num_seqs or inc"
                    "reasing gpu_memory_utilization or tensor_parallel_size.",
                    self._balance_counter)
            self._balance_counter = 0
            self._has_scheduled_decoding = False
            self._skip_batch_flag = False
            self._after_first_one_batch_empty = False

        return self._orig_schedule()

    def schedule(
        self: Scheduler
    ) -> Tuple[List[MineSequenceGroupMetadata], SchedulerOutputs, bool]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # Create input data structures.
        seq_group_metadata_list: List[MineSequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[
                self.cache_id].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(
                    seq_group)
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + num_computed_tokens
                        < seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = MineSequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    token_type_ids=seq_group.token_type_ids,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=(seq_group.multi_modal_data
                                      if scheduler_outputs.num_prefill_groups
                                      > 0 else None),
                    multi_modal_placeholders=(
                        seq_group.multi_modal_placeholders
                        if scheduler_outputs.num_prefill_groups > 0 else None),
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                    # [Parallel SD] Pass batch flag to PSD workers.
                    batch_flag=seq_group.batch_flag,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta,
                    seq_group.request_id,
                    block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(
                    seq_group)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group,
                scheduled_seq_group.token_chunk_size)

        self._seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # Return results
        return (seq_group_metadata_list, scheduler_outputs,
                allow_async_output_proc)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                if self._is_psd:
                    # [Parallel SD] Recycle the batch flag after finishing a running request.
                    self._recycle_batch_flag(seq_group.batch_flag)
                    seq_group.batch_flag = None
                    # [Parallel SD] Remove the finished request from deferred skip set.
                    self._has_deferred_skip.discard(seq_group.request_id)
                self.free_seq(seq)

    def free_finished_seq_groups(self) -> None:
        # [Parallel SD] Record request stats before freeing finished seq groups.
        for seq_group in self.running:
            if seq_group.is_finished():
                tid = rid_tid_map[seq_group.request_id]
                request_trace = TRACER.get(tid)
                request_trace.prompt_len = len(seq_group.prompt_token_ids)
                request_trace.gen_len = 0
                for seq in seq_group.get_seqs():
                    request_trace.gen_len += seq.get_output_len()
                request_trace.end_us = time.perf_counter() * 1e6
        self._orig_free_finished_seq_groups()

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: List[Tuple[int, int]],
        enable_chunking: bool = False,
    ) -> None:
        is_prefill: bool = seq_group.is_prefill()
        num_lookahead_slots: int = self._get_num_lookahead_slots(
            is_prefill, enable_chunking)

        seq_group.init_multi_step_from_lookahead_slots(
            num_lookahead_slots,
            num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
            is_multi_step=self.scheduler_config.is_multi_step,
            enable_chunking=enable_chunking,
        )

        seq_status: Optional[SequenceStatus] = SequenceStatus.RUNNING
        if self.scheduler_config.is_multi_step and enable_chunking:
            # In multi-step chunked-prefill any sequence type can have
            # slots appended.
            seq_status = None

        # [PEARL] Add more num_lookahead_slots to prevent CUDA illegal memory access
        if self._is_pearl and not is_prefill:
            num_lookahead_slots += num_lookahead_slots - 1

        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            if len(cows) > 0:
                blocks_to_copy.extend(cows)

    def _preempt(self, seq_group: SequenceGroup,
                 blocks_to_swap_out: List[Tuple[int, int]]) -> PreemptionMode:
        if self._is_psd:
            self._recycle_batch_flag(seq_group.batch_flag)
            seq_group.batch_flag = None
        return self._orig_preempt(seq_group=seq_group,
                                  blocks_to_swap_out=blocks_to_swap_out)

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        if self._is_psd:
            # [Parallel SD] Remove the preempted request from deferred skip set.
            self._has_deferred_skip.discard(seq_group.request_id)
        self._orig_preempt_by_recompute(seq_group=seq_group)


class SchedulerModulePatch(MinePatch[scheduler]):
    @staticmethod
    def scheduler_running_outputs_builder():
        return SchedulerRunningOutputs(decode_seq_groups=[],
                                       prefill_seq_groups=[],
                                       preempted=[],
                                       swapped_out=[],
                                       blocks_to_swap_out=[],
                                       blocks_to_copy=[],
                                       num_lookahead_slots=0,
                                       prefill_seq_groups_list=[],
                                       decode_seq_groups_list=[],
                                       preempted_requests_ids=[])