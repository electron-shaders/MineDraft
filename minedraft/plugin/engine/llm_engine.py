import time
from typing import List, Optional, Union

from vllm.engine.llm_engine import (
    LLMEngine,
    SchedulerContext,
    SchedulerOutputState,
    logger,
)
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import PoolingRequestOutput, RequestOutput, RequestOutputFactory
from vllm.sampling_params import RequestOutputKind
from vllm.sequence import SequenceGroup, SequenceGroupOutput
from vllm.worker.model_runner_base import InputProcessingError

import minedraft.benchmarks.trace as CTrace
from minedraft.benchmarks.trace import TRACER, Step
from minedraft.patching import MinePatch
from minedraft.plugin.core.scheduler import rid_tid_map
from minedraft.plugin.sequence import MineExecuteModelRequest


class LLMEnginePatch(MinePatch[LLMEngine]):
    _orig_init = LLMEngine.__init__

    def __init__(self, *args, **kwargs):
        self._orig_init(*args, **kwargs)
        if self.speculative_config and self.speculative_config.is_parallel:
            if not self.speculative_config.force_pearl:
                for sched in self.scheduler:
                    sched.set_use_pearl(False)
                    sched.set_max_num_seqs_for_psd(
                        self.scheduler_config.max_num_seqs * 2
                    )
            else:
                for sched in self.scheduler:
                    sched.set_use_pearl(True)

    def _process_model_outputs(self: LLMEngine,
                               ctx: SchedulerContext,
                               request_id: Optional[str] = None) -> None:

        now = time.time()

        if len(ctx.output_queue) == 0:
            return None

        # Get pending async postprocessor
        if request_id:
            # When we process only one request, no pop is required
            # (since later we will process all of the rest)
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output, skip) = ctx.output_queue[0]
        else:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output,
             skip) = ctx.output_queue.popleft()

        # Sanity check
        assert len(seq_group_metadata_list) == len(
            scheduler_outputs.scheduled_seq_groups)

        has_multiple_outputs: bool = len(outputs) > 1
        outputs_by_sequence_group: List[List[SequenceGroupOutput]]
        if has_multiple_outputs:
            assert self.scheduler_config.is_multi_step or \
                     self.speculative_config
            # Organize outputs by [step][sequence group] instead of
            # [sequence group][step].
            if self.scheduler_config.is_multi_step:
                outputs_by_sequence_group = create_output_by_sequence_group(
                    outputs, len(seq_group_metadata_list))
            elif self.speculative_config:
                # Decodes are multi-steps while prefills are not, outputting at
                # most 1 token. Separate them so that we can trigger chunk
                # processing without having to pad or copy over prompts K times
                # to match decodes structure (costly with prompt_logprobs).
                num_prefills = sum(sg.is_prompt
                                   for sg in seq_group_metadata_list)
                prefills, decodes = outputs[:num_prefills], outputs[
                    num_prefills:]
                outputs_by_sequence_group = create_output_by_sequence_group(
                    decodes,
                    num_seq_groups=len(seq_group_metadata_list) - num_prefills)
                outputs_by_sequence_group = [p.outputs for p in prefills
                                             ] + outputs_by_sequence_group
            # We have outputs for multiple steps submitted in a single burst,
            # so invalidate is_first_step_output.
            is_first_step_output = None
        else:
            outputs_by_sequence_group = outputs

        # Determine the requests we need to operate on
        if request_id:
            indices = []
            for i, seq_group_meta in enumerate(seq_group_metadata_list):
                if seq_group_meta.request_id == request_id:
                    assert i not in skip  # Cannot be called twice
                    indices.append(i)
                    break

            # If the request_id was not found, then it means that
            # this is a new request that has no pending async
            # postprocessor
            if not indices:
                return
        else:
            indices = range(len(seq_group_metadata_list))  # type: ignore

        finished_before: List[int] = []
        finished_now: List[int] = []
        for i in indices:
            if i in skip:
                continue

            seq_group_meta = seq_group_metadata_list[i]
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group: SequenceGroup = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                finished_before.append(i)
                continue

            output: List[SequenceGroupOutput]
            if has_multiple_outputs:
                output = outputs_by_sequence_group[i]
            else:
                output = [outputs_by_sequence_group[0][i]]

            # [Parallel SD] Skip decoding requests that are not scored.
            if len(output[0].samples) > 0 and output[0].samples[0].output_token == -1:
                continue

            if not is_async:
                if self.scheduler_config.is_multi_step:
                    # Updates happen only if the sequence is prefill
                    self._update_num_computed_tokens_for_multi_step_prefill(
                        seq_group, seq_group_meta, is_first_step_output)
                else:
                    seq_group.update_num_computed_tokens(
                        seq_group_meta.token_chunk_size or 0)

            if outputs:
                for o in outputs:
                    if (isinstance(o, SamplerOutput)
                            and seq_group.metrics is not None):
                        if seq_group.metrics.model_forward_time is not None:
                            seq_group.metrics.model_forward_time += (
                                o.model_forward_time or 0)
                        else:
                            seq_group.metrics.model_forward_time = (
                                o.model_forward_time)
                        if seq_group.metrics.model_execute_time is not None:
                            seq_group.metrics.model_execute_time += (
                                o.model_execute_time or 0)
                        else:
                            seq_group.metrics.model_execute_time = (
                                o.model_execute_time)

            if self.model_config.runner_type == "pooling":
                self._process_sequence_group_outputs(seq_group, output)
            else:
                self.output_processor.process_prompt_logprob(seq_group, output)
                if seq_group_meta.do_sample:
                    self.output_processor.process_outputs(
                        seq_group, output, is_async)

            if seq_group.is_finished():
                finished_now.append(i)

        # Generate outputs for the requests that finished this iteration
        for i in finished_now:
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # When we process a single request, we skip it for the next time,
        # and invoke the request output callback (if there was final output)
        if request_id:
            assert len(indices) == 1
            skip.append(indices[0])

            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Free currently finished requests
        if finished_now:
            for scheduler in self.scheduler:
                scheduler.free_finished_seq_groups()

        # For multi-step without streaming, don't create outputs each iteration
        if not is_last_step and not ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if (finished_now
                    and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Create the outputs
        for i in indices:
            if i in skip or i in finished_before or i in finished_now:
                continue  # Avoids double processing

            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]

            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.set_last_token_time(now)
            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)
            if request_output:
                ctx.request_outputs.append(request_output)

        # For multi-step with streaming, create outputs each iteration
        if not is_last_step and ctx.multi_step_stream_outputs:
            # Immediately process request outputs here (if callback is given)
            if self.process_request_outputs_callback is not None:
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        for seq_group in scheduler_outputs.ignored_seq_groups:
            params = seq_group.sampling_params
            if params is not None and params.output_kind == (
                    RequestOutputKind.DELTA) and not seq_group.is_finished():
                continue

            request_output = RequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs,
            )
            if request_output:
                ctx.request_outputs.append(request_output)

        # Immediately process request outputs here (if callback is given)
        if (ctx.request_outputs
                and self.process_request_outputs_callback is not None):
            self.process_request_outputs_callback(ctx.request_outputs)
            ctx.request_outputs.clear()

        # For async case, we need to record the stats here.
        # For non-async case, the stats are done in the
        # LLMEngine/AsyncLLMEngine directly
        if is_async:
            # Log stats.
            self.do_log_stats(scheduler_outputs, outputs, finished_before,
                              skip)

            # Tracing
            self.do_tracing(scheduler_outputs, finished_before)

        return None

    def step(self: LLMEngine) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        
        if self.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is only supported through AsyncLLMEngine "
                "as performance will be severely degraded otherwise.")

        # For llm_engine, there is no pipeline parallel support, so the engine
        # used is always 0.
        virtual_engine = 0

        # [Parallel SD] Add a trace for the step
        step_tid = TRACER.add(CTrace.Step)
        step_trace: Step = TRACER.get(step_tid)
        step_trace.start_us = time.perf_counter() * 1e6

        # These are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # Skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        # The scheduler is also skipped if a single request caused the last
        # engine step to fail, and the previous schedule needs to be rerun.
        if not self._has_remaining_steps(
                seq_group_metadata_list
        ) and not self._skip_scheduling_next_step:
            # Schedule iteration
            (seq_group_metadata_list, scheduler_outputs,
             allow_async_output_proc
             ) = self.scheduler[virtual_engine].schedule()

            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            finished_requests_ids = self.scheduler[
                virtual_engine].get_and_reset_finished_requests_ids()
            # When n>1, elements in self.seq_id_to_seq_group should be deleted
            # here, otherwise memory leaks.
            for finished_request_id in finished_requests_ids:
                if finished_request_id in self.seq_id_to_seq_group:
                    del self.seq_id_to_seq_group[finished_request_id]

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (self.scheduler_config.is_multi_step
                    and scheduler_outputs.num_lookahead_slots > 0):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine, seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc)
        else:
            finished_requests_ids = list()

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():
            # [Parallel SD] Update the step trace with scheduler outputs
            step_trace.is_prompt_run = scheduler_outputs.num_prefill_groups
            step_trace.batched_token_num = scheduler_outputs.num_batched_tokens
            step_trace.batched_requests = [
                rid_tid_map[r.seq_group.request_id]
                for r in scheduler_outputs.scheduled_seq_groups
            ]

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = MineExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # [Parallel SD] Pass the preempted requests ids
                preempted_requests_ids=scheduler_outputs.preempted_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                    virtual_engine]

            try:
                outputs = self.model_executor.execute_model(
                    execute_model_req=execute_model_req)
                self._skip_scheduling_next_step = False
            except InputProcessingError as e:
                # The input for this request cannot be processed, so we must
                # abort it. If there are remaining requests in the batch that
                # have been scheduled, they will be retried on the next step.
                invalid_request_id = e.request_id
                self._abort_and_cache_schedule(
                    request_id=invalid_request_id,
                    virtual_engine=virtual_engine,
                    seq_group_metadata_list=seq_group_metadata_list,
                    scheduler_outputs=scheduler_outputs,
                    allow_async_output_proc=allow_async_output_proc)
                # Raise so the caller is notified that this request failed
                raise

            # We need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
        else:
            # Nothing scheduled => If there is pending async postprocessor,
            # then finish it here.
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            # No outputs in this case
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        if not self._has_remaining_steps(seq_group_metadata_list):
            # clear the cache if we have finished all the steps.
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[0] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = False if not seq_group_metadata_list \
                else seq_group_metadata_list[0].state.num_steps == 1

            # Add results to the output_queue
            ctx.append_output(outputs=outputs,
                              seq_group_metadata_list=seq_group_metadata_list,
                              scheduler_outputs=scheduler_outputs,
                              is_async=allow_async_output_proc,
                              is_last_step=True,
                              is_first_step_output=is_first_step_output)

            if outputs and allow_async_output_proc:
                assert len(outputs) == 1, (
                    "Async postprocessor expects only a single output set")

                self._advance_to_next_step(
                    outputs[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)

            # Check if need to run the usual non-async path
            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)
        else:
            # Multi-step case
            # [Parallel SD] Record the end time of the current step
            step_trace.end_us = time.perf_counter() * 1e6
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            logger.debug("Stopping remote worker execution loop.")
            self.model_executor.stop_remote_worker_execution_loop()

        # [Parallel SD] Record the end time of the current step
        step_trace.end_us = time.perf_counter() * 1e6
        return ctx.request_outputs

    def dump(self, filename: str) -> None:
        TRACER.export(filename)