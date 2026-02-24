import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.distributed.parallel_state import get_tp_group, model_parallel_is_initialized
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler,
    SpecDecodeStochasticBaseSampler,
)
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler,
)
from vllm.platforms import current_platform
from vllm.sequence import (
    VLLM_INVALID_TOKEN_ID,
    HiddenStates,
    SequenceData,
    get_all_seq_ids,
)
from vllm.spec_decode import spec_decode_worker
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScores
from vllm.spec_decode.metrics import AsyncMetricsCollector
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.spec_decode.spec_decode_worker import (
    SpecDecodeWorker,
    logger,
    prepare_prefill_hidden_states,
)
from vllm.spec_decode.target_model_runner import TargetModelRunner
from vllm.spec_decode.util import Timer, nvtx_range, split_batch_by_proposal_len
from vllm.utils import resolve_obj_by_qualname
from vllm.worker.worker_base import WorkerBase

from minedraft.benchmarks.trace import TRACER, Step
from minedraft.patching import MinePatch
from minedraft.plugin.distributed.parallel_state import get_non_driver_tp_group
from minedraft.plugin.sequence import (
    MineExecuteModelRequest,
    MineSequenceGroupMetadata,
)
from minedraft.plugin.spec_decode.batch_expansion import (
    ParallelBatchExpansionTop1Scorer,
)
from minedraft.plugin.spec_decode.interfaces import ParallelSpeculativeScorer
from minedraft.plugin.spec_decode.mqa_scorer import ParallelMQAScorer
from minedraft.plugin.spec_decode.tetris import select_proposals_no_priority


class SpecDecodeWorkerModulePatch(MinePatch[spec_decode_worker]):
    @staticmethod
    def create_spec_worker(*args, **kwargs) -> SpecDecodeWorker:
        vllm_config: VllmConfig = kwargs.get("vllm_config")
        speculative_config: SpeculativeConfig = vllm_config.speculative_config
        assert speculative_config is not None

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("Speculative decoding is currently "
                                      "incompatible with pipeline parallelism")

        draft_worker_kwargs = kwargs.copy()

        kwargs["model_runner_cls"] = TargetModelRunner
        target_worker_config = copy.deepcopy(vllm_config)
        target_worker_config.parallel_config.worker_cls =\
            target_worker_config.parallel_config.sd_worker_cls
        cls = resolve_obj_by_qualname(
            target_worker_config.parallel_config.worker_cls)
        target_worker = cls(*args, **kwargs)
        # Set the disable_logprobs variable in the TargetModelRunner instance
        # as per its value specified in the SpeculativeConfig.
        target_worker.model_runner.disable_logprobs =\
            speculative_config.disable_logprobs

        draft_worker_config = copy.deepcopy(vllm_config)
        draft_worker_config.model_config = speculative_config.draft_model_config
        draft_worker_config.quant_config = VllmConfig._get_quantization_config(
            draft_worker_config.model_config,
            vllm_config.load_config,
        )
        speculative_config.draft_parallel_config.worker_cls =\
            draft_worker_config.parallel_config.sd_worker_cls
        draft_worker_config.parallel_config = speculative_config.draft_parallel_config  # noqa
        # TODO allow draft-model specific load config.

        # Override draft-model specific worker args.
        draft_worker_kwargs.update(
            vllm_config=draft_worker_config,
            ngram_prompt_lookup_max=speculative_config.prompt_lookup_max,
            ngram_prompt_lookup_min=speculative_config.prompt_lookup_min,
        )

        spec_decode_worker = SpecDecodeWorker.create_worker(
            scorer_worker=target_worker,
            draft_worker_kwargs=draft_worker_kwargs,
            disable_mqa_scorer=speculative_config.disable_mqa_scorer,
            disable_by_batch_size=speculative_config.disable_by_batch_size,
            draft_token_acceptance_method=speculative_config.acceptance_method,
            typical_acceptance_sampler_posterior_threshold=speculative_config.
            posterior_threshold,
            typical_acceptance_sampler_posterior_alpha=speculative_config.
            posterior_alpha,
            disable_logprobs=speculative_config.disable_logprobs,
            disable_log_stats=speculative_config.disable_log_stats,
            num_speculative_tokens=speculative_config.num_speculative_tokens,
            # [Parallel SD] Pass arguments of PSD
            tetris=speculative_config.tetris,
            tetris_extra_proposals=speculative_config.tetris_extra_proposals,
            tetris_turn_on_batch_size=speculative_config.tetris_turn_on_batch_size,
            tetris_capacity=speculative_config.tetris_capacity,
            is_parallel=speculative_config.is_parallel,
            force_mqa=speculative_config.force_mqa,
            force_pearl=speculative_config.force_pearl,
        )

        return spec_decode_worker


class SpecDecodeWorkerPatch(MinePatch[SpecDecodeWorker]):
    _orig_init = SpecDecodeWorker.__init__
    _orig_run_no_spec = SpecDecodeWorker.__dict__["_run_no_spec"].__wrapped__
    _orig_maybe_log_stage_times = SpecDecodeWorker._maybe_log_stage_times

    @classmethod
    def create_worker(
        cls,
        scorer_worker: WorkerBase,
        draft_worker_kwargs: Dict[str, Any],
        disable_mqa_scorer: bool,
        disable_by_batch_size: Optional[int],
        draft_token_acceptance_method: str,
        typical_acceptance_sampler_posterior_threshold: float,
        typical_acceptance_sampler_posterior_alpha: float,
        disable_logprobs: bool,
        disable_log_stats: bool,
        num_speculative_tokens: int,
        tetris: bool,
        tetris_extra_proposals: int,
        tetris_turn_on_batch_size: int,
        tetris_capacity: int,
        is_parallel: bool,
        force_mqa: bool = False,
        force_pearl: bool = False,
    ) -> SpecDecodeWorker:

        allow_zero_draft_token_step = True
        enable_lm_head_weight_load = False
        num_spec_prefill_steps = 1
        ngram_prompt_lookup_max = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_max"))
        ngram_prompt_lookup_min = (
            draft_worker_kwargs.pop("ngram_prompt_lookup_min"))
        draft_model_config = draft_worker_kwargs["vllm_config"].model_config
        draft_parallel_config: ParallelConfig = draft_worker_kwargs[
            'vllm_config'].parallel_config
        if ngram_prompt_lookup_max > 0:
            draft_worker_kwargs[
                "device_type"] = scorer_worker.device_config.device.type
            from vllm.spec_decode.ngram_worker import NGramWorker
            proposer_worker = NGramWorker(**draft_worker_kwargs)
            proposer_worker.set_ngram_window_size(ngram_prompt_lookup_min,
                                                  ngram_prompt_lookup_max)
        else:
            draft_tp = draft_parallel_config.tensor_parallel_size
            target_tp = scorer_worker.parallel_config.tensor_parallel_size

            if draft_model_config.hf_config.model_type == "mlp_speculator":
                from vllm.spec_decode.mlp_speculator_worker import MLPSpeculatorWorker
                proposer_worker = MLPSpeculatorWorker(**draft_worker_kwargs)
            elif draft_model_config.hf_config.model_type == "medusa":
                from vllm.spec_decode.medusa_worker import MedusaWorker
                proposer_worker = MedusaWorker(**draft_worker_kwargs)
            else:
                if draft_tp == 1:
                    if current_platform.is_cuda_alike():
                        from vllm.spec_decode.draft_model_runner import (
                            TP1DraftModelRunner,
                        )
                        draft_worker_kwargs["model_runner_cls"] = TP1DraftModelRunner
                else:
                    if draft_model_config.hf_config.model_type == "eagle":
                        raise NotImplementedError(
                            f"{draft_model_config.hf_config.model_type} "
                            "does not support TP > 1 yet")

                    allow_zero_draft_token_step = False

                # Load lm_head weight for eagle in init_device
                if draft_model_config.hf_config.model_type == "eagle":
                    enable_lm_head_weight_load = True

                from vllm.spec_decode.multi_step_worker import MultiStepWorker
                proposer_worker = MultiStepWorker(**draft_worker_kwargs)
                if draft_model_config.hf_config.model_type == "deepseek_mtp":
                    num_spec_prefill_steps = \
                        draft_model_config.hf_config.n_predict

            proposer_worker = SmallerTpProposerWorker.maybe_wrap_worker(
                proposer_worker, draft_tp, target_tp)

        logger.info("Configuring SpecDecodeWorker with proposer=%s",
                    type(proposer_worker))

        spec_decode_sampler: SpecDecodeBaseSampler = None
        if draft_token_acceptance_method == "rejection_sampler":
            spec_decode_sampler = RejectionSampler()
        elif draft_token_acceptance_method == "typical_acceptance_sampler":
            spec_decode_sampler = TypicalAcceptanceSampler(
                posterior_threshold=\
                    typical_acceptance_sampler_posterior_threshold,
                posterior_alpha=typical_acceptance_sampler_posterior_alpha,
            )
        logger.info(
            "[Speculative Decoding] Configuring"
            " SpecDecodeWorker with sampler=%s", type(spec_decode_sampler))

        if (not force_mqa) and (not disable_mqa_scorer):
            if scorer_worker.model_runner.attn_backend.get_name(
            ) != "FLASH_ATTN":
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "MQA is only available with flash attn backend.")

            if draft_model_config and \
                draft_model_config.max_model_len < \
                    scorer_worker.model_config.max_model_len:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "draft model max_model_len is smaller than the target "
                    "model max_model_len.")

            if not scorer_worker.model_runner.model_config.enforce_eager:
                disable_mqa_scorer = True
                logger.info(
                    "[Speculative Decoding] Disabling MQA scorer as the "
                    "target model is not running in eager mode.")

        if is_parallel:
            return ParallelSpecDecodeWorker(
                proposer_worker,
                scorer_worker,
                disable_mqa_scorer=disable_mqa_scorer,
                disable_logprobs=disable_logprobs,
                disable_log_stats=disable_log_stats,
                disable_by_batch_size=disable_by_batch_size,
                spec_decode_sampler=spec_decode_sampler,
                allow_zero_draft_token_step=allow_zero_draft_token_step,
                enable_lm_head_weight_load=enable_lm_head_weight_load,
                num_spec_prefill_steps=num_spec_prefill_steps,use_tetris=tetris,
                tetris_extra_proposals=tetris_extra_proposals,
                tetris_turn_on_batch_size=tetris_turn_on_batch_size,
                tetris_capacity=tetris_capacity,
                use_parallel=is_parallel,
                use_pearl=force_pearl,
            )

        return SpecDecodeWorker(
            proposer_worker,
            scorer_worker,
            disable_mqa_scorer=disable_mqa_scorer,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            disable_by_batch_size=disable_by_batch_size,
            spec_decode_sampler=spec_decode_sampler,
            allow_zero_draft_token_step=allow_zero_draft_token_step,
            enable_lm_head_weight_load=enable_lm_head_weight_load,
            num_spec_prefill_steps=num_spec_prefill_steps,use_tetris=tetris,
            tetris_extra_proposals=tetris_extra_proposals,
            tetris_turn_on_batch_size=tetris_turn_on_batch_size,
            tetris_capacity=tetris_capacity,
            use_parallel=is_parallel,
            use_pearl=force_pearl,
        )

    def __init__(
        self,
        proposer_worker: ProposerWorkerBase,
        scorer_worker: WorkerBase,
        spec_decode_sampler: SpecDecodeBaseSampler,
        disable_mqa_scorer: bool = False,
        disable_logprobs: bool = False,
        disable_log_stats: bool = False,
        metrics_collector: Optional[AsyncMetricsCollector] = None,
        disable_by_batch_size: Optional[int] = None,
        allow_zero_draft_token_step: Optional[bool] = True,
        enable_lm_head_weight_load: Optional[bool] = False,
        num_spec_prefill_steps: int = 1,
        use_tetris: Optional[bool] = None,
        tetris_extra_proposals: Optional[int] = None,
        tetris_turn_on_batch_size: Optional[int] = None,
        tetris_capacity: Optional[int] = None,
        use_parallel: Optional[bool] = None,
        use_pearl: Optional[bool] = None,
    ):
        self._orig_init(
            proposer_worker=proposer_worker,
            scorer_worker=scorer_worker,
            spec_decode_sampler=spec_decode_sampler,
            disable_mqa_scorer=disable_mqa_scorer,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
            metrics_collector=metrics_collector,
            disable_by_batch_size=disable_by_batch_size,
            allow_zero_draft_token_step=allow_zero_draft_token_step,
            enable_lm_head_weight_load=enable_lm_head_weight_load,
            num_spec_prefill_steps=num_spec_prefill_steps,
        )
        self.max_num_seqs = self.scorer_worker.scheduler_config.max_num_seqs
        self.use_tetris = use_tetris
        self.tetris_extra_proposals = tetris_extra_proposals
        self.tetris_turn_on_batch_size = tetris_turn_on_batch_size
        self.tetris_capacity = tetris_capacity
        self.use_parallel = use_parallel
        self.use_pearl = use_pearl
        self.sd_step = 0
        self.disable_step = 0
        if self.use_tetris:
            logger.info("[Speculative Decoding] Tetris is enabled.")

    @torch.inference_mode()
    def execute_model(
        self: SpecDecodeWorker,
        execute_model_req: Optional[MineExecuteModelRequest] = None,
        profile_time: bool = True,
    ) -> List[SamplerOutput]:
        if self.rank != self._driver_rank:
            self._run_non_driver_rank()
            return []

        # [Parallel SD] Record the start time of the batch
        cur_step_trace: Step = TRACER.current_step
        assert cur_step_trace is not None
        cur_step_trace.batch_start_us = time.perf_counter() * 1e6

        if execute_model_req is None:
            # This signals that there's no more requests to process for now.
            # All workers are running infinite loop with broadcast_tensor_dict,
            # and it stops the loop when the driver broadcasts an empty input.
            # Send an empty input to notify all other workers to stop their
            # execution loop.
            broadcast_tensor_dict({}, src=0)
            # [Parallel SD] Record the end time of the batch
            cur_step_trace.batch_end_us = time.perf_counter() * 1e6
            return []

        self._track_finished_requests(execute_model_req)
        # [Parallel SD] Update housekeeping data structures of
        # ParallelSpecDecodeWorker when certain requests are preempted.
        self._track_preempted_requests(execute_model_req)
        disable_all_speculation = self._should_disable_all_speculation(
            execute_model_req)
        num_lookahead_slots = execute_model_req.num_lookahead_slots
        all_prompt = True
        atleast_one_prompt = False
        all_zero_spec_tokens = True
        for sgm in execute_model_req.seq_group_metadata_list:
            all_prompt = all_prompt and sgm.is_prompt
            atleast_one_prompt = atleast_one_prompt or sgm.is_prompt
            all_zero_spec_tokens = all_zero_spec_tokens and (
                sgm.num_speculative_tokens == 0)

        if all_prompt and execute_model_req.seq_group_metadata_list:
            assert num_lookahead_slots == 0, (
                "Prompt only runs should have num_lookahead_slots equal to 0. "
                "This should never happen, please file a bug at "
                "https://github.com/vllm-project/vllm/issues")
        # Speculative decoding is disabled in the following cases:
        # 1. Prefill phase: Speculative decoding is not
        #    used during the prefill phase.
        # 2. Auto-disable enabled: The running queue size exceeds
        #    the specified threshold.
        # 3. No request: There are no requests in the batch, or
        #    none of the requests in the batch have spec decoding enabled.
        # In any of these cases, the proposer and scorer workers
        # are called normally.
        # We expect `num_speculative_tokens` to be None for prefills.
        no_spec = (num_lookahead_slots == 0 or disable_all_speculation
                   or all_zero_spec_tokens)

        # Broadcast how many lookahead slots are scheduled for this step, and
        # whether all speculation is disabled, to all non-driver workers.

        # This is required as if the number of draft model runs changes
        # dynamically, the non-driver workers won't know unless we perform a
        # communication to inform them.

        # no_spec is used to signal non-driver worker about prefill vs decode
        # stage. This is needed to ensure that order of execution of proposer
        # and scorer is same in both driver and non-driver workers (i.e.,
        # scorer -> proposer for prefill and proposer -> scorer in decode). This
        # order is needed to support models like EAGLE that take scorer states
        # as inputs.
        broadcast_dict = dict(
            num_lookahead_slots=num_lookahead_slots,
            no_spec=no_spec,
            disable_all_speculation=disable_all_speculation,
            # When both chunked prefill and speculative decoding are enabled
            # it is possible that the same batch contains both prefill
            # and decodes. If that happens in the scorer we run the batch
            # as one single forward pass. However, in the proposer we
            # run them as 2 different batches - one for prefill and
            # the other for decodes. The variable indicates to the non-driver
            # worker that there are prefills as part of the speculative batch
            # and hence it needs to run an extra prefill forward pass.
            run_spec_proposer_for_prefill=atleast_one_prompt,
        )
        # [Parallel SD] Broadcast in the non-blocking way if use_parallel is True.
        broadcast_tensor_dict(broadcast_dict, src=self._driver_rank,
                              async_op=self.use_parallel)

        assert execute_model_req.seq_group_metadata_list is not None, (
            "speculative decoding requires non-None seq_group_metadata_list")

        self._maybe_disable_speculative_tokens(
            disable_all_speculation, execute_model_req.seq_group_metadata_list)

        if no_spec:
            return self._run_no_spec(execute_model_req,
                                     skip_proposer=disable_all_speculation,
                                     profile_time=profile_time)
        return self._run_speculative_decoding_step(execute_model_req,
                                                   num_lookahead_slots,
                                                   profile_time)

    def _should_disable_all_speculation(
            self, execute_model_req: MineExecuteModelRequest) -> bool:
        # When the batch size is too large, disable speculative decoding
        # to stop trading off throughput for latency.
        if self.disable_by_batch_size and execute_model_req.running_queue_size >= self.disable_by_batch_size:
            self.disable_step += 1
        # Reset disable_step every 100 steps to avoid completely disabling spec decode
        if self.disable_step % 100 == 0:
            self.disable_by_batch_size = float("inf")
            self.disable_step = 0

        return (execute_model_req.running_queue_size
                >= self.disable_by_batch_size)

    @nvtx_range("spec_decode_worker._run_no_spec")
    def _run_no_spec(self,
                     execute_model_req: MineExecuteModelRequest,
                     skip_proposer: bool,
                     profile_time: bool = False) -> List[SamplerOutput]:

        with Timer(profile_time) as timer:
            ret = self._orig_run_no_spec(execute_model_req=execute_model_req,
                                         skip_proposer=skip_proposer)

        # [Parallel SD] Record the end time of the batch
        step_trace: Step = TRACER.current_step
        if profile_time:
            step_trace.measured_target_time = timer.elapsed_perf_time
        step_trace.batch_end_us = time.perf_counter() * 1e6

        return ret

    @nvtx_range("spec_decode_worker._run_speculative_decoding_step")
    def _run_speculative_decoding_step(
            self,
            execute_model_req: MineExecuteModelRequest,
            num_lookahead_slots: int,
            profile_time: bool = False) -> List[SamplerOutput]:

        assert not self.use_parallel, "SpecDecodeWorker does not support " + \
            "use_parallel == True, use ParallelSpecDecodeWorker instead"

        # With prefill chunking, expect requests to have prompts first
        # so that backend gets prefill|decode.
        assert num_lookahead_slots == execute_model_req.num_lookahead_slots
        self.sd_step += 1

        # Pass last hidden states from target model to proposer
        execute_model_req.previous_hidden_states = self.previous_hidden_states
        self.previous_hidden_states = None

        cur_step_trace: Step = TRACER.current_step

        # Pre-process proposals
        execute_model_req, num_lookahead_slots, proposal_len = \
            self._preprocess(execute_model_req, num_lookahead_slots, profile_time)
        cur_step_trace.proposed_len = proposal_len

        # Propose
        if proposal_len == 0:
            for seq_group_metadata \
                in execute_model_req.seq_group_metadata_list:
                seq_group_metadata.num_speculative_tokens = 0
            self.disable_by_batch_size = execute_model_req.running_queue_size
            return self._run_no_spec(execute_model_req, skip_proposer=True)

        with Timer(profile_time) as proposal_timer:
            # Generate proposals using draft worker.
            proposals = self.proposer_worker.get_spec_proposals(
                execute_model_req,
                self._seq_with_bonus_token_in_last_step,
            )

        cur_step_trace.match_count = (proposals.proposal_lens > 0).sum()

        if not self._allow_zero_draft_token_step and proposals.no_proposals:
            #TODO: Fix it #5814
            raise RuntimeError("Cannot handle cases where distributed draft "
                               "workers generate no tokens")

        execute_model_req.previous_hidden_states = None

        proposals, verify_len, tetris_time = self._postprocess(
            execute_model_req, proposal_len,
            cur_step_trace, proposals,
            profile_time)

        cur_step_trace.verify_len = proposals.proposal_lens.sum()

        # Score
        with Timer(profile_time) as scoring_timer:
            proposal_scores = self.scorer.score_proposals(
                execute_model_req,
                proposals,
            )

        _, (non_spec_seqs, non_spec_indices) = split_batch_by_proposal_len(
            execute_model_req.seq_group_metadata_list, proposals.proposal_lens)
        # With prefill chunking enabled, `non_spec_seqs` contains prefills too:
        # discard decodes that have already been processed by proposer.
        non_spec_indices = [
            idx for idx in non_spec_indices
            if execute_model_req.seq_group_metadata_list[idx].is_prompt
        ]
        if len(non_spec_indices):
            all_hidden_states = proposal_scores.hidden_states
            if all_hidden_states is not None:
                prefill_hidden_states = all_hidden_states[non_spec_indices]
                execute_model_req.previous_hidden_states = \
                    prepare_prefill_hidden_states(prefill_hidden_states)
            # Sync proposer KV cache for prefills.
            prefill_req = execute_model_req.clone(non_spec_seqs)
            # TODO avoid sampling here?
            self.proposer_worker.execute_model(prefill_req)

        with Timer(profile_time) as verification_timer:
            accepted_token_ids, target_logprobs = self._verify_tokens(
                execute_model_req.seq_group_metadata_list, proposal_scores,
                proposals, execute_model_req.num_lookahead_slots)

        stage_times = (proposal_timer.elapsed_time_ms / num_lookahead_slots,
                       scoring_timer.elapsed_time_ms,
                       verification_timer.elapsed_time_ms,
                       tetris_time)

        with Timer(profile_time) as overhead_timer:
            out = self._create_output_sampler_list(
                execute_model_req.seq_group_metadata_list,
                accepted_token_ids,
                target_logprobs=target_logprobs,
                prompt_logprobs=proposal_scores.prompt_logprobs
                if not self._disable_logprobs else None,
                k=execute_model_req.num_lookahead_slots,
                stage_times=stage_times)

        # [Parallel SD] Record stage times and end time of the batch
        if profile_time:
            cur_step_trace.measured_draft_time = proposal_timer.elapsed_perf_time
            cur_step_trace.measured_target_time = scoring_timer.elapsed_perf_time
            cur_step_trace.measured_overhead_time = overhead_timer.elapsed_perf_time + verification_timer.elapsed_perf_time
        cur_step_trace.batch_end_us = time.perf_counter() * 1e6
    
        return out

    def _preprocess(
        self,
        execute_model_req: MineExecuteModelRequest,
        num_lookahead_slots: int,
        profile_time: bool = False,
    ) -> Tuple[MineExecuteModelRequest, int, int]:
        batch_size = len(execute_model_req.seq_group_metadata_list)

        turn_on_tetris = self.use_tetris and batch_size >= self.tetris_turn_on_batch_size

        if self.use_tetris and turn_on_tetris:
            num_lookahead_slots -= self.tetris_extra_proposals

        proposal_len = num_lookahead_slots
        return execute_model_req, num_lookahead_slots, proposal_len

    def _postprocess(
        self,
        execute_model_req: MineExecuteModelRequest,
        proposal_len: int,
        cur_step_trace: Step,
        proposals: SpeculativeProposals,
        profile_time: bool = False,
    ) -> Tuple[SpeculativeProposals, int, float]:
        batch_size = len(execute_model_req.seq_group_metadata_list)
        max_num_seqs = self.max_num_seqs if not self.use_parallel else self.max_num_seqs // 2
        turn_on_tetris = self.use_tetris and batch_size >= max_num_seqs

        verify_len = proposal_len

        if turn_on_tetris:
            with Timer(profile_time) as tetris_timer:
                if self.tetris_capacity and self.tetris_capacity > 0:
                    capacity = self.tetris_capacity
                else:
                    capacity = int(proposal_len * max_num_seqs)
                proposals = select_proposals_no_priority(capacity, proposals)

        tetris_time = tetris_timer.elapsed_time_ms if turn_on_tetris else None
        return proposals, verify_len, tetris_time

    @nvtx_range("spec_decode_worker._verify_tokens")
    def _verify_tokens(
        self,
        seq_group_metadata_list: List[MineSequenceGroupMetadata],
        proposal_scores: SpeculativeScores,
        proposals: SpeculativeProposals,
        max_proposal_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        proposal_lens_list = proposals.proposal_lens.tolist()

        # vLLM currently only supports proposal lens equal to zero or the batch
        # proposal len. This adds some complexity (splitting the batch into spec
        # and non spec sequences) and should be removed in the future. It can be
        # done by supporting per-sequence proposal lens.
        (_, spec_indices), (_, non_spec_indices) = split_batch_by_proposal_len(
            seq_group_metadata_list, proposal_lens_list)
        original_indices = spec_indices + non_spec_indices

        # [Parallel SD] Get lengths of speculative proposals.
        spec_proposal_lens = proposals.proposal_lens[spec_indices]

        # Get probabilities of target model, including bonus tokens.
        proposal_verifier_probs = proposal_scores.probs[spec_indices]

        # Get non-speculative sampled tokens from target model.
        non_spec_token_ids = proposal_scores.token_ids[non_spec_indices]

        # Get bonus tokens from target model.
        # bonus_token_ids = proposal_scores.token_ids[spec_indices, -1:]
        # [Parallel SD] Edited as below to adapt varlen
        bonus_token_ids = proposal_scores.token_ids[spec_indices, spec_proposal_lens].unsqueeze(-1)

        # Get probabilities according to proposal method.
        proposal_probs = proposals.proposal_probs[spec_indices]

        # Get proposed tokens.
        proposal_token_ids = proposals.proposal_token_ids[spec_indices]

        # Sampler arguments
        sampler_extra_kwargs: Dict[str, Any] = {}
        if self.generators and isinstance(self.spec_decode_sampler,
                                          SpecDecodeStochasticBaseSampler):
            sampler_extra_kwargs["seeded_seqs"] = {
                idx: self.generators[sgm.request_id]
                for idx, sgm in enumerate(seq_group_metadata_list)
                if sgm.sampling_params.seed is not None
            }

        accepted_token_ids = self.spec_decode_sampler(
            target_with_bonus_probs=proposal_verifier_probs,
            bonus_token_ids=bonus_token_ids,
            draft_probs=proposal_probs,
            draft_token_ids=proposal_token_ids,
            proposal_lens=spec_proposal_lens,
            total_num_seqs=len(seq_group_metadata_list),
            **sampler_extra_kwargs,
        )

        # [Parallel SD] Record the number of accepted tokens
        cur_step_trace: Step = TRACER.current_step
        assert len(
            cur_step_trace.batched_requests) >= accepted_token_ids.shape[0]
        cur_step_trace.accepted_num = (
            (accepted_token_ids >= 0).sum() - accepted_token_ids.shape[0]
        )

        # Append output tokens from non-speculative sequences to
        # the accepted token ids tensor.
        non_spec_token_ids = non_spec_token_ids.expand(-1, max_proposal_len +
                                                       1).clone()
        non_spec_token_ids[:, 1:] = -1
        accepted_token_ids = torch.cat(
            [accepted_token_ids, non_spec_token_ids])
        # [Parallel SD] Record the number of generated tokens
        cur_step_trace.generated_num = (accepted_token_ids >= 0).sum()

        logprobs = proposal_scores.logprobs
        # Rearrange so that results are in the order of the original seq group
        # metadata.
        accepted_token_ids[original_indices] = accepted_token_ids.clone()

        # B x K+1 x D
        hidden_states = proposal_scores.hidden_states
        if hidden_states is not None:
            # Only get terminal hidden states for next step
            terminal_metadata = [
                sg for sg in seq_group_metadata_list if sg.do_sample
            ]

            # Contract hidden states based on accepted tokens
            hs_size = hidden_states.shape[-1]
            accepted_index = accepted_token_ids + 1  # Convert -1 to 0
            accepted_index = accepted_index.count_nonzero(dim=1).add_(-1)  # b
            # Drop non-terminal prefill chunks hidden states.
            hidden_states = hidden_states[accepted_index !=
                                          VLLM_INVALID_TOKEN_ID]
            accepted_index = accepted_index[accepted_index !=
                                            VLLM_INVALID_TOKEN_ID]
            assert len(accepted_index) == hidden_states.shape[0] == len(
                terminal_metadata)

            # [Parallel SD] Get second_last_token_hidden_states and hidden_states
            # in the varlen case.
            second_last_index = (proposals.proposal_lens - 2).clamp_min(0)[:, None, None].expand(-1, 1,
                                                                                                 hs_size)  # b x 1 x d
            second_last_token_hidden_states = hidden_states.gather(1, second_last_index).squeeze(1)        # b x d
            index = accepted_index[:, None, None].expand(-1, 1,
                                                         hs_size)  # b x 1 x d
            hidden_states = hidden_states.gather(1, index).squeeze(1)  # b x d
            # Store hidden states from target model for subsequent decode step
            self.previous_hidden_states = HiddenStates(
                hidden_states, terminal_metadata,
                second_last_token_hidden_states)
        return accepted_token_ids, logprobs
        

    def _maybe_log_stage_times(self, average_time_per_proposal_tok_ms: float,
                               scoring_time_ms: float,
                               verification_time_ms: float,
                               tetris_time_ms: Optional[float]) -> None:
        if not self._disable_log_stats and tetris_time_ms is not None:
            logger.info(
                "SpecDecodeWorker stage times: "
                "average_time_per_proposal_tok_ms=%.02f "
                "scoring_time_ms=%.02f verification_time_ms=%.02f tetris_time_ms=%.02f",
                average_time_per_proposal_tok_ms, scoring_time_ms,
                verification_time_ms, tetris_time_ms)
            return

        self._orig_maybe_log_stage_times(
            average_time_per_proposal_tok_ms=average_time_per_proposal_tok_ms,
            scoring_time_ms=scoring_time_ms,
            verification_time_ms=verification_time_ms,
        )

    def _track_preempted_requests(self, execute_model_req: MineExecuteModelRequest):
        pass


class ParallelSpecDecodeWorker(SpecDecodeWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_seqs = self.scorer_worker.scheduler_config.max_num_seqs * 2
        self.scorer: ParallelSpeculativeScorer
        if not self.use_pearl:
            self._send_batch_flag: bool = True
        self._previous_request_ids: Dict[str, int] = {}
        self._previous_proposals: Optional[SpeculativeProposals] = None
        self._valid_previous_proposals: Optional[torch.BoolTensor] = None
        logger.info("[Speculative Decoding] SSM/LLM Parallel is enabled.")

    def init_device(self) -> None:
        """Initialize both scorer and proposer models.
        """
        # The scorer worker model is initialized first in case the proposer
        # model has a smaller TP degree than the target worker.
        self.scorer_worker.init_device()
        self.proposer_worker.init_device()

        # NOTE(cade): load_model is not part of the WorkerBase interface.
        # [Parallel SD] Only non-driver ranks run the scorer.
        # Driver rank will load scorer model weights temporarily when
        # self._enable_lm_head_weight_load is True.
        if self._enable_lm_head_weight_load or self.rank != self._driver_rank:
            self.scorer_worker.load_model()
        self.proposer_worker.load_model()

        if self._enable_lm_head_weight_load:
            # NOTE(Shangming): gather lm_head weight when tp enabled
            target_lm_head_weight: torch.Tensor = get_tp_group().gather(
                self.scorer_worker.model_runner.model_runner.model.lm_head.\
                    weight.data,
                    dim=0,
            )

            if self.rank == self._driver_rank:
                start_index = target_lm_head_weight.size(0) // self.scorer_worker.parallel_config.tensor_parallel_size
                self.proposer_worker.maybe_load_lm_head_weight(
                    target_lm_head_weight[start_index:])

                # [Parallel SD] Remove scorer model on driver rank
                del self.scorer_worker.model_runner.model_runner.model

        self._metrics.init_tensors(self.rank, device_type=self.device)
        if model_parallel_is_initialized():
            self.spec_decode_sampler.init_tensors(get_tp_group().local_rank,
                                                  device_type=self.device)
        else:
            self.spec_decode_sampler.init_tensors(self.rank,
                                                  device_type=self.device)

        # [Parallel SD] Use parallel version of scorer
        scorer_cls: Type[ParallelSpeculativeScorer]
        if self.disable_mqa_scorer:
            scorer_cls = ParallelBatchExpansionTop1Scorer
            logger.info("[Speculative Decoding] Use batch "
                        "expansion for scoring proposals.")
        else:
            scorer_cls = ParallelMQAScorer
            logger.info(
                "[Speculative Decoding] Use MQA scorer for scoring proposals.")

        self.scorer = scorer_cls(scorer_worker=self.scorer_worker,
                                 device=self.device,
                                 vocab_size=self._vocab_size,
                                 probs_dtype=self.probs_dtype,
                                 token_id_dtype=self.token_id_dtype)

        self._configure_model_sampler_for_spec_decode()

    @property
    def is_repr_scorer(self):
        return get_non_driver_tp_group().is_first_rank

    def _configure_model_sampler_for_spec_decode(self):
        # [Parallel SD] Only non-driver ranks run the scorer
        if self.rank != self._driver_rank:
            (self.scorer_worker.model_runner.sampler.include_gpu_probs_tensor
            ) = True
            (self.scorer_worker.model_runner.sampler.
            should_modify_greedy_probs_inplace) = True
        self.proposer_worker.set_include_gpu_probs_tensor()
        self.proposer_worker.set_should_modify_greedy_probs_inplace()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        if self.rank == self._driver_rank:
            # [Parallel SD] Driver rank determines the number of
            # available blocks given the proposer if applicable.
            try:
                result = self.proposer_worker.determine_num_available_blocks()
                num_gpu_blocks, num_cpu_blocks = result
                assert num_gpu_blocks > 0 and num_cpu_blocks > 0
                return num_gpu_blocks, num_cpu_blocks
            except Exception:
                return (1 << 32) - 1, (1 << 32) - 1
        else:
            return self.scorer_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.scorer_worker.parallel_config.tensor_parallel_size -= 1

        if self.rank != self._driver_rank:
            self.scorer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                                num_cpu_blocks=num_cpu_blocks)
        self.proposer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)

        self.scorer_worker.parallel_config.tensor_parallel_size += 1

    def _run_non_driver_rank(self) -> bool:
        assert self.rank != self._driver_rank

        def send_sampler_output_to_driver(
            sampler_output: Union[List[SamplerOutput], None]
        ) -> None:
            if (
                type(sampler_output) is list
                and len(sampler_output) == 1
                and type(sampler_output[0]) is SamplerOutput
            ):
                sampler_output: SamplerOutput = sampler_output[0]
                size = len(sampler_output.sampled_token_ids)
                hidden_size = 0 if sampler_output.hidden_states is None else \
                    sampler_output.hidden_states.shape[-1]
                get_tp_group().send(
                    torch.tensor(
                        [size, hidden_size],
                        dtype=torch.int64,
                        device=self.device
                    ),
                    dst=0
                )
                get_tp_group().send(
                    sampler_output.sampled_token_ids,
                    dst=0
                )
                get_tp_group().send(
                    sampler_output.sampled_token_probs,
                    dst=0
                )
                get_tp_group().send(
                    sampler_output.logprobs,
                    dst=0
                )
                if sampler_output.hidden_states is not None:
                    get_tp_group().send(
                        sampler_output.hidden_states,
                        dst=0
                    )
                if sampler_output.prefill_hidden_states is not None:
                    prefill_hidden_size = sampler_output.prefill_hidden_states.shape[0]
                    get_tp_group().send(
                        torch.tensor(
                            [prefill_hidden_size],
                            dtype=torch.int64,
                            device=self.device
                        ),
                        dst=0
                    )
                    get_tp_group().send(
                        sampler_output.prefill_hidden_states,
                        dst=0
                    )
                else:
                    get_tp_group().send(
                        torch.zeros(1, dtype=torch.int64, device=self.device),
                        dst=0
                    )
            else:
                get_tp_group().send(
                    torch.zeros(2, dtype=torch.uint32, device=self.device),
                    dst=0
                )

        data = broadcast_tensor_dict(src=self._driver_rank)
        if not data:
            return False
        num_lookahead_slots = data["num_lookahead_slots"]

        # In case of prefill, scorer_worker has to be run before proposer so
        # that the hidden states can be propagated to proposer when needed.
        if data["no_spec"]:
            target_sampler_output = self.scorer_worker.execute_model()

        if not data["disable_all_speculation"]:
            # Even if num_lookahead_slots is zero, we want to run the
            # proposer model as it may have KV.
            #
            # We run the proposer once per lookahead slot. In the future we
            # should delegate how many times it runs to the proposer.
            for _ in range(max(num_lookahead_slots, 1)):
                self.proposer_worker.execute_model()

        if not data["no_spec"]:
            with nvtx_range("score"):
                target_sampler_output = self.scorer_worker.execute_model()
            if data["run_spec_proposer_for_prefill"]:
                self.proposer_worker.execute_model()

        if self.is_repr_scorer:
            # [Parallel SD] We are the representative scorer, send fields of
            # target sampler output needed by verification back to the driver rank.
            send_sampler_output_to_driver(target_sampler_output)
        return True

    def _split_batched_requests(
        self,
        is_sending_batch: torch.BoolTensor,
        execute_model_req: MineExecuteModelRequest,
    ) -> MineExecuteModelRequest:
        """Split the batched requests into two sub-batches according to
        batch_flag in MineSequenceGroupMetadata.
        """
        if self.use_pearl:
            execute_model_req.previous_hidden_states = self.previous_hidden_states
            self.previous_hidden_states = None
            return execute_model_req.clone([])

        # Corner case: one sub-batch is empty
        if not is_sending_batch.any() or is_sending_batch.all():
            is_sending_batch[:] = True
            execute_model_req.previous_hidden_states = self.previous_hidden_states
            self.previous_hidden_states = None
            return execute_model_req.clone([])

        seq_group_metadata_list1 = []
        seq_group_metadata_list2 = []
        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
            if seq_group_metadata.batch_flag == self._send_batch_flag:
                seq_group_metadata_list1.append(seq_group_metadata)
            else:
                seq_group_metadata_list2.append(seq_group_metadata)
        execute_model_req2 = execute_model_req.clone(seq_group_metadata_list2)
        execute_model_req.seq_group_metadata_list = seq_group_metadata_list1

        if execute_model_req.last_sampled_token_ids is not None:
            execute_model_req2.last_sampled_token_ids = \
                execute_model_req.last_sampled_token_ids[~is_sending_batch]
            execute_model_req.last_sampled_token_ids = \
                execute_model_req.last_sampled_token_ids[is_sending_batch]

        if self.previous_hidden_states is not None:
            execute_model_req2.previous_hidden_states = self.previous_hidden_states.clone(
                seq_group_metadata_list2
            )
            execute_model_req.previous_hidden_states = self.previous_hidden_states.clone(
                seq_group_metadata_list1
            )

        execute_model_req2.scoring_async_handle = None
        return execute_model_req2

    def _update_execute_model_req(
        self,
        execute_model_req: MineExecuteModelRequest,
        proposals: SpeculativeProposals,
        drafted: torch.Tensor
    ) -> MineExecuteModelRequest:
        """Use the proposal tokens to update the SequenceData
        as if they are verified proposals.
        """

        new_seq_group_metadata_list = []
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        for i, seq_group_meta in enumerate(seq_group_metadata_list):
            proposal_len = drafted[i].item()
            if proposal_len == 0:
                # This proposal was not verified in the previous step.
                # Keep the output tokens of the sequence.
                new_seq_group_metadata_list.append(seq_group_meta)
                continue

            assert len(seq_group_meta.seq_data) == 1
            seq_id = next(iter(seq_group_meta.seq_data.keys()))

            seq_data = seq_group_meta.seq_data[seq_id]
            new_seq_data = SequenceData.from_seqs(
                prompt_token_ids=seq_data.prompt_token_ids
            )
            new_seq_data._stage = seq_data.stage
            for token_id in seq_data.output_token_ids:
                logprob = seq_data.cumulative_logprob
                new_seq_data.append_token_id(token_id, logprob)
            for token_id in proposals.proposal_token_ids[i, :proposal_len]:
                new_seq_data.append_token_id(token_id.item(), 0.0)
            new_seq_data.update_num_computed_tokens(
                seq_data.get_num_computed_tokens() + proposal_len
            )
            new_seq_group_metadata_list.append(
                MineSequenceGroupMetadata(
                    request_id=seq_group_meta.request_id,
                    is_prompt=seq_group_meta.is_prompt,
                    seq_data={seq_id: new_seq_data},
                    sampling_params=seq_group_meta.sampling_params,
                    block_tables=seq_group_meta.block_tables,
                    state=seq_group_meta.state,
                    num_speculative_tokens=seq_group_meta.num_speculative_tokens,
                )
            )

        return execute_model_req.clone(new_seq_group_metadata_list)

    def _prepare_previous_proposals(
        self,
        execute_model_req: MineExecuteModelRequest) -> SpeculativeProposals:
        """Prepare proposals generated in the last step for speculative decoding
        """
        mask = self._valid_previous_proposals
        previous_proposals = self._previous_proposals
        self._previous_proposals = None

        if self.use_pearl:
            if previous_proposals is None or len(self._previous_request_ids) == 0 or not mask.any():
                bs = len(execute_model_req.seq_group_metadata_list)
                k = execute_model_req.num_lookahead_slots

                proposal_lens = torch.zeros((bs,), dtype=torch.int64,
                                            device=self.device)
                proposal_logprobs = torch.zeros((bs, k, self._vocab_size),
                                               dtype=self.probs_dtype,
                                               device=self.device)
                proposal_probs = torch.zeros((bs, k, self._vocab_size),
                                             dtype=self.probs_dtype,
                                             device=self.device)
                proposal_token_ids = torch.full((bs, k),
                                               fill_value=VLLM_INVALID_TOKEN_ID,
                                               dtype=self.token_id_dtype,
                                               device=self.device)
                return SpeculativeProposals(
                    proposal_lens=proposal_lens,
                    proposal_logprobs=proposal_logprobs,
                    proposal_probs=proposal_probs,
                    proposal_token_ids=proposal_token_ids,
                    no_proposals=True
                )

            bs = len(execute_model_req.seq_group_metadata_list)
            k = execute_model_req.num_lookahead_slots

            proposal_lens = previous_proposals.proposal_lens.new_zeros((bs,))
            proposal_logprobs = previous_proposals.proposal_logprobs.new_zeros((bs, k, self._vocab_size))
            proposal_probs = previous_proposals.proposal_probs.new_zeros((bs, k, self._vocab_size))
            proposal_token_ids = previous_proposals.proposal_token_ids.new_full((bs, k), fill_value=VLLM_INVALID_TOKEN_ID)

            for new_i, seq_metadata in enumerate(execute_model_req.seq_group_metadata_list):
                req_id = seq_metadata.request_id
                if (old_i := self._previous_request_ids.get(req_id)) is not None:
                    proposal_lens[new_i] = previous_proposals.proposal_lens[old_i]
                    proposal_logprobs[new_i] = previous_proposals.proposal_logprobs[old_i]
                    proposal_probs[new_i] = previous_proposals.proposal_probs[old_i]
                    proposal_token_ids[new_i] = previous_proposals.proposal_token_ids[old_i]

            return SpeculativeProposals(
                proposal_lens=proposal_lens,
                proposal_logprobs=proposal_logprobs,
                proposal_probs=proposal_probs,
                proposal_token_ids=proposal_token_ids,
                no_proposals=execute_model_req.num_lookahead_slots == 0
            )

        if len(execute_model_req.seq_group_metadata_list) != len(mask):
            # Parallel speculative decoding w/ chunked prefill (prefill + decodes)
            bs = len(execute_model_req.seq_group_metadata_list)
            k = execute_model_req.num_lookahead_slots
            non_spec_len = bs - len(mask)

            new_mask = mask.new_zeros((bs,))
            proposal_lens = previous_proposals.proposal_lens.new_zeros((bs,))
            proposal_logprobs = previous_proposals.proposal_logprobs.new_zeros((bs, k, self._vocab_size))
            proposal_probs = previous_proposals.proposal_probs.new_zeros((bs, k, self._vocab_size))
            proposal_token_ids = previous_proposals.proposal_token_ids.new_full((bs, k), fill_value=VLLM_INVALID_TOKEN_ID)

            new_mask[non_spec_len:] = mask
            proposal_lens[new_mask] = previous_proposals.proposal_lens
            proposal_logprobs[new_mask] = previous_proposals.proposal_logprobs
            proposal_probs[new_mask] = previous_proposals.proposal_probs
            proposal_token_ids[new_mask] = previous_proposals.proposal_token_ids
            return SpeculativeProposals(
                proposal_lens=proposal_lens,
                proposal_logprobs=proposal_logprobs,
                proposal_probs=proposal_probs,
                proposal_token_ids=proposal_token_ids
            )

        return SpeculativeProposals(
            proposal_lens=previous_proposals.proposal_lens[mask],
            proposal_logprobs=previous_proposals.proposal_logprobs[mask],
            proposal_probs=previous_proposals.proposal_probs[mask],
            proposal_token_ids=previous_proposals.proposal_token_ids[mask]
        )

    @nvtx_range("ParallelSpecDecodeWorker._run_no_spec")
    def _run_no_spec(self,
                     execute_model_req: MineExecuteModelRequest,
                     skip_proposer: bool,
                     profile_time: bool = False) -> List[SamplerOutput]:

        with Timer(profile_time) as timer:
            # [Parallel SD] Execute the scorer model.
            self.scorer_worker.execute_model(execute_model_req)

            sampler_output = self.scorer._recv_sampler_output_from_repr_scorer()

            # Store hidden states from target model execution, BxD.
            hidden_states = sampler_output.hidden_states
            if hidden_states is not None:
                # Only decodes and prefill terminal chunks need a hidden state.
                seq_group_meta_with_hidden = [
                    sg for sg in execute_model_req.seq_group_metadata_list
                    if sg.do_sample
                ]
                if any(seq.is_prompt for seq in seq_group_meta_with_hidden):
                    # Drop hidden_states with no prediction (eg non-terminal chunks)
                    hidden_states = hidden_states[
                        torch.where(sampler_output.sampled_token_ids -
                                    VLLM_INVALID_TOKEN_ID)[0]]
                if self.previous_hidden_states is None and len(
                        seq_group_meta_with_hidden):
                    self.previous_hidden_states = HiddenStates(
                        hidden_states, seq_group_meta_with_hidden)
                elif self.previous_hidden_states and len(
                        seq_group_meta_with_hidden):
                    self.previous_hidden_states.update(hidden_states,
                                                       seq_group_meta_with_hidden)
                    if not self.use_pearl:
                        seq_group_meta_with_hidden = [
                            sg for sg in self.previous_hidden_states.seq_group_metadata_list
                            if sg.batch_flag == self._send_batch_flag
                        ] + seq_group_meta_with_hidden
                        self.previous_hidden_states.seq_group_metadata_list = \
                            seq_group_meta_with_hidden
                    self.previous_hidden_states.prune(seq_group_meta_with_hidden)

            if not skip_proposer:
                # We prepare the prefill hidden states here so that there no
                # additional complexity in worker for spec_decode vs non_spec_decode
                # flow and execute_model doesn't need additional modifications.
                execute_model_req.previous_hidden_states = \
                    prepare_prefill_hidden_states(
                        sampler_output.prefill_hidden_states)
                execute_model_req.is_proposing = True
                for i in range(self._num_spec_prefill_steps):
                    execute_model_req.spec_step_idx = i
                    self.proposer_worker.execute_model(execute_model_req)
                execute_model_req.is_proposing = False

            sampler_output_to_return = (self._serialize_sampler_output_no_logprobs(
                execute_model_req=execute_model_req, sampler_output=sampler_output)
                                        if self._disable_logprobs else
                                        [sampler_output])

            # Clear device tensors from sampler output. This reduces communication
            # overhead when the engine runs in a different process than the workers.
            sampler_output.sampled_token_probs = None
            sampler_output.sampled_token_ids = None
            sampler_output.logprobs = None

        # [Parallel SD] Record the end time of the batch
        step_trace: Step = TRACER.current_step
        if profile_time:
            step_trace.measured_target_time = timer.elapsed_perf_time
        step_trace.batch_end_us = time.perf_counter() * 1e6

        return sampler_output_to_return

    @nvtx_range("ParallelSpecDecodeWorker._run_speculative_decoding_step")
    def _run_speculative_decoding_step(
        self,
        execute_model_req: MineExecuteModelRequest,
        num_lookahead_slots: int,
        profile_time: bool = False) -> List[SamplerOutput]:

        assert self.use_parallel, "ParallelSpecDecodeWorker does not support " \
            "use_parallel == False, use SpecDecodeWorker instead"
        # With prefill chunking, expect requests to have prompts first
        # so that backend gets prefill|decode.
        assert num_lookahead_slots == execute_model_req.num_lookahead_slots

        k = num_lookahead_slots
        tetris_time = None
        self.sd_step += 1
        cur_step_trace: Step = TRACER.current_step

        with Timer(profile_time) as prepare_timer:
            orig_seq_group_metadata_list = \
                execute_model_req.seq_group_metadata_list.copy()
            if not self.use_pearl:
                is_sending_batch = torch.tensor(
                    [
                        seq_group_metadata.batch_flag == self._send_batch_flag
                        for seq_group_metadata in orig_seq_group_metadata_list
                    ]
                )
            else:
                is_sending_batch = (
                    self._valid_previous_proposals if self._valid_previous_proposals is not None
                    else torch.zeros(len(orig_seq_group_metadata_list), dtype=torch.bool)
                )
        with Timer(profile_time) as split_timer:
            execute_model_req2 = self._split_batched_requests(
                is_sending_batch, execute_model_req)

        if self.use_pearl or self._previous_proposals is not None:
            cur_step_trace.is_parallelised = True

            # Pre-process proposals of the current batch to get proposal length
            # and number of lookahead slots.
            execute_model_req, num_lookahead_slots, proposal_len = \
                self._preprocess(execute_model_req, k, profile_time)
            cur_step_trace.proposed_len = proposal_len

            # Retrieve proposals generated in the previous step
            with Timer(profile_time) as proposal_timer:
                proposals = self._prepare_previous_proposals(
                    execute_model_req)

            drafted = proposals.proposal_lens
            cur_step_trace.match_count = (drafted > 0).sum()
            cur_step_trace.verify_len = drafted.sum()
            verify_len = drafted.max().item()
        else:
            cur_step_trace.is_parallelised = False

            # Pre-process proposals
            execute_model_req, num_lookahead_slots, proposal_len = \
                self._preprocess(execute_model_req, k, profile_time)
            cur_step_trace.proposed_len = proposal_len

            # Propose proposals
            if proposal_len == 0:
                for seq_group_metadata in orig_seq_group_metadata_list:
                    seq_group_metadata.num_speculative_tokens = 0
                self.disable_by_batch_size = execute_model_req.running_queue_size
                broadcast_dict = dict(
                    num_lookahead_slots=num_lookahead_slots,
                    no_spec=True,
                    disable_all_speculation=True,
                )
                broadcast_tensor_dict(broadcast_dict, src=self._driver_rank,
                                      async_op=self.use_parallel)
                return self._run_no_spec(execute_model_req, skip_proposer=True,
                                         profile_time=profile_time)

            execute_model_req.is_proposing = True
            with Timer(profile_time) as proposal_timer, nvtx_range("propose"):
                # Generate proposals using draft worker.
                proposals = self.proposer_worker.get_spec_proposals(
                    execute_model_req,
                    self._seq_with_bonus_token_in_last_step
                )

            cur_step_trace.match_count = (proposals.proposal_lens > 0).sum()

            if not self._allow_zero_draft_token_step and proposals.no_proposals:
                #TODO: Fix it #5814
                raise RuntimeError("Cannot handle cases where distributed draft "
                                "workers generate no tokens")

            execute_model_req.previous_hidden_states = None
            execute_model_req.is_proposing = False

            # Post-process proposals
            proposals, verify_len, tetris_time = self._postprocess(
                execute_model_req, proposal_len,
                cur_step_trace, proposals,
                profile_time)

            drafted = proposals.proposal_lens
            cur_step_trace.verify_len = drafted.sum()


        # Start scoring proposals of the current batch
        with Timer(profile_time) as start_scoring_timer:
            self.scorer.start_score_proposals(execute_model_req, proposals)

        if self.use_pearl:
            dummy_step_trace = Step()

            # Update execute_model_req with the last proposals
            mock_execute_model_req = self._update_execute_model_req(
                execute_model_req, proposals, drafted
            )

            mock_execute_model_req, _, proposal_len2 = self._preprocess(
                mock_execute_model_req, k, profile_time
            )

            # Propose next proposals
            if proposal_len2 > 0:
                mock_execute_model_req.is_proposing = True
                proposals2 = self.proposer_worker.get_spec_proposals(
                    mock_execute_model_req,
                    self._seq_with_bonus_token_in_last_step
                )

                if not self._allow_zero_draft_token_step and proposals2.no_proposals:
                    #TODO: Fix it #5814
                    raise RuntimeError("Cannot handle cases where distributed draft "
                                    "workers generate no tokens")

                mock_execute_model_req.previous_hidden_states = None
                mock_execute_model_req.is_proposing = False

                # Post-process next proposals
                proposals2, _, _ = self._postprocess(
                    mock_execute_model_req, proposal_len2,
                    dummy_step_trace, proposals2,
                    profile_time)
        else:
            # Pre-process proposals of the other batch
            # Note that we may not have proposals for the other batch
            if len(execute_model_req2.seq_group_metadata_list) > 0:
                dummy_step_trace = Step()
                execute_model_req2, _, proposal_len2 = (
                    self._preprocess(
                        execute_model_req2, k, profile_time
                    )
                )

                # Propose proposals of the other batch
                if proposal_len2 > 0:
                    execute_model_req2.is_proposing = True
                    with Timer(profile_time) as proposal2_timer, nvtx_range("propose"):
                        # Generate proposals using draft worker, i.e., driver rank.
                        proposals2 = self.proposer_worker.get_spec_proposals(
                            execute_model_req2,
                            self._seq_with_bonus_token_in_last_step,
                        )

                    if not self._allow_zero_draft_token_step and proposals2.no_proposals:
                        #TODO: Fix it #5814
                        raise RuntimeError("Cannot handle cases where distributed draft "
                                        "workers generate no tokens")

                    execute_model_req2.previous_hidden_states = None
                    execute_model_req2.is_proposing = False

                    # Post-process proposals of the other batch
                    proposals2, _, _ = self._postprocess(
                        execute_model_req2, proposal_len2,
                        dummy_step_trace, proposals2,
                        profile_time)

        # Get scores of proposals of the current batch
        with Timer(profile_time) as scoring_timer:
            proposal_scores = self.scorer.score_proposals(
                execute_model_req,
                proposals,
            )

        _, (non_spec_seqs, non_spec_indices) = split_batch_by_proposal_len(
            execute_model_req.seq_group_metadata_list, proposals.proposal_lens)
        # With prefill chunking enabled, `non_spec_seqs` contains prefills too:
        # discard decodes that have already been processed by proposer.
        non_spec_indices = [
            idx for idx in non_spec_indices
            if execute_model_req.seq_group_metadata_list[idx].is_prompt
        ]
        if len(non_spec_indices):
            all_hidden_states = proposal_scores.hidden_states
            if all_hidden_states is not None:
                prefill_hidden_states = all_hidden_states[non_spec_indices]
                execute_model_req.previous_hidden_states = \
                    prepare_prefill_hidden_states(prefill_hidden_states)
            # Sync proposer KV cache for prefills.
            prefill_req = execute_model_req.clone(non_spec_seqs)
            # TODO avoid sampling here?
            execute_model_req.is_proposing = True
            self.proposer_worker.execute_model(prefill_req)
            execute_model_req.is_proposing = False

        # Verify proposals of the current batch
        with Timer(profile_time) as verification_timer:
            accepted_token_ids, target_logprobs = self._verify_tokens(
                orig_seq_group_metadata_list,
                execute_model_req.seq_group_metadata_list,
                proposal_scores,
                proposals,
                k
            )

        stage_times = (
            proposal_timer.elapsed_time_ms / num_lookahead_slots,
            (
                start_scoring_timer.elapsed_time_ms +
                (proposal2_timer.elapsed_time_ms
                if len(execute_model_req2.seq_group_metadata_list) > 0 and proposal_len2 > 0 else 0.) +
                scoring_timer.elapsed_time_ms
            ),
            verification_timer.elapsed_time_ms,
            tetris_time
        )

        with Timer(profile_time) as overhead_timer:
            out = self._create_output_sampler_list(
                is_sending_batch,
                orig_seq_group_metadata_list,
                accepted_token_ids,
                target_logprobs=target_logprobs,
                prompt_logprobs=proposal_scores.prompt_logprobs
                if not self._disable_logprobs else None,
                k=k,
                stage_times=stage_times)

        # Store proposals of the other batch
        if self.use_pearl:
            if proposal_len2 > 0:
                self._previous_proposals = proposals2
                accepted = (accepted_token_ids >= 0).sum(dim=1)
                has_accepted_all = (
                    ((accepted > drafted) & (drafted > 0)) |
                    ((proposals2.proposal_token_ids[:, 0] == accepted_token_ids[:, 0]) & (drafted == 0))
                )
                self._previous_request_ids = {
                    seq_group_metadata.request_id: i
                    for i, seq_group_metadata in enumerate(
                        mock_execute_model_req.seq_group_metadata_list)
                }
                self._valid_previous_proposals = has_accepted_all
            else:
                self._previous_request_ids = None
                self._valid_previous_proposals = None
        else:
            if len(execute_model_req2.seq_group_metadata_list) > 0 and proposal_len2 > 0:
                self._previous_proposals = proposals2
                self._send_batch_flag = not self._send_batch_flag
                self._previous_request_ids = {
                    seq_group_metadata.request_id: i
                    for i, seq_group_metadata in enumerate(
                        execute_model_req2.seq_group_metadata_list)
                }
                self._valid_previous_proposals = torch.ones(
                    len(execute_model_req2.seq_group_metadata_list),
                    dtype=torch.bool
                )
            else:
                self._send_batch_flag = True
                self._previous_request_ids = None
                self._valid_previous_proposals = None

        if profile_time:
            cur_step_trace.measured_draft_time = proposal_timer.elapsed_perf_time
            cur_step_trace.measured_target_time = (
                start_scoring_timer.elapsed_perf_time +
                (proposal2_timer.elapsed_perf_time
                if len(execute_model_req2.seq_group_metadata_list) > 0 and proposal_len2 > 0 else 0.) +
                scoring_timer.elapsed_perf_time
            )
            cur_step_trace.measured_overhead_time = (
                prepare_timer.elapsed_perf_time +
                split_timer.elapsed_perf_time +
                verification_timer.elapsed_perf_time +
                overhead_timer.elapsed_perf_time
            )

        execute_model_req.seq_group_metadata_list = orig_seq_group_metadata_list
        cur_step_trace.batch_end_us = time.perf_counter() * 1e6
        return out

    @nvtx_range("ParallelSpecDecodeWorker._verify_tokens")
    def _verify_tokens(
        self,
        orig_seq_group_metadata_list: List[MineSequenceGroupMetadata],
        seq_group_metadata_list: List[MineSequenceGroupMetadata],
        proposal_scores: SpeculativeScores,
        proposals: SpeculativeProposals,
        max_proposal_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        proposal_lens_list = proposals.proposal_lens.tolist()

        # vLLM currently only supports proposal lens equal to zero or the batch
        # proposal len. This adds some complexity (splitting the batch into spec
        # and non spec sequences) and should be removed in the future. It can be
        # done by supporting per-sequence proposal lens.
        (_, spec_indices), (_, non_spec_indices) = split_batch_by_proposal_len(
            seq_group_metadata_list, proposal_lens_list)
        original_indices = spec_indices + non_spec_indices

        # [Parallel SD] Get lengths of speculative proposals.
        spec_proposal_lens = proposals.proposal_lens[spec_indices]

        # Get probabilities of target model, including bonus tokens.
        proposal_verifier_probs = proposal_scores.probs[spec_indices]

        # Get non-speculative sampled tokens from target model.
        non_spec_token_ids = proposal_scores.token_ids[non_spec_indices]

        # Get bonus tokens from target model.
        # bonus_token_ids = proposal_scores.token_ids[spec_indices, -1:]
        # [Parallel SD] Edited as below to adapt varlen
        bonus_token_ids = proposal_scores.token_ids[spec_indices, spec_proposal_lens].unsqueeze(-1)

        # Get probabilities according to proposal method.
        proposal_probs = proposals.proposal_probs[spec_indices]

        # Get proposed tokens.
        proposal_token_ids = proposals.proposal_token_ids[spec_indices]

        # Sampler arguments
        sampler_extra_kwargs: Dict[str, Any] = {}
        if self.generators and isinstance(self.spec_decode_sampler,
                                          SpecDecodeStochasticBaseSampler):
            sampler_extra_kwargs["seeded_seqs"] = {
                idx: self.generators[sgm.request_id]
                for idx, sgm in enumerate(seq_group_metadata_list)
                if sgm.sampling_params.seed is not None
            }

        accepted_token_ids = self.spec_decode_sampler(
            target_with_bonus_probs=proposal_verifier_probs,
            bonus_token_ids=bonus_token_ids,
            draft_probs=proposal_probs,
            draft_token_ids=proposal_token_ids,
            proposal_lens=spec_proposal_lens,
            total_num_seqs=len(seq_group_metadata_list),
            **sampler_extra_kwargs,
        )

        # [Parallel SD] Record the number of accepted tokens
        cur_step_trace: Step = TRACER.current_step
        assert len(
            cur_step_trace.batched_requests) >= accepted_token_ids.shape[0]
        cur_step_trace.accepted_num = (
            (accepted_token_ids >= 0).sum() - accepted_token_ids.shape[0]
        )

        # Append output tokens from non-speculative sequences to
        # the accepted token ids tensor.
        non_spec_token_ids = non_spec_token_ids.expand(-1, max_proposal_len +
                                                       1).clone()
        non_spec_token_ids[:, 1:] = -1
        accepted_token_ids = torch.cat(
            [accepted_token_ids, non_spec_token_ids])
        # [Parallel SD] Record the number of generated tokens
        cur_step_trace.generated_num = (accepted_token_ids >= 0).sum()

        logprobs = proposal_scores.logprobs
        # Rearrange so that results are in the order of the original seq group
        # metadata.
        accepted_token_ids[original_indices] = accepted_token_ids.clone()

        # B x K+1 x D
        hidden_states = proposal_scores.hidden_states
        if hidden_states is not None:
            # Only get terminal hidden states for next step
            terminal_metadata = [
                sg for sg in seq_group_metadata_list if sg.do_sample
            ]

            # Contract hidden states based on accepted tokens
            hs_size = hidden_states.shape[-1]
            accepted_index = accepted_token_ids + 1  # Convert -1 to 0
            accepted_index = accepted_index.count_nonzero(dim=1).add_(-1)  # b
            # Drop non-terminal prefill chunks hidden states.
            hidden_states = hidden_states[accepted_index !=
                                          VLLM_INVALID_TOKEN_ID]
            accepted_index = accepted_index[accepted_index !=
                                            VLLM_INVALID_TOKEN_ID]
            assert len(accepted_index) == hidden_states.shape[0] == len(
                terminal_metadata)

            # [Parallel SD] Get second_last_token_hidden_states and hidden_states
            # in the varlen case.
            second_last_index = (proposals.proposal_lens - 2).clamp_min(0)[:, None, None].expand(-1, 1,
                                                                                                 hs_size)  # b x 1 x d
            new_second_last_token_hidden_states = hidden_states.gather(1, second_last_index).squeeze(1)    # b x d
            index = accepted_index[:, None, None].expand(-1, 1, hs_size)   # b x 1 x d
            new_hidden_states = hidden_states.gather(1, index).squeeze(1)  # b x d

            # Store hidden states from target model for subsequent decode step
            if self.use_pearl or self.previous_hidden_states is None:
                self.previous_hidden_states = HiddenStates(
                    new_hidden_states, terminal_metadata,
                    new_second_last_token_hidden_states)
            elif len(self.previous_hidden_states.hidden_states) == len(orig_seq_group_metadata_list):
                terminal_seq_ids = set(get_all_seq_ids(terminal_metadata))
                prev_index = []
                for i, seq_id in enumerate(self.previous_hidden_states._seq_ids):
                    if seq_id in terminal_seq_ids:
                        prev_index.append(i)
                    if len(prev_index) == len(terminal_metadata):
                        break
                self.previous_hidden_states.hidden_states[prev_index] = new_hidden_states
                if self.previous_hidden_states.second_last_token_hidden_states is None:
                    self.previous_hidden_states.second_last_token_hidden_states = \
                        new_second_last_token_hidden_states.new_zeros(len(orig_seq_group_metadata_list), hs_size)
                self.previous_hidden_states.second_last_token_hidden_states[prev_index] = \
                    new_second_last_token_hidden_states
                for new_i, old_i in enumerate(prev_index):
                    self.previous_hidden_states.seq_group_metadata_list[old_i] = \
                        terminal_metadata[new_i]
            else:
                # Collect previous hidden states of alternative batch
                orig_seq_ids = set(get_all_seq_ids(orig_seq_group_metadata_list))
                prev_seq_ids = orig_seq_ids.difference(get_all_seq_ids(terminal_metadata))
                prev_index = []
                for i, seq_id in enumerate(self.previous_hidden_states._seq_ids):
                    if seq_id in prev_seq_ids:
                        prev_index.append(i)
                    if len(prev_index) == len(prev_seq_ids):
                        break

                # Construct seq_group_metadata_list of alternative batch by combining
                # the ones in previous_hidden_states and orig_seq_group_metadata_list
                alt_seq_group_metadata_list = []
                new_index = []
                old_sg_metadata = {
                    seq_id: sg for seq_id, sg in
                    zip(
                        self.previous_hidden_states._seq_ids,
                        self.previous_hidden_states.seq_group_metadata_list
                    )
                }
                for sg in orig_seq_group_metadata_list:
                    if sg.batch_flag == self._send_batch_flag:
                        continue

                    seq_id = next(iter(sg.seq_data.keys()))
                    if seq_id not in old_sg_metadata:
                        alt_seq_group_metadata_list.append(sg)
                        continue

                    new_index.append(len(alt_seq_group_metadata_list))
                    alt_seq_group_metadata_list.append(old_sg_metadata[seq_id])

                # Construct padded hidden states of alternative batch
                alt_hidden_states = self.previous_hidden_states.hidden_states.new_zeros(
                    len(alt_seq_group_metadata_list), hs_size)
                alt_second_last_token_hidden_states = \
                    self.previous_hidden_states.hidden_states.new_zeros(
                        len(alt_seq_group_metadata_list), hs_size)
                alt_hidden_states[new_index] = self.previous_hidden_states.hidden_states[prev_index]
                if self.previous_hidden_states.second_last_token_hidden_states is not None:
                    alt_second_last_token_hidden_states[new_index] = \
                        self.previous_hidden_states.second_last_token_hidden_states[prev_index]

                # Update previous_hidden_states to contain hidden states of both
                # alternative batch and scoring batch
                self.previous_hidden_states = HiddenStates(
                    new_hidden_states, terminal_metadata,
                    new_second_last_token_hidden_states)
                self.previous_hidden_states.update(alt_hidden_states,
                                                   alt_seq_group_metadata_list,
                                                   alt_second_last_token_hidden_states)
                self.previous_hidden_states.seq_group_metadata_list.extend(alt_seq_group_metadata_list)

        return accepted_token_ids, logprobs

    def _create_output_sampler_list(
        self,
        is_sending_batch: torch.BoolTensor,
        orig_seq_group_metadata_list: List[MineSequenceGroupMetadata],
        accepted_token_ids: torch.Tensor,  # shape: [batch_size, k+1]
        target_logprobs: torch.Tensor,  # shape: [batch_size, k+1, vocab_size]
        prompt_logprobs: Optional[torch.Tensor],
        k: int,
        stage_times: Tuple[float, float, float, Optional[float]]
    ) -> List[SamplerOutput]:
        """Given the accepted token ids of the current batch of requests,
        create a list of SamplerOutput where the requests in the other batch
        are padded with dummy values.

        The output is padded with -1 tokens such that each sequence has
        the same number of outputs.
        """
        if self.use_pearl:
            return super()._create_output_sampler_list(
                orig_seq_group_metadata_list,
                accepted_token_ids,
                target_logprobs=target_logprobs,
                prompt_logprobs=prompt_logprobs,
                k=k,
                stage_times=stage_times
            )

        orig_batch_size = len(orig_seq_group_metadata_list)
        padded_accepted_token_ids = torch.full(
            (orig_batch_size, k + 1),
            VLLM_INVALID_TOKEN_ID,
            dtype=self.token_id_dtype,
            device=self.device
        )
        padded_target_logprobs = torch.full(
            (orig_batch_size, k + 1, self._vocab_size),
            -torch.inf,
            dtype=self.probs_dtype,
            device=self.device
        )
        padded_accepted_token_ids[is_sending_batch] = accepted_token_ids
        padded_target_logprobs[is_sending_batch] = target_logprobs

        return super()._create_output_sampler_list(
            orig_seq_group_metadata_list,
            padded_accepted_token_ids,
            target_logprobs=padded_target_logprobs,
            prompt_logprobs=prompt_logprobs,
            k=k,
            stage_times=stage_times
        )

    def _track_finished_requests(self, execute_model_req: MineExecuteModelRequest):
        if not self.use_pearl or self._previous_request_ids is None:
            return super()._track_finished_requests(execute_model_req)
    
        for finished_request in execute_model_req.finished_requests_ids:
            for seq_id in self._request_id_seq_id_mapping[finished_request]:
                self._seq_with_bonus_token_in_last_step.discard(seq_id)
            del self._request_id_seq_id_mapping[finished_request]
            del self._previous_request_ids[finished_request]

    def _track_preempted_requests(self, execute_model_req: MineExecuteModelRequest):
        """
        Removes proposals of the preempted requests
        """
        if self._previous_request_ids is None:
            return
        for preempted_request in execute_model_req.preempted_requests_ids:
            if preempted_request in self._previous_request_ids:
                i = self._previous_request_ids[preempted_request]
                self._valid_previous_proposals[i] = False