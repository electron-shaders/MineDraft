from vllm.sequence import VLLM_INVALID_TOKEN_ID
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScores
from vllm.spec_decode.util import nvtx_range

from minedraft.plugin.sequence import MineExecuteModelRequest
from minedraft.plugin.spec_decode.interfaces import ParallelSpeculativeScorer


class ParallelBatchExpansionTop1Scorer(BatchExpansionTop1Scorer,
                                       ParallelSpeculativeScorer):
    @nvtx_range("ParallelBatchExpansionTop1Scorer.start_score_proposals")
    def start_score_proposals(
        self,
        execute_model_req: MineExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> None:

        # TODO(cade) perform this on GPU to remove blocking call.
        proposal_lens_list = proposals.proposal_lens.tolist()
        proposal_token_ids_list = proposals.proposal_token_ids.tolist()

        # Filter the list to ignore invalid proposals.
        proposal_token_ids_list_without_skips = [
            proposals for proposals in proposal_token_ids_list
            if VLLM_INVALID_TOKEN_ID not in proposals
        ]

        (spec_indices, non_spec_indices, target_seq_group_metadata_list,
         num_scoring_tokens) = self._expand_batch(
             seq_group_metadata_list=execute_model_req.seq_group_metadata_list,
             proposal_token_ids_list=proposal_token_ids_list_without_skips,
             proposal_lens_list=proposal_lens_list,
         )

        target_execute_model_req = execute_model_req.clone(
            seq_group_metadata_list=target_seq_group_metadata_list)
        self._scorer_worker.execute_model(execute_model_req=target_execute_model_req)
        execute_model_req.scoring_async_handle = target_execute_model_req.scoring_async_handle

    @nvtx_range("ParallelBatchExpansionTop1Scorer.score_proposals")
    def score_proposals(
        self,
        execute_model_req: MineExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        """Receive the score the proposed tokens from the representative scorer.
        
        This converts each input sequence to a set of k+1 target sequences. The
        target sequences have the unique continuations to be scored and a
        unique sequence ID that is different from all input sequence ids.

        If a speculative sequence length would exceed the max model length, then
        no speculation is produced for that sequence.

        Args:
            execute_model_req: The execution request.
            proposals: The speculative proposals to score.
        Returns:
            SpeculativeScores: The scores of each speculative token, along with
                which sequences were ignored during scoring.
        """
        work = execute_model_req.scoring_async_handle
        execute_model_req.scoring_async_handle = None
        assert work is not None, "No async handle found for scoring"
        work.wait()

        target_sampler_output = self._recv_sampler_output_from_repr_scorer()

        # TODO(cade) perform this on GPU to remove blocking call.
        proposal_lens_list = proposals.proposal_lens.tolist()
        proposal_token_ids_list = proposals.proposal_token_ids.tolist()

        # Filter the list to ignore invalid proposals.
        proposal_token_ids_list_without_skips = [
            proposals for proposals in proposal_token_ids_list
            if VLLM_INVALID_TOKEN_ID not in proposals
        ]

        (spec_indices, non_spec_indices, target_seq_group_metadata_list,
         num_scoring_tokens) = self._expand_batch(
             seq_group_metadata_list=execute_model_req.seq_group_metadata_list,
             proposal_token_ids_list=proposal_token_ids_list_without_skips,
             proposal_lens_list=proposal_lens_list,
         )

        if not non_spec_indices:
            # All sequence groups in batch have spec decoding enabled
            return self._contract_batch_all_spec(
                target_sampler_output=target_sampler_output,
                proposals=proposals,
            )
        else:
            # Batch has a mix of spec decode enabled and disabled seq groups
            return self._contract_batch(
                execute_model_req.seq_group_metadata_list,
                target_sampler_output=target_sampler_output,
                proposals=proposals,
                num_scoring_tokens=num_scoring_tokens,
                non_spec_indices=non_spec_indices,
                spec_indices=spec_indices,
                k=execute_model_req.num_lookahead_slots,
            )