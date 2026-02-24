from typing import List, Optional, Set, Tuple

import torch
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.spec_decode.util import sampler_output_to_torch

from minedraft.patching import MinePatch
from minedraft.plugin.sequence import MineExecuteModelRequest


class Top1ProposerPatch(MinePatch[Top1Proposer]):
    def get_spec_proposals(
        self: Top1Proposer,
        execute_model_req: MineExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:

        proposal_len = execute_model_req.num_lookahead_slots
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        # Split speculative- and non-speculative- sequences.
        (
            proposal_lens,
            nonzero_proposal_len_seqs,
            nonzero_proposal_len_indices,
        ) = self._split_by_proposal_len(seq_group_metadata_list, proposal_len)

        if nonzero_proposal_len_seqs:
            # Speculate tokens using the draft worker for the speculative
            # sequences.
            # If sampler_transposed is true, then maybe_sampler_output's
            # token_ids is like [batch] format in proposal_len size list,
            # while if it is false, the format would be [proposal_len]
            # in batch size list
            hidden_states = execute_model_req.previous_hidden_states
            if hidden_states is not None:
                hidden_states.prune(nonzero_proposal_len_seqs)
            nonzero_execute_model_req = MineExecuteModelRequest(
                seq_group_metadata_list=nonzero_proposal_len_seqs,
                num_lookahead_slots=proposal_len,
                previous_hidden_states=hidden_states,
                is_proposing=True,
            )
            maybe_sampler_output, transposed = self._worker.sampler_output(
                execute_model_req=nonzero_execute_model_req,
                sample_len=proposal_len,
                seq_ids_with_bonus_token_in_last_step=\
                    seq_ids_with_bonus_token_in_last_step,
            )
            (
                proposal_lens,
                maybe_sampler_output,
                nonzero_proposal_len_indices,
            ) = self._remove_no_proposal_seqs(proposal_lens,
                                              maybe_sampler_output,
                                              nonzero_proposal_len_indices,
                                              transposed)
        else:
            # If no sequences can be speculated, set sampler output to None.
            maybe_sampler_output = None
            transposed = False

        # Combine speculative- and non-speculative sequences into the same
        # representation.
        proposal_tokens, proposal_probs, proposal_lens, proposal_logprobs = self._merge_outputs(
            batch_size=len(seq_group_metadata_list),
            proposal_len=proposal_len,
            maybe_sampler_output=maybe_sampler_output,
            proposal_lens=proposal_lens,
            nonzero_proposal_len_indices=nonzero_proposal_len_indices,
            sampler_transposed=transposed,
        )

        proposals = SpeculativeProposals(proposal_token_ids=proposal_tokens,
                                         proposal_probs=proposal_probs,
                                         proposal_lens=proposal_lens,
                                         no_proposals=maybe_sampler_output is None,
                                         proposal_logprobs=proposal_logprobs)
        return proposals

    def _merge_outputs(
        self: Top1Proposer,
        batch_size: int,
        proposal_len: int,
        maybe_sampler_output: Optional[List[SamplerOutput]],
        proposal_lens: List[int],
        nonzero_proposal_len_indices: List[int],
        sampler_transposed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if maybe_sampler_output is None:
            # If no speculative tokens, the sampler output will be None.
            # In this case we return empty proposals.
            proposal_tokens = torch.tensor(-1,
                                           dtype=torch.long,
                                           device=self._device).expand(
                                               batch_size, proposal_len)
            proposal_probs = torch.tensor(0,
                                          dtype=torch.float32,
                                          device=self._device).expand(
                                              batch_size, proposal_len,
                                              self._vocab_size)
            proposal_lens_tensor = torch.tensor(0,
                                                dtype=torch.long,
                                                device=self._device).expand(
                                                    len(proposal_lens))
            return proposal_tokens, proposal_probs, proposal_lens_tensor, None

        sampler_output = maybe_sampler_output
        proposal_tokens, proposal_probs, proposal_logprobs, *_ = sampler_output_to_torch(
            sampler_output, sampler_transposed)

        # Now, reformat the output GPU tensors such that each sequence has
        # a proposal. the proposal can be empty, e.g. [-1, -1, -1]

        entire_proposal_tokens = proposal_tokens.new_full(
            size=(batch_size, *proposal_tokens.shape[1:]),
            fill_value=-1,
        )
        entire_proposal_tokens[nonzero_proposal_len_indices] = proposal_tokens
        entire_proposal_probs = proposal_probs.new_zeros(
            batch_size,
            *proposal_probs.shape[1:],
        )
        entire_proposal_probs[nonzero_proposal_len_indices] = proposal_probs
        entire_proposal_logprobs = proposal_logprobs.new_zeros(
            batch_size,
            *proposal_logprobs.shape[1:],
        )
        entire_proposal_logprobs[nonzero_proposal_len_indices] = proposal_logprobs

        proposal_tokens, proposal_probs, proposal_logprobs = (
            entire_proposal_tokens,
            entire_proposal_probs,
            entire_proposal_logprobs,
        )

        proposal_lens_tensor = torch.zeros(batch_size,
                                           dtype=torch.long,
                                           device=self._device)
        proposal_lens_tensor[nonzero_proposal_len_indices] = proposal_len

        return proposal_tokens, proposal_probs, proposal_lens_tensor, proposal_logprobs