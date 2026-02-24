from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposals, SpeculativeScorer
from vllm.worker.worker_base import WorkerBase

from minedraft.patching import MinePatch
from minedraft.plugin.sequence import MineExecuteModelRequest


@dataclass
class SpeculativeProposalsPatch(MinePatch[SpeculativeProposals]):
    """Datastructure used to represent proposal tokens from some proposer. It
    also tracks how many speculative tokens each sequence has.
    """

    # Speculative proposal tokens.
    proposal_token_ids: torch.Tensor

    # Probabilities of the proposal tokens according to the proposer.
    proposal_probs: torch.Tensor

    # The valid length of each proposal; can be zero.
    proposal_lens: torch.Tensor

    # A flag to mark that there's no available proposals
    no_proposals: bool = False

    # Optional logprobs.
    proposal_logprobs: Optional[torch.Tensor] = None

    def __repr__(self):
        return (f"SpeculativeProposals("
                f"proposal_token_ids={self.proposal_token_ids}, "
                f"proposal_probs={self.proposal_probs.shape}, "
                f"proposal_logprobs={self.proposal_logprobs.shape}, "
                f"proposal_lens={self.proposal_lens})")


class ParallelSpeculativeScorer(SpeculativeScorer):

    def __init__(self, scorer_worker: WorkerBase, device: str,
                 vocab_size: int, probs_dtype: torch.dtype,
                 token_id_dtype: torch.dtype):
        super().__init__(scorer_worker, device, vocab_size)
        self._probs_dtype = probs_dtype
        self._token_id_dtype = token_id_dtype
        scorer_runner = getattr(scorer_worker, "model_runner", None)
        self._return_hidden_states = scorer_runner.return_hidden_states if \
            scorer_runner else False
        self._hidden_size = scorer_worker.model_config.get_hidden_size()
        self._hidden_states_dtype = scorer_worker.model_config.dtype

    @abstractmethod
    def start_score_proposals(
        self,
        execute_model_req: MineExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> None:
        raise NotImplementedError

    def _recv_sampler_output_from_repr_scorer(
        self
    ) -> SamplerOutput:
        """Receive the sampler output from the speculative scorer."""
        size_tensor = \
            get_tp_group().recv(torch.Size((2,)), dtype=torch.int64, src=1)
        size = size_tensor[0].item()
        hidden_size = size_tensor[1].item()
        size_tensor = None
        if size == 0:
            return SamplerOutput(
                outputs=[],
                sampled_token_ids=None,
                sampled_token_probs=None,
                logprobs=None,
                hidden_states=None,
            )

        # Receive the sampler output from representation scorer.
        sampled_token_ids = get_tp_group().recv(
            torch.Size((size, 1)),
            dtype=self._token_id_dtype,
            src=1,
        )
        sampled_token_probs = get_tp_group().recv(
            torch.Size((size, self._vocab_size)),
            dtype=self._probs_dtype,
            src=1,
        )
        logprobs = get_tp_group().recv(
            torch.Size((size, self._vocab_size)),
            dtype=self._probs_dtype,
            src=1,
        )
        if hidden_size > 0:
            assert (
                self._return_hidden_states and hidden_size == self._hidden_size
            )
            hidden_states = get_tp_group().recv(
                torch.Size((size, self._hidden_size)),
                dtype=self._hidden_states_dtype,
                src=1,
            )
        prefill_hidden_size = get_tp_group().recv(
            torch.Size((1,)), dtype=torch.int64, src=1
        ).item()
        if prefill_hidden_size > 0:
            prefill_hidden_states = get_tp_group().recv(
                torch.Size((prefill_hidden_size, self._hidden_size)),
                dtype=self._hidden_states_dtype,
                src=1,
            )

        return SamplerOutput(
            outputs=[],
            sampled_token_ids=sampled_token_ids,
            sampled_token_probs=sampled_token_probs,
            logprobs=logprobs,
            hidden_states=hidden_states if hidden_size > 0 else None,
            prefill_hidden_states=prefill_hidden_states if prefill_hidden_size > 0 else None,
        )