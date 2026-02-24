from importlib.util import find_spec
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from vllm.model_executor.layers.rejection_sampler import RejectionSampler

from minedraft.patching import MinePatch

if find_spec("flashinfer"):
    """
    Consider utilizing the FlashInfer rejection sampling kernel initially,
    as it employs a dedicated kernel rather than relying on 
    Torch tensor operations. This design choice helps to fuse operations, 
    reduce memory I/O, and consequently enhances performance.
    """
    from flashinfer.sampling import chain_speculative_sampling
else:
    chain_speculative_sampling = None


class RejectionSamplerPatch(MinePatch[RejectionSampler], nn.Module):
    def forward(
        self: RejectionSampler,
        target_with_bonus_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
        proposal_lens: Optional[torch.Tensor] = None,
        total_num_seqs: Optional[int] = None,
    ) -> torch.Tensor:

        # Only perform shape/dtype/device checking in strict mode, as it adds
        # overhead.
        if self._strict_mode:
            self._raise_if_incorrect_input(target_with_bonus_probs,
                                           draft_token_ids, bonus_token_ids,
                                           draft_probs)

        batch_size, k, _ = draft_probs.shape

        # batch_size = 0 when all requests in the batch are
        # non_spec requests. In this case, output_token_ids is
        # just an empty tensor.
        if batch_size == 0:
            return torch.empty(0, k + 1, device=draft_probs.device, dtype=int)

        # If use Flashinfer chain_speculative_sampling kernel
        # for rejection sampling
        if self.use_flashinfer and chain_speculative_sampling is not None:
            batch_size, k, _ = draft_probs.shape

            (output_token_ids, accepted_token_num,
             emitted_token_num) = chain_speculative_sampling(
                 draft_probs,
                 draft_token_ids,
                 target_with_bonus_probs,
             )

            # num_emitted_tokens returned by flashinfer
            # does not include the bonus token
            # Flashinfer stops at the first token that violates
            # the condition p >= q and does not include recovery/bonus token.
            # Therefore, we need to add batch_size here.
            self.num_accepted_tokens += accepted_token_num.sum()
            self.num_emitted_tokens += emitted_token_num.sum() + batch_size
            self.num_draft_tokens += batch_size * k
        else:
            # [Parallel SD] with varlen, the bonus token is no longer always the last
            if proposal_lens is not None:
                # TODO: possible to avoid this overhead when it is not varlen
                # create a mask of zeros with the same shape as target_with_bonus_probs
                bonus_mask = torch.zeros_like(target_with_bonus_probs)

                # Use advanced indexing to set the specified indices to zero
                bonus_mask[torch.arange(batch_size), proposal_lens, :] = 1

                # Apply the mask to target_with_bonus_probs
                target_with_bonus_probs = target_with_bonus_probs * (1 - bonus_mask)

            accepted, recovered_token_ids = (
                self._batch_modified_rejection_sampling(
                    target_with_bonus_probs[:, :-1],
                    draft_probs,
                    draft_token_ids,
                    seeded_seqs,
                ))

            # [Parallel SD] Pass additional arguments to _create_output
            extra_kwargs: Dict[str, Any] = {}
            if proposal_lens is not None:
                extra_kwargs["proposal_lens_list"] = proposal_lens.tolist()
                extra_kwargs["total_num_seqs"] = total_num_seqs

            output_token_ids = self._create_output(
                accepted,
                recovered_token_ids,
                draft_token_ids,
                bonus_token_ids,
                **extra_kwargs,
            )

        return output_token_ids