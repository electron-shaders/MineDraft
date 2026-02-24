from typing import Optional, Union

import torch
from torch import nn
from vllm.model_executor.layers.spec_decode_base_sampler import SpecDecodeBaseSampler
from vllm.platforms import current_platform

from minedraft.patching import MinePatch


class SpecDecodeBaseSamplerPatch(MinePatch[SpecDecodeBaseSampler], nn.Module):
    _orig_init = SpecDecodeBaseSampler.__init__

    def __init__(self, *args, **kwargs):
        self._orig_init(*args, **kwargs)
        self.num_good_draft_tokens: Optional[torch.Tensor] = None
        self.num_verification_tokens: Optional[torch.Tensor] = None
        self.num_req: int = 0

    def init_gpu_tensors(self, device: Union[int, str]) -> None:
        assert self.num_accepted_tokens is None
        if isinstance(device, int):
            device = f"{current_platform.device_type}:{device}"
        elif not isinstance(device, str):
            raise ValueError(f"Device must be int or str, get {type(device)}")
        self.num_accepted_tokens = torch.tensor(0,
                                                dtype=torch.long,
                                                device=device)
        self.num_good_draft_tokens = torch.tensor(0,
                                                  dtype=torch.long,
                                                  device=device)
        self.num_verification_tokens = torch.tensor(0,
                                                    dtype=torch.long,
                                                    device=device)
        self.num_emitted_tokens = torch.tensor(0,
                                               dtype=torch.long,
                                               device=device)

    def init_tensors(self,
                     device: Union[int, str],
                     device_type: Union[torch.device, str] = 'cuda') -> None:
        assert self.num_accepted_tokens is None
        if isinstance(device_type, torch.device):
            device_type = device_type.type
        if isinstance(device, int):
            device = f"{device_type}:{device}"
        self.num_accepted_tokens = torch.tensor(0,
                                                dtype=torch.long,
                                                device=device)
        self.num_good_draft_tokens = torch.tensor(0,
                                                  dtype=torch.long,
                                                  device=device)
        self.num_verification_tokens = torch.tensor(0,
                                                    dtype=torch.long,
                                                    device=device)
        self.num_emitted_tokens = torch.tensor(0,
                                               dtype=torch.long,
                                               device=device)

    def _create_output(
            self,
            accepted: torch.Tensor,  # [batch_size, k]
            substitute_token_ids: torch.Tensor,  # [batch_size, k]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
            bonus_token_ids: torch.Tensor,  # [batch_size]
            proposal_lens_list: Optional[torch.Tensor] = None,  # [batch_size]
            total_num_seqs: Optional[int] = None, # total number of sequences including the current batch and the non-speculated sequences, necessary to calculate emitted tokens correctly
    ) -> torch.Tensor:
        batch_size, k = substitute_token_ids.shape
        bonus_token_ids = bonus_token_ids.squeeze(-1)
        # Determine the index of the first False value for each row.
        limits = (accepted == 0).max(1).indices
        limits[~(accepted == 0).any(1)] = k

        # Create masks using the indices.
        indices = torch.arange(k, device=accepted.device).unsqueeze(0)
        accepted_mask = indices < limits.unsqueeze(1)
        after_false_mask = indices == limits.unsqueeze(1)

        # Create an extended output tensor
        output_with_bonus_tokens = -torch.ones(
            (batch_size, k + self._num_bonus_tokens),
            dtype=self.token_id_dtype,
            device=accepted.device)
        output = output_with_bonus_tokens[:, :k]

        # Fill in the first k columns of the output tensor using masks and data
        # tensors.
        output[:, :k] = torch.where(accepted_mask, draft_token_ids,
                                    -torch.ones_like(draft_token_ids))

        if proposal_lens_list is not None:
            # [Parallel SD] NOTE: possible to skip these overhead when there is no verlen
            # varlen implementation
            output_with_bonus_tokens[torch.arange(batch_size), proposal_lens_list] = torch.where(
                output[torch.arange(batch_size), torch.tensor(proposal_lens_list) - 1] != -1,
                bonus_token_ids, -1)
            bonus_idx_mask = indices == torch.tensor(proposal_lens_list, device=accepted.device).unsqueeze(1)
            # do not substitute bonus tokens
            substitute_mask = after_false_mask * ~bonus_idx_mask
            # import debugpy; debugpy.breakpoint()
            # Fill the recovered token ids.
            output.mul_(~substitute_mask).add_(
                substitute_token_ids.mul(substitute_mask))

            # Update summary statistics
            # Actually this should be the number of verification tokens instead
            self.num_verification_tokens += sum(proposal_lens_list)
            self.num_emitted_tokens += (output_with_bonus_tokens != -1).sum() + (total_num_seqs - batch_size)

        else:
            # Orginal fixed k implementation

            # Fill the last column.
            # We check output directly as accepted may have True values inconsistent
            # with causal acceptance.
            output_with_bonus_tokens[:, -1] = torch.where(output[:, -1] != -1,
                                                          bonus_token_ids, -1)

            # Fill the recovered token ids.
            output.mul_(~after_false_mask).add_(
                substitute_token_ids.mul(after_false_mask))

            self.num_emitted_tokens += (output_with_bonus_tokens != -1).sum()
            self.num_verification_tokens += batch_size * k

        self.num_draft_tokens += batch_size * k
        self.num_accepted_tokens += accepted.sum()
        self.num_good_draft_tokens += limits.sum()
        self.num_req += total_num_seqs

        return output_with_bonus_tokens