from typing import Optional

import torch
import torch.nn as nn
from vllm.model_executor.models.eagle import EAGLE
from vllm.sequence import IntermediateTensors

from minedraft.patching import MinePatch


class EAGLEPatch(MinePatch[EAGLE], nn.Module):
    def forward(
        self: EAGLE,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        # Handle both empty previous_hidden_states
        # and mismatched batch size
        batch_size = inputs_embeds.size(0)
        if previous_hidden_states.size(0) == 0 or \
           previous_hidden_states.size(0) != batch_size:
            hidden_dim = self.config.model.hidden_size
            device = inputs_embeds.device
            # Create zero tensor with matching batch size
            # [BUGFIX] Use new_zeros to match dtype
            previous_hidden_states = \
                previous_hidden_states.new_zeros(batch_size, hidden_dim, device=device)

        if self.add_para_norm:
            inputs_embeds = torch.cat([
                self.enorm(inputs_embeds),
                self.hnorm(previous_hidden_states)
            ],
                                      dim=-1)
        else:
            inputs_embeds = torch.cat([inputs_embeds, previous_hidden_states],
                                      dim=-1)

        inputs_embeds = self.fc(inputs_embeds)

        inputs_embeds[positions == 0] = 0  # masking inputs at position=0

        hidden_states = self.model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
        )
        return hidden_states