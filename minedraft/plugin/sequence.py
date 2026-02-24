from typing import Any, Dict, List, Optional, Union

import msgspec
import torch
from vllm.sequence import (
    ExecuteModelRequest,
    HiddenStates,
    SequenceGroup,
    SequenceGroupMetadata,
    SequenceGroupMetadataDelta,
    get_all_seq_ids,
)

from minedraft.patching import MinePatch
from minedraft.plugin.distributed.parallel_state import Works


class SequenceGroupPatch(MinePatch[SequenceGroup]):
    _orig_init = SequenceGroup.__init__

    def __init__(self, *args, **kwargs):
        self.batch_flag = kwargs.pop('batch_flag', None)
        self._orig_init(*args, **kwargs)


class MineSequenceGroupMetadata(SequenceGroupMetadata):
    # The sub-batch this SequenceGroup belongs to.
    batch_flag: Optional[bool] = None


class HiddenStatesPatch(MinePatch[HiddenStates]):
    def clone(
        self: HiddenStates,
        seq_group_metadata_list: Optional[List[MineSequenceGroupMetadata]] = None,
    ) -> HiddenStates:
        new_seq_ids = get_all_seq_ids(seq_group_metadata_list)
        # Find sequence IDs that exist in self._seq_ids
        seq_ids = [seq_id for seq_id in new_seq_ids if seq_id in self._seq_ids]

        if seq_ids != self._seq_ids:
            index = [self._seq_ids.index(seq_id) for seq_id in seq_ids]
        else:
            index = slice(None)

        if len(seq_ids) == len(new_seq_ids):
            new = HiddenStates(
                hidden_states=self.hidden_states[index].clone(),
                _seq_ids=seq_ids,
                seq_group_metadata_list=seq_group_metadata_list
                if self.seq_group_metadata_list is not None else None,
                second_last_token_hidden_states=self.second_last_token_hidden_states[index].clone()
                if self.second_last_token_hidden_states is not None else None,
            )

        else:
            # There are some new sequences just transitioned to running.
            # Initialize their hidden states to zeros.
            mapping_index = [new_seq_ids.index(seq_id) for seq_id in seq_ids]

            hidden_states = self.hidden_states.new_zeros(
                len(seq_group_metadata_list), self.hidden_states.size(1))
            hidden_states[mapping_index] = self.hidden_states[index]

            second_last_token_hidden_states = None
            if self.second_last_token_hidden_states is not None:
                second_last_token_hidden_states = self.second_last_token_hidden_states.new_zeros(
                    len(seq_group_metadata_list), self.second_last_token_hidden_states.size(1))
                second_last_token_hidden_states[mapping_index] = \
                    self.second_last_token_hidden_states[index]

            new = HiddenStates(
                hidden_states=hidden_states,
                _seq_ids=new_seq_ids,
                seq_group_metadata_list=seq_group_metadata_list,
                second_last_token_hidden_states=second_last_token_hidden_states,
            )

        return new


class MineExecuteModelRequest(ExecuteModelRequest):
    # Preempted request ids since last step.
    preempted_requests_ids: List[str] = msgspec.field(default_factory=list)
    # Whether this request is for proposing.
    is_proposing: bool = False
    # The async handle for driver rank to wait on before getting
    # speculative scores in parallel speculative decoding 
    scoring_async_handle: Optional[Works[Dict[str, Union[torch.Tensor, Any]]]] = None

    def clone(
        self, seq_group_metadata_list: list[Union[MineSequenceGroupMetadata,
                                                  SequenceGroupMetadataDelta]]
    ) -> "MineExecuteModelRequest":

        return MineExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=self.blocks_to_swap_in.copy(),
            blocks_to_swap_out=self.blocks_to_swap_out.copy(),
            blocks_to_copy=self.blocks_to_copy.copy(),
            virtual_engine=self.virtual_engine,
            num_lookahead_slots=self.num_lookahead_slots,
            running_queue_size=self.running_queue_size,
            previous_hidden_states=self.previous_hidden_states,
            num_steps=self.num_steps,
            finished_requests_ids=self.finished_requests_ids,
            preempted_requests_ids=self.preempted_requests_ids,
            last_sampled_token_ids=self.last_sampled_token_ids.clone()
            if self.last_sampled_token_ids is not None else None,
            async_callback=self.async_callback,
            is_proposing=self.is_proposing,
            scoring_async_handle=self.scoring_async_handle
        )
