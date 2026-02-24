from typing import List, Optional, Union

import torch
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import IntermediateTensors
from vllm.worker.model_runner import (
    GPUModelRunnerBase,
    ModelInputForGPUWithSamplingMetadata,
    ModelRunner,
)

from minedraft.patching import MinePatch
from minedraft.plugin.distributed.parallel_state import get_non_driver_tp_group


class GPUModelRunnerBasePatch(MinePatch[GPUModelRunnerBase]):
    @property
    def is_driver(self: GPUModelRunnerBase) -> bool:
        if self.speculative_config and self.speculative_config.is_parallel:
            return get_non_driver_tp_group().is_first_rank
        return self.is_driver_worker


class ModelRunnerPatch(MinePatch[ModelRunner]):
    @torch.inference_mode()
    def execute_model(
        self: ModelRunner,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        self.attn_state.begin_forward(model_input)

        # Currently cuda graph is only supported by the decode phase.
        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        previous_hidden_states = kwargs.get("previous_hidden_states")
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            use_inputs_embeds = model_input.inputs_embeds is not None
            model_executable = self.graph_runners[virtual_engine][(
                graph_batch_size, use_inputs_embeds)]
            if previous_hidden_states is not None:
                previous_hidden_states = torch.cat([
                    previous_hidden_states,
                    torch.empty([
                        graph_batch_size - previous_hidden_states.shape[0],
                        *previous_hidden_states.shape[1:]
                    ],
                                dtype=previous_hidden_states.dtype,
                                device=previous_hidden_states.device)
                ])
        else:
            model_executable = self.model

        # Receive KV cache in distributed KV cache transfer setting
        # In disagg prefill setting, it will also recv hidden states and bypass
        # model forwarding
        # In KV cache database setting, it will change the model input so that
        # we can skip prefilling on tokens that successfully received KV caches
        # NOTE: The receive operation is blocking
        bypass_model_exec = False
        if self.need_recv_kv(model_input, kv_caches):
            hidden_or_intermediate_states, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    model_executable,
                    model_input,
                    kv_caches=kv_caches
                )

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        model_kwargs = {}
        if previous_hidden_states is not None:
            model_kwargs["previous_hidden_states"] = previous_hidden_states
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()

        if not bypass_model_exec:
            with set_forward_context(model_input.attn_metadata,
                                     self.vllm_config, virtual_engine):
                hidden_or_intermediate_states = model_executable(
                    input_ids=model_input.input_tokens,
                    inputs_embeds=model_input.inputs_embeds,
                    positions=model_input.input_positions,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(
                        multi_modal_kwargs,
                        device=self.device,
                    ),
                    **seqlen_agnostic_kwargs,
                    **model_kwargs,
                )

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        # Sending KV cache in distributed KV cache transfer setting
        # NOTE: the send operation is non-blocking
        if self.need_send_kv(model_input, kv_caches):
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            if (self.is_driver
                    and hidden_or_intermediate_states is not None
                    and isinstance(hidden_or_intermediate_states,
                                   IntermediateTensors)
                    and self.observability_config is not None
                    and self.observability_config.collect_model_forward_time):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = (
                    torch.tensor(model_forward_time + orig_model_forward_time))
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if self.is_driver:
            if model_input.async_callback is not None:
                model_input.async_callback()

            # Sample the next token.
            assert isinstance(self.sampler, Sampler)
            orig_include_gpu_probs = self.sampler.include_gpu_probs_tensor
            if model_input.inputs_embeds is not None:
                self.sampler.include_gpu_probs_tensor = True

            output: SamplerOutput = self.sampler(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )
            if (self.observability_config is not None
                    and self.observability_config.collect_model_forward_time
                    and output is not None):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                # If there are multiple workers, we are still tracking the
                # latency from the start time of the driver worker to the end
                # time of the driver worker. The model forward time will then
                # end up covering the communication time as well.
                output.model_forward_time = (orig_model_forward_time +
                                             model_forward_time)

        if model_input.inputs_embeds is not None:
            if self.is_driver:
                sampled = broadcast_tensor_dict(
                    {"token_ids": output.sampled_token_ids})
            else:
                sampled = broadcast_tensor_dict()
            if sampled["token_ids"] is not None:
                sampled_token_embeds = self.model.get_input_embeddings(
                    sampled["token_ids"].squeeze(1))
                if self.is_driver:
                    self.sampler.include_gpu_probs_tensor = \
                        orig_include_gpu_probs

                    output.sampled_token_embeds = sampled_token_embeds

                    for token_embed, sequence_group_output in zip(
                            output.sampled_token_embeds, output.outputs):
                        assert len(sequence_group_output.samples) == 1
                        sequence_group_output.samples[
                            0].output_embed = token_embed

        if not self.is_driver:
            return []

        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
                output.prefill_hidden_states = hidden_or_intermediate_states
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[:len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]