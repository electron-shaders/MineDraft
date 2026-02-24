from dataclasses import dataclass
from typing import Optional

import torch
from vllm import envs
from vllm.config import CompilationLevel, SpeculativeConfig, VllmConfig, logger
from vllm.utils import DEFAULT_MAX_NUM_BATCHED_TOKENS, random_uuid

from minedraft.patching import MinePatch


@dataclass
class MineSpeculativeConfig(SpeculativeConfig):

    draft_temperature: Optional[float] = None
    tetris: Optional[bool] = None
    tetris_extra_proposals: Optional[int] = None
    tetris_turn_on_batch_size: Optional[int] = None
    tetris_capacity: Optional[int] = None
    is_parallel: Optional[bool] = None
    force_mqa: Optional[bool] = None
    force_pearl: Optional[bool] = None


class SpeculativeConfigPatch(MinePatch[SpeculativeConfig]):

    _orig_from_dict = SpeculativeConfig.__dict__["from_dict"].__wrapped__
    _orig_post_init = SpeculativeConfig.__post_init__

    def __new__(cls, *args, **kwargs):
        # Override __new__ to return an MineSpeculativeConfig instead of a
        # SpeculativeConfig when creating a new instance of the class.
        if cls is SpeculativeConfig:
            return MineSpeculativeConfig.__new__(
                MineSpeculativeConfig, *args, **kwargs)
        return super(SpeculativeConfig, cls).__new__(cls)

    def __post_init__(self):
        if self.tetris:
            self.num_speculative_tokens += self.tetris_extra_proposals
        if self.is_parallel:
            if self.draft_tensor_parallel_size is None:
                self.draft_tensor_parallel_size = 1
            elif self.draft_tensor_parallel_size != 1:
                raise ValueError(
                    f"{self.draft_tensor_parallel_size=} cannot be "
                    f"other value than 1 if using Parallel SD")
            if self.enable_chunked_prefill:
                logger.warning_once(
                    "Chunked prefill is not fully supported in MineDraft so far. "
                    "It is highly likely that the evenness of sub-batch splitting "
                    "will be broken, resulting in degraded performance.")

        self._orig_post_init()

        self.draft_model_config.override_generation_config.update({
            "temperature": self.draft_temperature
        })

    @classmethod
    def from_dict(cls, dict_value: dict) -> SpeculativeConfig:
        """Parse the CLI value for the speculative config."""
        if cls is SpeculativeConfig:
            return SpeculativeConfigPatch._orig_from_dict(
                MineSpeculativeConfig, dict_value)
        return SpeculativeConfigPatch._orig_from_dict(cls, dict_value)


class VllmConfigPatch(MinePatch[VllmConfig]):

    def __post_init__(self: VllmConfig):
        self.try_verify_and_update_config()

        if self.model_config is not None:
            self.model_config.verify_async_output_proc(self.parallel_config,
                                                       self.speculative_config,
                                                       self.device_config)
            if self.speculative_config and self.speculative_config.is_parallel:
                # [Parallel SD] Restore tp size temporarily
                self.parallel_config.tensor_parallel_size -= 1
                self.model_config.verify_with_parallel_config(self.parallel_config)
                self.parallel_config.tensor_parallel_size += 1
            else:
                self.model_config.verify_with_parallel_config(self.parallel_config)
            self.model_config.verify_dual_chunk_attention_config(
                self.load_config)

        self.cache_config.verify_with_parallel_config(self.parallel_config)

        if self.lora_config is not None:
            self.lora_config.verify_with_cache_config(self.cache_config)
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_lora_support()
        if self.prompt_adapter_config is not None:
            self.prompt_adapter_config.verify_with_model_config(
                self.model_config)

        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config)

        from vllm.platforms import current_platform
        if self.model_config is not None and \
            self.scheduler_config.chunked_prefill_enabled and \
            self.model_config.dtype == torch.float32 and \
            current_platform.get_device_capability() == (7, 5):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels.")

        # async tp is built on top of sequence parallelism
        # and requires it to be enabled.
        if self.compilation_config.pass_config.enable_async_tp:
            self.compilation_config.pass_config.enable_sequence_parallelism = \
                True
        if self.compilation_config.pass_config.enable_sequence_parallelism:
            self.compilation_config.custom_ops.append("+rms_norm")
        if envs.VLLM_USE_V1 and self.model_config is not None and \
            not self.model_config.enforce_eager:
            # By default, V1 uses piecewise CUDA graphs. If full_cuda_graph
            # is set to True, full CUDA graphs will be used.
            self.compilation_config.cudagraph_num_of_warmups = 1
            self.compilation_config.level = CompilationLevel.PIECEWISE
            self.compilation_config.set_splitting_ops_for_v1()

        self._set_cudagraph_sizes()

        if self.cache_config.cpu_offload_gb > 0 and \
            self.compilation_config.level != CompilationLevel.NO_COMPILATION \
                and not envs.VLLM_USE_V1:
            logger.warning(
                "CPU offload is not supported with `torch.compile` in v0 yet."
                " Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        if ((not envs.VLLM_USE_V1) and self.lora_config is not None
                and self.compilation_config.level
                != CompilationLevel.NO_COMPILATION):
            logger.warning(
                "LoRA for V0 is not supported with `torch.compile` yet. "
                "Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        if self.compilation_config.full_cuda_graph and \
            not self.model_config.disable_cascade_attn:
            logger.info("full_cuda_graph is not supported with "
                        "cascade attention. Disabling cascade attention.")
            self.model_config.disable_cascade_attn = True

        disable_chunked_prefill_reasons: list[str] = []

        if self.model_config and self.model_config.pooler_config:
            pooling_type = self.model_config.pooler_config.pooling_type
            if pooling_type is None or pooling_type.lower() != "last":
                disable_chunked_prefill_reasons.append(
                    "Only \"last\" pooling supports chunked "
                    "prefill and prefix caching; disabling both.")

        if disable_chunked_prefill_reasons:
            for reason in disable_chunked_prefill_reasons:
                logger.info(reason)
            self.scheduler_config.chunked_prefill_enabled = False
            self.scheduler_config.long_prefill_token_threshold = 0
            self.scheduler_config.max_num_batched_tokens = max(
                self.scheduler_config.max_model_len,
                DEFAULT_MAX_NUM_BATCHED_TOKENS)

            if self.cache_config is not None:
                self.cache_config.enable_prefix_caching = False

        if (self.kv_events_config is not None
                and self.kv_events_config.enable_kv_cache_events
                and not self.cache_config.enable_prefix_caching):
            logger.warning(
                "KV cache events are on, but prefix caching is not enabled."
                "Use --enable-prefix-caching to enable.")
        if (self.kv_events_config is not None
                and self.kv_events_config.publisher != "null"
                and not self.kv_events_config.enable_kv_cache_events):
            logger.warning("KV cache events are disabled,"
                           "but the scheduler is configured to publish them."
                           "Modify KVEventsConfig.enable_kv_cache_events"
                           "to True to enable.")
        current_platform.check_and_update_config(self)

        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

        if (envs.VLLM_USE_V1
                and not self.scheduler_config.disable_hybrid_kv_cache_manager):
            # logger should only print warning message for hybrid models. As we
            # can't know whether the model is hybrid or not now, so we don't log
            # warning message here and will log it later.
            if not (current_platform.is_cuda() or current_platform.is_rocm()):
                # Hybrid KV cache manager is not supported on non-GPU platforms.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_transfer_config is not None:
                # Hybrid KV cache manager is not compatible with KV transfer.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_events_config is not None:
                # Hybrid KV cache manager is not compatible with KV events.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True