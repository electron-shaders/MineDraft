import gc
import os
from typing import Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.model_executor import set_random_seed
from vllm.utils import MemorySnapshot
from vllm.worker import worker
from vllm.worker.worker import (
    Worker,
    _check_if_gpu_supports_dtype,
    init_worker_distributed_environment,
)

from minedraft.patching import MinePatch


class WorkerModulePatch(MinePatch[worker]):
    @staticmethod
    def init_worker_distributed_environment(
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: Optional[str] = None,
        local_rank: int = -1,
        initialize_non_driver_tp_group: bool = False,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

        init_distributed_environment(parallel_config.world_size, rank,
                                    distributed_init_method, local_rank)
        ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                          parallel_config.pipeline_parallel_size,
                                          initialize_non_driver_tp_group=initialize_non_driver_tp_group)

        ensure_kv_transfer_initialized(vllm_config)


class WorkerPatch(MinePatch[Worker]):
    def init_device(self: Worker) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.baseline_snapshot = MemorySnapshot()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config, self.rank,
            self.distributed_init_method,
            self.local_rank,
            self.speculative_config and self.speculative_config.is_parallel)
        # Set random seed.
        set_random_seed(self.model_config.seed)