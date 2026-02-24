from typing import Any, Optional, Union

import torch
from vllm.distributed import communication_op
from vllm.distributed.parallel_state import get_tp_group

from minedraft.patching import MinePatch
from .parallel_state import get_non_driver_tp_group


class CommunicationOpModulePatch(MinePatch[communication_op]):
    @staticmethod
    def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
        """All-reduce the input tensor across model parallel group."""
        return get_non_driver_tp_group().all_reduce(input_)

    @staticmethod
    def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                         dim: int = -1) -> torch.Tensor:
        """All-gather the input tensor across model parallel group."""
        return get_non_driver_tp_group().all_gather(input_, dim)

    @staticmethod
    def tensor_model_parallel_reduce_scatter(input_: torch.Tensor,
                                             dim: int = -1) -> torch.Tensor:
        """Reduce-Scatter the input tensor across model parallel group."""
        return get_non_driver_tp_group().reduce_scatter(input_, dim)

    @staticmethod
    def tensor_model_parallel_gather(input_: torch.Tensor,
                                     dst: int = 0,
                                     dim: int = -1) -> Optional[torch.Tensor]:
        """Gather the input tensor across model parallel group."""
        return get_non_driver_tp_group().gather(input_, dst, dim)

    @staticmethod
    def broadcast_tensor_dict(tensor_dict: Optional[dict[Any, Union[torch.Tensor,
                                                                    Any]]] = None,
                              src: int = 0, async_op: bool = False):
        if not torch.distributed.is_initialized():
            return tensor_dict
        return get_tp_group().broadcast_tensor_dict(tensor_dict, src,
                                                    async_op=async_op)