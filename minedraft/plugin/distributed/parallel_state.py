from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch.distributed import Backend, ProcessGroup
from vllm.distributed import parallel_state
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    TensorMetadata,
    _get_unique_name,
    _register_group,
    _split_tensor_dict,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    get_world_group,
    init_model_parallel_group,
    initialize_model_parallel,
    logger,
    model_parallel_is_initialized,
)
from vllm.utils import resolve_obj_by_qualname

from minedraft.patching import MinePatch

T = TypeVar("T")

@dataclass
class Works(Generic[T]):
    result: T
    async_handles: List[torch.distributed.Work]

    def is_completed(self) -> bool:
        return all(
            async_handle.is_completed() for async_handle in self.async_handles
        )

    def wait(self) -> None:
        for async_handle in self.async_handles:
            async_handle.wait()

    def get(self) -> T:
        return self.result


class GroupCoordinatorPatch(MinePatch[GroupCoordinator]):
    is_non_driver_group_for_driver: bool  # whether this is a non-driver group
                                          # for a driver rank

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        is_non_driver_group: bool = False,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend)
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.is_non_driver_group_for_driver = False
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group
            elif is_non_driver_group and self.rank == ranks[0] - 1:
                # [Parallel SD] This is the driver rank but we are currently 
                # creating a non-driver group. We need to provide dummy field 
                # values for the driver rank.
                self.is_non_driver_group_for_driver = True
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = -1
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None
        assert self.device_group is not None

        from vllm.platforms import current_platform

        if current_platform.is_cuda_alike():
            self.device = torch.device(f"cuda:{local_rank}")
        elif current_platform.is_out_of_tree():
            self.device = torch.device(
                f"{current_platform.device_name}:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_device_communicator = use_device_communicator

        self.device_communicator: DeviceCommunicatorBase = None  # type: ignore
        if (not self.is_non_driver_group_for_driver and
            use_device_communicator and self.world_size > 1):
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls())
            self.device_communicator = device_comm_cls(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
        self.mq_broadcaster: Optional[MessageQueue] = None
        if (not self.is_non_driver_group_for_driver and
            use_message_queue_broadcaster and self.world_size > 1):
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6)

        from vllm.platforms import current_platform
        self.use_custom_op_call = (current_platform.is_cuda_alike()
                                   or current_platform.is_tpu())

    @overload
    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None,
        async_op: bool = False,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        ...

    @overload
    def broadcast_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None,
        async_op: bool = True,
    ) -> Works[Dict[str, Union[torch.Tensor, Any]]]:
        ...

    def broadcast_tensor_dict(
        self: GroupCoordinator,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None,
        async_op: bool = False,
    ) -> Union[Optional[Dict[str, Union[torch.Tensor, Any]]],
               Works[Dict[str, Union[torch.Tensor, Any]]]]:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if (not torch.distributed.is_initialized() or self.world_size == 1):
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"

        rank_in_group = self.rank_in_group
        if rank_in_group == src:
            metadata_list: List[Tuple[Any, Any]] = []
            assert isinstance(
                tensor_dict,
                dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=self.ranks[src],
                                                         group=metadata_group,
                                                         async_op=True)
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=self.ranks[src],
                                                         group=group,
                                                         async_op=True)
                async_handles.append(handle)

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(value.size,
                                         dtype=value.dtype,
                                         device=value.device)
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        tensor_dict[key] = tensor
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=metadata_group,
                            async_op=True)
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=group,
                            async_op=True)
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
        if not async_op:
            for async_handle in async_handles:
                async_handle.wait()
            return tensor_dict
        return Works(tensor_dict, async_handles)


def get_non_driver_tp_group() -> GroupCoordinator:
    if parallel_state._NON_DRIVER_TP is None:
        return get_tp_group()
    return parallel_state._NON_DRIVER_TP


class ParallelStateModulePatch(MinePatch[parallel_state]):
    _NON_DRIVER_TP: Optional[GroupCoordinator] = None

    @staticmethod
    def init_model_parallel_group(
        group_ranks: List[List[int]],
        local_rank: int,
        backend: str,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        is_non_driver_group: bool = False,
    ) -> GroupCoordinator:

        return GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            use_device_communicator=True,
            use_message_queue_broadcaster=use_message_queue_broadcaster,
            group_name=group_name,
            is_non_driver_group=is_non_driver_group,
        )

    @staticmethod
    def get_non_driver_tp_group() -> GroupCoordinator:
        return get_non_driver_tp_group()

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        backend: Optional[str] = None,
        initialize_non_driver_tp_group: bool = False,
    ) -> None:
        # Get world size and rank. Ensure some consistencies.
        assert torch.distributed.is_initialized()
        world_size: int = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        backend = backend or torch.distributed.get_backend(
            get_world_group().device_group)

        data_parallel_size = 1
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config()
        if config is not None:
            data_parallel_size = config.parallel_config.data_parallel_size

        # the layout order is: ExternalDP x DP x PP x TP
        # ExternalDP is the data parallel group that is not part of the model,
        # every dp rank can generate independently (in verl integration).
        # DP is the data parallel group that is part of the model,
        # all the ranks in the same DP group should generate simultaneously,
        # i.e. the `generate` call in the same DP group should be called together,
        # otherwise it will cause deadlock.
        # to get group_ranks for each dimension, transpose that dimension to the
        # last dimension, then reshape to 2D, then unbind the last dimension
        all_ranks = torch.arange(world_size).reshape(
            -1, data_parallel_size, pipeline_model_parallel_size,
            tensor_model_parallel_size)  # noqa

        # Build the tensor model-parallel groups.
        assert parallel_state._TP is None, ("tensor model parallel group is already initialized")
        group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]

        # message queue broadcaster is only used in tensor model parallel group
        parallel_state._TP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=True,
            group_name="tp")

        # [Parallel SD] Build the non-driver tensor model-parallel groups.
        if initialize_non_driver_tp_group:
            assert parallel_state._NON_DRIVER_TP is None, (
                "non-driver tensor model parallel group is already initialized")

            all_ranks = torch.arange(world_size).reshape(
                -1, data_parallel_size, pipeline_model_parallel_size,
                tensor_model_parallel_size)  # noqa
            group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
            group_ranks = [x.tolist()[1:] for x in group_ranks]
            parallel_state._NON_DRIVER_TP = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="non_driver_tp",
                is_non_driver_group=True
            )

        # Build the pipeline model-parallel groups.
        assert parallel_state._PP is None, (
            "pipeline model parallel group is already initialized")
        group_ranks = all_ranks.transpose(2, 3).reshape(
            -1, pipeline_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        parallel_state._PP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="pp")

        assert parallel_state._DP is None, ("data parallel group is already initialized")
        group_ranks = all_ranks.transpose(1,
                                          3).reshape(-1,
                                                     data_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        parallel_state._DP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="dp")

        assert parallel_state._EP is None, ("expert parallel group is already initialized")
        group_ranks = all_ranks.transpose(1, 2).reshape(
            -1, data_parallel_size * tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        parallel_state._EP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="ep")

        logger.info(
            "rank %s in world size %s is assigned as "
            "DP rank %s, PP rank %s, TP rank %s, EP rank %s", rank, world_size,
            parallel_state._DP.rank_in_group,
            parallel_state._PP.rank_in_group,
            parallel_state._TP.rank_in_group,
            parallel_state._EP.rank_in_group)

    @staticmethod
    def ensure_model_parallel_initialized(
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        backend: Optional[str] = None,
        initialize_non_driver_tp_group: bool = False,
    ) -> None:
        """Helper to initialize model parallel groups if they are not initialized,
        or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
        values if the model parallel groups are initialized.
        """
        backend = backend or torch.distributed.get_backend(
            get_world_group().device_group)
        if not model_parallel_is_initialized():
            initialize_model_parallel(
                tensor_model_parallel_size,
                pipeline_model_parallel_size, backend,
                initialize_non_driver_tp_group)
            return

        assert (
            get_tensor_model_parallel_world_size() == tensor_model_parallel_size
        ), ("tensor parallel group already initialized, but of unexpected size: "
            f"{get_tensor_model_parallel_world_size()=} vs. "
            f"{tensor_model_parallel_size=}")
        pp_world_size = get_pp_group().world_size
        assert (pp_world_size == pipeline_model_parallel_size), (
            "pipeline parallel group already initialized, but of unexpected size: "
            f"{pp_world_size=} vs. "
            f"{pipeline_model_parallel_size=}")

    @staticmethod
    def prepare_communication_buffer_for_model(model: torch.nn.Module):
        """Prepare the communication buffer for the model.
        Traditional communication libraries like NCCL are almost
        model agnostic. However, emerging new communication libraries like
        MoE all2all (DeepEP) usually allocate the communication buffer
        based on the model shape for optimal performance.
        """
        if parallel_state._TP is not None:
            parallel_state._TP.prepare_communication_buffer_for_model(model)
        if parallel_state._NON_DRIVER_TP is not None:
            parallel_state._NON_DRIVER_TP.prepare_communication_buffer_for_model(model)
        if parallel_state._PP is not None:
            parallel_state._PP.prepare_communication_buffer_for_model(model)
        if parallel_state._DP is not None:
            parallel_state._DP.prepare_communication_buffer_for_model(model)
        if parallel_state._EP is not None:
            parallel_state._EP.prepare_communication_buffer_for_model(model)

    @staticmethod
    @contextmanager
    def patch_tensor_parallel_group(tp_group: GroupCoordinator):
        """Patch the tp group temporarily until this function ends.

        This method is for draft workers of speculative decoding to run draft model
        with different tp degree from that of target model workers.

        Args:
            tp_group (GroupCoordinator): the tp group coordinator
        """
        assert not parallel_state._TP_STATE_PATCHED, "Should not call when it's already patched"

        parallel_state._TP_STATE_PATCHED = True
        old_tp_group = get_tp_group()
        parallel_state._TP = tp_group
        old_non_driver_tp_group = None
        if parallel_state._NON_DRIVER_TP is not None:
            old_non_driver_tp_group = get_non_driver_tp_group()
            parallel_state._NON_DRIVER_TP = tp_group
        try:
            yield
        finally:
            # restore the original state
            parallel_state._TP_STATE_PATCHED = False
            parallel_state._TP = old_tp_group
            if old_non_driver_tp_group is not None:
                parallel_state._NON_DRIVER_TP = old_non_driver_tp_group

    @staticmethod
    def get_tensor_model_parallel_world_size():
        """Return world size for the tensor model parallel group."""
        return get_non_driver_tp_group().world_size

    @staticmethod
    def get_tensor_model_parallel_rank():
        """Return my rank for the tensor model parallel group."""
        return get_non_driver_tp_group().rank_in_group

    @staticmethod
    def destroy_model_parallel():
        if parallel_state._TP:
            parallel_state._TP.destroy()
        parallel_state._TP = None

        if parallel_state._NON_DRIVER_TP:
            parallel_state._NON_DRIVER_TP.destroy()
        parallel_state._NON_DRIVER_TP = None

        if parallel_state._PP:
            parallel_state._PP.destroy()
        parallel_state._PP = None

        if parallel_state._DP:
            parallel_state._DP.destroy()
        parallel_state._DP = None

        if parallel_state._EP:
            parallel_state._EP.destroy()
        parallel_state._EP = None
