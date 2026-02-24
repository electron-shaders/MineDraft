import dataclasses
import time
from typing import Dict, List, Optional, Tuple

import torch
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import IntermediateTensors
from vllm.worker.model_runner_base import BroadcastableModelInput, ModelRunnerInputBase
from vllm.worker.worker_base import (
    LocalOrDistributedWorkerBase,
    WorkerInput,
    extract_previous_hidden_states,
)

from minedraft.patching import MinePatch
from minedraft.plugin.distributed.parallel_state import (
    Works,
    get_non_driver_tp_group,
)
from minedraft.plugin.sequence import MineExecuteModelRequest


class LocalOrDistributedWorkerBasePatch(MinePatch[LocalOrDistributedWorkerBase]):

    def _get_worker_input_from_broadcast(
        self: LocalOrDistributedWorkerBase,
        is_repr_scorer: bool = False
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
            str, torch.Tensor], None]]:

        assert self.do_metadata_broadcast
        assert not self.is_driver_worker
        broadcast_data = broadcast_tensor_dict(src=0)
        if not broadcast_data:
            return None

        worker_input = WorkerInput.from_broadcasted_tensor_dict(broadcast_data)

        # [Parallel SD] If this is a representation scorer, recover the SamplingMetadata
        # by re-constructing the model input.
        if is_repr_scorer:
            model_input = (
                self.model_runner.prepare_model_input(
                    broadcast_data.pop("seq_group_metadata_list"),
                    broadcast_data.pop("virtual_engine"),
                    broadcast_data.pop("finished_requests_ids")))
        else:
            del broadcast_data["seq_group_metadata_list"]
            model_input = (
                self.model_runner.make_model_input_from_broadcasted_tensor_dict(
                    broadcast_data))

        kwargs = extract_previous_hidden_states(broadcast_data)

        return model_input, worker_input, kwargs, None

    def _get_driver_input_and_broadcast(
        self: LocalOrDistributedWorkerBase,
        execute_model_req: MineExecuteModelRequest,
        is_psd: bool = False
    ) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor], Optional[Works]]:

        assert self.is_driver_worker

        worker_input: WorkerInput = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = (
            self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.virtual_engine,
                execute_model_req.finished_requests_ids))

        kwargs = {
            "seq_group_metadata_list": execute_model_req.seq_group_metadata_list,
        }
        if not is_psd:
            kwargs.update(extract_previous_hidden_states(execute_model_req))

        work = None

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_data.update(kwargs)
            work = broadcast_tensor_dict(broadcast_data, src=0, async_op=is_psd)

        if execute_model_req.async_callback:
            model_input = dataclasses.replace(  # type: ignore
                model_input,
                async_callback=execute_model_req.async_callback)

        return model_input, worker_input, kwargs, work

    def prepare_input(
        self,
        execute_model_req: Optional[MineExecuteModelRequest] = None
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
            str, torch.Tensor]]]:
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None
            if (
                self.speculative_config
                and self.speculative_config.is_parallel
                and not execute_model_req.is_proposing
            ):
                # [Parallel SD] Driver broadcasts the input to all scorers,
                # stores the async broadcast handle and returns directly.
                inputs = self._get_driver_input_and_broadcast(execute_model_req,
                                                              is_psd=True)
                if inputs is None:
                    return None
                _, worker_input, _, work = inputs
                worker_input = dataclasses.replace(worker_input,
                                                   num_seq_groups=0)
                return None, worker_input, None, work
            return self._get_driver_input_and_broadcast(execute_model_req)
        else:
            if (
                self.speculative_config
                and self.speculative_config.is_parallel
                and get_non_driver_tp_group().is_first_rank
            ):
                # [Parallel SD] Representative scorer receives the broadcasted input
                # and reconstructs the SamplingMetadata.
                return self._get_worker_input_from_broadcast(
                    is_repr_scorer=True
                )
            return self._get_worker_input_from_broadcast()

    def execute_model(
        self: LocalOrDistributedWorkerBase,
        execute_model_req: Optional[MineExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        start_time = time.perf_counter()

        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs, work = inputs
        num_steps = worker_input.num_steps
        if execute_model_req is not None and execute_model_req.spec_step_idx:
            kwargs["spec_step_idx"] = execute_model_req.spec_step_idx

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            if (self.is_driver_worker
                and self.speculative_config
                and self.speculative_config.is_parallel
                and not execute_model_req.is_proposing
            ):
                # [Parallel SD] Pass the work handle to the driver worker
                execute_model_req.scoring_async_handle = work
            return []

        intermediate_tensors = None
        orig_model_execute_time = 0.0
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                orig_model_execute_time = intermediate_tensors.tensors.get(
                    "model_execute_time", torch.tensor(0)).item()

        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
            **kwargs,
        )

        model_execute_time = time.perf_counter() - start_time
        if not get_pp_group().is_last_rank:
            # output is IntermediateTensors
            assert isinstance(output, IntermediateTensors)
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                output.tensors["model_execute_time"] = torch.tensor(
                    model_execute_time + orig_model_execute_time)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return [None]
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time
                and output is not None):
            for o in output:
                o.model_execute_time = (orig_model_execute_time +
                                        model_execute_time)

        # output is List[SamplerOutput]
        return output