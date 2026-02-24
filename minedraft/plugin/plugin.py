from .config import SpeculativeConfigPatch, VllmConfigPatch
from .core.scheduler import (
    SchedulerModulePatch,
    SchedulerOutputsPatch,
    SchedulerPatch,
    SchedulerRunningOutputsPatch,
)
from .distributed.communication_op import CommunicationOpModulePatch
from .distributed.parallel_state import GroupCoordinatorPatch, ParallelStateModulePatch
from .engine.async_llm_engine import AsyncLLMEnginePatch
from .engine.llm_engine import LLMEnginePatch
from .model_executor.layers.rejection_sampler import RejectionSamplerPatch
from .model_executor.layers.spec_decode_base_sampler import SpecDecodeBaseSamplerPatch
from .model_executor.models.eagle import EAGLEPatch
from .sequence import HiddenStatesPatch, SequenceGroupPatch
from .spec_decode.interfaces import SpeculativeProposalsPatch
from .spec_decode.metrics import AsyncMetricsCollectorPatch
from .spec_decode.mqa_scorer import MQAScorerPatch
from .spec_decode.spec_decode_worker import (
    SpecDecodeWorkerModulePatch,
    SpecDecodeWorkerPatch,
)
from .spec_decode.top1_proposer import Top1ProposerPatch
from .spec_decode.util import TimerPatch
from .worker.model_runner import GPUModelRunnerBasePatch, ModelRunnerPatch
from .worker.worker import WorkerModulePatch, WorkerPatch
from .worker.worker_base import LocalOrDistributedWorkerBasePatch


def minedraft_plugin():
    # Apply patches in distributed module
    GroupCoordinatorPatch.apply_patch()
    ParallelStateModulePatch.apply_patch()
    CommunicationOpModulePatch.apply_patch()

    # Apply patches for CLI arguments and configs
    SpeculativeConfigPatch.apply_patch()
    VllmConfigPatch.apply_patch()

    # Apply patches for worker module
    LocalOrDistributedWorkerBasePatch.apply_patch()
    WorkerModulePatch.apply_patch()
    WorkerPatch.apply_patch()
    GPUModelRunnerBasePatch.apply_patch()
    ModelRunnerPatch.apply_patch()

    # Apply patches in model_executor module
    SpecDecodeBaseSamplerPatch.apply_patch()
    RejectionSamplerPatch.apply_patch()
    EAGLEPatch.apply_patch()

    # Apply patches in sequence module
    SequenceGroupPatch.apply_patch()
    HiddenStatesPatch.apply_patch()

    # Apply patches for scheduler
    SchedulerOutputsPatch.apply_patch()
    SchedulerRunningOutputsPatch.apply_patch()
    SchedulerModulePatch.apply_patch()
    SchedulerPatch.apply_patch()

    # Apply patches for engines
    LLMEnginePatch.apply_patch()
    AsyncLLMEnginePatch.apply_patch()

    # Apply patches in spec_decode module
    TimerPatch.apply_patch()
    AsyncMetricsCollectorPatch.apply_patch()
    SpeculativeProposalsPatch.apply_patch()
    MQAScorerPatch.apply_patch()
    Top1ProposerPatch.apply_patch()
    SpecDecodeWorkerModulePatch.apply_patch()
    SpecDecodeWorkerPatch.apply_patch()