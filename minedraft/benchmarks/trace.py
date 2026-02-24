from dataclasses import asdict, dataclass
import json
from os import path
from pathlib import Path
from typing import Dict, List, Union
from collections import defaultdict
import torch
"""
One profiling process with same configuration generates one `TraceBundle`.
Each `TraceBundle` contains all collected `Trace`s with different types.
You may define your own trace type with a new class that inherits `Trace`.
This is the only required file for third-party profiling and shouldn't have any cllam dependency.
e.g.
    from cllam.analysis.trace import TRACER, Request
    tid = TRACER.add(trace.Request)
    trace: Request = TRACER.get(tid)
    trace.start_us = time.perf_counter() * 1_000_000
    TRACER.export("my_trace.json")
    TRACER.clear()
"""


@dataclass
class Trace:
    tid: str = None  # <type>:<index>
    type: str = None

    def __post_init__(self):
        self.type = self.__class__.__name__

    def asdict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TraceBundle:
    metadata: Dict = None
    traces: List[Trace] = None

    def __post_init__(self):
        if self.traces:
            self.traces = [
                globals()[trace["type"]](
                    **trace) if isinstance(trace, dict) else trace
                for trace in self.traces
            ]

    def asdict(self):
        return {
            "metadata": self.metadata,
            "traces": [trace.asdict() for trace in self.traces],
        }


@dataclass
class Step(Trace):
    start_us: int = None  # unit:microsec
    end_us: int = None  # unit:microsec
    batch_start_us: int = None  # unit:microsec
    batch_end_us: int = None  # unit:microsec
    is_prompt_run: bool = None
    batched_token_num: int = None
    context_token_num: int = None
    batched_requests: List[str] = None  # List of request trace id.
    preempted_requests: List[str] = None  # List of request trace id.
    available_slots: int = None
    num_blocks_to_swap_in: int = None
    num_blocks_to_swap_out: int = None
    prepare_duration: int = None
    use_cuda_graph: bool = None
    execute_model_duration: int = None
    sample_duration: int = None

    # SD metric
    is_parallelised: bool = False
    proposed_len: int = None
    verify_len: int = None
    accepted_num: int = None
    generated_num: int = None
    predicted_draft_time: int = None
    predicted_target_with_overhead_time: int = None
    predicted_acceptance_rate: float = None
    measured_draft_time: int = None
    measured_target_time: int = None
    measured_overhead_time: int = None
    measured_avg_seq_len: float = None

    match_count: int = None  # only for ngram


@dataclass
class Request(Trace):
    start_us: int = None  # unit:microsec
    end_us: int = None  # unit:microsec
    prompt_len: int = None
    gen_len: int = None


class Tracer:
    TRACE_FOLDER = Path(path.curdir) / "benchmarks" / "trace"

    def __init__(self):
        self.traces: Dict[str, Trace] = {}
        self.type_nums: Dict[str, int] = defaultdict(int)
        self.metadata: Dict = {}
        if not path.exists(self.TRACE_FOLDER):
            Path.mkdir(self.TRACE_FOLDER, parents=True)
        self.current_step = None

        self.acceptance_rates = []

    def add(self, trace_type: type) -> str:
        assert issubclass(trace_type,
                          Trace), f"Invalid trace type: {trace_type}"
        type_name = trace_type.__name__
        tid = f"{type_name}:{self.type_nums[type_name]}"
        self.traces[tid] = trace_type(tid)
        self.type_nums[type_name] += 1
        if trace_type == Step:
            self.current_step = self.traces[tid]
        return tid

    def get(self, tid: int) -> Union[Request, Trace]:
        assert tid in self.traces, f"Invalid trace id: {tid}"
        return self.traces[tid]

    def export(self, filename: str = "trace"):
        # Change all from tensor to list befure the dump
        for trace in self.traces.values():
            if isinstance(trace, Step):
                trace.verify_len = trace.verify_len.item() if isinstance(
                    trace.verify_len, torch.Tensor) else trace.verify_len
                trace.match_count = trace.match_count.item() if isinstance(
                    trace.match_count, torch.Tensor) else trace.match_count
                trace.accepted_num = trace.accepted_num.item() if isinstance(
                    trace.accepted_num, torch.Tensor) else trace.accepted_num
                trace.generated_num = trace.generated_num.item() if isinstance(
                    trace.generated_num, torch.Tensor) else trace.generated_num
                trace.predicted_acceptance_rate = trace.predicted_acceptance_rate.item(
                ) if isinstance(trace.predicted_acceptance_rate,
                                torch.Tensor) else trace.predicted_acceptance_rate

        bundle = TraceBundle(self.metadata, list(self.traces.values()))
        trace_path = path.join(self.TRACE_FOLDER, f"{filename}.jsonl")
        with open(trace_path, "a") as file:
            json.dump(bundle.asdict(), file)
            file.write("\n")
        print(f"Exported cllam trace file at {trace_path}")

        self.traces.clear()
        self.metadata.clear()
        self.type_nums.clear()


TRACER = Tracer()
