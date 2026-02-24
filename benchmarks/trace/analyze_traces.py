import json
import os
import re

import numpy as np


def load(filename):
    bundles = []
    with open(filename, "r") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {i} in {filename}: {e}")
                continue
            bundles.append(data["traces"])
    return bundles


def get_req_ttfts(request_traces, step_traces, req_tid_to_idx):
    req_ttfts = np.zeros(len(request_traces), dtype=np.float64)
    for trace in request_traces:
        i = req_tid_to_idx[trace["tid"]]
        if req_ttfts[i] == 0 or trace["start_us"] < req_ttfts[i]:
            req_ttfts[i] = trace["start_us"]

    processed = set()
    prompt_runs = 0
    for step_trace in step_traces:
        if not step_trace["is_prompt_run"]:
            continue
        prompt_runs += 1
        for r in step_trace["batched_requests"]:
            processed.add(r)
            i = req_tid_to_idx[r]
            req_ttfts[i] = (step_trace["end_us"] - req_ttfts[i]) / 1e6
    assert len(processed) == len(request_traces), (
        f"When calculating TTFTs, processed {len(processed)} requests, "
        f"but expected {len(request_traces)}. Some requests may finished "
        "without being executed due to insufficient memory"
    )
    return req_ttfts, prompt_runs


def get_request_latency(request_traces, req_tid_to_idx):
    request_latencies = np.zeros(len(request_traces), dtype=np.float64)
    num_steps = 0
    for trace in request_traces:
        num_steps += 1
        i = req_tid_to_idx[trace["tid"]]
        request_latencies[i] += (trace["end_us"] - trace["start_us"]) / 1e6
    return request_latencies, num_steps


def get_req_exec_times(step_traces, req_tid_to_idx):
    req_exec_time = np.zeros(len(req_tid_to_idx), dtype=np.float64)
    for trace in step_traces:
        step_duration = (trace["end_us"] - trace["start_us"]) / 1e6
        if "batched_requests" not in trace:
            # print('No batched_requests in trace', trace)
            continue
        for r in trace["batched_requests"]:
            i = req_tid_to_idx[r]
            req_exec_time[i] += step_duration
    return req_exec_time


def get_step_stats(step_traces, prompt_runs):
    drafted = np.zeros(len(step_traces) - prompt_runs, dtype=np.uint64)
    verified = np.zeros(len(step_traces) - prompt_runs, dtype=np.uint64)
    accepted = np.zeros(len(step_traces) - prompt_runs, dtype=np.uint64)
    generated = np.zeros(len(step_traces) - prompt_runs, dtype=np.uint64)
    preempted = np.zeros(len(step_traces) - prompt_runs, dtype=np.uint64)
    step_generation_times = np.zeros(len(step_traces) - prompt_runs, dtype=np.float64)
    i = 0
    for trace in step_traces:
        if not trace["is_prompt_run"]:
            drafted[i] = trace["proposed_len"]
            verified[i] = trace["verify_len"]
            accepted[i] = trace["accepted_num"]
            generated[i] = trace["generated_num"]
            preempted[i] = len(trace.get("preempted_requests", []))
            step_generation_times[i] = (trace["end_us"] - trace["start_us"]) / 1e6
            i += 1
    return drafted, verified, accepted, generated, preempted, step_generation_times


def analyze(filename):
    patt = r'^input=\d+_(.*)_(.*)_(.*)_(\d+)_(False|True)_parallel=(False|True|pearl)_k=(\d+)_t=(\d+)_n=(\d+)(?:_c=(\d+))?_warmup=(\d+)_runs=\d+.jsonl$'
    matched = re.match(patt, os.path.basename(filename))
    assert matched is not None
    dataset = matched.group(1)
    target_model = matched.group(2)
    draft_model = matched.group(3)
    batch_size = int(matched.group(4))
    tetris = matched.group(5) == 'True'
    is_parallel = matched.group(6) in ('True', 'pearl')
    is_pearl = is_parallel and matched.group(6) == "pearl"
    eagle = "eagle3" if "eagle3" in draft_model.lower() else "eagle" if "eagle" in draft_model.lower() else None
    k = int(matched.group(7))
    e = int(matched.group(8))
    n = int(matched.group(9))
    c = int(matched.group(10) or 0)
    warmup = int(matched.group(11))

    if is_pearl:
        if tetris:
            if eagle is not None:
                method = f"pearl_tetris_{eagle}"
            else:
                method = "pearl_tetris"
        else:
            if eagle is not None:
                method = f"pearl_{eagle}"
            else:
                method = "pearl_sd"
    elif is_parallel:
        if tetris:
            if eagle is not None:
                method = f"ptetris_{eagle}"
            else:
                method = "ptetris"
        else:
            if eagle is not None:
                method = f"p{eagle}"
            else:
                method = "psd"
    else:
        if tetris:
            if eagle is not None:
                method = f"tetris_{eagle}"
            else:
                method = "tetris"
        else:
            if eagle is not None:
                method = eagle
            else:
                method = "sd"

    # NOTE: batch_start_us and batch_end_us are closer to the points where vLLM
    # is about to execute the batch and where the execution is done, 
    # respectively. Whereas start_us and end_us are closer to the points where 
    # vLLM pass outputs to the user, i.e., there are overheads in between to do
    # logging, tracing, serialization, etc.

    bundles = load(filename)
    for i, bundle in enumerate(bundles):
        step_traces = [t for t in bundle if t["type"] == "Step"]
        request_traces = [
            t for t in bundle
            if t["type"] == "Request" and "end_us" in t
        ]
        req_tid_to_idx = {t["tid"]: i for i, t in enumerate(request_traces)}
        assert len(request_traces) == len(req_tid_to_idx), (
            f"Duplicate request TIDs found in bundle {i} of "
            f"trace file {filename}."
        )

        # max(end_us among all requests) - min(start_us among all requests)
        total_latency = (
            max(request['end_us'] for request in request_traces) -
            min(request['start_us'] for request in request_traces)) / 1e3
        # #requests / total elapsed time
        request_throughput = len(request_traces) / total_latency

        # end_us of the first step of a request - start_us of that request
        req_ttfts, prompt_runs = get_req_ttfts(
            request_traces, step_traces, req_tid_to_idx)
        # (end_us - start_us) of a request
        req_latencies, _ = get_request_latency(request_traces, req_tid_to_idx)
        # Sum of (end_us - start_us) of steps whose batch contains the request
        req_exec_times = get_req_exec_times(step_traces, req_tid_to_idx)
        req_wait_times = np.maximum(
            req_latencies - req_exec_times,
            np.zeros_like(req_latencies)
        )

        (
            step_drafted_tokens,  # proposed_len
            step_verified_tokens, # verify_len
            step_accepted_tokens, # accepted_num
            step_generated_tokens, # generated_num
            step_preempted_requests, # len(preempted_requests)
            step_generation_times # (end_us - start_us)
        ) = get_step_stats(step_traces, prompt_runs)

        bundles[i] = {
            "warmup": i < warmup,
            "dataset": dataset,
            "target_model": target_model,
            "draft_model": draft_model,
            "batch_size": batch_size,
            "method": method,
            "k": k,
            "e": e,
            "n": n,
            "c": c,
            "eagle": eagle,
            "reqs": (len(request_traces) // n) if n > 1 else len(request_traces),
            "total_latency": total_latency,
            "request_throughput": request_throughput,
            "req_ttfts": req_ttfts,
            "req_latencies": req_latencies,
            "req_exec_times": req_exec_times,
            "req_wait_times": req_wait_times,
            "step_generation_times": step_generation_times,
            "step_drafted_tokens": step_drafted_tokens,
            "step_verified_tokens": step_verified_tokens,
            "step_accepted_tokens": step_accepted_tokens,
            "step_generated_tokens": step_generated_tokens,
            "step_preempted_requests": step_preempted_requests
        }
    return bundles
