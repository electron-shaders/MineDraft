"""Benchmark the latency of processing a single batch of requests."""
import argparse
import dataclasses
import glob
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm.plugins import load_general_plugins

os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
_, config_path = tempfile.mkstemp(suffix=".json")
os.environ["VLLM_LOGGING_CONFIG_PATH"] = config_path
with open(config_path, "w") as f:
    json.dump({
        "formatters": {
            "vllm": {
                "class": "vllm.logging.NewLineFormatter",
                "datefmt": "%m-%d %H:%M:%S",
                "format": "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "vllm",
                "level": "INFO",
                "mode": "w",
                "filename": os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "vllm_benchmark.log"
                )
            },
            "stdout": {
                "class" : "logging.StreamHandler",
                "formatter": "vllm",
                "level": "INFO",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "vllm": {
                "handlers": ["stdout"],
                "level": "DEBUG",
                "propagate": True
            },
        },
        "root": {
            "handlers": ["file"],
            "level": "DEBUG"
        },
        "version": 1
    }, f, indent=4)


def sample_requests(tokenizer: PreTrainedTokenizerBase,
                    args: argparse.Namespace) -> List[str]:
    dataset_path: str = args.dataset
    num_requests: int = args.num_prompts
    input_len: int = args.input_len
    max_len: int = args.max_model_len
    assert input_len >= 4, "input_len too small"
    output_len: int = args.output_len
    assert output_len >= 4, "output_len too small"
    assert input_len + output_len <= max_len, "sequence too long"

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    prompts: List[str] = []
    for data in dataset:
        if len(prompts) == num_requests:
            break
        if "image" in data:
            continue

        # Only keep the first two turns of each conversation.
        prompt = data["conversations"][0]["value"]

        # Tokenize the prompts to ensure they are of at least input_len tokens.
        prompt_token_ids = tokenizer(prompt).input_ids
        if (
            len(prompt_token_ids) < input_len
            or len(prompt_token_ids) > max_len - output_len
        ):
            continue

        prompts.append(prompt)
    return prompts


def main(args: argparse.Namespace):
    print(args)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(script_dir, "trace")
    if args.output_trace_file and os.path.exists(os.path.join(trace_dir, args.output_trace_file)):
        print(f"Benchmark is skipped since the following trace file already exists:\n{args.output_trace_file}")
        return

    if args.speculative_config.get('is_parallel', False):
        if args.speculative_config.get('force_pearl', False):
            parallel = "pearl"
        else:
            parallel = True
    else:
        parallel = False

    matched = glob.glob(
        f"input={args.input_len}_"
        f"{os.path.splitext(os.path.basename(args.dataset))[0]}_"
        f"{os.path.basename(args.model)}_"
        f"{os.path.basename(args.speculative_config['model'])}_"
        f"{args.max_num_seqs}_"
        f"{args.speculative_config.get('tetris', False)}_"
        f"parallel={parallel}_"
        f"k={args.speculative_config.get('num_speculative_tokens')}_"
        f"t={args.speculative_config.get('tetris_extra_proposals', 0)}_"
        f"n={args.n}_"
        "*.jsonl",
        root_dir=trace_dir
    )
    if matched:
        print(f"Benchmark is skipped since the following trace files already exist:\n{', '.join(matched)}")
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import PromptType
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM(**dataclasses.asdict(engine_args))

    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    if args.dataset is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)
        prompts = sample_requests(tokenizer, args)
    else:
        dummy_prompt_token_ids = np.random.randint(10000,
                                                size=(args.num_prompts,
                                                      args.input_len))
        prompts: List[PromptType] = [{
            "prompt_token_ids": batch
        } for batch in dummy_prompt_token_ids.tolist()]


    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                llm.generate(prompts,
                             sampling_params=sampling_params,
                             use_tqdm=False)
            print(p.key_averages())
        else:
            llm.generate(prompts,
                         sampling_params=sampling_params,
                         use_tqdm=False)
        llm.llm_engine.dump(
            args.output_trace_file or
            f"input={args.input_len}_"
            f"{os.path.splitext(os.path.basename(args.dataset))[0]}_"
            f"{os.path.basename(args.model)}_"
            f"{os.path.basename(args.speculative_config['model'])}_"
            f"{args.max_num_seqs}_"
            f"{args.speculative_config.get('tetris', False)}_"
            f"parallel={parallel}_"
            f"k={args.speculative_config.get('num_speculative_tokens')}_"
            f"t={args.speculative_config.get('tetris_extra_proposals', 0)}_"
            f"n={args.n}_"
            f"c={args.speculative_config.get('tetris_capacity', 0)}_"
            f"warmup={args.num_iters_warmup}_runs={args.num_iters}"
        )


    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = Path(
                "."
            ) / "vllm_benchmark_result" / f"latency_result_{time.time()}"
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        return

    # Benchmark.
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        run_to_completion(profile_dir=None)


if __name__ == '__main__':
    from vllm.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset. The dataset is expected to "
                        "be a json in form of List[Dict[..., conversations: "
                        "List[Dict[..., value: <prompt_or_response>]]]]")
    parser.add_argument("--input-len",
                        type=int,
                        required=True,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        required=True,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.0,
                        help='Sampling temperature.')
    parser.add_argument('--top-p',
                        type=float,
                        default=1.0,
                        help='Top-p sampling parameter.')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument('--output-log-file',
                        type=str,
                        default=None,
                        help='Path to save the vLLM metrics log file.')
    parser.add_argument('--output-trace-file',
                        type=str,
                        default=None,
                        help='Path to save the SD trace file.')

    from vllm.engine.arg_utils import EngineArgs
    parser = EngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.speculative_config and args.speculative_config.get("is_parallel"):
        load_general_plugins()
    try:
        main(args)
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)
        default_log = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "vllm_benchmark.log"
        )
        if args.output_log_file:
            if os.path.exists(default_log):
                target_dir = os.path.dirname(args.output_log_file)
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)

                import shutil
                shutil.move(default_log, args.output_log_file)
        elif os.path.exists(default_log):
            os.remove(default_log)
