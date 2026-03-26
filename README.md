<div align="center">

<img src="pickaxe.PNG" alt="MineDraft Logo" width="80">

# MineDraft: A Framework for Batch Parallel Speculative Decoding

<a href="https://arxiv.org/abs/2603.18016" target="_blank"><img src="https://img.shields.io/badge/arXiv-2603.18016-b31b1b.svg?style=for-the-badge" alt="arXiv"></a>
<a href="https://arunv3rma.github.io/blogs/minedraft/minedraft.html" target="_blank"><img src="https://img.shields.io/badge/Project-Blog-green.svg?style=for-the-badge" alt="Project Blog"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![vLLM](https://img.shields.io/badge/plugin-vLLM-blue.svg?style=for-the-badge)](https://github.com/vllm-project/vllm)
[![Python](https://img.shields.io/badge/Python-3.9--3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

<br>

**MineDraft** accelerates large language model inference by *overlapping* the drafting and verification stages of speculative decoding, hiding latency and unlocking substantial throughput gains in batch settings.

<br>

| Metric | Improvement over Standard SD |
|:------:|:-----------------------------:|
| 🚀 Throughput | **up to +75%** |
| ⚡ End-to-end Latency | **up to −39%** |

<br>

</div>

---

## Overview

Speculative decoding (SD) uses a small *draft model* to propose candidate tokens that a larger *target model* then verifies — reducing the number of expensive forward passes. MineDraft leads this paradigm to **parallel execution** by overlapping the drafting and verification stages so that drafting latency is effectively hidden behind verification compute.


> *Experiments across Qwen3, Llama-3.3, and EAGLE models validate MineDraft's gains on ShareGPT, LMSYS Arena, and Spec-Bench benchmarks.*

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Results & Analysis](#results--analysis)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Requirements

### System

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (tested: Ubuntu 22.04) |
| **Python** | 3.9 – 3.12 (tested: 3.12) |
| **CUDA** | ≥ 11.8 (tested: 12.8) |
| **GPUs** | 5× NVIDIA with sufficient VRAM (A100 80GB / H100 / L40 recommended) |

### Core Dependencies

| Package | Version |
|---------|---------|
| [vLLM](https://github.com/vllm-project/vllm) | 0.9.2 |
| PyTorch | 2.7.0 |
| torch-scatter | 2.1.2 |

---

## Installation

**Step 1 — Create a virtual environment** (pick one):

<details>
<summary><b>venv</b></summary>

```bash
python -m venv venv
source venv/bin/activate
```

</details>

<details>
<summary><b>uv</b></summary>

```bash
uv venv --python 3.12 --seed
source venv/bin/activate
```

</details>

<details>
<summary><b>conda</b></summary>

```bash
conda create -n minedraft python=3.12 -y
conda activate minedraft
```

</details>

<br>

**Step 2 — Install vLLM:**

```bash
pip install vllm==0.9.2 --extra-index-url https://download.pytorch.org/whl/cu128
```

**Step 3 — Install MineDraft:**

```bash
pip install -e ".[benchmark]"
```

This installs:
- **Core**: `torch-scatter==2.1.2`
- **Benchmark**: `datasets`, `nvitop`, `pandas`, `numpy`, `matplotlib`, `IPython`, `tqdm`

---

## Dataset Preparation

```bash
mkdir -p benchmarks/datasets
python scripts/convert_datasets.py
```

| Output File | Source |
|-------------|--------|
| `ShareGPT.json` | ShareGPT_V3_unfiltered_cleaned_split |
| `arena.json` | LMSYS Chatbot Arena Conversations |
| `spec_bench.json` | Spec-Bench |
| `tough.json` | Domain-specific tough questions |

---

## Configuration

Experiments use various speculative decoding configurations set via `--speculative-config`:

```jsonc
{
    "method": null,
    // null = standard SD | "eagle" = EAGLE
    "model": "<draft_model>",
    // HuggingFace model ID for draft model
    "draft_tensor_parallel_size": 1,
    // TP size for draft model (always 1)
    "num_speculative_tokens": 5,
    // Number of draft tokens (k)
    "is_parallel": true,
    // Enable PSD (and MineDraft)
    "force_pearl": false,
    // Enable PEARL if is_parallel is true (disables MineDraft)
    "tetris": true,
    // Enable Tetris
    "tetris_turn_on_batch_size": 1,
    // Batch size threshold to activate Tetris
    "tetris_capacity": 0,
    // Tetris capacity, 0 → auto calculated from k × max_num_seqs
    "tetris_extra_proposals": 3
    // Extra draft tokens for Tetris
}
```

### Hardware Layout

| Mode | GPUs | Layout |
|------|------|--------|
| **Parallel** | 5 | 4 for target model TP + 1 for draft model |
| **Sequential** | 4 | All 4 for target model TP; drafter shares resources |

---

## Running Experiments

### Experiment Index

| Script | Model Setup |
|--------|-------------|
| `experiment_1_*.sh` | Qwen3-32B with draft models (0.6B, 1.7B, 4B) |
| `experiment_2_eagle_*.sh` | EAGLE — Vicuna-33B, Vicuna-13B |
| `experiment_2_llama_*.sh` | Llama-3.3-70B-AWQ with Llama-3.1-8B |
| `experiment_3_n_*.sh` | Multi-sample ablation |
| `experiment_4_bs_*.sh` | Batch size ablation (8, 16, 32, 64) |
| `experiment_5_tetris_*.sh` | Tetris VSR analysis |
| `experiment_6_qwen8b.sh` | Qwen3-32B with Qwen3-8B |
| `experiment_7_qwen235b.sh` | Qwen3-235B-A22B-FP8 with Qwen3-14B |
| `experiment_8_nsys.sh` | NVIDIA Nsight Systems profiling |

Each experiment ships with two variants: `*_parallel.sh` (5 GPUs) and `*_sequential.sh` (4 GPUs).

### Run All

```bash
cd scripts
bash run_all.sh        # parallel + sequential
bash run_parallel.sh   # parallel only
bash run_sequential.sh # sequential only
```

### Run Individual

```bash
cd scripts
bash experiment_1_parallel.sh          # Qwen3-32B parallel
bash experiment_2_eagle_sequential.sh  # EAGLE sequential
```

### GPU Bootstrap (optional)

Useful on shared clusters — waits for GPUs to become free before launching:

```bash
# First, comment out the `export CUDA_VISIBLE_DEVICES=` line in the target script, then:
python scripts/bootstrap.py bash scripts/experiment_1_parallel.sh
```

The bootstrap script monitors GPU availability, waits until 5 GPUs are free (<1% memory & utilization), then sets `CUDA_VISIBLE_DEVICES` and launches. You can adjust required GPU count and thresholds in the `main` function.

---

## Results & Analysis

| Artifact | Location |
|----------|----------|
| Benchmark traces | `benchmarks/trace/*.jsonl` |
| Nsight Systems profiling reports | `*.nsys-rep` (project root) |
| Trace analysis notebook | `benchmarks/trace/analyze_plots.ipynb` |
| Trace analysis utilities | `benchmarks/trace/analyze_traces.py` |

---

## Troubleshooting

<details>
<summary><b>Out of Memory (OOM)</b></summary>

- Reduce `--gpu-memory-utilization` (default: `0.65`)
- Reduce `--max-num-seqs` (batch size)
- Switch to a smaller draft or target model

</details>

<details>
<summary><b>CUDA Version Mismatch</b></summary>

Verify your CUDA installation:

```bash
nvcc --version
nvidia-smi
```

MineDraft requires CUDA ≥ 12.8 for the tested configuration.

</details>

<details>
<summary><b>Model Download Issues</b></summary>

Models are automatically downloaded from HuggingFace. Ensure you have:
- Sufficient disk space or quota
- HuggingFace access tokens for gated models (e.g., Llama)

For downloading gated models, run:
```bash
huggingface-cli login
```

</details>

<details>
<summary><b>NVIDIA Nsight Systems — Wrong event order error</b></summary>

If you see:
> *Wrong event order has been detected when adding events to the collection*

Upgrade to Nsight Systems ≥ 2024.2 from the [NVIDIA developer portal](https://developer.nvidia.com/tools-downloads#?search=nsight%20systems%202024.2&tx=$development_platform,linux).

</details>

---

## Citation

If you find MineDraft useful in your research, please cite:

```bibtex
@article{tang2026minedraft,
  title   = {MineDraft: A Framework for Batch Parallel Speculative Decoding},
  author  = {Tang, Zhenwei and Verma, Arun and Zhou, Zijian and Wu, Zhaoxuan
             and Prakash, Alok and Rus, Daniela and Low, Bryan Kian Hsiang},
  journal = {arXiv preprint arXiv:2603.18016},
  year    = {2026}
}
```

---
