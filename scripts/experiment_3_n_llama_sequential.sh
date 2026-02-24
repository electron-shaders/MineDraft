#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Model configuration
TARGET="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
DRAFT="meta-llama/Llama-3.1-8B-Instruct"

# Experiment configuration
WARMUP=1
REPEAT=3
METHOD=null # null or \"eagle\"
num_prompts=1000
CAPACITY=0 # 256
batch_size=64

for N in 1 2
do
    if [ "$N" -gt 1 ]; then
        TEMPERATURE=0.8
    else
        TEMPERATURE=1.0
    fi

    for dataset in "ShareGPT.json" "arena.json" "spec_bench.json" "tough.json"
    do
        if [ "$num_prompts" -gt 1000 ] && { [ "$dataset" = "spec_bench.json" ] || [ "$dataset" = "tough.json" ]; }; then
            continue
        fi
        if [ $dataset == "ShareGPT.json" ]; then
            input_len=256
        elif [ $dataset == "spec_bench.json" ]; then
            input_len=4
        else
            input_len=16
        fi

        for k in 1 2 3 4 5
        do
            # Sequential
            python benchmarks/benchmark_psd.py \
                --dataset benchmarks/datasets/$dataset \
                --max-num-seqs $batch_size \
                --gpu-memory-utilization 0.65 \
                --n $N \
                --temperature $TEMPERATURE \
                --max-model-len 2048 \
                --num-prompts $num_prompts \
                --model  $TARGET \
                --tensor-parallel-size 4 \
                --disable-async-output-proc \
                --enforce-eager \
                --use-v2-block-manager \
                --no-enable-chunked-prefill \
                --speculative-config '{
                    "method": '$METHOD',
                    "model": "'$DRAFT'",
                    "draft_tensor_parallel_size": 1,
                    "num_speculative_tokens": '$k'
                }' \
                --input-len $input_len \
                --output-len 256 \
                --num-iters-warmup $WARMUP \
                --num-iters $REPEAT
        done
    done
done
