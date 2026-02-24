#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,0,1,2,3

# Model configuration
TARGET="Qwen/Qwen3-32B"
DRAFT="Qwen/Qwen3-8B"

# Experiment configuration
WARMUP=1
REPEAT=2
N=2
TEMPERATURE=0.8
METHOD=null # null or \"eagle\"
num_prompts=1000
PEARL=false # true
CAPACITY=0 # 256
batch_size=64

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
        # Parallel
        python benchmarks/benchmark_psd.py \
            --dataset benchmarks/datasets/$dataset \
            --max-num-seqs $batch_size \
            --gpu-memory-utilization 0.65 \
            --n $N \
            --temperature $TEMPERATURE \
            --max-model-len 2048 \
            --num-prompts $num_prompts \
            --model  $TARGET \
            --tensor-parallel-size 5 \
            --disable-async-output-proc \
            --enforce-eager \
            --use-v2-block-manager \
            --no-enable-chunked-prefill \
            --speculative-config '{
                "method": '$METHOD',
                "model": "'$DRAFT'",
                "draft_tensor_parallel_size": 1,
                "num_speculative_tokens": '$k',
                "is_parallel": true,
                "force_pearl": '$PEARL'
            }' \
            --input-len $input_len \
            --output-len 256 \
            --num-iters-warmup $WARMUP \
            --num-iters $REPEAT
    done
done