#!/bin/bash

MODELS=(
    "Llama-2-7B-Chat-GPTQ"
    "Llama-2-13B-Chat-GPTQ"
    "Llama-2-70B-Chat-GPTQ"
    "Qwen2.5-7B-Instruct-GPTQ-Int4"
    "Qwen2.5-14B-Instruct-GPTQ-Int4"
    "Qwen2.5-32B-Instruct-GPTQ-Int4"
    "Qwen2.5-72B-Instruct-GPTQ-Int4"
)

MODELS_2BITS=(
    "Llama-2-13B-Chat-GPTQ"
    "Llama-2-70B-Chat-GPTQ"
    "Qwen2.5-7B-Instruct-GPTQ-Int4"
    "Qwen2.5-14B-Instruct-GPTQ-Int4"
    "Qwen2.5-32B-Instruct-GPTQ-Int4"
    "Qwen2.5-72B-Instruct-GPTQ-Int4"
)

VLLM_ARGS=(
    "--batch-size 1"
    "--input-len 16"
    "--output-len 128"
    "--max_model_len 256"
    "--gpu_memory_utilization 0.95"
    "--quantization qfactory"
)

rm -f qfactory.log

source ../../scripts/activate_env.sh ../../.venv-end2end

for MODEL in "${MODELS[@]}"; do
    srun --gres=gpu:H100:1 python3 ../../third_party/vllm/benchmarks/benchmark_latency.py --model ../../data/llmspecs/$MODEL ${VLLM_ARGS[@]} | tee -a qfactory.log
done
for MODEL in "${MODELS_2BITS[@]}"; do
    srun --gres=gpu:H100:1 python3 ../../third_party/vllm/benchmarks/benchmark_latency.py --model ../../data/llmspecs_2bit/$MODEL ${VLLM_ARGS[@]} | tee -a qfactory.log
done

deactivate
