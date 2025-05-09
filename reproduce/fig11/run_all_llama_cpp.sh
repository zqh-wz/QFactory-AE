#!/bin/bash

GPUSPEC='--gres=gpu:H100:1'

rm -f llama_cpp.log

export MOCK=4

export BLOCK_COUNT=32
export EMBEDDING_LENGTH=4096
export FEED_FORWARD_LENGTH=11008
export HEAD_COUNT=32
export HEAD_COUNT_KV=32

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/llama-2-7b-chat.Q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=40
export EMBEDDING_LENGTH=5120
export FEED_FORWARD_LENGTH=13824
export HEAD_COUNT=40
export HEAD_COUNT_KV=40

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/llama-2-7b-chat.Q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=80
export EMBEDDING_LENGTH=8192
export FEED_FORWARD_LENGTH=28672
export HEAD_COUNT=64
export HEAD_COUNT_KV=8

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/llama-2-7b-chat.Q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

# ================================================================================

export BLOCK_COUNT=28
export CONTEXT_LENGTH=131072
export EMBEDDING_LENGTH=3584
export FEED_FORWARD_LENGTH=18944
export HEAD_COUNT=28
export HEAD_COUNT_KV=4

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/qwen2.5-7b-instruct-q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=48
export CONTEXT_LENGTH=131072
export EMBEDDING_LENGTH=5120
export FEED_FORWARD_LENGTH=13824
export HEAD_COUNT=40
export HEAD_COUNT_KV=8

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/qwen2.5-7b-instruct-q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=64
export CONTEXT_LENGTH=131072
export EMBEDDING_LENGTH=5120
export FEED_FORWARD_LENGTH=27648
export HEAD_COUNT=40
export HEAD_COUNT_KV=8

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/qwen2.5-7b-instruct-q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=80
export CONTEXT_LENGTH=32768
export EMBEDDING_LENGTH=8192
export FEED_FORWARD_LENGTH=29696
export HEAD_COUNT=64
export HEAD_COUNT_KV=8

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/qwen2.5-7b-instruct-q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log


# ================================================================================


export MOCK=2

export BLOCK_COUNT=32
export EMBEDDING_LENGTH=4096
export FEED_FORWARD_LENGTH=11008
export HEAD_COUNT=32
export HEAD_COUNT_KV=32

# srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/llama-2-7b-chat.Q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=40
export EMBEDDING_LENGTH=5120
export FEED_FORWARD_LENGTH=13824
export HEAD_COUNT=40
export HEAD_COUNT_KV=40

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/llama-2-7b-chat.Q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=80
export EMBEDDING_LENGTH=8192
export FEED_FORWARD_LENGTH=28672
export HEAD_COUNT=64
export HEAD_COUNT_KV=8

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/llama-2-7b-chat.Q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

# ================================================================================

export BLOCK_COUNT=28
export CONTEXT_LENGTH=131072
export EMBEDDING_LENGTH=3584
export FEED_FORWARD_LENGTH=18944
export HEAD_COUNT=28
export HEAD_COUNT_KV=4

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/qwen2.5-7b-instruct-q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=48
export CONTEXT_LENGTH=131072
export EMBEDDING_LENGTH=5120
export FEED_FORWARD_LENGTH=13824
export HEAD_COUNT=40
export HEAD_COUNT_KV=8

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/qwen2.5-7b-instruct-q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=64
export CONTEXT_LENGTH=131072
export EMBEDDING_LENGTH=5120
export FEED_FORWARD_LENGTH=27648
export HEAD_COUNT=40
export HEAD_COUNT_KV=8

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/qwen2.5-7b-instruct-q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log

export BLOCK_COUNT=80
export CONTEXT_LENGTH=32768
export EMBEDDING_LENGTH=8192
export FEED_FORWARD_LENGTH=29696
export HEAD_COUNT=64
export HEAD_COUNT_KV=8

srun $GPUSPEC ../../third_party/llama.cpp/build/bin/llama-bench -m ../../third_party/llama.cpp/models/qwen2.5-7b-instruct-q4_0.gguf -b 1 -n 128 | tee -a llama_cpp.log
