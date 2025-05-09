#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <CUDA_ARCHITECTURES>"
    exit 1
fi

CUDA_ARCHITECTURES=$1

spack load cmake

cd third_party/llama.cpp
git apply ../llama-cpp.patch

# build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Release -j16

# download base model
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q4_0.gguf --local-dir ./models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q4_0*" --local-dir ./models
./build/bin/llama-gguf-split --merge models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf models/qwen2.5-7b-instruct-q4_0.gguf
