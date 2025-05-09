#!/bin/bash

# 1. Prepare Kernel Performance Environment (.venv-kernel)

bash ./scripts/create_env.sh .venv-kernel
source ./scripts/activate_env.sh .venv-kernel

./scripts/install_h100.sh

./scripts/modify_bitblas.sh # Fix bitblas on small-batch kernel benchmarking senarios

deactivate

# 2. Prepare End-to-end Performance Environment (.venv-end2end)

bash ./scripts/create_env.sh .venv-end2end
source ./scripts/activate_env.sh .venv-end2end

cd third_party/vllm
uv pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/0.6.4/vllm-0.6.4-cp38-abi3-manylinux1_x86_64.whl
python3 python_only_dev.py
cd ../..

./scripts/install_e2e.sh

deactivate

# 3. Prepare BitBLAS End-to-end Performance Environment (.venv-bitblas)

bash ./scripts/create_env.sh .venv-bitblas
source ./scripts/activate_env.sh .venv-bitblas

cd third_party/vllm-bitblas
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev3930+g3703449.d20250102 VLLM_PRECOMPILED_WHEEL_LOCATION=https://vllm-wheels.s3.us-west-2.amazonaws.com/a0f7d53beb176034546c6deb328a3d49e94e1f6d/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl uv pip install -e .
uv pip install bitblas -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ../..

deactivate

# 4. Install llama.cpp

# Select according to GPU compute capability
./scripts/install_llama_cpp.sh 90a

# 5. Download model specifications

./scripts/download_llm_specs.sh
