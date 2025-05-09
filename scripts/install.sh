#!/bin/bash

# Install dependencies
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install Marlin
cd third_party/marlin
TORCH_CUDA_ARCH_LIST="7.0 8.0 9.0a" uv pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation
cd ../..

# Install QFactory
uv pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple --reinstall

# Test the installation
python3 tests/import_test.py
