# QFactory-AE

## A. Abstract

This repository contains the code for the reproduction of the paper "QFactory: Accelerating Quantized Large Language Model Serving with Qtile Graphs" at USENIX ATC'25.

The reproduction is divided into two parts:

- Kernel performance reproduction (Figure 6, 7, 8, 9, 10; Table 3)
- End-to-end performance reproduction (Figure 11, 12, 13)

## B. Prepare Hardware Environment

To reproduce this work, a GPU server with NVIDIA V100, A100, and H100 GPUs is required.

**For AE Reviewers, please check the HotCRP website for instructions on how to access the provided GPU servers.**

Due to our limited resources, we provide two servers with different GPU configurations: Server A has NVIDIA V100 and A100 GPUs, while server B has NVIDIA H100 GPUs. Therefore, the reproduction need to be performed seperately on both servers.

To avoid issues of environment, we strongly recommend reviewers to use our provided environment.

## C. Prepare Software Environment

**For AE reviewers, please skip this step and use the provided environment.**

### C1. Prepare codebase

Download this repository and its submodules:

```bash
git clone --recursive https://github.com/zqh-wz/QFactory-AE.git
git submodule update --init --recursive
```

Then, apply nessesary patches to submodules.

```bash
# For intergation with vllm
cd third_party/vllm
git apply ../vllm-qfactory.patch
cd ../..

# For fixing bugs and skip weight loading, has been intergrated into codebase due to reasons below
# cd third_party/vllm-bitblas
# git apply ../vllm-bitblas.patch
# cd ../..

# (Only apply to H100) For fixing runtime errors on H100
cd third_party/marlin
git apply ../marlin-hopper.patch
cd ../..

# For skip weight loading and easier benchmarking
cd third_party/llama.cpp
git apply ../llama-cpp.patch
cd ../..
```

**Notes:**

- The commit we previously used for testing BitBLAS's end-to-end performance has been removed from the repository by its authors. Therefore, we directly include them in this repository. The original commit ID was `3703449`.

### C2. Installation

We manage python virtual environments with `uv`. We need three virtual environments to examine the performance:

- `.venv-kernel`: for kernel performance evaluation
- `.venv-end2end`: for end-to-end performance evaluation
- `.venv-bitblas`: for BitBLAS end-to-end performance evaluation

#### 1. Prepare Kernel Performance Environment (`.venv-kernel`)

```bash
bash ./scripts/create_env.sh .venv-kernel
source ./scripts/activate_env.sh .venv-kernel

./scripts/install.sh

./scripts/modify_bitblas.sh # Fix bitblas on small-batch kernel benchmarking senarios

deactivate
```

#### 2. Prepare End-to-end Performance Environment (`.venv-end2end`)

```bash
bash ./scripts/create_env.sh .venv-end2end
source ./scripts/activate_env.sh .venv-end2end

cd third_party/vllm
uv pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/0.6.4/vllm-0.6.4-cp38-abi3-manylinux1_x86_64.whl
python3 python_only_dev.py
cd ../..

./scripts/install_e2e.sh

deactivate
```

#### 3. Prepare BitBLAS End-to-end Performance Environment (`.venv-bitblas`)

```bash
bash ./scripts/create_env.sh .venv-bitblas
source ./scripts/activate_env.sh .venv-bitblas

cd third_party/vllm-bitblas
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev3930+g3703449.d20250102 VLLM_PRECOMPILED_WHEEL_LOCATION=https://vllm-wheels.s3.us-west-2.amazonaws.com/a0f7d53beb176034546c6deb328a3d49e94e1f6d/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl uv pip install -e .
uv pip install bitblas -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ../..

deactivate
```

#### 4. Install llama.cpp

```bash
# Select according to GPU compute capability
./scripts/install_llama_cpp.sh 80
./scripts/install_llama_cpp.sh 90a
```

#### 5. Download model specifications

```bash
./scripts/download_llm_specs.sh
```

## D. Reproduce Experimental Results

### D1. Reproduce V100 and A100 Results (Server A)

First, connect to the server with V100 and A100 GPUs.

#### Figure 7: Kernel performance on A100

```bash
./scripts/reproduce.sh fig7
```

#### Figure 8: Kernel performance on V100

```bash
./scripts/reproduce.sh fig8
```

#### Figure 12: End-to-end performance on A100

```bash
./scripts/reproduce.sh fig12
```

### D2. Reproduce H100 Results (Server B)

First, connect to the server with H100 GPUs.

#### Figure 6: Kernel performance on H100

```bash
./scripts/reproduce.sh fig6
```

#### Figure 9: Scaling matrix sizes

```bash
./scripts/reproduce.sh fig9
```

#### Figure 10: Scaling bit-widths

```bash
./scripts/reproduce.sh fig10
```

#### Figure 11: End-to-end performance on H100

```bash
./scripts/reproduce.sh fig11
```

#### Figure 13: Performance breakdown

```bash
./scripts/reproduce.sh fig13
```

#### Table 3: Varing batch size

```bash
./scripts/reproduce.sh tab3
```
