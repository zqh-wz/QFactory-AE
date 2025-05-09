#!/bin/bash

export QFACTORY_CONFIG=best_config.json
export QFACTORY_ARCH=70
source ../../scripts/activate_env.sh ../../.venv-kernel

srun --gres=gpu:v100:1 -w ja1 -p long -c 32 python3 bench_kernels.py

deactivate
