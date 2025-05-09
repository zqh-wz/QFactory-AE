#!/bin/bash

export QFACTORY_CONFIG=best_config.json
export QFACTORY_ARCH=90a
source ../../scripts/activate_env.sh ../../.venv-kernel

srun --gres=gpu:H100:1 -c 32 python3 bench_kernels.py

deactivate
