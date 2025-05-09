#!/bin/bash

export QFACTORY_CONFIG=best_config.json
export QFACTORY_ARCH=80
source ../../scripts/activate_env.sh ../../.venv-kernel

srun --gres=gpu:a100:1 -w octave -p long -c 32 python3 bench_kernels.py

deactivate
