#!/bin/bash

export QFACTORY_CONFIG=best_config.json
export QFACTORY_ARCH=80

. run_all_qfactory.sh
. run_all_vllm.sh
. run_all_bitblas.sh
. run_all_llama_cpp.sh

source ../../scripts/activate_env.sh ../../.venv-kernel
python3 plot_end2end.py
deactivate
