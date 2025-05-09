#!/bin/bash

export QFACTORY_CONFIG=best_config.json
export QFACTORY_ARCH=90a

. run_all_qfactory.sh
. run_all_vllm.sh
. run_all_bitblas.sh
. run_all_llama_cpp.sh
