#!/bin/bash

# Check if the argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 'venv_name'"
    exit 1
fi

# Prepare the environment
spack load cuda@12.4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
source $1/bin/activate
