#!/bin/bash

# Check if the argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 'venv_name'"
    exit 1
fi

# Prepare the environment
uv venv $1 --python 3.12
