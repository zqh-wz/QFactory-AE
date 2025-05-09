#!/bin/bash

# Check if the argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 'fig6'"
    exit 1
fi

echo "Starting at: $(date)"

cd reproduce/$1
bash run.sh
cd -

echo "Completed at: $(date)"
