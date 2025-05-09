#!/bin/bash

for task in fig7 fig8 fig12; do
    echo "Running $task"
    bash ./scripts/reproduce.sh "$script"
done
