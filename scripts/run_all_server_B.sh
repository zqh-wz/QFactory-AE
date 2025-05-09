#!/bin/bash

for task in fig6 fig9 fig10 fig11 fig13 tab3; do
    echo "Running $task"
    bash ./scripts/reproduce.sh "$script"
done
